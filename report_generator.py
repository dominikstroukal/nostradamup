"""
report_generator.py
===================
Generuje čtvrtletní prognostický report ve stylu MUP:
  - Fan charty v barvách mup.cz
  - Automatický textový komentář přes Anthropic API (claude-sonnet-4-20250514)
  - Výstup: PNG grafy + Markdown/PDF report

Spuštění:
    python report_generator.py

Závislosti:
    pip install anthropic pandas matplotlib numpy
    export ANTHROPIC_API_KEY="tvuj_klic"
"""

import os
import json
import textwrap
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────
# MUP DESIGN SYSTÉM
# Barvy dle mup.cz (navy, mid-blue, accent, šedá)
# ─────────────────────────────────────────────

MUP = {
    "navy":      "#00205B",
    "blue":      "#003DA5",
    "blue_mid":  "#2255A4",
    "blue_3":    "#3A6FBF",
    "blue_4":    "#5B8DB8",
    "blue_5":    "#7BAFD4",
    "gray_dark": "#4A4A4A",
    "gray_mid":  "#9E9E9E",
    "gray_light":"#EEF2FA",
    "white":     "#FFFFFF",
}

_BLUE = "#003DA5"
VAR_COLORS = {
    "gdp_qoq":   {"line": _BLUE, "fan": _BLUE, "hist": MUP["navy"]},
    "gdp_yoy":   {"line": _BLUE, "fan": _BLUE, "hist": MUP["navy"]},
    "hicp_yoy":  {"line": _BLUE, "fan": _BLUE, "hist": MUP["navy"]},
    "cpi_yoy":   {"line": _BLUE, "fan": _BLUE, "hist": MUP["navy"]},
    "wages_yoy": {"line": _BLUE, "fan": _BLUE, "hist": MUP["navy"]},
}

LABELS_CZ = {
    "gdp_qoq":   "HDP – mezikvartální růst (%)",
    "gdp_yoy":   "HDP – meziroční růst (%)",
    "hicp_yoy":  "Inflace HICP – meziroční změna (%)",
    "cpi_yoy":   "Inflace CPI ČSÚ – meziroční změna (%)",
    "wages_yoy": "Průměrné mzdy – meziroční růst (%)",
}

# ─────────────────────────────────────────────
# Pomocná funkce: AR(2) prognóza (bez statsmodels)
# Po instalaci statsmodels nahraď VAR modelem z var_model.py
# ─────────────────────────────────────────────

def _current_quarter_start() -> pd.Timestamp:
    """Vrátí začátek aktuálního čtvrtletí."""
    today = pd.Timestamp.now()
    q_month = ((today.month - 1) // 3) * 3 + 1
    return pd.Timestamp(today.year, q_month, 1)


def _extend_to_present(series: pd.Series) -> pd.Series:
    """Doplní sérii flat-forward do aktuálního čtvrtletí."""
    s = series.dropna()
    if len(s) == 0:
        return series
    last = s.index[-1]
    current_q = _current_quarter_start()
    if last >= current_q:
        return series
    fill_idx = pd.date_range(start=last + pd.offsets.QuarterBegin(1),
                             end=current_q, freq="QS")
    fill = pd.Series(s.iloc[-1], index=fill_idx, name=series.name)
    return pd.concat([series, fill]).sort_index()



def ar_forecast(
    series: pd.Series,
    steps: int = 8,
    n_sims: int = 2000,
    p: int = 2,
    # Měnověpolitická zpětná vazba (inflace)
    is_inflation: bool = False,
    pribor_path: list | None = None,
    neutral_rate: float = 3.0,
    inflation_target: float = 2.0,
    mp_sensitivity: float = 0.25,
    mp_lag: int = 4,
    # DSGE zpětné vazby (cross-variable)
    wages_path: list | None = None,       # mzdy ovlivňují inflaci (cost-push)
    unempl_path: list | None = None,      # nezaměstnanost ovlivňuje mzdy (Phillipsova křivka)
    gdp_path: list | None = None,         # HDP gap ovlivňuje inflaci a nezaměstnanost
    is_wages: bool = False,
    # Dynamická IS křivka (poptávkový kanál měnové politiky)
    is_gdp: bool = False,                 # HDP: reaguje na reálnou sazbu (dynamická IS)
    gdp_cumulative: bool = False,         # True pro YoY sérii (kumuluje QoQ efekt přes 4Q)
    inflation_path: list | None = None,   # očekávaná inflace pro ex-ante reálnou sazbu
    real_neutral: float = 1.0,            # přirozená reálná sazba r* (ČNB odhad ~1 %)
    potential_gdp_qoq: float = 0.55,      # potenciální růst HDP QoQ (~2.2 % ročně, ČR)
    is_sensitivity: float = 0.05,         # citlivost HDP QoQ na 1 pp reálné mezery
                                          # (umírněná, ČNB-style; backtest: kanál
                                          # nezlepšuje bodovou predikci HDP - IS puzzle -
                                          # ale je nutný pro koherentní transmisi a scénáře)
    is_lag: int = 4,                      # transmise do poptávky: 1-4Q (ČNB: 12-18 měsíců)
    is_unempl: bool = False,
    wages_infl_pass: float = 0.08,        # 1 pp růstu mezd -> +0.08 pp inflace (cost-push)
    phillips_slope: float = 0.15,         # 1 pp nižší nezaměstnanost -> +0.15 pp mzdy
    phillips_convexity: float = 0.8,      # konvexita PC: zestrmení sklonu při napjatém trhu (0=lineární)
    nairu: float = 2.8,                   # strukturální míra nezaměstnanosti (NAIRU) pro měření napjatosti
    okun_coef: float = 0.4,               # 1 pp nižší HDP -> +0.4 pp nezaměstnanosti
    # Kanál inflačních očekávání (hybridní novokeynesiánská Phillipsova křivka)
    expect_weight: float = 0.35,          # γ_f: váha dopředu hledících očekávání (0–1)
    anchoring: float = 0.75,              # λ: ukotvenost očekávání k jádru (0=de-ukotveno, 1=plně ukotveno)
    # Lepkavé jádro inflace (Stock-Watson trend + Galí-Gertler náklady)
    core_persistence: float = 0.85,       # ρ: perzistence jádra (lepkavost služeb/nájmů)
    housing_services_pressure: float = 0.5,  # domácí tlak (nemovitosti+služby) v pp nad cíl
    wage_norm: float = 4.0,               # mzdový růst konzistentní s 2% inflací (cíl+produktivita)
    wage_to_core: float = 0.15,           # přenos nadměrného mzdového růstu do jádra
    # Kurzový pass-through (ERPT)
    eurczk_path: list | None = None,      # prognóza EUR/CZK na 'steps' čtvrtletí
    erpt_coef: float = 0.15,              # pass-through: % změny CPI na 1 % oslabení koruny (za rok)
    erpt_lag: int = 4,                    # rozložení efektu přes čtvrtletí
    erpt_state: float = 0.02,             # stavová závislost: násobek koef. dle velikosti depreciace
    extend: bool = True,                  # natáhnout sérii do současnosti (False pro backtest)
) -> pd.DataFrame:
    """
    AR(p) prognóza s DSGE-inspirovanými zpětnými vazbami:

    HDP:
      <- dynamická IS křivka (reálná sazba nad r* => slabší poptávka, lag 1-4Q)

    Inflace:
      <- měnová politika (PRIBOR > neutrál => dezinflace, lag 4Q)
      <- mzdový cost-push (vyšší mzdy => vyšší inflace, lag 1Q)
      <- HDP gap (Phillipsova křivka – vyšší output => vyšší inflace)
      <- inflační očekávání ukotvená k lepkavému jádru (Stock-Watson trend)
      <- kurzový pass-through (slabší koruna => dovozní inflace, lag 4Q, stavově závislý)

    Mzdy:
      <- konvexní Phillipsova křivka (nižší nezaměstnanost => vyšší mzdy, lag 1Q;
         sklon se zestrmuje při napjatém trhu práce, Benigno-Eggertsson 2023)

    Nezaměstnanost:
      <- Okunův zákon (slabší HDP => vyšší nezaměstnanost, lag 1Q)
    """
    if extend:
        series = _extend_to_present(series)
    vals = series.values.astype(float)
    X = np.array([vals[i-p:i][::-1] for i in range(p, len(vals))])
    y = vals[p:]
    Xd = np.column_stack([np.ones(len(X)), X])
    coef = np.linalg.lstsq(Xd, y, rcond=None)[0]
    intercept, ar_coefs = coef[0], coef[1:]
    residuals = y - (intercept + X @ ar_coefs)
    sigma = residuals.std()

    # ── Deterministické korekce ───────────────────────────────────────────────
    correction = np.zeros(steps)

    # A0) Dynamická IS křivka: reálná sazba -> HDP (poptávkový kanál).
    #     Třetí rovnice novokeynesiánského jádra (Galí 2015, kap. 3):
    #     výstup reaguje na ex-ante reálnou sazbu vůči přirozené sazbě r*.
    #     r_real = PRIBOR - očekávaná inflace; kladná mezera tlumí poptávku,
    #     záporná (reálné sazby pod r*) ji stimuluje. Efekt rozložen 1-4Q.
    if is_gdp and pribor_path is not None:
        pribor_arr = np.array(pribor_path[:steps], dtype=float)
        if inflation_path is not None:
            infl_arr = np.array(inflation_path[:steps], dtype=float)
        else:
            infl_arr = np.full(steps, inflation_target)
        n_avail = min(len(pribor_arr), len(infl_arr), steps)
        real_gap = pribor_arr[:n_avail] - infl_arr[:n_avail] - real_neutral
        # Kvartální impulz do QoQ růstu (rozložené zpoždění)
        impulse = np.zeros(steps)
        for t in range(steps):
            for lag in range(1, is_lag + 1):
                src_q = t - lag
                if 0 <= src_q < n_avail:
                    impulse[t] -= is_sensitivity * real_gap[src_q] / is_lag
        if gdp_cumulative:
            # YoY je klouzavý 4Q součet QoQ, kumuluj impulz přes 4 čtvrtletí
            for t in range(steps):
                correction[t] += float(np.sum(impulse[max(0, t - 3):t + 1]))
        else:
            correction += impulse

    # A) Měnová politika -> inflace
    if is_inflation and pribor_path is not None:
        pribor_arr = np.array(pribor_path[:steps])
        excess = np.maximum(0.0, pribor_arr - neutral_rate)
        for t in range(steps):
            for lag in range(1, mp_lag + 1):
                src_q = t - lag
                if 0 <= src_q < len(excess):
                    weight = lag / mp_lag
                    correction[t] -= mp_sensitivity * excess[src_q] * weight / mp_lag

    # B) Mzdový cost-push -> inflace (lag 1Q)
    #    Nadměrný růst mezd NAD NORMOU (wage_norm = růst konzistentní s cílem
    #    + produktivita, ~4 %) tlačí inflaci. Reference je fixní norma, ne
    #    průměr cesty - jinak by trvale vysoké mzdy cost-push "vynulovaly".
    if is_inflation and wages_path is not None:
        w_arr = np.array(wages_path[:steps])
        for t in range(steps):
            src_q = t - 1
            if 0 <= src_q < len(w_arr):
                w_excess = w_arr[src_q] - wage_norm
                correction[t] += wages_infl_pass * w_excess * 0.25  # čtvrtletní efekt

    # C) HDP gap -> inflace (Phillipsova křivka, lag 1Q)
    #    Gap = růst NAD POTENCIÁLEM (potential_gdp_qoq ~ 0.55 % QoQ = ~2.2 %
    #    ročně pro ČR). Reference je potenciál, ne průměr prognózní cesty -
    #    jinak trvalý růst nad potenciálem nevytvoří žádný inflační tlak.
    if is_inflation and gdp_path is not None:
        g_arr = np.array(gdp_path[:steps])
        for t in range(steps):
            src_q = t - 1
            if 0 <= src_q < len(g_arr):
                gap = g_arr[src_q] - potential_gdp_qoq
                correction[t] += 0.05 * gap   # malý přímý efekt HDP gap na inflaci

    # C2) Kurzový pass-through (ERPT): EUR/CZK -> inflace
    #     Oslabení koruny (EUR/CZK roste) zdraží dovoz a se zpožděním zvýší CPI.
    #     Pass-through je stavově závislý: roste s velikostí depreciace
    #     (firmy přenášejí velké pohyby více než malé – moderní ERPT literatura).
    if is_inflation and eurczk_path is not None and len(eurczk_path) >= 2:
        fx = np.array(eurczk_path[:steps])
        # Procentní čtvrtletní změna kurzu (kladná = oslabení koruny)
        fx_chg = np.zeros(len(fx))
        fx_chg[1:] = (fx[1:] - fx[:-1]) / fx[:-1] * 100.0
        for t in range(steps):
            # Pass-through s rozloženým zpožděním přes erpt_lag čtvrtletí
            for lag in range(1, erpt_lag + 1):
                src_q = t - lag
                if 0 <= src_q < len(fx_chg):
                    depr = fx_chg[src_q]
                    # Stavová závislost: efektivní koeficient roste s |depreciací|
                    erpt_eff = erpt_coef * (1.0 + erpt_state * abs(depr))
                    # Rovnoměrné rozložení efektu přes lag období
                    correction[t] += erpt_eff * depr / erpt_lag

    # D) Konvexní Phillipsova křivka: nezaměstnanost -> mzdy (lag 1Q)
    #    Benigno-Eggertsson (2023): sklon se zestrmuje když je trh práce napjatý.
    #    Napjatost = jak hluboko je u pod strukturální mírou NAIRU.
    if is_wages and unempl_path is not None:
        u_arr = np.array(unempl_path[:steps])
        for t in range(steps):
            src_q = t - 1
            if 0 <= src_q < len(u_arr):
                u_gap = nairu - u_arr[src_q]   # nižší u než NAIRU => kladný gap (napjatý trh)
                # Konvexní zestrmení: efektivní sklon roste s napjatostí.
                # Při slabém trhu (u_gap<0) zůstává plochý (lineární spodní větev).
                tightness = max(0.0, u_gap)
                slope_eff = phillips_slope * (1.0 + phillips_convexity * tightness)
                correction[t] += slope_eff * u_gap * 0.25

    # E) Okunův zákon: HDP -> nezaměstnanost (lag 1Q)
    if is_unempl and gdp_path is not None:
        g_arr = np.array(gdp_path[:steps])
        g_trend = float(np.mean(g_arr))
        for t in range(steps):
            src_q = t - 1
            if 0 <= src_q < len(g_arr):
                gdp_dev = g_arr[src_q] - g_trend
                correction[t] -= okun_coef * gdp_dev * 0.25   # nižší HDP => vyšší u

    # ── Lepkavé jádro inflace (precompute) ───────────────────────────────────
    # Stock-Watson (2007): inflace se vrací k pomalu se měnícímu trendu (jádru),
    # ne k fixnímu cíli. Jádro je tažené domácími náklady (Galí-Gertler):
    # jednotkové náklady práce (mzdy nad normou) + tlak nemovitostí/služeb.
    # Vysoká perzistence ρ = lepkavost služeb a nájmů (Fuhrer 2011).
    core_path = None
    if is_inflation:
        core_path = []
        # Počáteční jádro = vyhlazená nedávná inflace (poslední 4 čtvrtletí)
        core_prev = float(np.mean(vals[-4:])) if len(vals) >= 4 else float(vals[-1])
        for t in range(steps):
            # Mzdový tlak: nadměrný růst mezd nad normou tlačí jádro nahoru
            if wages_path is not None and t < len(wages_path):
                wage_excess = max(0.0, float(wages_path[t]) - wage_norm)
            else:
                wage_excess = 0.0
            core_anchor = (inflation_target
                           + wage_to_core * wage_excess
                           + housing_services_pressure)
            # Perzistentní vývoj jádra (AR(1) k domácí kotvě)
            core_prev = core_persistence * core_prev + (1.0 - core_persistence) * core_anchor
            core_path.append(core_prev)

    # ── Simulace ─────────────────────────────────────────────────────────────
    all_paths = []
    for _ in range(n_sims):
        path = list(vals[-p:])
        for t in range(steps):
            fv = intercept + sum(ar_coefs[i] * path[-(i+1)] for i in range(p))
            fv += correction[t]

            # Kanál inflačních očekávání (hybridní NKPC) – path-dependent.
            # Očekávání se ukotví k LEPKAVÉMU JÁDRU (ne k fixnímu cíli):
            # π^e = λ·jádro_t + (1-λ)·π_{t-1}; inflace tažena k π^e vahou γ_f.
            # Když je jádro nad cílem (mzdy+nemovitosti), inflace se vrací výš.
            if is_inflation and expect_weight > 0:
                last_infl = path[-1]
                core_t = core_path[t] if core_path else inflation_target
                pi_e = anchoring * core_t + (1.0 - anchoring) * last_infl
                fv = (1.0 - expect_weight) * fv + expect_weight * pi_e

            dist_from_target = abs(fv - inflation_target) if is_inflation else 1.0
            noise_scale = sigma * min(1.0, 0.5 + 0.5 * dist_from_target / 5.0)
            fv += np.random.normal(0, noise_scale)
            path.append(fv)
        all_paths.append(path[p:])

    all_paths = np.array(all_paths)
    fut = pd.date_range(
        start=series.index[-1] + pd.offsets.QuarterBegin(1),
        periods=steps, freq="QS"
    )
    return pd.DataFrame({
        "lower_90": np.percentile(all_paths, 5,  axis=0),
        "lower_50": np.percentile(all_paths, 25, axis=0),
        "median":   np.percentile(all_paths, 50, axis=0),
        "upper_50": np.percentile(all_paths, 75, axis=0),
        "upper_90": np.percentile(all_paths, 95, axis=0),
    }, index=fut)


# ─────────────────────────────────────────────
# MUP Fan Chart
# ─────────────────────────────────────────────

def plot_mup_fan_chart(
    history: pd.Series,
    intervals: pd.DataFrame,
    variable: str,
    quarter_label: str,
    save_path: str,
    history_years: int = 6,
) -> None:
    """
    Fan chart v MUP grafickém stylu.
    Barvy: navy / modrá / šedé pozadí, čistá mřížka.
    """
    colors = VAR_COLORS.get(variable, {
        "line": MUP["blue"], "fan": MUP["blue_4"], "hist": MUP["navy"]
    })

    hist_plot = history[history.index >= history.index[-1] - pd.DateOffset(years=history_years)]

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(MUP["white"])
    ax.set_facecolor(MUP["gray_light"])

    # Mřížka – jemná, horizontální
    ax.yaxis.grid(True, color=MUP["white"], linewidth=1.2, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)

    # Spines – jen dolní a levá, tmavomodré
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color(MUP["gray_mid"])
        ax.spines[spine].set_linewidth(0.8)

    # Nulová linie
    ax.axhline(0, color=MUP["gray_mid"], linewidth=0.8, zorder=1)

    # Historická data
    ax.plot(
        hist_plot.index, hist_plot.values,
        color=colors["hist"], linewidth=2.2, zorder=5,
        solid_capstyle="round",
    )

    # Fan pásma – napoj od posledního hist bodu
    _last_v  = hist_plot.values[-1]
    _last_dt = hist_plot.index[-1]
    _fan_idx = pd.DatetimeIndex([_last_dt]).append(intervals.index)
    _lo90 = np.concatenate([[_last_v], intervals["lower_90"].values])
    _hi90 = np.concatenate([[_last_v], intervals["upper_90"].values])
    _lo50 = np.concatenate([[_last_v], intervals["lower_50"].values])
    _hi50 = np.concatenate([[_last_v], intervals["upper_50"].values])
    _med  = np.concatenate([[_last_v], intervals["median"].values])

    ax.fill_between(_fan_idx, _lo90, _hi90, alpha=0.15, color=colors["fan"], zorder=2)
    ax.fill_between(_fan_idx, _lo50, _hi50, alpha=0.30, color=colors["fan"], zorder=3)
    ax.plot(_fan_idx, _med, color=colors["line"], linewidth=2.2, linestyle="--",
            zorder=6, solid_capstyle="round")

    # COVID šedý pás (Q1–Q3 2020)
    import matplotlib.dates as _mdates
    _covid_s = pd.Timestamp("2020-01-01")
    _covid_e = pd.Timestamp("2020-09-30")
    if hist_plot.index[0] <= _covid_s and _covid_e <= _last_dt:
        ax.axvspan(_covid_s, _covid_e, alpha=0.08, color="#888888", zorder=1)
        # Souřadnice z dat (ylim ještě není finální v tomto bodě)
        _all_v = np.concatenate([hist_plot.values, _lo90, _hi90])
        _all_v = _all_v[np.isfinite(_all_v)]
        _ybot = float(_all_v.min()) if len(_all_v) else 0.0
        _ytop = float(_all_v.max()) if len(_all_v) else 1.0
        ax.text(_covid_s + pd.Timedelta(days=45),
                _ybot + (_ytop - _ybot) * 0.04,
                "COVID", fontsize=7, color=MUP["gray_mid"], va="bottom")

    ax.axvline(_last_dt, color=MUP["gray_mid"], linewidth=1.0, linestyle=":", zorder=4)

    ax.annotate(f"{_last_v:+.1f}", xy=(_last_dt, _last_v),
                xytext=(-36, 10), textcoords="offset points",
                fontsize=9, color=MUP["navy"], fontweight="bold")
    ax.annotate(f"{_med[-1]:+.1f}", xy=(intervals.index[-1], _med[-1]),
                xytext=(6, 0), textcoords="offset points",
                fontsize=9, color=colors["line"], fontweight="bold")

    # Nadpis na střed, opravená osa X
    ax.set_title(
        LABELS_CZ.get(variable, variable),
        fontsize=24, fontweight="bold", color=MUP["navy"],
        loc="center", pad=14,
    )
    ax.set_ylabel("%", fontsize=10, color=MUP["gray_dark"], rotation=0, labelpad=12)
    ax.tick_params(axis="both", labelsize=9, colors=MUP["gray_dark"])
    ax.xaxis.set_major_formatter(_mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(_mdates.YearLocator())
    fig.autofmt_xdate(rotation=0, ha="center")

    # Legenda
    legend_elements = [
        Line2D([0], [0], color=colors["hist"], linewidth=2, label="Skutečnost"),
        Line2D([0], [0], color=colors["line"], linewidth=2, linestyle="--", label="Prognóza (medián)"),
        plt.Rectangle((0, 0), 1, 1, fc=colors["fan"], alpha=0.30, label="50% interval"),
        plt.Rectangle((0, 0), 1, 1, fc=colors["fan"], alpha=0.15, label="90% interval"),
    ]
    ax.legend(
        handles=legend_elements, loc="upper left", fontsize=8.5,
        framealpha=0.9, edgecolor=MUP["gray_mid"],
        facecolor=MUP["white"],
    )

    # Patička na střed
    fig.text(
        0.5, 0.01,
        f"Zdroj: ČSÚ, Eurostat, ČNB  |  Prognóza {quarter_label}  |  MUP",
        ha="center", va="bottom", fontsize=7.5, color=MUP["gray_mid"],
    )

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=MUP["white"])
    plt.close(fig)
    print(f"  Graf uložen: {save_path}")


# ─────────────────────────────────────────────
# AI Komentář přes Anthropic API
# ─────────────────────────────────────────────

def generate_commentary(
    data: pd.DataFrame,
    forecast: pd.DataFrame,
    quarter_label: str,
    api_key: str | None = None,
    fin_df: 'pd.DataFrame | None' = None,
    fin_intervals: dict | None = None,
    repo_freeze: bool = False,
) -> str:
    """
    Volá Claude API a vrátí stručný analytický komentář ke čtvrtletní prognóze.
    Pokud není API klíč, vrátí šablonový text.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    var_desc = {
        "gdp_qoq":   "HDP mezikvartální růst (%)",
        "gdp_yoy":   "HDP meziroční růst (%)",
        "hicp_yoy":  "Inflace HICP meziroční (%)",
        "cpi_yoy":   "Inflace CPI ČSÚ meziroční (%)",
        "wages_yoy": "Mzdy meziroční růst (%)",
        "repo_rate": "Repo sazba ČNB (%)",
        "unempl":    "Míra nezaměstnanosti (%)",
        "pribor3m":  "PRIBOR 3M (%)",
    }
    col_labels = {c: var_desc.get(c, c) for c in data.columns}
    # Přidej finanční proměnné do přehledu pokud jsou dostupné
    data_extended = data.copy()
    forecast_extended = forecast.copy()
    if fin_intervals:
        for fvar in ["repo_rate", "unempl", "pribor3m"]:
            if fvar in fin_intervals:
                # Přidej poslední hodnotu z fin_df
                try:
                    data_extended[fvar] = fin_df[fvar]
                except Exception:
                    pass
                forecast_extended[fvar] = fin_intervals[fvar]["median"].values

    hist_named = data_extended.tail(6).rename(columns=col_labels).round(2).to_string()
    fc_named   = forecast_extended.rename(columns=col_labels).round(2).to_string()

    prompt = f"""Jsi ekonomický analytik. Na základě níže uvedených dat napiš stručný čtvrtletní komentář k makroekonomické prognóze České republiky pro {quarter_label}.

Historická data (posledních 6 čtvrtletí):
{{hist_named}}

Prognóza na příštích 8 čtvrtletí:
{{fc_named}}

Pokyny:
- Celková délka: 4 odstavce (cca 200–250 slov celkem)
- Odstavec 1: Aktuální stav – HDP (QoQ i YoY), obě inflace, mzdy, nezaměstnanost
- Odstavec 2: Výhled HDP a inflace na 2 roky; srovnej HICP vs CPI ČSÚ; roli repo sazby ČNB
- Odstavec 3: Výhled mezd a reálných mezd; trh práce (nezaměstnanost + mzdy dohromady)
- Odstavec 4: PRIBOR a repo sazba – výhled uvolňování/utahování; hlavní rizika
- Tón: střízlivý, akademický, faktografický
- Piš česky
- Nepoužívej bullet pointy, jen plynulé odstavce
- Nezačínaj slovy "Samozřejmě", "Jistě" apod.
"""

    if not api_key:
        print("  ANTHROPIC_API_KEY není nastaven – generuji šablonový komentář.")
        return _template_commentary(data, forecast, quarter_label, fin_df=fin_df, fin_intervals=fin_intervals, repo_freeze=repo_freeze)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        print("  AI komentář vygenerován.")
        return text
    except ImportError:
        print("  Knihovna 'anthropic' není nainstalována.")
        print("  Nainstaluj: pip install anthropic")
        return _template_commentary(data, forecast, quarter_label, fin_df=fin_df, fin_intervals=fin_intervals, repo_freeze=repo_freeze)
    except Exception as e:
        print(f"  Chyba API: {e}")
        return _template_commentary(data, forecast, quarter_label, fin_df=fin_df, fin_intervals=fin_intervals, repo_freeze=repo_freeze)


def _template_commentary(
    data: pd.DataFrame,
    forecast: pd.DataFrame,
    quarter_label: str,
    fin_df=None,
    fin_intervals=None,
    repo_freeze: bool = False,
) -> str:
    """Záložní komentář – pracuje se všemi ukazateli."""
    last = data.iloc[-1]
    fc1  = forecast.iloc[0]
    fc4  = forecast.iloc[3] if len(forecast) > 3 else forecast.iloc[-1]

    def _g(df, col, fmt=".1f"):
        return format(df[col], fmt) if col in df.index else "n/a"

    # Makro
    gdp_qoq   = _g(last, "gdp_qoq")
    gdp_yoy   = _g(last, "gdp_yoy")
    hicp      = _g(last, "hicp_yoy")
    cpi       = _g(last, "cpi_yoy")
    wages     = _g(last, "wages_yoy")
    fc1_qoq   = _g(fc1,  "gdp_qoq")
    fc4_qoq   = _g(fc4,  "gdp_qoq")
    fc1_yoy   = _g(fc1,  "gdp_yoy")
    fc4_hicp  = _g(fc4,  "hicp_yoy")
    fc4_cpi   = _g(fc4,  "cpi_yoy")
    fc4_wages = _g(fc4,  "wages_yoy")

    real_wages_now = float(last.get("wages_yoy", 0)) - float(last.get("cpi_yoy", last.get("hicp_yoy", 0)))
    real_wages_fc  = float(fc4.get("wages_yoy", 0)) - float(fc4.get("cpi_yoy", fc4.get("hicp_yoy", 0)))
    rw_trend    = "kladné" if real_wages_now > 0 else "záporné"
    rw_fc_trend = "kladné" if real_wages_fc > 0 else "záporné"
    gdp_trend   = "oživení" if float(fc1.get("gdp_qoq", 0)) > float(last.get("gdp_qoq", 0)) else "zpomalení"

    # Finanční proměnné z fin_df / fin_intervals
    def _fin_last(col):
        if fin_df is not None and col in fin_df.columns:
            return format(fin_df[col].dropna().iloc[-1], ".2f")
        return "n/a"
    def _fin_fc4(col):
        if fin_intervals and col in fin_intervals:
            vals = fin_intervals[col]["median"].values
            return format(vals[3] if len(vals) > 3 else vals[-1], ".2f")
        return "n/a"

    repo_now   = _fin_last("repo_rate")
    repo_fc4   = _fin_fc4("repo_rate")
    unempl_now = _fin_last("unempl")
    unempl_fc4 = _fin_fc4("unempl")
    pribor_now = _fin_last("pribor3m")
    pribor_fc4 = _fin_fc4("pribor3m")
    eurczk_now = _fin_last("eurczk")
    eurczk_fc4 = _fin_fc4("eurczk")
    eurusd_now = _fin_last("eurusd")
    eurusd_fc4 = _fin_fc4("eurusd")

    # Kurzy – zmínit jen pokud jsou k dispozici
    def _fx_str(now, fc, label, unit):
        if now == "n/a" or fc == "n/a":
            return ""
        try:
            diff = float(fc) - float(now)
            # Práh pro "beze změny": pohyb menší než 0.5 % kurzu
            threshold = abs(float(now)) * 0.005
            if abs(diff) < threshold:
                direction = "přibližně beze změny"
            elif (label == "EUR/CZK" and diff < 0) or (label == "EUR/USD" and diff > 0):
                direction = "posílení koruny" if label == "EUR/CZK" else "posílení eura"
            else:
                direction = "oslabení koruny" if label == "EUR/CZK" else "oslabení eura"
            return f"{label} {now} → {fc} {unit} ({direction})"
        except Exception:
            return ""

    czk_str = _fx_str(eurczk_now, eurczk_fc4, "EUR/CZK", "CZK")
    usd_str = _fx_str(eurusd_now, eurusd_fc4, "EUR/USD", "USD")
    fx_parts = [s for s in [czk_str, usd_str] if s]
    fx_sentence = ("Devizový výhled: " + "; ".join(fx_parts) + ".") if fx_parts else ""

    # PRIBOR výhled (bez koncové tečky, věta se skládá do souvětí)
    if pribor_now != "n/a" and pribor_fc4 != "n/a":
        try:
            pribor_diff = float(pribor_fc4) - float(pribor_now)
            pribor_trend = "vzroste" if pribor_diff > 0.1 else ("klesne" if pribor_diff < -0.1 else "zůstane stabilní")
            pribor_sentence = f"PRIBOR 3M {pribor_trend} z {pribor_now} % na {pribor_fc4} %"
        except Exception:
            pribor_sentence = ""
    else:
        pribor_sentence = ""

    # Směr inflace: reflektuj lepkavé jádro. Pokud výhled drží nad cílem,
    # neříkej "vrátila blíže k cíli", ale popiš setrvání nad cílem.
    try:
        _fc_infl = float(fc4_hicp) if fc4_hicp != "n/a" else None
    except Exception:
        _fc_infl = None
    # Aktuální inflace pro určení SMĚRU (stoupá od cíle vs klesá k cíli)
    try:
        _cur_infl = float(hicp) if hicp != "n/a" else None
    except Exception:
        _cur_infl = None

    if _fc_infl is not None and _fc_infl >= 2.5:
        inflation_sentence = (f"Inflace podle modelu setrvá nad cílem ČNB, ve čtyřkvartálním "
                              f"horizontu na {fc4_hicp} % HICP / {fc4_cpi} % CPI. Lepkavá jádrová "
                              f"složka, tažená mzdovou dynamikou a tlakem z nemovitostí a služeb, "
                              f"brání rychlejšímu návratu k 2 %")
    elif (_fc_infl is not None and _fc_infl > 2.1
          and _cur_infl is not None and _cur_infl <= 2.15):
        # Inflace je NA cíli a prognóza stoupá - vzdaluje se, ne přibližuje
        inflation_sentence = (f"Inflace se od cíle mírně odchýlí směrem vzhůru, na "
                              f"{fc4_hicp} % HICP / {fc4_cpi} % CPI za čtyři čtvrtletí, "
                              f"vlivem domácích nákladových tlaků (mzdy, nemovitosti a služby)")
    elif _fc_infl is not None and _fc_infl > 2.1:
        inflation_sentence = (f"Inflace se bude k cíli ČNB přibližovat jen pozvolna, na "
                              f"{fc4_hicp} % HICP / {fc4_cpi} % CPI za čtyři čtvrtletí")
    else:
        inflation_sentence = (f"Inflace by se po {fc4_hicp} % HICP / {fc4_cpi} % CPI za čtyři "
                              f"čtvrtletí vrátila blíže k cíli ČNB")

    if repo_freeze:
        repo_context = (f"{inflation_sentence}, a to při zmrazené repo sazbě na {repo_now} % "
                        f"(scénář beze změny sazeb ČNB). {pribor_sentence} ve čtyřkvartálním horizontu."
                        if pribor_sentence else
                        f"{inflation_sentence}, a to při zmrazené repo sazbě na {repo_now} % "
                        f"(scénář beze změny sazeb ČNB).")
        risk_sentence = f"Hlavními riziky jsou vývoj cen energií, tempo dezinflace v eurozóně a dynamika mzdových dohod ve veřejném sektoru. {fx_sentence} Klíčovým ukazatelem pro příští čtvrtletí bude vývoj inflace, neboť přesah prognózy by ČNB postavil pod tlak přehodnotit scénář zmrazených sazeb."
    else:
        repo_dir = "uvolňováním" if repo_fc4 != "n/a" and float(repo_fc4) < float(repo_now.replace("n/a","999")) else "utahováním nebo stabilizací"
        _pribor_part = f"; {pribor_sentence}" if pribor_sentence else ""
        repo_context = (f"{inflation_sentence}. Výhled je podmíněn pokračujícím {repo_dir} "
                        f"měnové politiky (repo {repo_now} % → {repo_fc4} %{_pribor_part}).")
        risk_sentence = f"Hlavními riziky jsou vývoj cen energií, tempo dezinflace v eurozóně a dynamika mzdových dohod ve veřejném sektoru. {fx_sentence} Klíčovým ukazatelem pro příští čtvrtletí bude rozhodnutí ČNB o repo sazbě."

    # Směr nezaměstnanosti: dynamicky podle prognózy, ne natvrdo "pokles"
    try:
        _u_now = float(unempl_now) if unempl_now != "n/a" else None
        _u_fc  = float(unempl_fc4) if unempl_fc4 != "n/a" else None
    except Exception:
        _u_now, _u_fc = None, None
    if _u_now is not None and _u_fc is not None:
        _du = _u_fc - _u_now
        if _du > 0.05:
            u_phrase = f"při mírném růstu nezaměstnanosti na {unempl_fc4} %"
            labor_sentence = ("Restriktivní měnová politika trh práce postupně ochlazuje, "
                              "napjatost však zůstává vysoká a mzdový cost-push kanál dál "
                              "působí proinflačně.")
        elif _du < -0.05:
            u_phrase = f"při poklesu nezaměstnanosti na {unempl_fc4} %"
            labor_sentence = ("Trh práce tak zůstane klíčovým proinflačním faktorem "
                              "prostřednictvím cost-push kanálu, který model explicitně modeluje.")
        else:
            u_phrase = f"při stabilní nezaměstnanosti kolem {unempl_fc4} %"
            labor_sentence = ("Napjatý trh práce zůstává klíčovým proinflačním faktorem "
                              "prostřednictvím cost-push kanálu, který model explicitně modeluje.")
    else:
        u_phrase = f"při nezaměstnanosti {unempl_fc4} %"
        labor_sentence = ("Trh práce zůstává klíčovým proinflačním faktorem prostřednictvím "
                          "cost-push kanálu.")

    return textwrap.dedent(f"""
    Česká ekonomika zakončila sledované období s mezikvartálním růstem HDP {gdp_qoq} % a meziročním růstem {gdp_yoy} %. Inflace CPI ČSÚ dosahovala {cpi} % (HICP: {hicp} %) a mzdy rostly {wages} % meziročně, reálné mzdy byly {rw_trend}. Míra nezaměstnanosti se pohybovala na úrovni {unempl_now} %.

    Model signalizuje {gdp_trend} s výhledem HDP QoQ na {fc4_qoq} % a YoY na {fc1_yoy} % v prvním horizontu. {repo_context}

    Mzdový výhled ({fc4_wages} % meziročně po čtyřech čtvrtletích) {u_phrase} naznačuje, že reálné mzdy zůstanou {rw_fc_trend}. {labor_sentence}

    {risk_sentence}
    """).strip()


# ─────────────────────────────────────────────
# Sestavení Markdown reportu
# ─────────────────────────────────────────────

def build_markdown_report(
    quarter_label: str,
    commentary: str,
    forecast: pd.DataFrame,
    chart_paths: dict[str, str],
    output_path: str,
    cnb_table: str = "",
) -> None:
    """Sestaví Markdown soubor s komentářem, tabulkou prognóz a odkazy na grafy."""

    def _ql(dt):
        return f"{dt.year}-Q{(dt.month-1)//3+1}"

    # Tabulka prognóz
    rows = []
    for dt in forecast.index:
        rows.append({
            "Čtvrtletí": _ql(dt),
            "HDP QoQ (%)": f"{forecast.loc[dt,'gdp_qoq']:+.2f}",
            "Inflace YoY (%)": f"{forecast.loc[dt,'hicp_yoy']:+.2f}",
            "Mzdy YoY (%)": f"{forecast.loc[dt,'wages_yoy']:+.2f}",
        })
    table_df = pd.DataFrame(rows)

    # Markdown
    md_lines = [
        f"# Makroekonomická prognóza – {quarter_label}",
        f"*Metropolitní univerzita Praha  |  {datetime.date.today().strftime('%-d. %-m. %Y')}*",
        "",
        "---",
        "",
        "## Komentář analytika",
        "",
        commentary,
        "",
        "---",
        "",
        "## Prognóza klíčových veličin",
        "",
        table_df.to_markdown(index=False),
        "",
        "> **Metodická poznámka:** Prognózy vycházejí ze semi-strukturálního "
        "novokeynesiánského modelu (dynamická IS křivka, konvexní Phillipsova křivka, "
        "Taylorovo pravidlo) s AR jádrem a kanály zpětných vazeb. Intervaly spolehlivosti "
        "z Monte Carlo simulace (n=2 000).",
        "",
        (cnb_table + "\n") if cnb_table else "",
        "",
        "---",
        "",
        "## Fan charty",
        "",
    ]

    for var, path in chart_paths.items():
        label = LABELS_CZ.get(var, var)
        rel_path = os.path.relpath(path, os.path.dirname(output_path))
        md_lines += [f"### {label}", f"", f"![{label}]({rel_path})", ""]

    md_lines += [
        "---",
        "",
        f"*Dokument vygenerován automaticky systémem makroekonomického modelu MUP. "
        f"Verze dat: {datetime.date.today().isoformat()}.*",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"  Report uložen: {output_path}")


# ─────────────────────────────────────────────
# Hlavní funkce
# ─────────────────────────────────────────────


def inflation_decomposition(series, steps, pribor_path, wages_path, gdp_path,
                            eurczk_path, args, horizons=(4, 8)):
    """
    Dekompozice inflační prognózy na příspěvky kanálů (styl ČNB).

    Marginální příspěvek kanálu = prognóza se všemi kanály minus prognóza
    s daným kanálem vypnutým. Stejný seed náhodných šoků pro všechny varianty
    zajistí, že rozdíly mediánů jsou čisté efekty kanálů, ne MC šum.
    Zbytek do celkové prognózy = AR dynamika + interakce kanálů.
    """
    def _run(**overrides):
        kw = dict(
            steps=steps, is_inflation=True,
            pribor_path=pribor_path, wages_path=wages_path,
            gdp_path=gdp_path, eurczk_path=eurczk_path,
            neutral_rate=3.0, inflation_target=2.0,
            mp_sensitivity=0.25, mp_lag=4,
            expect_weight=getattr(args, "expect_weight", 0.35),
            anchoring=getattr(args, "anchoring", 0.75),
            erpt_coef=getattr(args, "erpt_coef", 0.15),
            housing_services_pressure=getattr(args, "housing_pressure", 0.5),
        )
        kw.update(overrides)
        np.random.seed(1234)          # identické šoky napříč variantami
        return ar_forecast(series, **kw)["median"].values

    base = _run()
    channels = {
        "Mzdový cost-push":     _run(wages_path=None),
        "Kurz (ERPT)":          _run(erpt_coef=0.0),
        "Měnová politika":      _run(mp_sensitivity=0.0),
        "HDP mezera":           _run(gdp_path=None),
        "Očekávání + jádro":    _run(expect_weight=0.0),
    }
    out = {}
    for name, variant in channels.items():
        out[name] = {h: float(base[h-1] - variant[h-1]) for h in horizons}
    out["_base"] = {h: float(base[h-1]) for h in horizons}
    return out


def plot_decomposition(decomp, ql, save_path):
    """Sloupcový graf příspěvků kanálů k inflační prognóze (Q+4 a Q+8)."""
    import matplotlib.pyplot as plt
    names = [k for k in decomp if not k.startswith("_")]
    h4 = [decomp[n][4] for n in names]
    h8 = [decomp[n][8] for n in names]

    fig, ax = plt.subplots(figsize=(9.5, 5.2), facecolor="#EEF2FA")
    ax.set_facecolor("#EEF2FA")
    y = np.arange(len(names))
    ax.barh(y + 0.19, h4, height=0.36, color="#003DA5", label="za 4 čtvrtletí")
    ax.barh(y - 0.19, h8, height=0.36, color="#7A9BD4", label="za 8 čtvrtletí")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=10)
    ax.axvline(0, color="#00205B", lw=1)
    ax.grid(True, axis="x", color="#D8E0F0", ls="--", lw=0.7)
    ax.set_axisbelow(True)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("příspěvek k inflaci (procentní body)", fontsize=10)
    ax.set_title(f"Příspěvky kanálů k inflační prognóze  |  {ql}",
                 fontsize=13, color="#00205B", fontweight="bold")
    b4 = decomp["_base"][4]; b8 = decomp["_base"][8]
    fig.text(0.5, 0.005,
             f"Celková prognóza HICP: {b4:.1f} % (4Q), {b8:.1f} % (8Q). "
             "Příspěvek = plný model minus model bez daného kanálu (identické šoky). "
             "Zbytek = AR dynamika.",
             fontsize=8.5, color="#5A6478", ha="center")
    ax.legend(fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#EEF2FA")
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NOSTRADAMUP – generátor reportu")
    parser.add_argument("--output-dir",   type=str, default=None,
                        help="Výstupní složka pro grafy")
    parser.add_argument("--repo-freeze",   action="store_true", default=False,
                        help="Zmrazit repo sazbu na aktuální úrovni")
    parser.add_argument("--repo-neutral",  type=float, default=3.5,
                        help="Neutrální repo sazba pro Taylorovo pravidlo (default: 3.5)")
    parser.add_argument("--pribor-eq",     type=float, default=3.0,
                        help="Rovnovážná sazba PRIBOR (default: 3.0)")
    parser.add_argument("--pribor-speed",  type=float, default=0.30,
                        help="Rychlost konvergence PRIBOR (default: 0.30)")
    parser.add_argument("--expect-weight", type=float, default=0.35,
                        help="Váha inflačních očekávání γ_f v hybridní NKPC (0–1, default: 0.35)")
    parser.add_argument("--anchoring",     type=float, default=0.75,
                        help="Ukotvenost očekávání λ k cíli ČNB (0=de-ukotveno, 1=plně ukotveno, default: 0.75)")
    parser.add_argument("--phillips-convexity", type=float, default=0.8,
                        help="Konvexita Phillipsovy křivky – zestrmení při napjatém trhu (0=lineární, default: 0.8)")
    parser.add_argument("--erpt-coef",     type=float, default=0.15,
                        help="Kurzový pass-through: změna CPI na 1%% oslabení koruny za rok (default: 0.15)")
    parser.add_argument("--horizon", type=int, default=12,
                        help="Prognózní horizont ve čtvrtletích (default: 12 = 3 roky pro ČNB)")
    parser.add_argument("--housing-pressure", type=float, default=0.5,
                        help="Domácí tlak nemovitostí+služeb na jádrovou inflaci v pp nad cíl (default: 0.5)")
    parser.add_argument("--is-sensitivity", type=float, default=0.05,
                        help="Dynamická IS křivka: citlivost HDP QoQ na 1 pp reálné sazbové mezery "
                             "(default: 0.05, kalibrováno backtestem - HDP poráží RW; 0 = čistě statistické HDP)")
    parser.add_argument("--bond5y", type=float, default=4.1,
                        help="5Y výnos českého stát. dluhopisu pro IRS (default 4.1; ruční override tržní kotace)")
    parser.add_argument("--bond10y", type=float, default=4.7,
                        help="10Y výnos českého stát. dluhopisu pro IRS (default 4.7; ruční override tržní kotace)")
    parser.add_argument("--swap-spread", type=float, default=0.0,
                        help="Swapový (asset-swap) spread IRS nad govvie výnosem v pp (default 0.0)")
    args, _ = parser.parse_known_args()

    # Adresáře
    base       = os.path.dirname(os.path.abspath(__file__))
    charts_dir = (
        str(__import__("pathlib").Path(args.output_dir).expanduser().resolve())
        if args.output_dir else
        os.path.join(base, "outputs", "charts")
    )
    report_dir = (
        os.path.join(os.path.dirname(charts_dir), "reports")
        if args.output_dir else
        os.path.join(base, "outputs", "reports")
    )
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    # Označení čtvrtletí
    now = datetime.date.today()
    quarter_label = f"{now.year}-Q{(now.month-1)//3+1}"

    # ---- Načti data ----
    try:
        # Živá data (nebo čerstvý cache z data_fetch.py) místo záložních demo funkcí.
        from data_fetch import load_cz_dataset
        data = load_cz_dataset()
        data = data.resample("QS").mean().dropna()
        # Doplň každou sérii flat-forward do aktuálního čtvrtletí
        data = pd.DataFrame(
            {col: _extend_to_present(data[col]) for col in data.columns}
        ).sort_index().dropna()
    except ImportError as _e:
        print("\n" + "!"*60)
        print("  CHYBA: nelze načíst load_cz_dataset z data_fetch.py")
        print(f"  Detail: {_e}")
        print("  PŘÍČINA: data_fetch.py je nejspíš starší verze bez funkce")
        print("  load_cz_dataset (přidána ve v2.0). Aktualizuj VŠECHNY soubory")
        print("  ze stejného snapshotu (v2.4), ne jen některé.")
        print("!"*60 + "\n")
        return

    # ---- Prognóza ----
    steps    = getattr(args, "horizon", 12)
    forecast = pd.DataFrame(index=pd.date_range(
        start=data.index[-1] + pd.offsets.QuarterBegin(1), periods=steps, freq="QS"
    ))
    # Načti finanční data a prognózy – preferuj uložené intervaly z financial_data.py
    fin_df = None
    fin_intervals = None
    try:
        from financial_data import build_financial_dataset, load_intervals
        import os as _os
        fin_df = build_financial_dataset(use_cache=True)
        # Pokus se načíst předpočítané intervaly
        _ipath = _os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", "fin_intervals.json")
        if _os.path.exists(_ipath):
            fin_intervals = load_intervals(_ipath)
            # Zarovnej horizont na délku cache (fin a makro musí mít stejné kroky)
            _any = next(iter(fin_intervals.values()), None)
            if _any is not None and len(_any) != steps:
                steps = len(_any)
                forecast = pd.DataFrame(index=pd.date_range(
                    start=data.index[-1] + pd.offsets.QuarterBegin(1),
                    periods=steps, freq="QS"))
            print(f"  Finanční intervaly načteny z cache ({steps} čtvrtletí).")
        else:
            # Fallback: přepočítej s aktuálními parametry
            from financial_data import forecast_financial
            fin_intervals = forecast_financial(
                fin_df, steps=steps,
                pribor_long_run=args.pribor_eq,
                pribor_speed=args.pribor_speed,
                repo_neutral=args.repo_neutral,
                repo_freeze=args.repo_freeze,
            )
        pribor_path = fin_intervals["pribor3m"]["median"].tolist()
    except Exception as e:
        print(f"  Varování: finanční data nedostupná ({e})")
        pribor_path = None

    intervals = {}
    inflation_vars = {"hicp_yoy", "cpi_yoy"}

    # Vstupy nezávislé na iteraci
    unempl_baseline = [3.0] * steps
    nairu_est = 2.8
    if fin_intervals and "unempl" in fin_intervals:
        unempl_baseline = fin_intervals["unempl"]["median"].tolist()
        try:
            if fin_df is not None and "unempl" in fin_df.columns:
                nairu_est = float(fin_df["unempl"].dropna().tail(20).mean())
        except Exception:
            pass
    eurczk_path = None
    if fin_intervals and "eurczk" in fin_intervals:
        eurczk_path = fin_intervals["eurczk"]["median"].tolist()

    _is_sens = getattr(args, "is_sensitivity", 0.15)

    # ── Tříravnicové NK jádro řešené iterativně (fixed point) ────────────────
    # HDP potřebuje očekávanou inflaci (ex-ante reálná sazba v IS křivce),
    # inflace potřebuje HDP (výstupová mezera, mzdy). Dvě iterace stačí:
    # 0) startovní odhad inflace bez vazeb, 1) plný řetěz, 2) zpřesnění.
    infl_guess = None
    if "hicp_yoy" in data.columns:
        _g = ar_forecast(data["hicp_yoy"], steps=steps, is_inflation=True,
                         inflation_target=2.0,
                         anchoring=getattr(args, "anchoring", 0.75),
                         housing_services_pressure=getattr(args, "housing_pressure", 0.5))
        infl_guess = _g["median"].tolist()

    for _iter in range(2):
        # 1) HDP: dynamická IS křivka (reálná sazba tlumí/stimuluje poptávku)
        for var in ["gdp_qoq", "gdp_yoy"]:
            if var not in data.columns:
                continue
            ival = ar_forecast(data[var], steps=steps,
                               is_gdp=True,
                               gdp_cumulative=(var == "gdp_yoy"),
                               pribor_path=pribor_path,
                               inflation_path=infl_guess,
                               is_sensitivity=_is_sens,
                               neutral_rate=3.0)
            intervals[var] = ival
            forecast[var]  = ival["median"].values

        gdp_path = forecast.get("gdp_qoq", pd.Series(dtype=float))
        gdp_path_list = list(gdp_path) if hasattr(gdp_path, "__iter__") else None

        # 2) Mzdy: konvexní Phillipsova křivka
        if "wages_yoy" in data.columns:
            ival = ar_forecast(data["wages_yoy"], steps=steps,
                               is_wages=True,
                               unempl_path=unempl_baseline,
                               gdp_path=gdp_path_list,
                               phillips_convexity=getattr(args, "phillips_convexity", 0.8),
                               nairu=nairu_est)
            intervals["wages_yoy"] = ival
            forecast["wages_yoy"]  = ival["median"].values

        wages_path = list(forecast.get("wages_yoy", pd.Series([5.0]*steps, dtype=float)))

        # 3) Inflace: plný kanálový mix (NKPC + jádro + ERPT + cost-push)
        for var in ["hicp_yoy", "cpi_yoy"]:
            if var not in data.columns:
                continue
            ival = ar_forecast(
                data[var], steps=steps,
                is_inflation=True,
                pribor_path=pribor_path,
                wages_path=wages_path,
                gdp_path=gdp_path_list,
                eurczk_path=eurczk_path,
                neutral_rate=3.0,
                inflation_target=2.0,
                mp_sensitivity=0.25,
                mp_lag=4,
                expect_weight=getattr(args, "expect_weight", 0.35),
                anchoring=getattr(args, "anchoring", 0.75),
                erpt_coef=getattr(args, "erpt_coef", 0.15),
                housing_services_pressure=getattr(args, "housing_pressure", 0.5),
            )
            intervals[var] = ival
            forecast[var]  = ival["median"].values

        # Zpřesni inflační očekávání pro další iteraci IS křivky
        if "hicp_yoy" in forecast.columns:
            infl_guess = list(forecast["hicp_yoy"].values)

    # ---- Dekompozice inflační prognózy (příspěvky kanálů, styl ČNB) ----
    decomp = None
    if "hicp_yoy" in data.columns and steps >= 8:
        try:
            decomp = inflation_decomposition(
                data["hicp_yoy"], steps=steps,
                pribor_path=pribor_path, wages_path=wages_path,
                gdp_path=gdp_path_list, eurczk_path=eurczk_path, args=args,
            )
        except Exception as _e:
            print(f"  (dekompozice přeskočena: {_e})")

    print(f"\nGeneruji report pro {quarter_label}...")

    # ---- Grafy ----
    print("\n[1/3] Generuji fan charty...")
    chart_paths = {}
    for var in ["gdp_qoq", "gdp_yoy", "hicp_yoy", "cpi_yoy", "wages_yoy"]:
        if var not in forecast.columns:
            continue
        path = os.path.join(charts_dir, f"mup_{var}_{quarter_label.replace('-','')}.png")
        plot_mup_fan_chart(
            history=data[var],
            intervals=intervals[var],
            variable=var,
            quarter_label=quarter_label,
            save_path=path,
        )
        chart_paths[var] = path

    # ---- AI komentář ----
    print("\n[2/3] Generuji analytický komentář...")
    commentary = generate_commentary(data, forecast, quarter_label,
                                       fin_df=fin_df, fin_intervals=fin_intervals,
                                       repo_freeze=getattr(args, "repo_freeze", False))

    # ---- Markdown report ----
    print("\n[3/3] Sestavuji report...")
    report_path = os.path.join(report_dir, f"prognoza_{quarter_label.replace('-','')}.md")
    # ---- Graf dekompozice ----
    if decomp is not None:
        try:
            _dpath = os.path.join(charts_dir, f"mup_infl_dekompozice_{quarter_label.replace('-','')}.png")
            plot_decomposition(decomp, quarter_label, _dpath)
            chart_paths["dekompozice"] = _dpath
            print(f"  Dekompozice inflace uložena: {_dpath}")
        except Exception as _e:
            print(f"  (graf dekompozice přeskočen: {_e})")

    # ČNB tabulka - postavená před reportem, vložená přímo do těla (spolehlivé)
    try:
        from cnb_survey import build_cnb_table
        cnb_md = build_cnb_table(data, forecast, fin_intervals or {}, fin_df=fin_df,
                                 bond5y=getattr(args, 'bond5y', 4.1),
                                 bond10y=getattr(args, 'bond10y', 4.7),
                                 swap_spread=getattr(args, 'swap_spread', 0.0))
    except Exception as _e:
        cnb_md = ""
        import traceback
        print("\n" + "!"*60)
        print("  ČNB TABULKA SELHALA – nebude v reportu. Důvod:")
        traceback.print_exc()
        print("!"*60 + "\n")

    build_markdown_report(
        quarter_label=quarter_label,
        commentary=commentary,
        forecast=forecast,
        chart_paths=chart_paths,
        output_path=report_path,
        cnb_table=cnb_md,
    )

    # ---- Kombinovaný přehled 2×3 ----
    print("\n[4/4] Generuji kombinovaný přehled...")
    overview_path = os.path.join(charts_dir, f"prehled_{quarter_label.replace('-','')}.png")
    build_combined_overview(
        macro_paths=chart_paths,
        fin_dir=charts_dir,
        quarter_label=quarter_label,
        save_path=overview_path,
    )

    if cnb_md:
        print("\n" + "="*60)
        print("  OČEKÁVÁNÍ HLAVNÍCH INFLAČNÍCH VELIČIN (také zapsáno na konci md reportu):")
        print("="*60)
        print(cnb_md)
        print("="*60)

    print(f"\n✓ Hotovo. Čtvrtletní report: {report_path}")
    print(f"  Grafy: {charts_dir}")
    print(f"  Přehled: {overview_path}")
    print(f"\nKomentář:\n{'─'*60}")
    print(commentary)
    print('─'*60)



# ─────────────────────────────────────────────
# Kombinovaný přehled 6 grafů (2 sloupce × 3 řádky)
# ─────────────────────────────────────────────

def build_combined_overview(
    macro_paths: dict,
    fin_dir: str,
    quarter_label: str,
    save_path: str,
) -> None:
    """
    Složí grafy do přehledu ve dvou sloupcích – vždy tematické páry:
      Řádek 1: HDP YoY  |  HDP QoQ
      Řádek 2: Inflace CPI ČSÚ  |  Inflace HICP
      Řádek 3: EUR/CZK  |  EUR/USD
      Řádek 4: Mzdy  |  PRIBOR
    """
    from PIL import Image, ImageDraw, ImageFont

    ql = quarter_label.replace("-", "")

    # Páry: (levý, pravý) – cesta k PNG
    pairs = [
        (macro_paths.get("gdp_yoy",   ""), macro_paths.get("gdp_qoq",   "")),
        (macro_paths.get("cpi_yoy",   ""), macro_paths.get("hicp_yoy",  "")),
        (os.path.join(fin_dir, "fin_eurczk.png"), os.path.join(fin_dir, "fin_eurusd.png")),
        (macro_paths.get("wages_yoy", ""), os.path.join(fin_dir, "fin_pribor3m.png")),
        (os.path.join(fin_dir, "fin_unempl.png"), os.path.join(fin_dir, "fin_repo_rate.png")),
    ]

    # Rozměry z prvního dostupného obrázku
    sample = None
    for lp, rp in pairs:
        for p in [lp, rp]:
            if p and os.path.exists(p):
                sample = Image.open(p)
                break
        if sample:
            break
    if sample is None:
        print("  Žádný zdrojový graf nenalezen.")
        return

    cell_w, cell_h = sample.size
    pad      = 28
    header_h = 160   # místo pro velký nadpis (96px font)

    total_w = 2 * cell_w + 3 * pad
    total_h = header_h + len(pairs) * cell_h + (len(pairs) + 1) * pad

    canvas = Image.new("RGB", (total_w, total_h), color=(238, 242, 250))
    draw   = ImageDraw.Draw(canvas)

    # ── Fonty ──────────────────────────────────────────────────────────
    def _font(size, bold=False):
        candidates = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
                else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for c in candidates:
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                continue
        return ImageFont.load_default()

    font_title = _font(96, bold=True)    # 4× původní, 2× větší než nadpisy grafů (24px)
    font_sub   = _font(18)
    font_col   = _font(15)

    # ── Záhlaví ────────────────────────────────────────────────────────
    title_text = f"Makroekonomický přehled – {quarter_label}"
    sub_text   = "Metropolitní univerzita Praha  |  Makroekonomický model"

    def centered_text(text, font, y, color):
        try:
            bb = draw.textbbox((0, 0), text, font=font)
            tw = bb[2] - bb[0]
        except Exception:
            tw = len(text) * (font.size if hasattr(font, "size") else 10)
        draw.text(((total_w - tw) // 2, y), text, font=font, fill=color)

    centered_text(title_text, font_title, 10,  (0, 32, 91))
    centered_text(sub_text,   font_sub,   118, (158, 158, 158))

    # Záhlaví sloupců
    col_labels = ["Meziroční pohled", "Mezikvartální / doplňkový pohled"]
    for c, col_label in enumerate(col_labels):
        cx = pad + c * (cell_w + pad)
        try:
            bb = draw.textbbox((0,0), col_label, font=font_col)
            lw = bb[2] - bb[0]
        except Exception:
            lw = len(col_label) * 8
        draw.text((cx + (cell_w - lw) // 2, header_h - 22),
                  col_label, font=font_col, fill=(91, 141, 184))

    # ── Vlož grafy ─────────────────────────────────────────────────────
    for r, (lp, rp) in enumerate(pairs):
        y = header_h + pad + r * (cell_h + pad)
        for c, path in enumerate([lp, rp]):
            x = pad + c * (cell_w + pad)
            if path and os.path.exists(path):
                img = Image.open(path).convert("RGB")
                if img.size != (cell_w, cell_h):
                    img = img.resize((cell_w, cell_h), Image.LANCZOS)
                canvas.paste(img, (x, y))
            else:
                draw.rectangle([x, y, x+cell_w, y+cell_h],
                               fill=(242,244,247), outline=(200,200,200))

    # ── Patička s logem MUP (text) ─────────────────────────────────────
    footer_y = total_h - 36
    footer_text = f"Zdroj: ČSÚ, Eurostat, ČNB, ECB  |  Prognóza {quarter_label}"
    try:
        bb = draw.textbbox((0,0), footer_text, font=font_col)
        fw = bb[2] - bb[0]
    except Exception:
        fw = len(footer_text) * 8
    draw.text(((total_w - fw) // 2, footer_y),
              footer_text, font=font_col, fill=(158,158,158))

    # Logo MUP – stylizovaný text vpravo
    logo_text = "MUP"
    font_logo = _font(22, bold=True)
    try:
        bb = draw.textbbox((0,0), logo_text, font=font_logo)
        lw = bb[2] - bb[0]
    except Exception:
        lw = 50
    draw.text((total_w - lw - pad, footer_y - 2),
              logo_text, font=font_logo, fill=(0, 32, 91))
    # Podtržení loga (simulace MUP design)
    draw.rectangle([total_w - lw - pad, footer_y + 26,
                    total_w - pad, footer_y + 29],
                   fill=(91, 141, 184))

    canvas.save(save_path, dpi=(150, 150))
    print(f"  Kombinovaný přehled uložen: {save_path}")


if __name__ == "__main__":
    main()
