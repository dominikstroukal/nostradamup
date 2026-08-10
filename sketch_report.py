"""
sketch_report.py
================
Generuje alternativní přehled 6 grafů ve stylu "nakresleno rukou".

Techniky:
  - plt.xkcd() kontext  → wobbly osy, skicovitý font
  - Gaussovský šum na datové linky → žádná přímá pravítková linka
  - Šrafování fan pásem (hatch) místo průhledné výplně
  - Pozadí čistě bílé (#FFFFFF)
  - Ručně psaný styl nadpisů a popisků

Výběr 6 grafů (stejná data jako hlavní report):
  HDP YoY  |  HDP QoQ
  CPI ČSÚ  |  HICP
  EUR/CZK  |  PRIBOR 3M

Spuštění:
    python sketch_report.py

Výstup:
    outputs/charts/sketch_prehled_YYYYQN.png
"""

import os
import logging
import warnings
import datetime
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D

# Potlač fontové warningy – xkcd font se instaluje přes install_font.py
warnings.filterwarnings("ignore")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(BASE_DIR, "outputs", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

def _detect_sketch_font() -> str | None:
    """
    Zkontroluje jestli je nainstalován xkcd nebo jiný ručně psaný font.
    Pokud ne, vrátí None – plt.xkcd() styl stále funguje (wobbly čáry),
    jen bez hand-written fontu. Nainstaluj font přes: python install_font.py
    """
    available = {f.name for f in fm.fontManager.ttflist}
    # Upřednostni typewriter/monospace fonty, pak xkcd
    for name in (
        "American Typewriter",    # macOS – nejlepší volba
        "Courier New",            # Windows / cross-platform
        "Courier",                # Unix fallback
        "Latin Modern Mono",      # LaTeX distribuce
        "Liberation Mono",        # Linux
        "xkcd Script", "xkcd", "Humor Sans",   # po install_font.py
        "DejaVu Sans Mono",       # vždy dostupný v matplotlib
    ):
        if name in available:
            return name
    return "monospace"   # Vždy vrátí něco použitelného

_SKETCH_FONT = _detect_sketch_font()

# ── Barvy skicovitého stylu ────────────────────────────────────────────────────
SK = {
    "paper":   "#FFFFFF",    # papírové pozadí
    "ink":     "#1A1A2E",    # hlavní linka (skoro černá, mírně modrá)
    "blue":    "#2255A4",    # prognóza
    "grid":    "#E8E8E8",    # mřížka (jako blok papíru)
    "covid":   "#C8C0A8",    # COVID pás
    "text":    "#1A1A2E",
    "muted":   "#7A7060",
}

LABELS = {
    "gdp_yoy":   "HDP – meziroční růst (%)",
    "gdp_qoq":   "HDP – mezikvartální růst (%)",
    "hicp_yoy":  "Inflace HICP – meziroční (%)",
    "cpi_yoy":   "Inflace CPI ČSÚ – meziroční (%)",
    "eurczk":    "EUR/CZK",
    "eurusd":    "EUR/USD",
    "wages_yoy": "Průměrné mzdy – meziroční růst (%)",
    "pribor3m":  "PRIBOR 3M (%)",
    "repo_rate": "Repo sazba ČNB (%)",
    "unempl":    "Míra nezaměstnanosti (%)",
}

PAIRS = [
    ("gdp_yoy",  "gdp_qoq"),
    ("cpi_yoy",  "hicp_yoy"),
    ("eurczk",   "pribor3m"),
]


# ── Pomocné funkce pro "ruční" efekt ──────────────────────────────────────────

def _wobble(x: np.ndarray, y: np.ndarray, noise: float = 0.012) -> tuple:
    """Přidá drobný Gaussovský šum na y-osu – linka vypadá kresleně."""
    rng = np.random.default_rng(42)   # seed = reprodukovatelnost
    span = np.ptp(y) if np.ptp(y) > 0 else 1.0
    jitter = rng.normal(0, noise * span, size=len(y))
    # Kumulativní klouzavý průměr = plynulý šum, ne random chaos
    smooth = np.convolve(jitter, np.ones(3)/3, mode="same")
    return x, y + smooth


def _wobble_series(s: pd.Series, noise: float = 0.012) -> pd.Series:
    x_num = np.arange(len(s), dtype=float)
    _, y_w = _wobble(x_num, s.values, noise)
    return pd.Series(y_w, index=s.index)


def _sketch_fan(ax, idx, lo90, hi90, lo50, hi50, color: str):
    """
    Šrafované fan páso místo průhledné výplně.
    Vypadá jako ručně šrafované perem.
    """
    ax.fill_between(idx, lo90, hi90,
                    facecolor="none", edgecolor=color,
                    hatch="///", linewidth=0.0, alpha=0.35, zorder=2)
    ax.fill_between(idx, lo50, hi50,
                    facecolor=color, alpha=0.12, zorder=3,
                    linewidth=0)


def _sketch_axes(ax):
    """Nastav osy pro skicovitý look."""
    ax.set_facecolor(SK["paper"])
    ax.yaxis.grid(True, color=SK["grid"], linewidth=0.8, linestyle="--", zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    # Silnější spodní spine jako pero
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.8)
    ax.spines["bottom"].set_color(SK["ink"])
    ax.spines["left"].set_linewidth(1.8)
    ax.spines["left"].set_color(SK["ink"])
    ax.tick_params(labelsize=9, colors=SK["muted"])


def _covid_band(ax, hist_index, all_vals=None):
    """
    Vykreslí COVID pás a label.
    all_vals: numpy array hodnot pro výpočet y-souřadnice (nezávislé na ax.get_ylim()).
    """
    cs = pd.Timestamp("2020-01-01")
    ce = pd.Timestamp("2020-09-30")
    if len(hist_index) == 0 or hist_index[0] > cs or ce > hist_index[-1]:
        return
    ax.axvspan(cs, ce, alpha=0.18, color=SK["covid"], zorder=1)
    # Y-souřadnice z dat (ax.get_ylim() není spolehlivé před set_ylim)
    if all_vals is not None and len(all_vals) > 0:
        finite = all_vals[np.isfinite(all_vals)]
        if len(finite) > 0:
            ybot = float(finite.min())
            ytop = float(finite.max())
            span = ytop - ybot if ytop > ybot else abs(ybot) * 0.2 or 1.0
            y_label = ybot + span * 0.04
        else:
            y_label = 0.0
    else:
        # Fallback: transform z axes fraction
        y_label = ax.transData.inverted().transform(
            ax.transAxes.transform([0, 0.04])
        )[1]
    ax.text(cs + pd.Timedelta(days=50), y_label,
            "COVID", fontsize=8, color=SK["muted"], va="bottom",
            style="italic", zorder=10)


# ── Hlavní funkce jednoho sketche ─────────────────────────────────────────────

def plot_sketch_fan(
    ax,
    history: pd.Series,
    intervals: pd.DataFrame,
    variable: str,
    history_years: int = 6,
):
    """Vykreslí jeden hand-drawn fan chart do zadané osy."""
    s = history.dropna()
    cutoff = s.index[-1] - pd.DateOffset(years=history_years)
    hist = s[s.index >= cutoff]

    last_v  = hist.values[-1]
    last_dt = hist.index[-1]

    # Fan pásma – napoj od posledního hist bodu
    fan_idx = pd.DatetimeIndex([last_dt]).append(intervals.index)
    lo90 = np.concatenate([[last_v], intervals["lower_90"].values])
    hi90 = np.concatenate([[last_v], intervals["upper_90"].values])
    lo50 = np.concatenate([[last_v], intervals["lower_50"].values])
    hi50 = np.concatenate([[last_v], intervals["upper_50"].values])
    med  = np.concatenate([[last_v], intervals["median"].values])

    _sketch_axes(ax)

    # COVID pás před ostatními prvky
    ax.axhline(0, color=SK["ink"], linewidth=0.6, alpha=0.3, zorder=1)

    # Fan
    _sketch_fan(ax, fan_idx, lo90, hi90, lo50, hi50, SK["blue"])

    # Prognóza – wobbly linka
    med_w = _wobble(np.arange(len(fan_idx), dtype=float), med, noise=0.008)[1]
    ax.plot(fan_idx, med_w, color=SK["blue"], linewidth=2.0,
            linestyle=(0, (6, 2)),   # dlouhé čárky jako pero
            zorder=6, solid_capstyle="round")

    # Historická data – wobbly
    hist_w = _wobble_series(hist, noise=0.010)
    ax.plot(hist_w.index, hist_w.values, color=SK["ink"],
            linewidth=2.2, zorder=5, solid_capstyle="round")

    # Svislá čára "Nyní" – ručně kreslená (mírně nakloněná přes path effect)
    ax.axvline(last_dt, color=SK["muted"], linewidth=1.2,
               linestyle=(0, (4, 3)), zorder=4, alpha=0.7)

    # ylim a COVID – musí být až po vykreslení dat
    all_vals = np.concatenate([hist.values, lo90, hi90])
    all_vals = all_vals[np.isfinite(all_vals)]

    # COVID pás s y-souřadnicemi z dat
    _covid_band(ax, hist.index, all_vals)

    # Anotace – poslední hodnota
    fmt = ".2f" if variable in ("eurczk", "pribor3m") else "+.1f"
    ax.annotate(format(last_v, fmt),
                xy=(last_dt, last_v),
                xytext=(-38, 10), textcoords="offset points",
                fontsize=9, color=SK["ink"], fontweight="bold")

    # Anotace – konec prognózy
    ax.annotate(format(med[-1], fmt),
                xy=(intervals.index[-1], med[-1]),
                xytext=(6, 0), textcoords="offset points",
                fontsize=9, color=SK["blue"])

    # Osa X
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.get_xticklabels(), ha="center")

    # Y label – va="center" zajistí centrování na střed osy Y
    ylabel = "CZK" if variable == "eurczk" else "%"
    ax.set_ylabel(ylabel, fontsize=9, color=SK["muted"],
                  rotation=0, labelpad=14, va="center")

    # ylim – nastavíme z already-computed all_vals
    if len(all_vals) > 0:
        vmin, vmax = all_vals.min(), all_vals.max()
        span = vmax - vmin if vmax > vmin else abs(vmax) * 0.2 or 1.0
        margin = span * 0.18
        ax.set_ylim(vmin - margin, vmax + margin)

    # Nadpis – set_title s pad zajistí správnou vertikální pozici
    ax.set_title(LABELS.get(variable, variable),
                 fontsize=13, color=SK["ink"], pad=10, loc="center")


# ── Sestavení přehledu 3×2 ────────────────────────────────────────────────────

def build_sketch_overview(
    macro_data: pd.DataFrame,
    macro_intervals: dict,
    fin_data: pd.DataFrame,
    fin_intervals: dict,
    quarter_label: str,
    save_path: str,
    history_years: int = 6,
):
    """
    Sestaví přehled 6 grafů (3 řádky × 2 sloupce) ve sketch stylu.
    """
    with plt.xkcd(scale=0.8, length=120, randomness=1):
        # Nastav typewriter/monospace font pro celý přehled
        font = _SKETCH_FONT or "monospace"
        matplotlib.rcParams["font.family"] = font
        matplotlib.rcParams["font.monospace"] = [font, "Courier New", "Courier", "DejaVu Sans Mono"]

        fig, axes = plt.subplots(
            5, 2,
            figsize=(20, 30),
            facecolor=SK["paper"],
        )
        fig.patch.set_facecolor(SK["paper"])

        # ── Záhlaví ──────────────────────────────────────────────────────────
        fig.text(
            0.5, 0.975,
            f"Makroekonomický přehled – {quarter_label}",
            ha="center", va="top",
            fontsize=32, color=SK["ink"], fontweight="bold",
        )
        fig.text(
            0.5, 0.958,
            "Metropolitní univerzita Praha",
            ha="center", va="top",
            fontsize=14, color=SK["muted"], style="italic",
        )

        # ── Data pro každý panel ─────────────────────────────────────────────
        panels = [
            # (řádek, sloupec, proměnná, zdroj dat, zdroj intervalů)
            (0, 0, "gdp_yoy",   macro_data, macro_intervals),
            (0, 1, "gdp_qoq",   macro_data, macro_intervals),
            (1, 0, "cpi_yoy",   macro_data, macro_intervals),
            (1, 1, "hicp_yoy",  macro_data, macro_intervals),
            (2, 0, "eurczk",    fin_data,   fin_intervals),
            (2, 1, "eurusd",    fin_data,   fin_intervals),
            (3, 0, "wages_yoy", macro_data, macro_intervals),
            (3, 1, "pribor3m",  fin_data,   fin_intervals),
            (4, 0, "unempl",    fin_data,   fin_intervals),
            (4, 1, "repo_rate", fin_data,   fin_intervals),
        ]

        for row, col, var, data_src, ival_src in panels:
            ax = axes[row, col]
            if var not in data_src.columns or var not in ival_src:
                ax.set_visible(False)
                continue
            plot_sketch_fan(
                ax=ax,
                history=data_src[var],
                intervals=ival_src[var],
                variable=var,
                history_years=history_years,
            )

        # ── Legenda (jednou, dole) ────────────────────────────────────────────
        legend_elements = [
            Line2D([0], [0], color=SK["ink"],  linewidth=2,   label="Skutečnost"),
            Line2D([0], [0], color=SK["blue"], linewidth=2,
                   linestyle=(0, (6, 2)),                     label="Prognóza (medián)"),
            mpatches.Patch(facecolor="none", edgecolor=SK["blue"],
                           hatch="///", alpha=0.5,            label="90% interval"),
            mpatches.Patch(facecolor=SK["blue"], alpha=0.12,  label="50% interval"),
        ]
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=4,
            fontsize=11,
            frameon=True,
            facecolor=SK["paper"],
            edgecolor=SK["grid"],
            bbox_to_anchor=(0.5, 0.030),
        )

        # ── Patička ───────────────────────────────────────────────────────────
        fig.text(
            0.5, 0.004,
            f"Zdroj: ČSÚ, Eurostat, ČNB, ECB  |  Prognóza {quarter_label}  |  MUP",
            ha="center", va="bottom",
            fontsize=9, color=SK["muted"], style="italic",
        )

        fig.subplots_adjust(top=0.92, bottom=0.07, hspace=0.45, wspace=0.28)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=SK["paper"])
        plt.close(fig)
        print(f"  Sketch přehled uložen: {save_path}")


# ── Individuální grafy ve sketch stylu ────────────────────────────────────────

def save_sketch_individual(
    macro_data: pd.DataFrame,
    macro_intervals: dict,
    fin_data: pd.DataFrame,
    fin_intervals: dict,
    quarter_label: str,
    out_dir: str,
    history_years: int = 6,
):
    """Uloží každý ze 10 grafů samostatně jako sketch_<var>_<ql>.png."""
    panels = [
        ("gdp_yoy",   macro_data, macro_intervals),
        ("gdp_qoq",   macro_data, macro_intervals),
        ("cpi_yoy",   macro_data, macro_intervals),
        ("hicp_yoy",  macro_data, macro_intervals),
        ("eurczk",    fin_data,   fin_intervals),
        ("eurusd",    fin_data,   fin_intervals),
        ("wages_yoy", macro_data, macro_intervals),
        ("pribor3m",  fin_data,   fin_intervals),
        ("unempl",    fin_data,   fin_intervals),
        ("repo_rate", fin_data,   fin_intervals),
    ]
    ql_safe = quarter_label.replace("-", "")

    with plt.xkcd(scale=0.8, length=120, randomness=1):
        font = _SKETCH_FONT or "monospace"
        matplotlib.rcParams["font.family"] = font
        matplotlib.rcParams["font.monospace"] = [font, "Courier New", "Courier", "DejaVu Sans Mono"]

        for var, data_src, ival_src in panels:
            if var not in data_src.columns or var not in ival_src:
                continue

            fig, ax = plt.subplots(figsize=(9, 5), facecolor=SK["paper"])
            fig.patch.set_facecolor(SK["paper"])

            plot_sketch_fan(
                ax=ax,
                history=data_src[var],
                intervals=ival_src[var],
                variable=var,
                history_years=history_years,
            )

            # Legenda uvnitř grafu (kompaktní)
            legend_elements = [
                Line2D([0], [0], color=SK["ink"],  linewidth=2, label="Skutečnost"),
                Line2D([0], [0], color=SK["blue"], linewidth=2,
                       linestyle=(0, (6, 2)), label="Prognóza"),
                mpatches.Patch(facecolor="none", edgecolor=SK["blue"],
                               hatch="///", alpha=0.5, label="90% int."),
                mpatches.Patch(facecolor=SK["blue"], alpha=0.12, label="50% int."),
            ]
            ax.legend(
                handles=legend_elements,
                loc="upper left", fontsize=8, frameon=True,
                facecolor=SK["paper"], edgecolor=SK["grid"], framealpha=0.85,
            )

            fig.text(0.5, 0.01, f"NOSTRADAMUP  ·  {quarter_label}",
                     ha="center", va="bottom", fontsize=8,
                     color=SK["muted"], style="italic")

            fig.tight_layout(rect=[0, 0.04, 1, 1])
            path = os.path.join(out_dir, f"sketch_{var}_{ql_safe}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SK["paper"])
            plt.close(fig)
            print(f"  sketch_{var}_{ql_safe}.png")


# ── Spuštění ──────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="NOSTRADAMUP – sketch přehled")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Výstupní složka pro grafy")
    parser.add_argument("--expect-weight", type=float, default=0.35,
                        help="Váha inflačních očekávání γ_f (default: 0.35)")
    parser.add_argument("--anchoring",     type=float, default=0.75,
                        help="Ukotvenost očekávání λ (default: 0.75)")
    parser.add_argument("--phillips-convexity", type=float, default=0.8,
                        help="Konvexita Phillipsovy křivky (0=lineární, default: 0.8)")
    parser.add_argument("--erpt-coef",     type=float, default=0.15,
                        help="Kurzový pass-through (default: 0.15)")
    parser.add_argument("--housing-pressure", type=float, default=0.5,
                        help="Domácí tlak nemovitostí+služeb na jádro v pp (default: 0.5)")
    args, _ = parser.parse_known_args()
    global CHARTS_DIR
    if args.output_dir:
        import pathlib
        CHARTS_DIR = str(pathlib.Path(args.output_dir).expanduser().resolve())
        os.makedirs(CHARTS_DIR, exist_ok=True)

    now = datetime.date.today()
    ql  = f"{now.year}-Q{(now.month-1)//3+1}"

    print(f"Generuji sketch přehled pro {ql}...")

    # ── Importy musí být první – _extend_to_present potřebujeme hned ──────────
    try:
        from data_fetch import load_cz_dataset
        from financial_data import build_financial_dataset, load_intervals,                                    forecast_financial, _extend_to_present
        from report_generator import ar_forecast
    except ImportError as e:
        print(f"Chyba importu: {e}")
        return

    # ── Načti makro data a extenduj do současnosti ───────────────────────────
    try:
        # Živá data (nebo čerstvý cache z data_fetch.py) místo záložních demo funkcí
        macro_data = load_cz_dataset().resample("QS").mean().dropna()
        # Doplň každou sérii flat-forward do aktuálního čtvrtletí
        macro_data = pd.DataFrame(
            {col: _extend_to_present(macro_data[col]) for col in macro_data.columns}
        ).sort_index().dropna()
    except Exception as e:
        print(f"Chyba makro dat: {e}")
        return

    # ── Finanční data a intervaly ─────────────────────────────────────────────
    try:
        fin_data = build_financial_dataset(use_cache=True)
        _ipath = os.path.join(BASE_DIR, "data", "raw", "fin_intervals.json")
        if os.path.exists(_ipath):
            fin_intervals = load_intervals(_ipath)
        else:
            fin_intervals = forecast_financial(fin_data, steps=8)
        pribor_path = fin_intervals["pribor3m"]["median"].tolist()
    except Exception as e:
        print(f"  Varování: finanční data nedostupná ({e})")
        fin_data      = pd.DataFrame()
        fin_intervals = {}
        pribor_path   = None

    _any = next(iter(fin_intervals.values()), None) if fin_intervals else None
    SKETCH_STEPS = len(_any) if _any is not None else 12
    macro_intervals = {}
    inflation_vars  = {"hicp_yoy", "cpi_yoy"}
    _ew = getattr(args, "expect_weight", 0.35)
    _anc = getattr(args, "anchoring", 0.75)
    _pcx = getattr(args, "phillips_convexity", 0.8)
    _erpt = getattr(args, "erpt_coef", 0.15)
    _hsp = getattr(args, "housing_pressure", 0.5)
    # Cesty pro křížové vazby
    _unempl_path = (fin_intervals["unempl"]["median"].tolist()
                    if fin_intervals and "unempl" in fin_intervals else None)
    _eurczk_path = (fin_intervals["eurczk"]["median"].tolist()
                    if fin_intervals and "eurczk" in fin_intervals else None)
    _gdp_path = None
    if "gdp_qoq" in macro_data.columns:
        _gtmp = ar_forecast(macro_data["gdp_qoq"], steps=SKETCH_STEPS, pribor_path=pribor_path)
        _gdp_path = _gtmp["median"].tolist()

    for var in ["gdp_qoq", "gdp_yoy", "hicp_yoy", "cpi_yoy", "wages_yoy"]:
        if var not in macro_data.columns:
            continue
        macro_intervals[var] = ar_forecast(
            macro_data[var], steps=SKETCH_STEPS,
            is_inflation=(var in inflation_vars),
            is_wages=(var == "wages_yoy"),
            pribor_path=pribor_path,
            unempl_path=_unempl_path if var == "wages_yoy" else None,
            gdp_path=_gdp_path if var == "wages_yoy" else None,
            eurczk_path=_eurczk_path if var in inflation_vars else None,
            neutral_rate=3.0, inflation_target=2.0,
            mp_sensitivity=0.25, mp_lag=4,
            expect_weight=_ew, anchoring=_anc,
            phillips_convexity=_pcx,
            erpt_coef=_erpt,
            housing_services_pressure=_hsp,
        )

    # ── Vygeneruj přehled ────────────────────────────────────────────────────
    save_path = os.path.join(CHARTS_DIR, f"sketch_prehled_{ql.replace('-','')}.png")
    build_sketch_overview(
        macro_data=macro_data,
        macro_intervals=macro_intervals,
        fin_data=fin_data,
        fin_intervals=fin_intervals,
        quarter_label=ql,
        save_path=save_path,
    )
    print(f"✓ Přehled: {save_path}")

    # ── Individuální grafy ───────────────────────────────────────────────────
    print("\nGeneruji individuální sketch grafy...")
    save_sketch_individual(
        macro_data=macro_data,
        macro_intervals=macro_intervals,
        fin_data=fin_data,
        fin_intervals=fin_intervals,
        quarter_label=ql,
        out_dir=CHARTS_DIR,
    )
    print(f"\n✓ Hotovo. Grafy v: {CHARTS_DIR}")


if __name__ == "__main__":
    main()
