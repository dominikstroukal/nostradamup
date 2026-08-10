"""
backtest.py
===========
Out-of-sample validace modelu NOSTRADAMUP.

Pseudo out-of-sample backtest: pro každé historické čtvrtletí T se model
postaví "do minulosti" (vidí jen data do T), vygeneruje prognózu T+1..T+H
a porovná ji se skutečností, která pak nastala.

Metriky:
  - RMSE a MAE podle horizontu (1Q, 2Q, ... HQ)
  - Srovnání proti naivnímu benchmarku (random walk = "beze změny")
  - Theilův U poměr (RMSE_model / RMSE_random_walk); U < 1 => model je lepší

Spuštění:
    python backtest.py                  # všechny proměnné, 12 oken
    python backtest.py --windows 16     # 16 historických oken
    python backtest.py --var hicp_yoy   # jen jedna proměnná

Výstup:
    outputs/reports/backtest_YYYYMMDD.md   – tabulky metrik
    outputs/charts/backtest_<var>.png      – overlay prognóza vs skutečnost
"""

import os
import argparse
import datetime
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib").setLevel(logging.ERROR)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR  = os.path.join(BASE_DIR, "outputs", "charts")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
for d in [CHARTS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

MUP = {"navy": "#00205B", "blue": "#003DA5", "bg": "#EEF2FA",
       "grid": "#D8E0F0", "actual": "#1A1A2E", "rw": "#B0B7C5"}

LABELS = {
    "gdp_qoq":   "HDP mezikvartální (%)",
    "gdp_yoy":   "HDP meziroční (%)",
    "hicp_yoy":  "Inflace HICP (%)",
    "cpi_yoy":   "Inflace CPI ČSÚ (%)",
    "wages_yoy": "Mzdy meziroční (%)",
    "eurczk":    "EUR/CZK",
    "eurusd":    "EUR/USD",
    "pribor3m":  "PRIBOR 3M (%)",
    "repo_rate": "Repo sazba (%)",
    "unempl":    "Nezaměstnanost (%)",
}


# ── Načtení dat ───────────────────────────────────────────────────────────────

def load_all_data() -> pd.DataFrame:
    """Sestaví kompletní dataset všech proměnných (makro + finanční)."""
    from data_fetch import load_cz_dataset
    from financial_data import build_financial_dataset

    macro = load_cz_dataset().resample("QS").mean()
    fin = build_financial_dataset(use_cache=True)
    df = macro.join(fin, how="outer").sort_index()
    return df


# ── Jádro backtestu ─────────────────────────────────────────────────────────

def _forecast_at(series: pd.Series, var: str, horizon: int,
                 repo_series: pd.Series | None = None,
                 aux: dict | None = None):
    """
    Vygeneruje prognózu pro 'var' z dat 'series' (již zkrácených do bodu T).
    Vrátí pole mediánů délky 'horizon'. Bez extend (nepřidávat současnost).

    Testuje SKUTEČNÉ produkční modely: PRIBOR navázaný na repo cestu
    a nezaměstnanost navázanou na měnovou restrikci, ne zastaralé verze.
    repo_series: repo sazba zkrácená do stejného bodu T (pro pribor a unempl).
    """
    from report_generator import ar_forecast
    from financial_data import (_forecast_rw, _forecast_taylor_repo,
                                 _forecast_pribor_linked, _forecast_unemployment,
                                 _forecast_mean_reversion)

    macro_vars = {"gdp_qoq", "gdp_yoy", "hicp_yoy", "cpi_yoy", "wages_yoy"}
    inflation_vars = {"hicp_yoy", "cpi_yoy"}

    if var in macro_vars:
        # HDP: testuj produkční konfiguraci s dynamickou IS křivkou.
        # Reálná sazba z repo cesty (proxy PRIBOR, spread ~0) a rychlého
        # inflačního odhadu z dat dostupných v bodě T (žádný leak budoucnosti).
        gdp_kwargs = {}
        if var in ("gdp_qoq", "gdp_yoy") and aux is not None:
            repo_s = aux.get("repo")
            hicp_s = aux.get("hicp")
            if repo_s is not None and len(repo_s.dropna()) >= 8:
                try:
                    rp = _forecast_taylor_repo(
                        repo_s.dropna(), steps=horizon, neutral_rate=3.5,
                    )["median"].tolist()
                    ip = None
                    if hicp_s is not None and len(hicp_s.dropna()) >= 12:
                        ip = ar_forecast(hicp_s.dropna(), steps=horizon,
                                         is_inflation=True, extend=False,
                                         )["median"].tolist()
                    gdp_kwargs = dict(
                        is_gdp=True,
                        gdp_cumulative=(var == "gdp_yoy"),
                        pribor_path=rp,
                        inflation_path=ip,
                        is_sensitivity=aux.get("is_sensitivity", 0.05),
                    )
                except Exception:
                    gdp_kwargs = {}
        iv = ar_forecast(
            series, steps=horizon,
            is_inflation=(var in inflation_vars),
            inflation_target=2.0,
            extend=False,          # KLÍČOVÉ pro backtest
            **gdp_kwargs,
        )
        return iv["median"].values

    # Finanční proměnné - stejné modely jako v produkci (forecast_financial)
    if var in ("eurczk", "eurusd"):
        return _forecast_rw(series, steps=horizon)["median"].values

    # Repo cesta z Taylorova pravidla (společný vstup pro repo/pribor/unempl)
    repo_path = None
    if repo_series is not None and len(repo_series.dropna()) >= 4:
        try:
            repo_path = _forecast_taylor_repo(
                repo_series.dropna(), steps=horizon, neutral_rate=3.5,
            )["median"].tolist()
        except Exception:
            repo_path = None

    if var == "repo_rate":
        return _forecast_taylor_repo(series, steps=horizon, neutral_rate=3.5)["median"].values
    if var == "pribor3m":
        if repo_path is not None and repo_series is not None:
            return _forecast_pribor_linked(
                pribor=series.dropna(), repo=repo_series.dropna(),
                repo_path=repo_path, steps=horizon, speed=0.30,
            )["median"].values
        return _forecast_mean_reversion(series, steps=horizon,
                                        long_run_mean=3.0, speed=0.30)["median"].values
    if var == "unempl":
        return _forecast_unemployment(
            unempl=series.dropna(), repo_path=repo_path,
            steps=horizon, neutral_rate=3.5,
        )["median"].values
    return None


def backtest_variable(df: pd.DataFrame, var: str, n_windows: int = 12,
                      horizon: int = 8, exclude_covid: bool = True) -> dict:
    """
    Walk-forward backtest jedné proměnné.
    Pro posledních n_windows čtvrtletí vždy odřízne data v bodě T,
    prognózuje horizon kroků, porovná se skutečností.

    exclude_covid: vynechá z evaluace okna, jejichž prognózní cíl zasahuje
                   do covidového propadu 2020 (standardní praxe – 2020 je
                   extrémní outlier, který znehodnocuje srovnání).
    """
    s = df[var].dropna()
    if len(s) < n_windows + horizon + 8:
        n_windows = max(4, len(s) - horizon - 8)

    covid_start = pd.Timestamp("2020-01-01")
    covid_end   = pd.Timestamp("2020-12-31")

    # Sběr chyb: errors[h] = seznam chyb pro horizont h
    model_err = {h: [] for h in range(1, horizon + 1)}
    rw_err    = {h: [] for h in range(1, horizon + 1)}

    last_fc = None
    last_origin = None

    start_idx = len(s) - n_windows - horizon
    start_idx = max(8, start_idx)

    for origin in range(start_idx, len(s) - 1):
        train = s.iloc[:origin + 1]
        actual_future = s.iloc[origin + 1:origin + 1 + horizon]
        h_avail = len(actual_future)
        if h_avail < 1:
            continue

        try:
            repo_train = None
            aux = None
            if var in ("pribor3m", "unempl") and "repo_rate" in df.columns:
                repo_train = df["repo_rate"].dropna()
                repo_train = repo_train[repo_train.index <= train.index[-1]]
            if var in ("gdp_qoq", "gdp_yoy"):
                aux = {}
                if "repo_rate" in df.columns:
                    _r = df["repo_rate"].dropna()
                    aux["repo"] = _r[_r.index <= train.index[-1]]
                if "hicp_yoy" in df.columns:
                    _h = df["hicp_yoy"].dropna()
                    aux["hicp"] = _h[_h.index <= train.index[-1]]
            fc = _forecast_at(train, var, horizon, repo_series=repo_train, aux=aux)
        except Exception:
            continue
        if fc is None:
            continue

        rw_value = float(train.iloc[-1])
        origin_date = train.index[-1]
        origin_in_covid = covid_start <= origin_date <= covid_end

        for h in range(1, h_avail + 1):
            target_date = actual_future.index[h - 1]
            # Vynech pokud počátek NEBO cíl prognózy spadá do covidu 2020.
            # Okna s počátkem v 2020 mají trénink kontaminovaný extrémními
            # hodnotami (-8,7 % / +6,7 %), cíle v 2020 jsou neprognózovatelné.
            if exclude_covid and (origin_in_covid or
                                  covid_start <= target_date <= covid_end):
                continue
            a = float(actual_future.iloc[h - 1])
            model_err[h].append(fc[h - 1] - a)
            rw_err[h].append(rw_value - a)

        last_fc = fc
        last_origin = train.index[-1]

    # Spočítej metriky
    def rmse(errs): return float(np.sqrt(np.mean(np.square(errs)))) if errs else np.nan
    def mae(errs):  return float(np.mean(np.abs(errs))) if errs else np.nan

    metrics = {"horizon": [], "rmse": [], "mae": [], "rmse_rw": [], "theil_u": [], "n": []}
    for h in range(1, horizon + 1):
        r_m = rmse(model_err[h])
        r_rw = rmse(rw_err[h])
        metrics["horizon"].append(h)
        metrics["rmse"].append(r_m)
        metrics["mae"].append(mae(model_err[h]))
        metrics["rmse_rw"].append(r_rw)
        metrics["theil_u"].append(r_m / r_rw if r_rw and r_rw > 0 else np.nan)
        metrics["n"].append(len(model_err[h]))

    return {"var": var, "metrics": metrics, "series": s,
            "last_fc": last_fc, "last_origin": last_origin, "horizon": horizon}


# ── Vizualizace ───────────────────────────────────────────────────────────────

def plot_backtest(result: dict, save_path: str):
    """Overlay poslední prognózy vs skutečnost + Theilovo U podle horizontu."""
    var = result["var"]
    s = result["series"]
    fc = result["last_fc"]
    origin = result["last_origin"]
    m = result["metrics"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5),
                                   facecolor=MUP["bg"],
                                   gridspec_kw={"width_ratios": [2, 1]})
    for ax in (ax1, ax2):
        ax.set_facecolor(MUP["bg"])
        ax.grid(True, color=MUP["grid"], linewidth=0.7, linestyle="--")
        ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

    # Levý panel: skutečnost + poslední backtest prognóza
    hist = s[s.index <= origin].tail(20)
    ax1.plot(hist.index, hist.values, color=MUP["actual"], lw=2, label="Skutečnost")
    if fc is not None:
        fut_idx = pd.date_range(start=origin + pd.offsets.QuarterBegin(1),
                                periods=len(fc), freq="QS")
        ax1.plot(fut_idx, fc, color=MUP["blue"], lw=2, ls=(0, (5, 2)),
                 marker="o", ms=3, label="Prognóza modelu")
        # Skutečnost v prognózním okně
        actual_fut = s[(s.index > origin)].head(len(fc))
        if len(actual_fut):
            ax1.plot(actual_fut.index, actual_fut.values, color="#C0392B",
                     lw=1.5, marker="x", ms=4, label="Skutečnost (ex-post)")
        ax1.axvline(origin, color=MUP["rw"], lw=1, ls=":")
    ax1.set_title(f"{LABELS.get(var, var)} – poslední backtest okno",
                  fontsize=12, color=MUP["navy"])
    ax1.legend(fontsize=8, loc="best")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Pravý panel: Theilovo U podle horizontu
    horizons = m["horizon"]
    theil = m["theil_u"]
    colors = [MUP["blue"] if (u is not None and u < 1) else "#C0392B" for u in theil]
    ax2.bar(horizons, theil, color=colors, alpha=0.8)
    ax2.axhline(1.0, color=MUP["actual"], lw=1.2, ls="--", label="Random walk (U=1)")
    ax2.set_title("Theilovo U podle horizontu", fontsize=12, color=MUP["navy"])
    ax2.set_xlabel("Horizont (čtvrtletí)", fontsize=9)
    ax2.set_ylabel("RMSE model / RMSE RW", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.text(0.5, 0.95, "U < 1: model lepší než RW", transform=ax2.transAxes,
             fontsize=8, color=MUP["navy"], ha="center", va="top", style="italic")

    fig.suptitle(f"NOSTRADAMUP backtest – {LABELS.get(var, var)}",
                 fontsize=14, color=MUP["navy"], fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=MUP["bg"])
    plt.close(fig)


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(results: list, path: str, exclude_covid: bool = True):
    """Sestaví Markdown report s tabulkami metrik."""
    lines = ["# NOSTRADAMUP – Backtest (out-of-sample validace)", ""]
    lines.append(f"Datum: {datetime.date.today().isoformat()}")
    lines.append("")
    lines.append("Pseudo out-of-sample walk-forward backtest. Pro každé historické "
                 "čtvrtletí model viděl jen data do daného bodu a prognózoval vpřed. "
                 "Theilovo U = RMSE modelu / RMSE random walk; **U < 1 znamená, že "
                 "model překonává naivní benchmark.**")
    lines.append("")
    if exclude_covid:
        lines.append("> **Poznámka:** Covidová čtvrtletí 2020 jsou z evaluace vyloučena "
                     "jako extrémní outlier (standardní praxe ČNB, MMF i akademických "
                     "studií). Propad HDP o −8,7 % a následný odraz nejsou prognózovatelné "
                     "žádným modelem a znehodnocují srovnání. Spusťte s `--include-covid` "
                     "pro zahrnutí.")
        lines.append("")

    # Souhrnná tabulka – Theilovo U na klíčových horizontech
    lines.append("## Souhrn – Theilovo U (model vs. random walk)")
    lines.append("")
    lines.append("| Proměnná | 1Q | 4Q | 8Q | Verdikt |")
    lines.append("|----------|-----|-----|-----|---------|")
    for r in results:
        m = r["metrics"]
        def u_at(h):
            try:
                i = m["horizon"].index(h)
                v = m["theil_u"][i]
                return f"{v:.2f}" if v == v else "—"
            except (ValueError, IndexError):
                return "—"
        u4 = m["theil_u"][3] if len(m["theil_u"]) > 3 else None
        verdikt = ("✅ lepší než RW" if (u4 is not None and u4 == u4 and u4 < 1)
                   else "⚠️ horší než RW" if (u4 is not None and u4 == u4) else "—")
        lines.append(f"| {LABELS.get(r['var'], r['var'])} | "
                     f"{u_at(1)} | {u_at(4)} | {u_at(8)} | {verdikt} |")
    lines.append("")

    # Detailní tabulky RMSE/MAE
    for r in results:
        m = r["metrics"]
        lines.append(f"## {LABELS.get(r['var'], r['var'])}")
        lines.append("")
        lines.append("| Horizont | RMSE model | MAE model | RMSE RW | Theil U | N oken |")
        lines.append("|----------|-----------|-----------|---------|---------|--------|")
        for i, h in enumerate(m["horizon"]):
            def fmt(x): return f"{x:.3f}" if (x is not None and x == x) else "—"
            lines.append(f"| {h}Q | {fmt(m['rmse'][i])} | {fmt(m['mae'][i])} | "
                         f"{fmt(m['rmse_rw'][i])} | {fmt(m['theil_u'][i])} | {m['n'][i]} |")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NOSTRADAMUP backtest")
    parser.add_argument("--windows", type=int, default=12,
                        help="Počet historických oken (default: 12)")
    parser.add_argument("--horizon", type=int, default=8,
                        help="Prognózní horizont ve čtvrtletích (default: 8)")
    parser.add_argument("--var", type=str, default=None,
                        help="Jen jedna proměnná (default: všechny)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Výstupní složka")
    parser.add_argument("--include-covid", action="store_true", default=False,
                        help="Zahrnout covidová čtvrtletí 2020 do evaluace (jinak vyloučena jako outlier)")
    args = parser.parse_args()
    exclude_covid = not args.include_covid

    global CHARTS_DIR, REPORTS_DIR
    if args.output_dir:
        import pathlib
        CHARTS_DIR = str(pathlib.Path(args.output_dir).expanduser().resolve())
        REPORTS_DIR = CHARTS_DIR
        os.makedirs(CHARTS_DIR, exist_ok=True)

    print("Načítám data...")
    df = load_all_data()

    all_vars = ["gdp_qoq", "gdp_yoy", "hicp_yoy", "cpi_yoy", "wages_yoy",
                "eurczk", "eurusd", "pribor3m", "repo_rate", "unempl"]
    vars_to_test = [args.var] if args.var else all_vars

    print(f"Backtest {len(vars_to_test)} proměnných, {args.windows} oken, "
          f"horizont {args.horizon}Q...\n")

    results = []
    for var in vars_to_test:
        if var not in df.columns:
            print(f"  {var}: chybí v datech, přeskakuji")
            continue
        print(f"  Backtest: {var} ...", end=" ")
        r = backtest_variable(df, var, n_windows=args.windows, horizon=args.horizon,
                              exclude_covid=exclude_covid)
        results.append(r)
        # Klíčové U na 4Q
        u4 = r["metrics"]["theil_u"][3] if len(r["metrics"]["theil_u"]) > 3 else None
        verdikt = "lepší než RW" if (u4 and u4 < 1) else "horší než RW" if u4 else "?"
        print(f"U(4Q)={u4:.2f} ({verdikt})" if u4 and u4 == u4 else "hotovo")
        # Graf
        plot_backtest(r, os.path.join(CHARTS_DIR, f"backtest_{var}.png"))

    # Report
    ts = datetime.date.today().strftime("%Y%m%d")
    report_path = os.path.join(REPORTS_DIR, f"backtest_{ts}.md")
    write_report(results, report_path, exclude_covid=exclude_covid)
    print(f"\n✓ Report: {report_path}")
    print(f"  Grafy: {CHARTS_DIR}/backtest_*.png")


if __name__ == "__main__":
    main()
