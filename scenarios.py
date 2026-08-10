"""
scenarios.py
============
Scénářová analýza NOSTRADAMUP ve stylu alternativních scénářů ČNB.

Vedle základního scénáře počítá dva alternativní, které se liší kalibrací
moderních kanálů modelu (ukotvenost očekávání, domácí nákladové tlaky):

  1. Základní          - výchozí kalibrace (anchoring 0.75, tlak 0.5 pp)
  2. Inflační tlaky    - částečné de-ukotvení očekávání, silný tlak
                         z nemovitostí a služeb (anchoring 0.55, tlak 1.2 pp)
  3. Rychlá dezinflace - vysoká kredibilita ČNB, domácí tlaky odeznívají
                         (anchoring 0.92, tlak 0.1 pp)

Vnitřní konzistence: inflace a měnová politika se řeší iterativně.
Nejdřív se spočítá inflace při neutrální sazbové cestě, pak repo sazba
Taylorovým pravidlem reagujícím na tuto inflaci, pak PRIBOR navázaný na
repo, a nakonec finální inflace při této sazbové cestě. Dvě iterace
stačí k praktické konvergenci a scénáře tak zachycují simultánnost
inflace a sazeb (jestřábí scénář má vyšší inflaci I vyšší sazby).

Spuštění:
    python scenarios.py
    python scenarios.py --steps 8 --repo-neutral 3.5 --output-dir /cesta

Výstup:
    outputs/charts/scenare_YYYYQN.png   - srovnávací graf (inflace + repo)
    outputs/reports/scenare_YYYYQN.md   - tabulka scénářů
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
       "grid": "#D8E0F0", "hist": "#1A1A2E"}

# Definice scénářů: název, barva, parametry moderních kanálů
SCENARIOS = [
    {
        "key": "zakladni",
        "name": "Základní",
        "color": MUP["blue"],
        "anchoring": 0.75,
        "housing_pressure": 0.5,
        "popis": "Výchozí kalibrace. Očekávání převážně ukotvená, domácí "
                 "nákladové tlaky mírné.",
    },
    {
        "key": "inflacni",
        "name": "Inflační tlaky",
        "color": "#C0392B",
        "anchoring": 0.55,
        "housing_pressure": 1.2,
        "popis": "Částečné de-ukotvení očekávání po letech nad cílem, silný "
                 "tlak z nemovitostí, nájmů a služeb. ČNB reaguje vyššími sazbami.",
    },
    {
        "key": "dezinflace",
        "name": "Rychlá dezinflace",
        "color": "#2E7D32",
        "anchoring": 0.92,
        "housing_pressure": 0.1,
        "popis": "Vysoká kredibilita ČNB, očekávání pevně u 2 %, domácí "
                 "nákladové tlaky odeznívají. Prostor pro dřívější snižování sazeb.",
    },
]


def run_scenario(macro: pd.DataFrame, fin: pd.DataFrame, params: dict,
                 steps: int = 8, repo_neutral: float = 3.5,
                 n_policy_iters: int = 2) -> dict:
    """
    Spočítá jeden scénář s iterací inflace a měnové politiky.

    Kauzální řetěz v každé iteraci:
      repo (Taylor na inflaci z minulé iterace) -> PRIBOR (navázaný)
      -> nezaměstnanost (restrikce) -> mzdy (konvexní PC)
      -> inflace (očekávání + jádro + ERPT + cost-push)
    """
    from report_generator import ar_forecast
    from financial_data import (_forecast_taylor_repo, _forecast_pribor_linked,
                                 _forecast_unemployment, _forecast_rw)

    anchoring = params["anchoring"]
    housing   = params["housing_pressure"]

    repo_s   = fin["repo_rate"].dropna()
    pribor_s = fin["pribor3m"].dropna()
    unempl_s = fin["unempl"].dropna()
    eurczk_s = fin["eurczk"].dropna()

    # Kurz: společný napříč scénáři (random walk s driftem)
    eurczk_path = _forecast_rw(eurczk_s, steps=steps)["median"].tolist()

    # Startovní inflační cesta pro Taylorovo pravidlo (iterace 0):
    # jednoduché AR bez sazbové zpětné vazby
    infl0 = ar_forecast(macro["hicp_yoy"], steps=steps, is_inflation=True,
                        inflation_target=2.0, anchoring=anchoring,
                        housing_services_pressure=housing)
    inflation_path = infl0["median"].tolist()

    result = {}
    for _ in range(n_policy_iters):
        # 1) Repo: Taylor reagující na scénářovou inflaci
        repo_iv = _forecast_taylor_repo(
            repo_s, inflation_path=inflation_path, steps=steps,
            neutral_rate=repo_neutral,
            anchor_to_neutral=False,   # scénáře: sazby reagují na úroveň inflace
            smoothing=0.7,             # setrvačnost: ČNB reaguje postupně
        )
        repo_path = repo_iv["median"].tolist()

        # 2) PRIBOR navázaný na repo
        pribor_iv = _forecast_pribor_linked(
            pribor=pribor_s, repo=repo_s, repo_path=repo_path, steps=steps,
        )
        pribor_path = pribor_iv["median"].tolist()

        # 3) Nezaměstnanost: restrikce chladí trh práce
        unempl_iv = _forecast_unemployment(
            unempl=unempl_s, repo_path=repo_path, steps=steps,
            neutral_rate=repo_neutral,
        )
        unempl_path = unempl_iv["median"].tolist()

        # 3b) HDP: dynamická IS křivka reaguje na sazbovou cestu scénáře.
        #     Restriktivnější scénář = vyšší reálné sazby = slabší poptávka.
        gdp_iv = ar_forecast(
            macro["gdp_qoq"], steps=steps, is_gdp=True,
            pribor_path=pribor_path, inflation_path=inflation_path,
            is_sensitivity=0.05,
        )
        gdp_iv_yoy = ar_forecast(
            macro["gdp_yoy"], steps=steps, is_gdp=True, gdp_cumulative=True,
            pribor_path=pribor_path, inflation_path=inflation_path,
            is_sensitivity=0.05,
        )

        # 4) Mzdy: konvexní Phillipsova křivka
        wages_iv = ar_forecast(
            macro["wages_yoy"], steps=steps, is_wages=True,
            unempl_path=unempl_path,
            gdp_path=gdp_iv["median"].tolist(),
        )
        wages_path = wages_iv["median"].tolist()

        # 5) Inflace: plný kanálový mix se scénářovými parametry
        hicp_iv = ar_forecast(
            macro["hicp_yoy"], steps=steps, is_inflation=True,
            pribor_path=pribor_path, wages_path=wages_path,
            eurczk_path=eurczk_path,
            inflation_target=2.0,
            anchoring=anchoring,
            housing_services_pressure=housing,
        )
        inflation_path = hicp_iv["median"].tolist()

        result = {
            "hicp": hicp_iv, "repo": repo_iv, "pribor": pribor_iv,
            "unempl": unempl_iv, "wages": wages_iv,
            "gdp": gdp_iv, "gdp_yoy": gdp_iv_yoy,
        }

    # CPI: stejná dynamika, vlastní série
    result["cpi"] = ar_forecast(
        macro["cpi_yoy"], steps=steps, is_inflation=True,
        pribor_path=result["pribor"]["median"].tolist(),
        wages_path=result["wages"]["median"].tolist(),
        eurczk_path=eurczk_path,
        inflation_target=2.0, anchoring=anchoring,
        housing_services_pressure=housing,
    )
    return result


def plot_scenarios(macro: pd.DataFrame, fin: pd.DataFrame,
                   results: dict, ql: str, save_path: str):
    """Srovnávací graf: inflace HICP (levý panel) a repo sazba (pravý)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), facecolor=MUP["bg"])
    for ax in (ax1, ax2):
        ax.set_facecolor(MUP["bg"])
        ax.grid(True, color=MUP["grid"], linewidth=0.7, linestyle="--")
        ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

    hist_i = macro["hicp_yoy"].dropna().tail(16)
    ax1.plot(hist_i.index, hist_i.values, color=MUP["hist"], lw=2,
             label="Skutečnost")
    hist_r = fin["repo_rate"].dropna().tail(16)
    ax2.plot(hist_r.index, hist_r.values, color=MUP["hist"], lw=2,
             label="Skutečnost")

    for sc in SCENARIOS:
        r = results[sc["key"]]
        iv_i, iv_r = r["hicp"], r["repo"]
        ax1.plot(iv_i.index, iv_i["median"].values, color=sc["color"], lw=2,
                 ls=(0, (5, 2)), marker="o", ms=3, label=sc["name"])
        ax1.fill_between(iv_i.index, iv_i["lower_50"], iv_i["upper_50"],
                         color=sc["color"], alpha=0.10)
        ax2.plot(iv_r.index, iv_r["median"].values, color=sc["color"], lw=2,
                 ls=(0, (5, 2)), marker="o", ms=3, label=sc["name"])

    ax1.axhline(2.0, color=MUP["navy"], lw=1, ls=":", alpha=0.6)
    ax1.text(hist_i.index[0], 2.05, "cíl ČNB 2 %", fontsize=8,
             color=MUP["navy"], alpha=0.8)
    ax1.axvline(hist_i.index[-1], color="#999", lw=1, ls=":")
    ax2.axvline(hist_r.index[-1], color="#999", lw=1, ls=":")

    ax1.set_title("Inflace HICP podle scénářů (%)", fontsize=13,
                  color=MUP["navy"], fontweight="bold")
    ax2.set_title("Repo sazba ČNB podle scénářů (%)", fontsize=13,
                  color=MUP["navy"], fontweight="bold")
    for ax in (ax1, ax2):
        ax.legend(fontsize=9, loc="best")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle(f"NOSTRADAMUP scénářová analýza  |  {ql}",
                 fontsize=15, color=MUP["navy"], fontweight="bold")
    fig.text(0.5, 0.005,
             "Scénáře se liší ukotvením inflačních očekávání a domácím "
             "nákladovým tlakem. Sazby reagují Taylorovým pravidlem.",
             fontsize=8.5, color="#5A6478", ha="center")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=MUP["bg"])
    plt.close(fig)


def write_scenario_report(results: dict, ql: str, path: str):
    """Markdown tabulka scénářů."""
    lines = [f"# NOSTRADAMUP scénářová analýza | {ql}", ""]
    lines.append("Alternativní scénáře ve stylu ČNB. Liší se ukotvením "
                 "inflačních očekávání (kredibilita ČNB) a silou domácích "
                 "nákladových tlaků (nemovitosti, nájmy, služby). Sazby v "
                 "každém scénáři endogenně reagují Taylorovým pravidlem, "
                 "takže inflačnější svět znamená i vyšší sazby.")
    lines.append("")
    lines.append("> Pozn.: Ve scénářovém režimu je měnová politika plně endogenní "
                 "(inertní Taylorovo pravidlo reaguje na úroveň inflace). Základní "
                 "scénář se proto může mírně lišit od hlavní prognózy, která repo "
                 "ukotvuje k uživatelem zadané neutrální sazbě.")
    lines.append("")
    for sc in SCENARIOS:
        lines.append(f"- **{sc['name']}** (λ={sc['anchoring']}, "
                     f"tlak {sc['housing_pressure']} pp): {sc['popis']}")
    lines.append("")

    def _v(r, var, q):
        try:
            return f"{r[var]['median'].values[q-1]:.2f}"
        except Exception:
            return "-"

    for horizon_q, label in [(4, "4 čtvrtletí"), (8, "8 čtvrtletí")]:
        lines.append(f"## Horizont {label}")
        lines.append("")
        lines.append("| Scénář | HDP YoY (%) | HICP (%) | CPI (%) | Repo (%) | PRIBOR 3M (%) | Nezam. (%) | Mzdy (%) |")
        lines.append("|--------|-------------|----------|---------|----------|---------------|------------|----------|")
        for sc in SCENARIOS:
            r = results[sc["key"]]
            lines.append(f"| {sc['name']} | {_v(r,'gdp_yoy',horizon_q)} | {_v(r,'hicp',horizon_q)} | "
                         f"{_v(r,'cpi',horizon_q)} | {_v(r,'repo',horizon_q)} | "
                         f"{_v(r,'pribor',horizon_q)} | {_v(r,'unempl',horizon_q)} | "
                         f"{_v(r,'wages',horizon_q)} |")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="NOSTRADAMUP scénářová analýza")
    parser.add_argument("--steps", type=int, default=8,
                        help="Prognózní horizont ve čtvrtletích (default: 8)")
    parser.add_argument("--repo-neutral", type=float, default=3.5,
                        help="Neutrální repo sazba pro Taylorovo pravidlo")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Výstupní složka")
    args, _ = parser.parse_known_args()

    global CHARTS_DIR, REPORTS_DIR
    if args.output_dir:
        import pathlib
        CHARTS_DIR = str(pathlib.Path(args.output_dir).expanduser().resolve())
        REPORTS_DIR = CHARTS_DIR
        os.makedirs(CHARTS_DIR, exist_ok=True)

    from data_fetch import load_cz_dataset
    from financial_data import build_financial_dataset, _extend_to_present

    print("Načítám data...")
    macro = load_cz_dataset().resample("QS").mean().dropna()
    macro = pd.DataFrame(
        {c: _extend_to_present(macro[c]) for c in macro.columns}
    ).sort_index().dropna()
    fin = build_financial_dataset(use_cache=True)

    now = datetime.date.today()
    ql = f"{now.year}-Q{(now.month - 1) // 3 + 1}"

    results = {}
    print(f"Počítám {len(SCENARIOS)} scénáře (horizont {args.steps}Q, "
          f"2 iterace inflace a sazeb)...")
    for sc in SCENARIOS:
        print(f"  {sc['name']} (λ={sc['anchoring']}, tlak {sc['housing_pressure']} pp)")
        results[sc["key"]] = run_scenario(
            macro, fin, sc, steps=args.steps, repo_neutral=args.repo_neutral,
        )

    chart_path = os.path.join(CHARTS_DIR, f"scenare_{ql.replace('-','')}.png")
    plot_scenarios(macro, fin, results, ql, chart_path)
    print(f"\n✓ Srovnávací graf: {chart_path}")

    report_path = os.path.join(REPORTS_DIR, f"scenare_{ql.replace('-','')}.md")
    write_scenario_report(results, ql, report_path)
    print(f"✓ Tabulka scénářů: {report_path}")

    # Souhrn do konzole
    print(f"\nInflace HICP za 4Q podle scénářů:")
    for sc in SCENARIOS:
        v = results[sc["key"]]["hicp"]["median"].values[3]
        r = results[sc["key"]]["repo"]["median"].values[3]
        print(f"  {sc['name']:<18} HICP {v:.2f} %  |  repo {r:.2f} %")


if __name__ == "__main__":
    main()
