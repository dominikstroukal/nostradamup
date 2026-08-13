"""
export_web.py
=============
Exportuje výstupy modelu do sebe-popisného JSON pro statický web.

KLÍČOVÝ PRINCIP: JSON obsahuje i METADATA (seznam proměnných, jejich názvy,
jednotky, parametry modelu). Frontend nic natvrdo nezná - jen vykreslí, co
najde. Když do modelu přidáš proměnnou nebo kanál, objeví se na webu sama,
bez úprav HTML/JS.

Spuštění:
    python export_web.py                      # do web/data/
    python export_web.py --out web/data       # explicitně

Výstup:
    web/data/latest.json     - aktuální prognóza (vše co web potřebuje)
    web/data/history.json    - archiv vintages (jak se prognóza měnila)
"""

import os
import json
import glob
import argparse
import datetime
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _ser(intervals_df):
    """DataFrame intervalů -> list dictů se stringovými datumy (JSON safe)."""
    out = []
    for idx, row in intervals_df.iterrows():
        out.append({
            "date": pd.Timestamp(idx).strftime("%Y-%m-%d"),
            "median": round(float(row["median"]), 3),
            "lower_50": round(float(row["lower_50"]), 3),
            "upper_50": round(float(row["upper_50"]), 3),
            "lower_90": round(float(row["lower_90"]), 3),
            "upper_90": round(float(row["upper_90"]), 3),
        })
    return out


def _hist(series, n=24):
    """Historie série -> list dictů."""
    s = series.dropna().tail(n)
    return [{"date": pd.Timestamp(i).strftime("%Y-%m-%d"), "value": round(float(v), 3)}
            for i, v in s.items()]


def build_payload(args) -> dict:
    from data_fetch import load_cz_dataset
    from financial_data import build_financial_dataset, load_intervals, forecast_financial
    from report_generator import (ar_forecast, _extend_to_present,
                                  inflation_decomposition, generate_commentary)
    from cnb_survey import build_cnb_table

    macro = load_cz_dataset().resample("QS").mean().dropna()
    macro = pd.DataFrame({c: _extend_to_present(macro[c]) for c in macro.columns}
                         ).sort_index().dropna()

    # ── Nowcast HDP: kotva aktuálního čtvrtletí ────────────────────────────
    # Nowcast z měsíčních indikátorů nahradí flat-forward odhad běžného
    # čtvrtletí, takže prognóza i inflační kanály startují z DAT, ne z kopie
    # minulého Q. Sjednocuje nowcast s prognózou. Výpadek export neshodí.
    nowcast = None
    if not args.no_nowcast:
        try:
            from nowcast import run_nowcast
            nowcast = run_nowcast()
            cq = pd.Period(nowcast["current_quarter"], "Q").to_timestamp()
            if cq in macro.index:
                macro.loc[cq, "gdp_qoq"] = nowcast["estimate"]
                macro.loc[cq, "gdp_yoy"] = float(macro["gdp_qoq"].loc[:cq].tail(4).sum())
                print(f"  nowcast {nowcast['current_quarter']} = {nowcast['estimate']} % QoQ "
                      f"→ kotva startu prognózy")
            else:
                print(f"  nowcast {nowcast['current_quarter']} = {nowcast['estimate']} % "
                      f"(mimo macro index, jen zobrazeno)")
        except Exception as e:
            print(f"  (nowcast přeskočen: {e})")

    fin = build_financial_dataset(use_cache=True)

    ipath = os.path.join(BASE_DIR, "data", "raw", "fin_intervals.json")
    if os.path.exists(ipath):
        fin_iv = load_intervals(ipath)
    else:
        fin_iv = forecast_financial(fin, steps=args.horizon)

    steps = len(next(iter(fin_iv.values())))
    pribor_path = fin_iv["pribor3m"]["median"].tolist() if "pribor3m" in fin_iv else None
    eurczk_path = fin_iv["eurczk"]["median"].tolist() if "eurczk" in fin_iv else None

    # ── Makro prognózy (stejný řetěz jako report) ───────────────────────────
    forecast = pd.DataFrame(index=pd.date_range(
        start=macro.index[-1] + pd.offsets.QuarterBegin(1), periods=steps, freq="QS"))
    macro_iv = {}

    unempl_path = fin_iv["unempl"]["median"].tolist() if "unempl" in fin_iv else None
    infl_guess = None
    for _ in range(2):
        for var in ["gdp_qoq", "gdp_yoy"]:
            if var in macro.columns:
                iv = ar_forecast(macro[var], steps=steps, is_gdp=True,
                                 gdp_cumulative=(var == "gdp_yoy"),
                                 pribor_path=pribor_path, inflation_path=infl_guess,
                                 is_sensitivity=args.is_sensitivity)
                macro_iv[var] = iv
                forecast[var] = iv["median"].values
        gdp_path = list(forecast["gdp_qoq"]) if "gdp_qoq" in forecast else None
        if "wages_yoy" in macro.columns:
            iv = ar_forecast(macro["wages_yoy"], steps=steps, is_wages=True,
                             unempl_path=unempl_path, gdp_path=gdp_path,
                             phillips_convexity=args.phillips_convexity)
            macro_iv["wages_yoy"] = iv
            forecast["wages_yoy"] = iv["median"].values
        wages_path = list(forecast.get("wages_yoy", pd.Series([5.0] * steps)))
        for var in ["hicp_yoy", "cpi_yoy"]:
            if var in macro.columns:
                iv = ar_forecast(macro[var], steps=steps, is_inflation=True,
                                 pribor_path=pribor_path, wages_path=wages_path,
                                 gdp_path=gdp_path, eurczk_path=eurczk_path,
                                 anchoring=args.anchoring,
                                 expect_weight=args.expect_weight,
                                 erpt_coef=args.erpt_coef,
                                 housing_services_pressure=args.housing_pressure)
                macro_iv[var] = iv
                forecast[var] = iv["median"].values
        if "hicp_yoy" in forecast:
            infl_guess = list(forecast["hicp_yoy"].values)

    # ── Metadata proměnných (frontend je čte, nic nezná natvrdo) ────────────
    META = {
        "hicp_yoy":  ("Inflace HICP", "%", "inflace", True),
        "cpi_yoy":   ("Inflace CPI ČSÚ", "%", "inflace", True),
        "gdp_yoy":   ("HDP meziročně", "%", "aktivita", True),
        "gdp_qoq":   ("HDP mezikvartálně", "%", "aktivita", False),
        "wages_yoy": ("Průměrné mzdy", "%", "aktivita", False),
        "unempl":    ("Nezaměstnanost", "%", "aktivita", False),
        "repo_rate": ("Repo sazba ČNB", "%", "sazby", True),
        "pribor3m":  ("PRIBOR 3M", "%", "sazby", False),
        "pribor12m": ("PRIBOR 12M", "%", "sazby", False),
        "eurczk":    ("EUR/CZK", "CZK", "kurzy", False),
        "eurusd":    ("EUR/USD", "USD", "kurzy", False),
    }

    variables = {}
    for var, iv in {**macro_iv, **fin_iv}.items():
        label, unit, group, headline = META.get(var, (var, "", "ostatní", False))
        hist_src = macro[var] if var in macro.columns else (
            fin[var] if var in fin.columns else None)
        variables[var] = {
            "label": label, "unit": unit, "group": group, "headline": headline,
            "history": _hist(hist_src) if hist_src is not None else [],
            "forecast": _ser(iv),
        }

    # ── Headline čísla (inflace 1Y a 3Y = jádro sdělení) ────────────────────
    def _at(var, q):
        try:
            return round(float(variables[var]["forecast"][q - 1]["median"]), 1)
        except Exception:
            return None

    headline = {
        "inflace_1y": _at("cpi_yoy", 4),
        "inflace_3y": _at("cpi_yoy", 12),
        "cil": 2.0,
        "hdp_1y": _at("gdp_yoy", 4),
        "repo_1y": _at("repo_rate", 4),
    }
    if headline["inflace_3y"] is not None:
        d = headline["inflace_3y"] - 2.0
        headline["ukotvenost"] = (
            "ukotvená" if abs(d) <= 0.3 else
            ("nad cílem" if d > 0 else "pod cílem"))
        headline["ukotvenost_odchylka"] = round(d, 1)

    # ── Dekompozice inflace ────────────────────────────────────────────────
    decomp = None
    try:
        d = inflation_decomposition(
            macro["hicp_yoy"], steps=steps, pribor_path=pribor_path,
            wages_path=wages_path, gdp_path=gdp_path, eurczk_path=eurczk_path,
            args=args)
        decomp = {"base": d.pop("_base"),
                  "channels": {k: {str(h): round(v, 3) for h, v in vals.items()}
                               for k, vals in d.items()}}
    except Exception as e:
        print(f"  (dekompozice: {e})")

    # ── Scénáře ────────────────────────────────────────────────────────────
    scenarios = None
    try:
        from scenarios import SCENARIOS, run_scenario
        scenarios = []
        for sc in SCENARIOS:
            r = run_scenario(macro, fin, sc, steps=min(steps, 8),
                             repo_neutral=args.repo_neutral)
            scenarios.append({
                "key": sc["key"], "name": sc["name"], "color": sc["color"],
                "popis": sc["popis"],
                "anchoring": sc["anchoring"],
                "housing_pressure": sc["housing_pressure"],
                "hicp": _ser(r["hicp"]), "repo": _ser(r["repo"]),
            })
    except Exception as e:
        print(f"  (scénáře: {e})")

    # ── Komentář + ČNB tabulka ─────────────────────────────────────────────
    now = datetime.date.today()
    ql = f"{now.year}-Q{(now.month - 1) // 3 + 1}"
    try:
        commentary = generate_commentary(macro, forecast, ql,
                                         fin_df=fin, fin_intervals=fin_iv)
    except Exception as e:
        commentary = ""
        print(f"  (komentář: {e})")
    try:
        cnb_table = build_cnb_table(macro, forecast, fin_iv, fin_df=fin,
                                    bond5y=args.bond5y, bond10y=args.bond10y,
                                    swap_spread=args.swap_spread)
    except Exception as e:
        cnb_table = ""
        print(f"  (ČNB tabulka: {e})")

    return {
        "meta": {
            "vintage": ql,
            "generated": now.isoformat(),
            "generated_ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "horizon_quarters": steps,
            "model": "NOSTRADAMUP",
            "institution": "Metropolitní univerzita Praha",
            "data_through": macro.index[-1].strftime("%Y-%m-%d"),
        },
        # Parametry se exportují automaticky - přidáš CLI flag, objeví se na webu
        "parameters": {k: v for k, v in vars(args).items()
                       if k not in ("out", "no_history", "no_nowcast")},
        "headline": headline,
        "nowcast": nowcast,
        "variables": variables,
        "decomposition": decomp,
        "scenarios": scenarios,
        "commentary": commentary,
        "cnb_table": cnb_table,
    }


def main():
    p = argparse.ArgumentParser(description="Export NOSTRADAMUP pro web")
    p.add_argument("--out", default=os.path.join(BASE_DIR, "web", "data"))
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--anchoring", type=float, default=0.75)
    p.add_argument("--expect-weight", type=float, default=0.35)
    p.add_argument("--phillips-convexity", type=float, default=0.8)
    p.add_argument("--erpt-coef", type=float, default=0.15)
    p.add_argument("--housing-pressure", type=float, default=0.5)
    p.add_argument("--is-sensitivity", type=float, default=0.05)
    p.add_argument("--repo-neutral", type=float, default=3.5)
    p.add_argument("--bond5y", type=float, default=4.1)
    p.add_argument("--bond10y", type=float, default=4.7)
    p.add_argument("--swap-spread", type=float, default=0.0)
    p.add_argument("--no-history", action="store_true")
    p.add_argument("--no-nowcast", action="store_true", help="přeskočit nowcast blok")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("Sestavuji export...")
    payload = build_payload(args)

    latest = os.path.join(args.out, "latest.json")
    with open(latest, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"✓ {latest}")

    # ── Archiv vintages: jak se prognóza měnila v čase ──────────────────────
    if not args.no_history:
        hpath = os.path.join(args.out, "history.json")
        hist = []
        if os.path.exists(hpath):
            try:
                hist = json.load(open(hpath))
            except Exception:
                hist = []
        entry = {
            "vintage": payload["meta"]["vintage"],
            "generated": payload["meta"]["generated"],
            "headline": payload["headline"],
        }
        # Nahraď stejný vintage, jinak přidej
        hist = [h for h in hist if h.get("vintage") != entry["vintage"]]
        hist.append(entry)
        hist.sort(key=lambda h: h.get("generated", ""))
        with open(hpath, "w") as f:
            json.dump(hist[-40:], f, ensure_ascii=False, indent=1)
        print(f"✓ {hpath} ({len(hist)} vintages)")

    hl = payload["headline"]
    print(f"\nInflace 1Y: {hl.get('inflace_1y')} % | 3Y: {hl.get('inflace_3y')} % "
          f"({hl.get('ukotvenost')})")


if __name__ == "__main__":
    main()
