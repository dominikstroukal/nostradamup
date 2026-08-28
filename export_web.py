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

    # Poslední SKUTEČNĚ pozorované Q per proměnná (pro poctivé rozdělení
    # historie vs prognóza – nezobrazovat flat-forward jako skutečnost).
    from data_fetch import load_last_obs
    LAST_OBS = {k: pd.Timestamp(v) for k, v in load_last_obs().items()}
    LAST_COMPLETE = macro.index[-1]   # poslední kompletní Q (např. 2026Q2)

    # Veličiny, jejichž ZDROJ ze své podstaty zaostává za posledním kompletním
    # Q (ČSÚ zveřejní mzdy/nezaměstnanost až po Q1, Eurostat HICP je pozadu).
    # Jen ty smějí mít prognózu startující dřív než aktuální Q. Ostatní veličiny
    # jsou vždy aktuální do posledního kompletního Q – i kdyby byl last_obs.json
    # zastaralý, netvoříme u nich mezeru mezi historií a prognózou.
    _LAGGING = {"wages_yoy", "unempl", "hicp_yoy"}

    def obs_end(var):
        """Poslední SKUTEČNĚ pozorované Q veličiny, zastropované na kompletní Q."""
        if var not in _LAGGING:
            return LAST_COMPLETE
        return min(LAST_OBS.get(var, LAST_COMPLETE), LAST_COMPLETE)

    # ── Nowcast HDP: kotva aktuálního čtvrtletí ────────────────────────────
    # Nowcast (z měsíčních indikátorů) přepíše první prognózní bod HDP
    # (aktuální Q), takže prognóza i inflační kanály startují z dat. Nowcast
    # aktuální Q je přesnější než čistě AR odhad. Výpadek export neshodí.
    nowcast = None
    if not args.no_nowcast:
        try:
            from nowcast import run_nowcast
            nowcast = run_nowcast()
            print(f"  nowcast {nowcast['current_quarter']} = {nowcast['estimate']} % QoQ "
                  f"→ přepíše první prognózní bod HDP")
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
        # Nowcast kotva: přepiš první prognózní bod HDP (aktuální Q) nowcastem,
        # aby z něj startovaly i inflační kanály (gdp_path).
        if nowcast and "gdp_qoq" in forecast:
            _q3 = forecast.index[0]
            forecast.loc[_q3, "gdp_qoq"] = nowcast["estimate"]
            macro_iv["gdp_qoq"].loc[macro_iv["gdp_qoq"].index[0], "median"] = nowcast["estimate"]
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

    # ── Poctivé zarovnání: prognóza každé veličiny startuje jejím posledním
    #    SKUTEČNÝM čtvrtletím, ne uniformně aktuálním Q ──────────────────────
    # Historie ukazuje jen skutečně pozorovaná Q (do obs_end). Veličiny, jejichž
    # živý zdroj zaostává (mzdy a nezaměstnanost = ČSÚ zveřejní až Q1, HICP =
    # Eurostat pozadu), se přeforecastují od reálného okraje, aby se flat-forward
    # NEzobrazoval jako skutečnost a model sám vyplnil mezeru. Ostatní veličiny
    # jsou reálné do posledního kompletního Q, jejich prognóza startuje aktuálním
    # Q beze změny. Pořadí přeforecastu = pořadí závislostí (nezam. → mzdy → HICP).
    END = forecast.index[-1]
    _all_iv = {**macro_iv, **fin_iv}

    def _real(var):
        src = macro[var] if var in macro.columns else fin[var]
        return src.loc[:obs_end(var)].dropna()

    def _grid(var):
        return pd.date_range(obs_end(var) + pd.offsets.QuarterBegin(1), END, freq="QS")

    def _path_on(driver, grid):
        """Driver (skutečnost do obs_end + medián prognózy) zarovnaný na grid dle data."""
        iv = _all_iv.get(driver)
        parts = [_real(driver)]
        if iv is not None:
            parts.append(iv["median"][iv["median"].index > obs_end(driver)])
        s = pd.concat(parts).sort_index()
        return list(s.reindex(grid).ffill().bfill().values)

    def _lags(var):
        return obs_end(var) < LAST_COMPLETE

    # 1) Nezaměstnanost (driver mezd přes Phillipsovu křivku).
    if "unempl" in fin.columns and _lags("unempl"):
        from financial_data import _forecast_unemployment
        g = _grid("unempl")
        fin_iv["unempl"] = _forecast_unemployment(
            _real("unempl"), repo_path=_path_on("repo_rate", g),
            steps=len(g), neutral_rate=3.5)
        _all_iv["unempl"] = fin_iv["unempl"]

    # 2) Mzdy (potřebují nezaměstnanost + HDP).
    if "wages_yoy" in macro.columns and _lags("wages_yoy"):
        g = _grid("wages_yoy")
        macro_iv["wages_yoy"] = ar_forecast(
            _real("wages_yoy"), steps=len(g), is_wages=True, extend=False,
            unempl_path=_path_on("unempl", g), gdp_path=_path_on("gdp_qoq", g),
            phillips_convexity=args.phillips_convexity)
        _all_iv["wages_yoy"] = macro_iv["wages_yoy"]

    # 3) HICP (mzdy, sazby, kurz, HDP).
    if "hicp_yoy" in macro.columns and _lags("hicp_yoy"):
        g = _grid("hicp_yoy")
        macro_iv["hicp_yoy"] = ar_forecast(
            _real("hicp_yoy"), steps=len(g), is_inflation=True, extend=False,
            pribor_path=_path_on("pribor3m", g), wages_path=_path_on("wages_yoy", g),
            gdp_path=_path_on("gdp_qoq", g), eurczk_path=_path_on("eurczk", g),
            anchoring=args.anchoring, expect_weight=args.expect_weight,
            erpt_coef=args.erpt_coef,
            housing_services_pressure=args.housing_pressure)
        _all_iv["hicp_yoy"] = macro_iv["hicp_yoy"]

    variables = {}
    for var, iv in {**macro_iv, **fin_iv}.items():
        label, unit, group, headline = META.get(var, (var, "", "ostatní", False))
        variables[var] = {
            "label": label, "unit": unit, "group": group, "headline": headline,
            "history": _hist(_real(var)),
            "forecast": _ser(iv),
        }

    # ── Roční (kalendářní) prognóza — srovnatelná s ČBA/ČNB/MF/KB ────────────
    # Růstové/inflační veličiny = průměr 4 kvartálů roku; sazby = konec roku.
    # Kombinuje skutečnost (historie) + prognózu; jen kompletní roky (4 Q).
    ANNUAL_ROWS = [
        ("gdp_yoy",   "HDP – reálný růst",       "%",   "mean", 1),
        ("cpi_yoy",   "Inflace CPI – průměr",    "%",   "mean", 1),
        ("unempl",    "Nezaměstnanost – průměr", "%",   "mean", 1),
        ("wages_yoy", "Mzdy nominální – růst",   "%",   "mean", 1),
        ("repo_rate", "2T repo ČNB – konec roku","%",   "eoy",  2),
        ("pribor3m",  "PRIBOR 3M – průměr",      "%",   "mean", 2),
        ("eurczk",    "EUR/CZK – průměr",        "CZK", "mean", 1),
    ]

    def _annual_table():
        # kandidátní roky: od posledního plně skutečného roku po konec prognózy
        this_year = datetime.date.today().year
        years = [this_year - 1, this_year, this_year + 1, this_year + 2]
        actual_end = {}  # rok -> je plně skutečný? (pro označení prognózních)
        rows = []
        for var, label, unit, agg, dec in ANNUAL_ROWS:
            if var not in variables:
                continue
            v = variables[var]
            series, is_fc = {}, {}
            for p in v["history"]:
                series[p["date"][:7]] = p["value"]; is_fc[p["date"][:7]] = False
            for p in v["forecast"]:
                series[p["date"][:7]] = p["median"]; is_fc[p["date"][:7]] = True
            vals = {}
            for y in years:
                qs = sorted((ym, val) for ym, val in series.items() if int(ym[:4]) == y)
                if len(qs) < 4:
                    continue
                x = qs[-1][1] if agg == "eoy" else sum(val for _, val in qs) / len(qs)
                vals[str(y)] = round(float(x), dec)
                # rok je "prognóza", pokud aspoň jedno čtvrtletí je z prognózy
                actual_end[y] = actual_end.get(y, True) and not any(is_fc[ym] for ym, _ in qs)
            rows.append({"var": var, "label": label, "unit": unit, "dec": dec, "values": vals})
        shown = [str(y) for y in years if any(str(y) in r["values"] for r in rows)]
        forecast_years = [str(y) for y in years if not actual_end.get(y, True)]
        return {"years": shown, "rows": rows, "forecast_years": forecast_years}

    annual = _annual_table()

    # ── Realizované roky: OFICIÁLNÍ roční data z Eurostatu ─────────────────
    # Náš dopočet z kvartálů driftuje (řetězení zaokrouhlených QoQ), takže
    # "skutečnost" nesedí s ČSÚ/MF. U hotových let bereme oficiální roční
    # hodnotu (nama_10_gdp/A, une_rt_a) – ta je autoritativní.
    def _official_annual():
        import requests as _rq
        base = ("https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/")
        specs = {
            "gdp_yoy": ("nama_10_gdp", "unit=CLV_PCH_PRE&na_item=B1GQ", 1),
            "unempl":  ("une_rt_a", "age=Y15-74&unit=PC_ACT&sex=T", 1),
        }
        out = {}
        for var, (ds, params, dec) in specs.items():
            try:
                js = _rq.get(f"{base}{ds}?format=JSON&lang=EN&freq=A&{params}&geo=CZ"
                             f"&sinceTimePeriod=2018", timeout=30,
                             headers={"Accept": "application/json"}).json()
                inv = {v: k for k, v in js["dimension"]["time"]["category"]["index"].items()}
                out[var] = {inv[int(k)]: round(float(v), dec) for k, v in js.get("value", {}).items()}
            except Exception:
                pass
        return out

    try:
        off = _official_annual()
        realized = [y for y in annual["years"] if y not in set(annual.get("forecast_years", []))]
        for r in annual["rows"]:
            ov = off.get(r["var"])
            if ov:
                for y in realized:
                    if y in ov:
                        r["values"][y] = ov[y]
        print(f"  oficiální roční data pro realizované roky: {realized}")
    except Exception as e:
        print(f"  (oficiální roční data přeskočena: {e})")

    # ── Srovnání s MF ČR (automaticky stažená predikce) ────────────────────
    # ČNB/ČBA publikují jen PDF (bez API), proto nejsou. Graceful: výpadek
    # jen vynechá srovnání, neshodí export.
    try:
        from external_forecasts import fetch_mf_forecast
        mf = fetch_mf_forecast()
        if mf:
            # Srovnání jen na PROGNÓZNÍCH letech. Minulé roky jsou skutečnost
            # (settled) – tam je případný rozdíl jen šum mezi zdroji (Eurostat
            # vs ČSÚ), ne neshoda prognóz, takže MF u nich nezobrazujeme.
            fy = set(annual.get("forecast_years", []))
            for r in annual["rows"]:
                mv = mf["values"].get(r["var"])
                if mv:
                    r["mf"] = {y: mv[y] for y in annual["years"] if y in mv and y in fy}
            annual["external"] = {"mf": {"label": mf["label"], "date": mf["date"]}}
            print(f"  srovnání MF: {mf['label']}")
    except Exception as e:
        print(f"  (srovnání MF přeskočeno: {e})")

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
        "annual": annual,
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
