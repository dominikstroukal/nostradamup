"""
nowcast.py
==========
Nowcast HDP ČR pomocí mixed-frequency dynamického faktorového modelu (DFM).

Myšlenka (Bok et al. 2018, NY Fed Nowcast; Mariano-Murasawa 2003):
  Existuje malý počet latentních faktorů (zde 1 "stav ekonomiky"), na které
  nakládají všechny pozorované indikátory. Měsíční indikátory vychází dřív
  (konfidence ~42 dní) než tvrdá data (~72 dní) a HDP až ~s velkým zpožděním.
  Kalmanův filtr dopočítá faktor i z "roztřepeného okraje" (ragged edge –
  část nejnovějších dat ještě nevyšla) a z něj implikuje HDP za probíhající
  čtvrtletí = NOWCAST.

Fáze 1: datová vrstva + fit DFMQ + bodový nowcast + sanity check vůči historii.
(Bez webu a news-dekompozice – ty přijdou ve Fázi 2 a 3.)

Spuštění:
    python nowcast.py
"""

import os
import logging
import requests
import numpy as np
import pandas as pd
import statsmodels.api as sm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"

# ── Konfigurace indikátorů ────────────────────────────────────────────────
# transform: jak sérii převést na stacionární pro DFM.
#   "logdiff" = 100*Δlog (měsíční % růst indexu)  – pro tvrdé indexy
#   "diff"    = první diference (pp změna)          – pro nezaměstnanost
#   "level"   = ponech úroveň (konfidence jsou stacionární kolem průměru)
MONTHLY = [
    ("ip",          "sts_inpr_m",   {"indic_bt":"PRD","nace_r2":"B-D","s_adj":"SCA","unit":"I21"}, "logdiff"),
    ("retail",      "sts_trtu_m",   {"indic_bt":"VOL_SLS","nace_r2":"G47","s_adj":"SCA","unit":"I21"}, "logdiff"),
    ("construction","sts_copr_m",   {"indic_bt":"PRD","nace_r2":"F","s_adj":"SCA","unit":"I21"}, "logdiff"),
    ("unempl",      "une_rt_m",     {"age":"TOTAL","unit":"PC_ACT","s_adj":"SA","sex":"T"}, "diff"),
    ("esi",         "ei_bssi_m_r2", {"indic":"BS-ESI-I","s_adj":"SA"}, "level"),
    ("conf_ind",    "ei_bssi_m_r2", {"indic":"BS-ICI-BAL","s_adj":"SA"}, "level"),
    ("conf_cons",   "ei_bssi_m_r2", {"indic":"BS-CCI-BAL","s_adj":"SA"}, "level"),
    ("conf_retail", "ei_bssi_m_r2", {"indic":"BS-RCI-BAL","s_adj":"SA"}, "level"),
]
GDP = ("namq_10_gdp", {"unit":"CLV_PCH_PRE","s_adj":"SCA","na_item":"B1GQ"})  # HDP QoQ %

# COVID: dubnový propad 2020 a odraz jsou extrémní odlehlé hodnoty, které by
# zkreslily odhad faktoru. U tvrdých dat je pro fit maskujeme (Kalman je bere
# jako chybějící). Poctivé outlier-handling, ne zametání – nowcast má popisovat
# normální dynamiku, ne krátký šok lockdownů.
COVID_MASK = (pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-01"))


def _parse_period(p):
    if "-Q" in p:
        y, q = p.split("-Q"); return pd.Timestamp(f"{y}-{(int(q)-1)*3+1:02d}-01")
    if "-M" in p:
        a, b = p.split("-M"); return pd.Timestamp(f"{a}-{int(b):02d}-01")
    return pd.Timestamp(f"{p}-01-01")


def fetch(dataset, dims, freq):
    """Stáhne jednu řadu z Eurostat statistics 1.0 (JSON-stat)."""
    params = "&".join([f"freq={freq}", "geo=CZ"] + [f"{k}={v}" for k, v in dims.items()])
    url = f"{BASE}{dataset}?format=JSON&lang=EN&{params}&sinceTimePeriod=2010"
    r = requests.get(url, timeout=60, headers={"Accept": "application/json"})
    r.raise_for_status()
    js = r.json()
    if not js.get("value"):
        raise ValueError(f"Žádná data pro {dataset} {dims}")
    ids, sz = js["id"], js["size"]
    tp = ids.index("time")
    pos2t = {v: k for k, v in js["dimension"]["time"]["category"]["index"].items()}
    strides = [1] * len(sz)
    for i in range(len(sz) - 2, -1, -1):
        strides[i] = strides[i + 1] * sz[i + 1]
    data = {}
    for k, v in js["value"].items():
        if v is None:
            continue
        rem = int(k); coords = []
        for s in strides:
            coords.append(rem // s); rem %= s
        if all(c == 0 for i, c in enumerate(coords) if i != tp):
            lbl = pos2t.get(coords[tp])
            if lbl:
                data[lbl] = float(v)
    s = pd.Series(data)
    s.index = pd.DatetimeIndex([_parse_period(p) for p in s.index])
    return s.sort_index()


def _transform(s, how):
    if how == "logdiff":
        return 100.0 * np.log(s).diff()
    if how == "diff":
        return s.diff()
    return s  # level


def build_nowcast_data(mask_covid=True):
    """Vrátí (monthly_df stacionární, gdp_q) připravené pro DFMQ."""
    log.info("Stahuji měsíční indikátory + HDP ...")
    cols = {}
    for name, ds, dims, how in MONTHLY:
        raw = fetch(ds, dims, "M")
        s = _transform(raw, how)
        if mask_covid and how == "logdiff":
            s.loc[COVID_MASK[0]:COVID_MASK[1]] = np.nan
        cols[name] = s
    monthly = pd.DataFrame(cols).resample("MS").mean()
    monthly = monthly.loc["2010":]
    monthly.index = monthly.index.to_period("M")

    gdp = fetch(GDP[0], GDP[1], "Q").rename("gdp_qoq")
    if mask_covid:
        gdp.loc["2020-01-01":"2020-06-01"] = np.nan
    gdp = gdp.loc["2010":]
    gdp.index = gdp.index.to_period("Q")
    return monthly, gdp


def fit_nowcast(monthly, gdp, factors=1, factor_order=2):
    """Nafituje mixed-frequency DFM (statsmodels DynamicFactorMQ)."""
    log.info("Fituji DynamicFactorMQ (faktory=%d, řád=%d) ...", factors, factor_order)
    mod = sm.tsa.DynamicFactorMQ(
        monthly,
        endog_quarterly=gdp.to_frame(),
        factors=factors,
        factor_orders=factor_order,
        idiosyncratic_ar1=True,
    )
    res = mod.fit(disp=False)
    return res


# Hezké názvy indikátorů pro web/výstup
LABELS = {
    "ip": "Průmyslová produkce", "retail": "Maloobchod", "construction": "Stavebnictví",
    "unempl": "Nezaměstnanost", "esi": "ESI sentiment", "conf_ind": "Konfidence: průmysl",
    "conf_cons": "Konfidence: spotřebitel", "conf_retail": "Konfidence: maloobchod",
}


def _to_quarterly(s):
    """Měsíčně mapovaná predikce HDP -> čtvrtletní (hodnota v posledním měsíci Q)."""
    s = s[[p.month % 3 == 0 for p in s.index]]
    s.index = pd.PeriodIndex([f"{p.year}Q{(p.month - 1) // 3 + 1}" for p in s.index], freq="Q")
    return s


def nowcast_series(res):
    """Čtvrtletní odhad HDP (in-sample fit + forecast do konce příštího Q)."""
    ins = res.predict()["gdp_qoq"]
    fut = res.forecast(steps=6)["gdp_qoq"]
    return _to_quarterly(pd.concat([ins, fut]))


def news_decomposition(res, monthly, gdp):
    """
    Rozpad 'co pohnulo nowcastem' (Bańbura-Modugno news framework).

    Porovná dvě datové vintage se STEJNÝMI parametry modelu (jen jiná data):
      previous = bez posledního měsíce dat, updated = se vším.
    Rozdíl v nowcastu HDP se rozloží na příspěvky jednotlivých nově příchozích
    pozorování: impact = news (překvapení = observed - očekávané) × weight
    (Kalmanův zisk přepočtený na HDP). To je poctivý analog druhého grafu
    z gdpdynamics ('Changes in Contributions').
    """
    target_q = (gdp.dropna().index[-1] + 1)                 # první nepublikované Q
    impact_month = str(target_q.asfreq("M", how="end"))     # jeho poslední měsíc

    monthly_prev = monthly.copy()
    monthly_prev.iloc[-1] = np.nan                          # stav před posledním releasem
    res_prev = res.apply(monthly_prev, endog_quarterly=gdp.to_frame(), retain_standardization=True)
    res_upd  = res.apply(monthly,      endog_quarterly=gdp.to_frame(), retain_standardization=True)

    news = res_upd.news(res_prev, impact_date=impact_month,
                        impacted_variable="gdp_qoq", comparison_type="previous")

    det = news.details_by_impact.reset_index()
    contrib = det.groupby("updated variable")["impact"].sum().sort_values(key=abs, ascending=False)
    prev_f = float(news.prev_impacted_forecasts["gdp_qoq"].iloc[0])
    post_f = float(news.post_impacted_forecasts["gdp_qoq"].iloc[0])
    return target_q, contrib, prev_f, post_f


def run_nowcast(hist_q=12):
    """
    Spočítá nowcast a vrátí JSON-safe dict pro web (export_web.py).
    Obsahuje: odhad probíhajícího + příštího Q, historii model vs skutečnost,
    a news-rozpad posledního releasu.
    """
    monthly, gdp = build_nowcast_data()
    res = fit_nowcast(monthly, gdp)
    gdp_hat = nowcast_series(res)
    actual = gdp.dropna()

    comp = pd.DataFrame({"actual": actual, "model": gdp_hat}).dropna().tail(hist_q)
    nowcasts = gdp_hat[gdp_hat.index > actual.index[-1]].head(2)
    target_q, contrib, prev_f, post_f = news_decomposition(res, monthly, gdp)

    data_through = monthly.dropna(how="all").index[-1]
    return {
        "current_quarter": str(nowcasts.index[0]),
        "estimate": round(float(nowcasts.iloc[0]), 2),
        "next_quarter": str(nowcasts.index[1]) if len(nowcasts) > 1 else None,
        "next_estimate": round(float(nowcasts.iloc[1]), 2) if len(nowcasts) > 1 else None,
        "unit": "% QoQ",
        "data_through": str(data_through),
        "last_actual_q": str(actual.index[-1]),
        "last_actual": round(float(actual.iloc[-1]), 2),
        "history": [{"q": str(q), "actual": round(float(r.actual), 2),
                     "model": round(float(r.model), 2)} for q, r in comp.iterrows()],
        "last_release": {
            "quarter": str(target_q),
            "prev": round(prev_f, 3), "post": round(post_f, 3),
            "delta": round(post_f - prev_f, 3),
            "contributions": [{"indicator": k, "label": LABELS.get(k, k),
                               "impact": round(float(v), 4)} for k, v in contrib.items()],
        },
    }


def main():
    monthly, gdp = build_nowcast_data()
    log.info("Měsíční matice: %s, HDP: %d čtvrtletí (do %s)",
             monthly.shape, gdp.dropna().shape[0], gdp.dropna().index[-1])
    res = fit_nowcast(monthly, gdp)

    # HDP je v DFMQ interně mapované na POSLEDNÍ měsíc čtvrtletí. Nowcast
    # probíhajícího čtvrtletí = predikce v jeho posledním měsíci (Kalman
    # kondicionuje na všech dostupných měsíčních datech, zbytek čtvrtletí dopočte).
    gdp_hat = nowcast_series(res)

    print("\n" + "=" * 62)
    print("  NOWCAST HDP ČR — sanity check (model vs. skutečnost)")
    print("=" * 62)
    actual = gdp.dropna()
    comp = pd.DataFrame({"skutečnost": actual, "model": gdp_hat}).dropna().tail(8)
    print(comp.to_string(float_format=lambda x: f"{x:+.2f}"))
    rmse = np.sqrt(((comp["skutečnost"] - comp["model"]) ** 2).mean())
    print(f"In-sample RMSE (posl. 8Q): {rmse:.2f} pp")

    # Aktuální nowcast = první čtvrtletí ZA posledním publikovaným HDP
    last_actual_q = actual.index[-1]
    nowcasts = gdp_hat[gdp_hat.index > last_actual_q]
    print(f"\nPoslední publikované HDP: {last_actual_q} = {actual.iloc[-1]:+.2f} %")
    print("Nowcast dosud nepublikovaných čtvrtletí:")
    for q, v in nowcasts.head(2).items():
        print(f"   {q}:  {v:+.2f} % QoQ")

    # ── News dekompozice posledního releasu ──────────────────────────────
    target_q, contrib, prev_f, post_f = news_decomposition(res, monthly, gdp)
    print("\n" + "=" * 62)
    print(f"  CO POHNULO NOWCASTEM {target_q} (poslední batch dat)")
    print("=" * 62)
    print(f"  před releasem: {prev_f:+.3f} %   →   po releasu: {post_f:+.3f} %"
          f"   (Δ {post_f - prev_f:+.3f} pp)")
    print("  Příspěvky podle indikátoru (pp):")
    for name, val in contrib.items():
        bar = "▇" * int(round(abs(val) / max(contrib.abs().max(), 1e-9) * 20))
        print(f"    {name:12s} {val:+.4f}  {bar}")


if __name__ == "__main__":
    main()
