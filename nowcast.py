"""
nowcast.py
==========
Nowcast HDP ČR pomocí BRIDGE REGRESE.

Myšlenka:
  Měsíční indikátory (průmysl, maloobchod, konfidence…) se agregují na čtvrtletí
  a HDP QoQ se na ně regreduje. Bridge využívá SOUČASNOU vazbu indikátor–HDP
  (průmyslová produkce koreluje s HDP QoQ +0,88), takže odhad míří tam, kam
  ekonomika teď, a trefuje směr. Ragged edge (část dat za probíhající Q ještě
  nevyšla) se řeší průměrem dostupných měsíců; chybějící indikátor = neutrální
  (průměr).

Proč ne dynamický faktorový model (DFM): testovaný DynamicFactorMQ vyráběl
odhad posunutý o kvartál (corr s HDP[t-1] = 0,95) a směr změny pletl v ~⅔
případů (směrová shoda 28-33 %). Bridge na týchž datech trefuje směr
(corr(Δ) +0,51, směrová shoda 59 %) a není posunutý. Viz git historie.

Dekompozice "co pohnulo odhadem" je u bridge přímočará: příspěvek indikátoru
= β_i × změna (standardizovaného) indikátoru.

Spuštění:
    python nowcast.py
"""

import os
import logging
import requests
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"

# ── Konfigurace indikátorů ────────────────────────────────────────────────
# transform: převod na stacionární / srovnatelnou veličinu.
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
INDICATORS = [m[0] for m in MONTHLY]

# COVID: propad a odraz 2020 jsou extrémní odlehlé hodnoty; maskujeme je u
# tvrdých dat i HDP a vynecháme z fitu regrese, aby nezkreslily koeficienty.
COVID_MASK = (pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-01"))
COVID_YEARS = (2020, 2021)

# Hezké názvy indikátorů pro web/výstup
LABELS = {
    "ip": "Průmyslová produkce", "retail": "Maloobchod", "construction": "Stavebnictví",
    "unempl": "Nezaměstnanost", "esi": "ESI sentiment", "conf_ind": "Konfidence: průmysl",
    "conf_cons": "Konfidence: spotřebitel", "conf_retail": "Konfidence: maloobchod",
}


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
    """Vrátí (monthly_df, gdp_q): měsíční indikátory (transformované) + HDP QoQ."""
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


# ── Bridge regrese ─────────────────────────────────────────────────────────

def to_quarterly(monthly, min_months=1):
    """Agreguje měsíční indikátory na čtvrtletí (průměr dostupných měsíců).
    min_months=3 → jen kompletní čtvrtletí (pro fit); 1 → i ragged aktuální Q."""
    qidx = monthly.index.asfreq("Q")
    out = {}
    for c in monthly.columns:
        g = monthly[c].groupby(qidx)
        out[c] = g.mean().where(g.count() >= min_months)
    return pd.DataFrame(out)


def fit_bridge(monthly, gdp, alpha=1.0):
    """Ridge regrese HDP QoQ na čtvrtletně agregované indikátory (standardizované,
    COVID roky vynechány). Vrací koeficienty + standardizaci."""
    Qc = to_quarterly(monthly, min_months=3)
    df = Qc[INDICATORS].join(gdp.rename("gdp")).dropna()
    df = df[~df.index.year.isin(COVID_YEARS)]
    X = df[INDICATORS]
    y = df["gdp"].values
    mu = X.mean()
    sd = X.std().replace(0, 1.0)
    Xs = ((X - mu) / sd).values
    Xd = np.column_stack([np.ones(len(Xs)), Xs])
    R = np.eye(Xd.shape[1]) * alpha
    R[0, 0] = 0.0  # nepenalizuj intercept
    beta = np.linalg.solve(Xd.T @ Xd + R, Xd.T @ y)
    return {"beta": beta, "mu": mu, "sd": sd, "cols": list(INDICATORS), "n": len(df)}


def bridge_predict(fit, q_row):
    """Nowcast pro jedno čtvrtletí. Chybějící indikátor (ragged edge) = průměr."""
    z = ((q_row[fit["cols"]] - fit["mu"]) / fit["sd"]).astype(float).fillna(0.0).values
    return float(fit["beta"][0] + fit["beta"][1:] @ z)


def fitted_quarterly(fit, monthly):
    """In-sample odhad HDP za každé KOMPLETNÍ čtvrtletí (model vs skutečnost)."""
    Qc = to_quarterly(monthly, min_months=3)
    idx = [q for q in Qc.index if Qc.loc[q, INDICATORS].notna().all()]
    return pd.Series({q: bridge_predict(fit, Qc.loc[q]) for q in idx})


def news_decomposition(fit, monthly, cur_q):
    """Rozpad 'co pohnulo odhadem' probíhajícího Q: příspěvek indikátoru
    = β_i × (standardizovaná změna indikátoru mezi vintage před/po posledním
    měsíci dat). Přímočařejší než u faktorového modelu."""
    Qu = to_quarterly(monthly).loc[cur_q]
    m_prev = monthly.copy()
    m_prev.loc[monthly.dropna(how="all").index[-1]] = np.nan
    Qp_all = to_quarterly(m_prev)
    Qp = Qp_all.loc[cur_q] if cur_q in Qp_all.index else Qu * np.nan

    def _z(row):
        return ((row[fit["cols"]] - fit["mu"]) / fit["sd"]).astype(float).fillna(0.0)

    zu, zp = _z(Qu), _z(Qp)
    contrib = {c: float(fit["beta"][1 + i] * (zu[c] - zp[c])) for i, c in enumerate(fit["cols"])}
    return cur_q, bridge_predict(fit, Qp), bridge_predict(fit, Qu), contrib


def run_nowcast(hist_q=12):
    """Spočítá nowcast a vrátí JSON-safe dict pro web (export_web.py)."""
    monthly, gdp = build_nowcast_data()
    fit = fit_bridge(monthly, gdp)
    actual = gdp.dropna()
    fitted = fitted_quarterly(fit, monthly)

    cur_q = actual.index[-1] + 1
    Qall = to_quarterly(monthly)
    estimate = bridge_predict(fit, Qall.loc[cur_q]) if cur_q in Qall.index else float("nan")

    comp = pd.DataFrame({"actual": actual, "model": fitted}).dropna().tail(hist_q)
    tq, prev_f, post_f, contrib = news_decomposition(fit, monthly, cur_q)
    contrib_sorted = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))

    return {
        "current_quarter": str(cur_q),
        "estimate": round(estimate, 2),
        "unit": "% QoQ",
        "data_through": str(monthly.dropna(how="all").index[-1]),
        "last_actual_q": str(actual.index[-1]),
        "last_actual": round(float(actual.iloc[-1]), 2),
        "history": [{"q": str(q), "actual": round(float(r.actual), 2),
                     "model": round(float(r.model), 2)} for q, r in comp.iterrows()],
        "last_release": {
            "quarter": str(tq), "prev": round(prev_f, 3), "post": round(post_f, 3),
            "delta": round(post_f - prev_f, 3),
            "contributions": [{"indicator": k, "label": LABELS.get(k, k),
                               "impact": round(v, 4)} for k, v in contrib_sorted],
        },
    }


def main():
    monthly, gdp = build_nowcast_data()
    fit = fit_bridge(monthly, gdp)
    actual = gdp.dropna()
    fitted = fitted_quarterly(fit, monthly)
    comp = pd.DataFrame({"actual": actual, "model": fitted}).dropna()

    print("\n" + "=" * 62)
    print("  NOWCAST HDP ČR — bridge regrese (model vs. skutečnost)")
    print("=" * 62)
    print(comp.tail(8).to_string(float_format=lambda x: f"{x:+.2f}"))

    c = comp.loc["2015":]
    da, dm = c["actual"].diff(), c["model"].diff()
    hit = (np.sign(da) == np.sign(dm)).mean()
    print(f"\nSměrová diagnostika (2015+, n={len(c)}):")
    print(f"  corr(model, skutečnost[t])   = {c['model'].corr(c['actual']):+.2f}")
    print(f"  corr(Δmodel, Δskutečnost)     = {da.corr(dm):+.2f}   (kladná = trefuje směr)")
    print(f"  směrová shoda                = {100*hit:.0f} %")

    cur_q = actual.index[-1] + 1
    Qall = to_quarterly(monthly)
    if cur_q in Qall.index:
        est = bridge_predict(fit, Qall.loc[cur_q])
        print(f"\nNowcast {cur_q}: {est:+.2f} % QoQ  (posl. publikované {actual.index[-1]} = {actual.iloc[-1]:+.2f})")
        tq, prev_f, post_f, contrib = news_decomposition(fit, monthly, cur_q)
        print(f"Co pohnulo odhadem {tq}: {prev_f:+.3f} → {post_f:+.3f} (Δ {post_f-prev_f:+.3f} pp)")
        for k, v in sorted(contrib.items(), key=lambda kv: -abs(kv[1])):
            if abs(v) > 1e-4:
                print(f"    {LABELS.get(k,k):24s} {v:+.4f}")


if __name__ == "__main__":
    main()
