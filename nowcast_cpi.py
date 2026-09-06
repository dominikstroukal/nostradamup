"""
NOWCAST MĚSÍČNÍ INFLACE (národní CPI, ČSÚ)
==========================================

Odhaduje inflaci za měsíc, který ČSÚ ještě nezveřejnil. Typické použití: na
začátku října víme ceny pohonných hmot za celé září, ale index spotřebitelských
cen za září vyjde až v polovině října. Nowcast tu mezeru vyplní.

Proč se predikuje MEZIMĚSÍČNÍ změna, a ne rovnou meziroční
-----------------------------------------------------------
Meziroční inflace za měsíc m je z velké části daná už známou loňskou
základnou; jediná nová informace v každém zveřejnění je meziměsíční změna.
Predikovat rovnou meziroční hodnotu znamená předpovídat i to, co už dávno víme,
což zbytečně rozmělní chybu. Model proto odhaduje MoM a meziroční se z něj
dopočítá přes bazický index:

    I_m = I_{m-1} * (1 + MoM_m/100)      ->      YoY_m = (I_m / I_{m-12} - 1) * 100

Co model vlastně predikuje
--------------------------
Měsíční inflaci ovládá sezónnost (lednové přeceňování, sezóna potravin,
dovolené). Naivní odhad "průměrná MoM tohoto kalendářního měsíce" je proto
překvapivě silný benchmark. Model tedy necílí na MoM jako takovou, ale na
ODCHYLKU od sezónní normy, a tu vysvětluje cenami pohonných hmot. Ty ČSÚ
publikuje týdně, tedy s velkým předstihem před CPI.

Krizové roky 2022-23 se z tréninku vynechávají. Energetický šok měl rozptyl
odchylek dvakrát větší než normální režim a natrénovaný na něm model ztrácel
signál i v klidných letech (stejná logika, proč nowcast HDP vynechává covid).
Backtest: Theil U vůči sezónní normě 0,83 a směrová shoda 71 % v současném
režimu; včetně krizových let U 0,97, protože energetický šok se z cen benzinu
predikovat nedá. To je poctivá mez modelu, ne chyba.

Zbývá do budoucna: ceny potravin (~20 % koše) by odchylku vysvětlily nejvíc,
ale předběžné měsíční šetření ČSÚ (CEN0101F) je v opendatech jen do konce
minulého roku, takže včas k dispozici není.
"""

import numpy as np
import pandas as pd

from data_fetch import fetch_csu_cpi_monthly, fetch_csu_fuel

# Energetický šok: netrénovat na něm (viz docstring).
CRISIS_YEARS = (2022, 2023)
# Sezónní norma z posledních N let, aby odrážela současný cenový režim.
SEAS_YEARS = 10
# Prediktory odchylky od sezónní normy.
FEATURES = ["fuel_mom", "fuel_mom_l1"]
RIDGE_ALPHA = 1.0
# Kolik týdnů šetření PHM má úplný měsíc (pro míru spolehlivosti).
FULL_WEEKS = 4

LABELS = {
    "fuel_mom":    "Ceny pohonných hmot (m/m)",
    "fuel_mom_l1": "Ceny pohonných hmot (zpožděné)",
}


# ── Data ──────────────────────────────────────────────────────────────────────

def build_cpi_nowcast_data() -> pd.DataFrame:
    """Měsíční tabulka: mom, yoy, idx (CPI) + fuel_mom, weeks (PHM).

    Index je SJEDNOCENÍ obou zdrojů, aby v něm byl i měsíc, kde už máme ceny
    PHM, ale CPI ještě ne. Právě ten se nowcastuje.
    """
    cpi = fetch_csu_cpi_monthly()
    fuel = fetch_csu_fuel()

    ix = cpi.index.union(fuel.index)
    d = pd.DataFrame(index=ix)
    for c in ("mom", "yoy", "idx"):
        d[c] = cpi[c].reindex(ix)
    d["fuel"] = fuel["fuel"].reindex(ix)
    d["weeks"] = fuel["weeks"].reindex(ix)
    d["fuel_mom"] = d["fuel"].pct_change() * 100
    d["fuel_mom_l1"] = d["fuel_mom"].shift(1)
    d["m"] = d.index.month
    return d


# ── Model ─────────────────────────────────────────────────────────────────────

def _train_window(train: pd.DataFrame) -> pd.DataFrame:
    """Trénink: posledních SEAS_YEARS let bez krizových roků."""
    t = train[~train.index.year.isin(CRISIS_YEARS)]
    if len(t):
        t = t[t.index >= t.index[-1] - pd.DateOffset(years=SEAS_YEARS)]
    return t


def seasonal_norm(train: pd.DataFrame) -> pd.Series:
    """Průměrná meziměsíční změna podle kalendářního měsíce (benchmark)."""
    return _train_window(train).groupby("m")["mom"].mean()


def fit_predict(train: pd.DataFrame, target: pd.Series, alpha: float = RIDGE_ALPHA):
    """Ridge na ODCHYLKU od sezónní normy. Vrací (mom, seas, příspěvky)."""
    seas = seasonal_norm(train)
    tr = _train_window(train).dropna(subset=FEATURES + ["mom"]).copy()
    if len(tr) < 24:
        raise ValueError("málo trénovacích pozorování pro nowcast CPI")

    tr["dev"] = tr["mom"] - tr["m"].map(seas)
    X, y = tr[FEATURES].values.astype(float), tr["dev"].values.astype(float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    ybar = y.mean()
    beta = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - ybar))

    if target[FEATURES].isna().any():
        raise ValueError("chybí prediktory pro cílový měsíc")
    z = (target[FEATURES].values.astype(float) - mu) / sd
    s = float(seas.get(int(target["m"]), np.nan))
    contrib = dict(zip(FEATURES, (z * beta).tolist()))
    return s + float(ybar + z @ beta), s, contrib


def backtest(d: pd.DataFrame, start: str = "2018-01", skip_crisis: bool = False) -> dict:
    """Expanding-window backtest proti sezónní normě (relevantní benchmark)."""
    rows = []
    for t in d.index[d.index >= pd.Timestamp(start)]:
        if pd.isna(d.loc[t, "mom"]):
            continue
        if skip_crisis and t.year in CRISIS_YEARS:
            continue
        try:
            p, s, _ = fit_predict(d[d.index < t], d.loc[t])
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not np.isfinite(p) or not np.isfinite(s):
            continue
        rows.append({"act": d.loc[t, "mom"], "model": p, "seas": s})
    r = pd.DataFrame(rows).dropna()
    if r.empty:
        return {"n": 0}

    def _rmse(a, b):
        return float(np.sqrt(((a - b) ** 2).mean()))

    rm, rs = _rmse(r["act"], r["model"]), _rmse(r["act"], r["seas"])
    return {
        "n": int(len(r)),
        "rmse": round(rm, 3),
        "rmse_seasonal": round(rs, 3),
        "theil_u": round(rm / rs, 2) if rs else None,
        "directional": round(float(
            (np.sign(r["act"] - r["seas"]) == np.sign(r["model"] - r["seas"])).mean()), 2),
    }


# ── Výstup pro web ────────────────────────────────────────────────────────────

def run_cpi_nowcast(hist_m: int = 12, d: pd.DataFrame | None = None) -> dict:
    """Spočítá nowcast inflace a vrátí JSON-safe dict pro web (export_web.py).

    'd' lze předat, když už je tabulka sestavená (ušetří stahování ČSÚ).
    """
    if d is None:
        d = build_cpi_nowcast_data()

    last = d["mom"].last_valid_index()               # poslední zveřejněný měsíc
    target = last + pd.DateOffset(months=1)          # první nezveřejněný
    if target not in d.index:
        raise ValueError(f"pro {target.date()} nejsou data o PHM")

    mom, seas, contrib = fit_predict(d[d.index < target], d.loc[target])

    # Meziroční z bazického indexu (viz docstring).
    base = d["idx"].loc[target - pd.DateOffset(months=12)]
    yoy = (d["idx"].loc[last] * (1 + mom / 100) / base - 1) * 100

    bt = backtest(d, "2024-01")
    weeks = int(d.loc[target, "weeks"]) if pd.notna(d.loc[target, "weeks"]) else 0
    # Nekompletní měsíc PHM = širší interval (průměr z 1-2 týdnů je nereprezentativní).
    rmse = bt.get("rmse", 0.3) or 0.3
    band = rmse * (1.0 if weeks >= FULL_WEEKS else 1.6)

    comp = []
    for t in d.index[d.index <= last][-hist_m:]:
        if pd.isna(d.loc[t, "mom"]):
            continue
        try:
            p, _, _ = fit_predict(d[d.index < t], d.loc[t])
        except (ValueError, np.linalg.LinAlgError):
            continue
        comp.append({"month": t.strftime("%Y-%m"),
                     "actual": round(float(d.loc[t, "mom"]), 2),
                     "model": round(float(p), 2)})

    return {
        "target_month": target.strftime("%Y-%m"),
        "mom": round(float(mom), 2),
        "yoy": round(float(yoy), 1),
        "yoy_lower": round(float(yoy - band), 1),
        "yoy_upper": round(float(yoy + band), 1),
        "seasonal_norm": round(float(seas), 2),
        "unit": "% m/m",
        "last_actual_month": last.strftime("%Y-%m"),
        "last_actual_mom": round(float(d.loc[last, "mom"]), 2),
        "last_actual_yoy": round(float(d.loc[last, "yoy"]), 1),
        "fuel_weeks": weeks,
        "fuel_complete": bool(weeks >= FULL_WEEKS),
        "fuel_mom": round(float(d.loc[target, "fuel_mom"]), 2),
        "backtest": bt,
        "history": comp,
        "contributions": [
            {"indicator": k, "label": LABELS.get(k, k), "impact": round(float(v), 3)}
            for k, v in sorted(contrib.items(), key=lambda kv: -abs(kv[1]))
        ],
    }


def main():
    d = build_cpi_nowcast_data()
    r = run_cpi_nowcast(d=d)

    print("\n" + "=" * 64)
    print("  NOWCAST MĚSÍČNÍ INFLACE ČR (národní CPI, ČSÚ)")
    print("=" * 64)
    print(f"  poslední zveřejněný měsíc : {r['last_actual_month']}  "
          f"({r['last_actual_mom']:+.2f} % m/m, {r['last_actual_yoy']:.1f} % r/r)")
    print(f"  NOWCAST na                : {r['target_month']}")
    print(f"    meziměsíčně             : {r['mom']:+.2f} %  "
          f"(sezónní norma {r['seasonal_norm']:+.2f} %)")
    print(f"    MEZIROČNĚ               : {r['yoy']:.1f} %  "
          f"({r['yoy_lower']:.1f} až {r['yoy_upper']:.1f})")
    print(f"  ceny PHM za cílový měsíc  : {r['fuel_mom']:+.2f} % m/m "
          f"({r['fuel_weeks']} týdnů{'' if r['fuel_complete'] else ', NEúplný měsíc'})")
    print("\n  příspěvky k odchylce od sezónní normy:")
    for c in r["contributions"]:
        print(f"    {c['label']:34s} {c['impact']:+.3f} pp")

    print("\n  backtest (expanding window, proti sezónní normě):")
    for lbl, kw in [("2018+ vč. krize", dict(start="2018-01")),
                    ("2018+ bez 2022-23", dict(start="2018-01", skip_crisis=True)),
                    ("2024+ současný režim", dict(start="2024-01"))]:
        b = backtest(d, **kw)
        if b.get("n"):
            print(f"    {lbl:22s} n={b['n']:3d}  RMSE {b['rmse']:.3f} vs {b['rmse_seasonal']:.3f}"
                  f"  Theil U={b['theil_u']:.2f}  směr {b['directional']:.0%}")

    print("\n  model vs. skutečnost (posledních 12 měsíců, % m/m):")
    for h in r["history"]:
        print(f"    {h['month']}  skutečnost {h['actual']:+.2f}   model {h['model']:+.2f}")
    print()


if __name__ == "__main__":
    main()
