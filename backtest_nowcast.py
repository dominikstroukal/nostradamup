"""
backtest_nowcast.py
===================
Pseudo-real-time backtest nowcastu HDP.

Pro každé cílové čtvrtletí Q a několik referenčních okamžiků (jak hluboko do Q
už jsme) zrekonstruuje, co bylo v daný moment ZNÁMO (respektuje publikační
zpoždění -> ragged edge), udělá nowcast a porovná se skutečným HDP. Kritérium:
Theil U vůči naivnímu random walku (U < 1 = lepší než RW), konzistentně
s backtest.py hlavního modelu.

Tři horizonty ukazují jádro hodnoty nowcastu: chyba klesá, jak dobíhají data.

POCTIVÉ CAVEATY (číst před interpretací):
  1. Parametry modelu jsou odhadnuté JEDNOU na celém vzorku (fixní). Filtrování
     a data jsou per-vintage out-of-sample, ale parametry "vidí" i budoucnost
     -> mírně optimistické. Striktní verze (expanding-window refit) je budoucí
     upgrade; EM refit v každém kroku je pomalý.
  2. Používáme REVIDOVANÁ data, ne první odhady (nemáme skutečné vintage).
     Zachycujeme ragged edge (timing), ne datové revize. Proto "pseudo"-real-time.
  3. Publikační zpoždění jsou přibližná (v měsících).

Spuštění:
    python backtest_nowcast.py
"""

import logging
import numpy as np
import pandas as pd

from nowcast import build_nowcast_data, fit_nowcast

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Publikační zpoždění v měsících: za kolik měsíců po skončení měsíce M je
# hodnota za M dostupná. Konfidence/ESI vychází rychle (~1M), tvrdá data ~2M.
LAG = {"ip": 2, "retail": 2, "construction": 2, "unempl": 2,
       "esi": 1, "conf_ind": 1, "conf_cons": 1, "conf_retail": 1}

# Referenční okamžiky (offset v měsících od začátku cílového čtvrtletí Q):
#   0 = před Q (žádná data z Q; čistý "forecast" z faktoru a předchozích dat)
#   1 = brzy v Q (2. měsíc; první konfidence Q)
#   2 = konec Q (3. měsíc; + první tvrdá data Q)
#   4 = po Q (~měsíc po konci Q; tvrdá data za celé Q, těsně před vydáním HDP)
HORIZONS = {"před Q": 0, "brzy v Q": 1, "konec Q": 2, "po Q": 4}


def _truncate(monthly, gdp, asof_m):
    """Ořízne data na to, co by bylo publikováno k referenčnímu měsíci asof_m."""
    m = monthly.copy()
    for col in m.columns:
        cutoff = asof_m - LAG[col]
        m.loc[[p > cutoff for p in m.index], col] = np.nan
    g = gdp.copy()
    keep = [q for q in g.index if (q.asfreq("M", how="end") + 2) <= asof_m]
    g[[q not in keep for q in g.index]] = np.nan
    return m, g


def _nowcast_for(res, monthly_v, gdp_v, target_q):
    """Nowcast HDP za target_q z jedné vintage (fixní parametry, jen filtrování)."""
    res_v = res.apply(monthly_v, endog_quarterly=gdp_v.to_frame(),
                      retain_standardization=True)
    end_m = target_q.asfreq("M", how="end")
    pred = res_v.get_prediction(end=end_m).predicted_mean["gdp_qoq"]
    return float(pred.loc[end_m])


def main(start="2016Q1"):
    monthly, gdp = build_nowcast_data()
    log.info("Fituji parametry (jednou, na celém vzorku)...")
    res = fit_nowcast(monthly, gdp)

    actual = gdp.dropna()
    test_qs = [q for q in actual.index
               if q >= pd.Period(start, "Q") and q.year != 2020]  # COVID ven ze statistik

    rows = []
    for q in test_qs:
        q_start = q.asfreq("M", how="start")
        rw = actual.loc[q - 1] if (q - 1) in actual.index else np.nan  # naivní RW
        for hname, off in HORIZONS.items():
            asof = q_start + off
            mv, gv = _truncate(monthly, gdp, asof)
            try:
                nc = _nowcast_for(res, mv, gv, q)
            except Exception:
                nc = np.nan
            rows.append({"q": str(q), "horizon": hname,
                         "actual": float(actual.loc[q]), "nowcast": nc, "rw": rw})

    df = pd.DataFrame(rows).dropna()
    print("\n" + "=" * 70)
    print(f"  BACKTEST NOWCASTU — pseudo-real-time, {start}+ (COVID 2020 vyňat)")
    print(f"  {len(test_qs)} čtvrtletí × {len(HORIZONS)} horizonty")
    print("=" * 70)
    print(f"\n{'Horizont':<12} {'n':>4} {'RMSE nowcast':>13} {'RMSE RW':>9} {'Theil U':>9}  {'hodnocení'}")
    print("-" * 70)
    for hname in HORIZONS:
        sub = df[df["horizon"] == hname]
        e_nc = np.sqrt(((sub["nowcast"] - sub["actual"]) ** 2).mean())
        e_rw = np.sqrt(((sub["rw"] - sub["actual"]) ** 2).mean())
        U = e_nc / e_rw if e_rw > 0 else np.nan
        verdict = "✅ lepší než RW" if U < 1 else "❌ horší než RW"
        print(f"{hname:<12} {len(sub):>4} {e_nc:>13.3f} {e_rw:>9.3f} {U:>9.3f}  {verdict}")

    print("\nPoznámka: parametry fixní z celého vzorku (mírně optimistické), "
          "revidovaná data.\nZachycuje ragged edge, ne datové revize.")
    return df


if __name__ == "__main__":
    main()
