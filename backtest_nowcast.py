"""
backtest_nowcast.py
===================
Pseudo-real-time backtest nowcastu HDP (bridge regrese).

Pro každé cílové čtvrtletí Q a několik referenčních okamžiků zrekonstruuje, co
bylo v daný moment ZNÁMO (respektuje publikační zpoždění → ragged edge), udělá
nowcast bridge regresí a porovná se skutečným HDP. Metriky:
  - Theil U vůči naivnímu random walku (U < 1 = lepší než RW)
  - SMĚROVÁ shoda: trefil nowcast, jestli HDP vůči minulému Q vzroste/klesne?
    (to je hlavní důvod přechodu z faktorového modelu, který směr pletl)

Tři horizonty ukazují, jak přesnost roste, jak dobíhají data.

POCTIVÉ CAVEATY:
  1. Koeficienty regrese odhadnuté JEDNOU na celém vzorku (fixní) → mírně
     optimistické. Striktní verze = expanding-window refit.
  2. Revidovaná data, ne skutečné vintage → zachycuje ragged edge (timing),
     ne datové revize. Proto "pseudo".
  3. Publikační zpoždění přibližná (v měsících).

Spuštění:
    python backtest_nowcast.py
"""

import logging
import numpy as np
import pandas as pd

from nowcast import build_nowcast_data, fit_bridge, bridge_predict, to_quarterly, INDICATORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Publikační zpoždění v měsících (za kolik měsíců po skončení měsíce M je M dostupné).
LAG = {"ip": 2, "retail": 2, "construction": 2, "unempl": 2,
       "esi": 1, "conf_ind": 1, "conf_cons": 1, "conf_retail": 1}

# Referenční okamžiky (offset v měsících od začátku cílového Q):
HORIZONS = {"před Q": 0, "brzy v Q": 1, "konec Q": 2, "po Q": 4}


def _vintage_q_row(monthly, asof_m, target_q):
    """Čtvrtletně agregované indikátory pro target_q z toho, co bylo k asof_m publikováno."""
    m = monthly.copy()
    for col in m.columns:
        cutoff = asof_m - LAG[col]
        m.loc[[p > cutoff for p in m.index], col] = np.nan
    Q = to_quarterly(m)
    if target_q in Q.index:
        return Q.loc[target_q]
    return pd.Series(index=INDICATORS, dtype=float)


def main(start="2016Q1"):
    monthly, gdp = build_nowcast_data()
    log.info("Fituji bridge (jednou, na celém vzorku bez COVIDu)...")
    fit = fit_bridge(monthly, gdp)

    actual = gdp.dropna()
    test_qs = [q for q in actual.index
               if q >= pd.Period(start, "Q") and q.year != 2020 and (q - 1) in actual.index]

    rows = []
    for q in test_qs:
        q_start = q.asfreq("M", how="start")
        prev = float(actual.loc[q - 1])
        for hname, off in HORIZONS.items():
            row = _vintage_q_row(monthly, q_start + off, q)
            nc = bridge_predict(fit, row)
            rows.append({"q": str(q), "horizon": hname, "actual": float(actual.loc[q]),
                         "nowcast": nc, "prev": prev})

    df = pd.DataFrame(rows).dropna()
    print("\n" + "=" * 74)
    print(f"  BACKTEST NOWCASTU (bridge) — pseudo-real-time, {start}+ (COVID 2020 vyňat)")
    print(f"  {len(test_qs)} čtvrtletí × {len(HORIZONS)} horizonty")
    print("=" * 74)
    print(f"\n{'Horizont':<11} {'n':>3} {'RMSE':>6} {'RMSE RW':>8} {'Theil U':>8} {'směr. shoda':>12}")
    print("-" * 74)
    for hname in HORIZONS:
        s = df[df["horizon"] == hname]
        e_nc = np.sqrt(((s["nowcast"] - s["actual"]) ** 2).mean())
        e_rw = np.sqrt(((s["prev"] - s["actual"]) ** 2).mean())
        U = e_nc / e_rw if e_rw > 0 else np.nan
        # směrová shoda: znaménko (nowcast - prev) vs (actual - prev)
        dir_nc = np.sign(s["nowcast"] - s["prev"])
        dir_ac = np.sign(s["actual"] - s["prev"])
        hit = (dir_nc == dir_ac).mean()
        print(f"{hname:<11} {len(s):>3} {e_nc:>6.3f} {e_rw:>8.3f} {U:>8.3f} {100*hit:>10.0f} %")

    print("\nPoznámka: koeficienty fixní z celého vzorku (mírně optimistické), "
          "revidovaná data.\nSměrová shoda > 50 % = nowcast trefuje směr HDP vůči minulému Q.")
    return df


if __name__ == "__main__":
    main()
