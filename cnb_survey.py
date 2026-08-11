"""
cnb_survey.py
=============
Extrakce prognóz do formátu dotazníku ČNB pro analytiky.

ČNB se ptá na konkrétní indikátory a horizonty (viz tabulka níže).
Tento modul z čtvrtletních prognóz modelu vytáhne přesně ty hodnoty,
které ČNB dotazník požaduje - včetně agregace na kalendářní roky
(letošní / příští rok) u HDP a mezd.

Stav pokrytí (v této verzi):
  ✓ Inflace CPI      1Y (Q+4) a 3Y (Q+12)
  ✓ Růst HDP         letošní a příští rok (roční průměr YoY)
  ✓ Nominální mzdy   letošní a příští rok (roční průměr YoY)
  ✓ CZK/EUR          1Y (Q+4)
  ✓ 2T repo          1Y (Q+4)
  — CZK/EUR 1M, repo 1M         (chybí sub-čtvrtletní nowcast)
  — 12M PRIBOR 1M/1Y            (model počítá 3M PRIBOR; nutno odvodit)
  — 5Y a 10Y IRS 1M/1Y          (chybí model výnosové křivky)
"""

import numpy as np
import pandas as pd





def _irs_market(bond_yield, swap_spread, intervals, q, repo_beta=0.4):
    """
    Úrokový swap z TRŽNÍHO govvie výnosu: IRS = výnos stát. dluhopisu +
    swapový (asset-swap) spread. Zdroj výnosu = trh (FRED / panel / ruční vstup).

    Prognóza:
      q=1 (1 měsíc): spot IRS (dlouhé sazby se přes měsíc prakticky nemění)
      q=4 (1 rok):   spot + repo_beta * (repo za rok - repo nyní), tj. dlouhý
                     konec částečně následuje očekávaný pohyb krátkých sazeb.
    """
    if bond_yield is None:
        return None
    irs_spot = float(bond_yield) + float(swap_spread)
    if q <= 1:
        return irs_spot
    # Posun podle repo cyklu za rok
    try:
        repo = intervals["repo_rate"]["median"].values
        repo_now = float(repo[0])
        repo_1y  = float(repo[min(3, len(repo) - 1)])
        return irs_spot + repo_beta * (repo_1y - repo_now)
    except Exception:
        return irs_spot


def _pribor_12m(intervals: dict, q: int, term_premium: float = 0.10):
    """
    12M PRIBOR přes hypotézu očekávání: průměr očekávaných 3M sazeb přes
    následující 4 čtvrtletí (1 rok) + termínová prémie (~10 bps pro CZK).
    q: horizont, od kterého se průměruje (1 = nejbližší 12M rate).
    """
    try:
        path = intervals["pribor3m"]["median"].values
        window = path[q - 1:q - 1 + 4]
        if len(window) < 1:
            return None
        return float(np.mean(window)) + term_premium
    except Exception:
        return None


def _nowcast_1m(spot, q1_value, kind="fx"):
    """
    1M nowcast mezi spotem (nyní) a Q+1 (3 měsíce).
    fx:   lineární interpolace 1/3 cesty k Q+1 (krátkodobý drift kurzu).
    rate: aktuální úroveň - sazby jsou diskrétní a mezi zasedáními ČNB drží,
          přes 1 měsíc se prakticky nemění (proxy = spot).
    """
    if spot is None:
        return q1_value
    if kind == "rate":
        return spot
    if q1_value is None:
        return spot
    return spot + (q1_value - spot) / 3.0


def _annual_avg(actual: pd.Series, forecast: pd.Series, year: int):
    """Průměr meziroční řady za kalendářní rok (kombinace skutečnosti a prognózy)."""
    combined = pd.concat([actual.dropna(), forecast.dropna()])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    yr = combined[combined.index.year == year]
    return float(yr.mean()) if len(yr) else None


def _fc_at(intervals: dict, var: str, q: int):
    """Hodnota mediánu prognózy v Q+q (1-indexováno)."""
    try:
        return float(intervals[var]["median"].values[q - 1])
    except Exception:
        return None


def _series_at(forecast: pd.DataFrame, var: str, q: int):
    """Hodnota makro prognózy (forecast DataFrame) v Q+q."""
    try:
        return float(forecast[var].values[q - 1])
    except Exception:
        return None


def build_cnb_table(data: pd.DataFrame, forecast: pd.DataFrame,
                    fin_intervals: dict, fin_df: pd.DataFrame | None = None,
                    bond5y: float | None = None, bond10y: float | None = None,
                    swap_spread: float = 0.0) -> str:
    """
    Sestaví markdown tabulku ve formátu dotazníku ČNB.
    data:          skutečnost (makro), pro roční agregaci a aktuální rok
    forecast:      makro prognóza (gdp_yoy, cpi_yoy, hicp_yoy, wages_yoy)
    fin_intervals: finanční prognózy (eurczk, repo_rate, pribor3m)
    """
    # Aktuální a příští rok podle prvního prognózního čtvrtletí
    try:
        first_fc_year = forecast.index[0].year
    except Exception:
        import datetime as _dt
        first_fc_year = _dt.date.today().year
    yr_now, yr_next = first_fc_year, first_fc_year + 1

    def f(x, dec=1, unit=""):
        return f"{x:.{dec}f}{unit}" if x is not None else "—"

    # Inflace CPI: 1Y = Q+4, 3Y = Q+12
    cpi_1y = _series_at(forecast, "cpi_yoy", 4)
    cpi_3y = _series_at(forecast, "cpi_yoy", 12)

    # HDP a mzdy: roční průměr YoY
    gdp_now  = _annual_avg(data.get("gdp_yoy", pd.Series(dtype=float)),
                           forecast.get("gdp_yoy", pd.Series(dtype=float)), yr_now)
    gdp_next = _annual_avg(data.get("gdp_yoy", pd.Series(dtype=float)),
                           forecast.get("gdp_yoy", pd.Series(dtype=float)), yr_next)
    wage_now  = _annual_avg(data.get("wages_yoy", pd.Series(dtype=float)),
                            forecast.get("wages_yoy", pd.Series(dtype=float)), yr_now)
    wage_next = _annual_avg(data.get("wages_yoy", pd.Series(dtype=float)),
                            forecast.get("wages_yoy", pd.Series(dtype=float)), yr_next)

    # Kurz a repo: 1Y = Q+4
    czk_1y  = _fc_at(fin_intervals, "eurczk", 4)
    repo_1y = _fc_at(fin_intervals, "repo_rate", 4)

    # 1M nowcast ze spotu (poslední skutečnost) a Q+1
    def _spot(var):
        try:
            return float(fin_df[var].dropna().iloc[-1]) if fin_df is not None else None
        except Exception:
            return None
    czk_1m  = _nowcast_1m(_spot("eurczk"),  _fc_at(fin_intervals, "eurczk", 1), "fx")
    repo_1m = _nowcast_1m(_spot("repo_rate"), _fc_at(fin_intervals, "repo_rate", 1), "rate")

    # 12M PRIBOR: preferuj TRŽNÍ prognózu (fin_intervals["pribor12m"] z ČNB
    # fixingu), jinak fallback na hypotézu očekávání z 3M dráhy.
    if "pribor12m" in fin_intervals:
        pribor12m_1m = _fc_at(fin_intervals, "pribor12m", 1)
        pribor12m_1y = _fc_at(fin_intervals, "pribor12m", 4)
    else:
        pribor12m_1m = _pribor_12m(fin_intervals, 1)
        pribor12m_1y = _pribor_12m(fin_intervals, 4)

    # 5Y a 10Y IRS z TRŽNÍCH govvie výnosů (+ swapový spread), prognóza podle
    # repo cyklu. bond5y/bond10y = výnos stát. dluhopisu (panel/FRED/ruční vstup).
    irs5_1m  = _irs_market(bond5y, swap_spread, fin_intervals, 1)
    irs5_1y  = _irs_market(bond5y, swap_spread, fin_intervals, 4)
    irs10_1m = _irs_market(bond10y, swap_spread, fin_intervals, 1)
    irs10_1y = _irs_market(bond10y, swap_spread, fin_intervals, 4)

    L = []
    L.append("## Očekávání hlavních inflačních veličin")
    L.append("")
    L.append("| Indikátor | Horizont | Hodnota |")
    L.append("|-----------|----------|---------|")
    L.append(f"| Inflace CPI (% YoY) | 1 rok | {f(cpi_1y)} |")
    L.append(f"| Inflace CPI (% YoY) | 3 roky | {f(cpi_3y)} |")
    L.append(f"| Růst HDP (% YoY) | letošní rok ({yr_now}) | {f(gdp_now)} |")
    L.append(f"| Růst HDP (% YoY) | příští rok ({yr_next}) | {f(gdp_next)} |")
    L.append(f"| Nominální mzdy (% YoY) | letošní rok ({yr_now}) | {f(wage_now)} |")
    L.append(f"| Nominální mzdy (% YoY) | příští rok ({yr_next}) | {f(wage_next)} |")
    L.append(f"| CZK/EUR (kurz) | 1 měsíc | {f(czk_1m, 2)} |")
    L.append(f"| CZK/EUR (kurz) | 1 rok | {f(czk_1y, 2)} |")
    L.append(f"| 2T repo sazba (% p.a.) | 1 měsíc | {f(repo_1m, 2)} |")
    L.append(f"| 2T repo sazba (% p.a.) | 1 rok | {f(repo_1y, 2)} |")
    L.append(f"| 12M PRIBOR (% p.a.) | 1 měsíc | {f(pribor12m_1m, 2)} |")
    L.append(f"| 12M PRIBOR (% p.a.) | 1 rok | {f(pribor12m_1y, 2)} |")
    L.append(f"| 5Y IRS (% p.a.) | 1 měsíc | {f(irs5_1m, 2)} |")
    L.append(f"| 5Y IRS (% p.a.) | 1 rok | {f(irs5_1y, 2)} |")
    L.append(f"| 10Y IRS (% p.a.) | 1 měsíc | {f(irs10_1m, 2)} |")
    L.append(f"| 10Y IRS (% p.a.) | 1 rok | {f(irs10_1y, 2)} |")
    L.append("")
    L.append("> 12M PRIBOR: tržní fixing ČNB. 5Y/10Y IRS = tržní výnos státních "
             "dluhopisů + swapový spread (výnos z FRED / ručního vstupu v panelu); "
             "prognóza 1Y podle repo cyklu.")

    # Kontrola ukotvenosti: 3Y inflace vs cíl 2 %
    if cpi_3y is not None:
        odchylka = cpi_3y - 2.0
        if abs(odchylka) <= 0.3:
            kotva = "blízko cíle (očekávání ukotvená)"
        elif odchylka > 0.3:
            kotva = f"o {odchylka:+.1f} pp nad cílem (riziko de-ukotvení)"
        else:
            kotva = f"o {odchylka:+.1f} pp pod cílem"
        L.append("")
        L.append(f"> **Ukotvenost:** tříletá inflace {cpi_3y:.1f} % je {kotva}.")

    return "\n".join(L)
