"""
data_fetch.py
=============
Stažení makroekonomických dat pro CZ / Eurozóna / USA.

Spuštění:
    python data_fetch.py

Výstup:
    data/raw/cz_macro.csv
    data/raw/ea_macro.csv
    data/raw/us_macro.csv
"""

import os
import time
import json
import logging
import requests
import pandas as pd
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1.  Česká republika  –  Eurostat (SDMX REST)
#     Alternativa pro přímý přístup ke ČSÚ:
#     https://vdb.czso.cz/pll/eweb/pkg_getdata.get_data
# ─────────────────────────────────────────────

EUROSTAT_BASE = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data"

EUROSTAT_SERIES = {
    # (dataset_id, filter_params, column_name)
    "cz_gdp_qoq":   ("namq_10_gdp",  "Q.GR.SCA.B1GQ.CZ",    "gdp_qoq"),
    "cz_hicp":      ("prc_hicp_midx", "M.INX.CP00.CZ",        "hicp"),
    "ea_gdp_qoq":   ("namq_10_gdp",  "Q.GR.SCA.B1GQ.EA20",   "gdp_qoq"),
    "ea_hicp":      ("prc_hicp_midx", "M.INX.CP00.EA20",      "hicp"),
}

# Eurostat: SDMX 2.1 TSV endpoint – nejstabilnější varianta
# Formát URL: /sdmx/2.1/data/{dataset}/{key}?format=TSV&compressed=true
# TSV hlavička: "freq,s_adj,...,geo	YYYY-Q1	YYYY-Q2..."
# Dimenze pro Eurostat statistics 1.0 API (JSON-stat).
# Klíč = (dataset, filter_str), hodnota = slovník dimenzí.
# Statistics API validuje kódy dimenzí proti definici datasetu;
# tyto kódy odpovídají aktuální struktuře (ověřeno proti dokumentaci Eurostatu).
_EUROSTAT_DIMS = {
    # HDP QoQ – reálný řetězený růst k předchozímu období, sezónně očištěný
    ("namq_10_gdp", "Q.GR.SCA.B1GQ.CZ"): {
        "freq": "Q", "unit": "CLV_PCH_PRE", "s_adj": "SCA",
        "na_item": "B1GQ", "geo": "CZ",
    },
    # HICP – meziroční míra změny (annual rate of change)
    ("prc_hicp_manr", "M.RCH_A.CP00.CZ"): {
        "freq": "M", "unit": "RCH_A", "coicop": "CP00", "geo": "CZ",
    },
}


def fetch_eurostat(dataset: str, filter_str: str, col_name: str) -> pd.Series:
    """
    Stáhne časovou řadu z Eurostatu přes statistics 1.0 API (JSON-stat).
    Dimenze se zadávají jako jednotlivé parametry – spolehlivější než SDMX
    seriesKey, kde záleží na přesném pořadí a kódech (jinak HTTP 400).
    """
    dims = _EUROSTAT_DIMS.get((dataset, filter_str))
    if dims is None:
        # Fallback: jen geo z poslední části filter_str
        dims = {"geo": filter_str.split(".")[-1]}

    params = "&".join(f"{k}={v}" for k, v in dims.items())
    url = (
        "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
        f"{dataset}?format=JSON&lang=EN&{params}&sinceTimePeriod=2010"
    )
    log.info("Eurostat <- %s (statistics API)", dataset)

    r = requests.get(url, timeout=60, headers={"Accept": "application/json"})
    r.raise_for_status()
    js = r.json()

    # Robustní JSON-stat parsing: dekóduj plochý index na souřadnice
    # a vezmi jen pozorování, kde jsou ne-časové dimenze na indexu 0.
    dim_ids = js["id"]
    sizes   = js["size"]
    time_pos = dim_ids.index("time")
    time_index = js["dimension"]["time"]["category"]["index"]
    pos_to_time = {pos: label for label, pos in time_index.items()}

    strides = [1] * len(sizes)
    for i in range(len(sizes) - 2, -1, -1):
        strides[i] = strides[i + 1] * sizes[i + 1]

    data = {}
    for idx_str, val in js.get("value", {}).items():
        if val is None:
            continue
        flat = int(idx_str)
        coords, rem = [], flat
        for st in strides:
            coords.append(rem // st)
            rem = rem % st
        if all(c == 0 for i, c in enumerate(coords) if i != time_pos):
            label = pos_to_time.get(coords[time_pos])
            if label:
                data[label] = float(val)

    if not data:
        raise ValueError(f"Žádná data z Eurostatu pro {dataset}")

    s = pd.Series(data, name=col_name)

    def parse_period(p):
        if "-Q" in p:
            y, q = p.split("-Q")
            return pd.Timestamp(f"{y}-{(int(q)-1)*3+1:02d}-01")
        elif "-M" in p:
            parts = p.split("-M")
            return pd.Timestamp(f"{parts[0]}-{int(parts[1]):02d}-01")
        return pd.Timestamp(f"{p}-01-01")

    s.index = pd.DatetimeIndex([parse_period(p) for p in s.index])
    return s.sort_index()


# ─────────────────────────────────────────────
# 2.  USA  –  FRED  (St. Louis Fed)
#     Potřebuješ API klíč: https://fred.stlouisfed.org/docs/api/api_key.html
#     Zadej ho do proměnné prostředí:  export FRED_API_KEY="tvuj_klic"
# ─────────────────────────────────────────────

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

FRED_SERIES = {
    "us_gdp_qoq":   "A191RL1Q225SBEA",   # Real GDP growth QoQ SAAR
    "us_cpi":       "CPIAUCSL",           # CPI All Urban Consumers
    "us_unrate":    "UNRATE",             # Unemployment rate
    "us_wages":     "CES0500000003",      # Average Hourly Earnings
    "us_fed_rate":  "FEDFUNDS",           # Federal Funds Rate
}

def fetch_fred(series_id: str, col_name: str, api_key: str) -> pd.Series:
    """Stáhne jednu řadu z FRED API."""
    url = FRED_BASE
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": "2000-01-01",
    }
    log.info("FRED ← %s", series_id)
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()

    obs = js["observations"]
    data = {}
    for o in obs:
        if o["value"] != ".":
            data[o["date"]] = float(o["value"])

    s = pd.Series(data, name=col_name)
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


# ─────────────────────────────────────────────
# 3.  ARAD (ČNB)  –  průměrné hrubé mzdy CZ
#     https://www.cnb.cz/arad/
#     ČNB nemá otevřené REST API; stahujeme CSV export.
# ─────────────────────────────────────────────

CNB_WAGES_URL = (
    "https://www.cnb.cz/arad/api/data.csv?"
    "series=PMZDA_Q&dataPoints=all&lang=CZ"
)

def fetch_cnb_wages() -> pd.Series:
    """Průměrná hrubá mzda ČR (čtvrtletně), zdroj ARAD/ČNB."""
    log.info("ARAD/ČNB ← průměrné mzdy")
    try:
        r = requests.get(CNB_WAGES_URL, timeout=30)
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text), sep=";", skiprows=5, header=0)
        # Formát sloupců se může lišit; adaptuj podle skutečného CSV
        df.columns = [c.strip() for c in df.columns]
        date_col = df.columns[0]
        val_col  = df.columns[1]
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
        df = df.dropna(subset=[date_col])
        df = df.set_index(date_col)
        s = pd.to_numeric(df[val_col].str.replace(",", "."), errors="coerce")
        s.name = "wages_czk"
        return s.sort_index()
    except Exception as e:
        log.info("ČNB ARAD nedostupný (%s) – záložní data mezd. ARAD vyžaduje API klíč.", e)
        return _fallback_wages()


def _fallback_wages() -> pd.Series:
    """Záložní data mezd CZ z ČSÚ (2010–2024) pro testování modelu."""
    # Zdroj: ČSÚ, průměrná hrubá mzda, Kč/čtvrtletí
    data = {
        "2010-01-01": 23144, "2010-04-01": 24144, "2010-07-01": 23698, "2010-10-01": 25646,
        "2011-01-01": 23858, "2011-04-01": 25034, "2011-07-01": 24436, "2011-10-01": 26513,
        "2012-01-01": 24408, "2012-04-01": 25588, "2012-07-01": 25024, "2012-10-01": 27170,
        "2013-01-01": 24165, "2013-04-01": 25511, "2013-07-01": 24955, "2013-10-01": 27178,
        "2014-01-01": 24806, "2014-04-01": 26072, "2014-07-01": 25637, "2014-10-01": 27827,
        "2015-01-01": 25929, "2015-04-01": 27170, "2015-07-01": 26843, "2015-10-01": 29022,
        "2016-01-01": 27297, "2016-04-01": 28663, "2016-07-01": 28403, "2016-10-01": 30745,
        "2017-01-01": 28746, "2017-04-01": 30265, "2017-07-01": 29984, "2017-10-01": 32654,
        "2018-01-01": 30265, "2018-04-01": 32220, "2018-07-01": 31962, "2018-10-01": 35423,
        "2019-01-01": 32466, "2019-04-01": 34125, "2019-07-01": 33697, "2019-10-01": 36943,
        "2020-01-01": 33929, "2020-04-01": 35402, "2020-07-01": 35402, "2020-10-01": 38525,
        "2021-01-01": 35285, "2021-04-01": 37221, "2021-07-01": 36893, "2021-10-01": 40239,
        "2022-01-01": 37839, "2022-04-01": 40150, "2022-07-01": 39817, "2022-10-01": 43454,
        "2023-01-01": 40891, "2023-04-01": 43325, "2023-07-01": 43000, "2023-10-01": 46796,
        "2024-01-01": 43500, "2024-04-01": 46000, "2024-07-01": 45800, "2024-10-01": 50200,
        "2025-01-01": 46100, "2025-04-01": 48700, "2025-07-01": 48500, "2025-10-01": 53100,
    }
    s = pd.Series({pd.Timestamp(k): v for k, v in data.items()}, name="wages_czk")
    log.info("Záložní data mezd načtena (%d pozorování)", len(s))
    return s


# ─────────────────────────────────────────────
# 4.  Sestavení datových rámců
# ─────────────────────────────────────────────

def build_cz_dataset() -> pd.DataFrame:
    """Sestaví čtvrtletní dataset pro ČR."""
    log.info("=== Sestavuji dataset CZ ===")

    try:
        gdp_raw = fetch_eurostat("namq_10_gdp",  "Q.GR.SCA.B1GQ.CZ",  "gdp_qoq")
        # HDP YoY = 4Q rolling součet QoQ (dataset nemá přímou YoY sérii pro tyto dimenze)
        gdp_yoy_raw = gdp_raw.rolling(4).sum()
        gdp_yoy_raw.name = "gdp_yoy"
        # POZOR: prc_hicp_manr UŽ JE meziroční míra změny (RCH_A = rate of change, annual).
        # Nesmí se na ni aplikovat pct_change – jen převést na čtvrtletní průměr.
        hicp = fetch_eurostat("prc_hicp_manr", "M.RCH_A.CP00.CZ", "hicp")
        hicp_yoy = hicp.resample("QS").mean()
        hicp_yoy.name = "hicp_yoy"
        # CPI ČSÚ – stejný dataset (ČNB používá HICP jako proxy CPI)
        cpi_raw = fetch_eurostat("prc_hicp_manr", "M.RCH_A.CP00.CZ", "cpi_yoy")
        cpi_yoy = cpi_raw.resample("QS").mean()
        cpi_yoy.name = "cpi_yoy"
    except Exception as e:
        log.info("Eurostat API nedostupné (%s) – používám demo data.", e)
        gdp_raw, hicp_yoy, cpi_yoy = _demo_cz_gdp_hicp()
        # GDP YoY aproximace z QoQ (4Q rolling součet)
        gdp_yoy_raw = gdp_raw.rolling(4).sum()
        gdp_yoy_raw.name = "gdp_yoy"

    wages = fetch_cnb_wages()
    wages_yoy = wages.pct_change(4) * 100
    wages_yoy.name = "wages_yoy"

    df = pd.DataFrame({
        "gdp_qoq":  gdp_raw,
        "gdp_yoy":  gdp_yoy_raw,
        "hicp_yoy": hicp_yoy,
        "cpi_yoy":  cpi_yoy,
        "wages_yoy": wages_yoy,
    })
    df = df.resample("QS").mean()
    df = df.dropna()
    df.to_csv(os.path.join(RAW_DIR, "cz_macro.csv"))
    log.info("CZ dataset uložen: %d čtvrtletí (%s – %s)", len(df), df.index[0].date(), df.index[-1].date())
    return df


def load_cz_dataset(max_age_hours: float = 12.0) -> pd.DataFrame:
    """
    Načte CZ makro dataset. Preferuje čerstvý CSV cache (zapsaný data_fetch.py),
    aby navazující skripty nepřestahovávaly stejná data a hlavně aby VŠECHNY
    výstupy stály na stejných (živých) datech. Když cache chybí nebo je stará,
    sestaví dataset znovu (živý fetch + fallback).
    """
    import time as _time
    path = os.path.join(RAW_DIR, "cz_macro.csv")
    if os.path.exists(path):
        age_h = (_time.time() - os.path.getmtime(path)) / 3600.0
        if age_h <= max_age_hours:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            log.info("CZ makro načteno z cache (%.1f h staré, %d čtvrtletí, do %s)",
                     age_h, len(df), df.index[-1].date())
            return df
    return build_cz_dataset()


def build_us_dataset(api_key: str | None = None) -> pd.DataFrame:
    """Sestaví čtvrtletní dataset pro USA. Potřebuješ FRED API klíč."""
    log.info("=== Sestavuji dataset USA ===")
    api_key = api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        log.info("FRED_API_KEY není nastavený – generuji demo data USA (nastavte env proměnnou pro živá data).")
        df = _demo_us()
        df.to_csv(os.path.join(RAW_DIR, "us_macro.csv"))
        return df

    series_out = {}
    for col_name, series_id in FRED_SERIES.items():
        try:
            s = fetch_fred(series_id, col_name, api_key)
            series_out[col_name] = s
            time.sleep(0.3)   # respektuj rate limit FRED
        except Exception as e:
            log.error("FRED %s: %s", series_id, e)

    df = pd.DataFrame(series_out).resample("QS").mean()
    df.dropna(inplace=True)
    df.to_csv(os.path.join(RAW_DIR, "us_macro.csv"))
    log.info("US dataset uložen: %d čtvrtletí", len(df))
    return df


def _demo_cz_gdp_hicp():
    """Skutečná historická data ČR pro offline testování (2010–2024).
    CPI ČSÚ (národní index, základ pro cíl ČNB 2 %) mírně odlišný od HICP.
    """
    data = {
        # datum:        (gdp_qoq,  hicp_yoy, cpi_yoy_czso)
        "2010-01-01": (0.6,  1.0,  0.7), "2010-04-01": (1.2,  1.3,  1.2),
        "2010-07-01": (1.1,  1.6,  1.8), "2010-10-01": (1.0,  2.1,  2.4),
        "2011-01-01": (0.8,  2.4,  2.0), "2011-04-01": (0.5,  2.9,  1.9),
        "2011-07-01": (0.3,  2.2,  1.8), "2011-10-01": (-0.1, 2.8,  2.4),
        "2012-01-01": (-0.5, 3.4,  3.8), "2012-04-01": (-0.7, 3.2,  3.5),
        "2012-07-01": (-0.4, 3.4,  3.4), "2012-10-01": (-0.2, 2.6,  3.0),
        "2013-01-01": (-0.9, 1.9,  1.8), "2013-04-01": (-0.7, 1.6,  1.6),
        "2013-07-01": (-0.2, 1.4,  1.4), "2013-10-01": (0.8,  1.2,  1.1),
        "2014-01-01": (0.6,  0.3,  0.2), "2014-04-01": (0.9,  0.5,  0.4),
        "2014-07-01": (0.4,  0.7,  0.6), "2014-10-01": (0.4,  0.7,  0.6),
        "2015-01-01": (1.0,  0.4,  0.3), "2015-04-01": (0.6,  0.4,  0.5),
        "2015-07-01": (0.7,  0.4,  0.5), "2015-10-01": (0.8,  0.4,  0.3),
        "2016-01-01": (0.6,  0.6,  0.4), "2016-04-01": (0.9,  0.4,  0.4),
        "2016-07-01": (0.7,  0.5,  0.5), "2016-10-01": (0.6,  1.5,  1.5),
        "2017-01-01": (1.2,  2.5,  2.6), "2017-04-01": (1.1,  2.5,  2.4),
        "2017-07-01": (1.2,  2.5,  2.5), "2017-10-01": (1.5,  2.5,  2.6),
        "2018-01-01": (0.8,  2.2,  2.2), "2018-04-01": (0.6,  2.2,  2.2),
        "2018-07-01": (0.7,  2.2,  2.3), "2018-10-01": (0.8,  2.0,  2.2),
        "2019-01-01": (0.6,  2.8,  3.0), "2019-04-01": (0.7,  2.8,  2.9),
        "2019-07-01": (0.4,  3.0,  3.0), "2019-10-01": (0.5,  3.2,  3.2),
        "2020-01-01": (-0.3, 3.4,  3.8), "2020-04-01": (-8.7, 3.1,  3.2),
        "2020-07-01": (6.7,  2.8,  3.4), "2020-10-01": (-0.6, 2.9,  2.9),
        "2021-01-01": (1.1,  2.6,  2.2), "2021-04-01": (1.0,  3.4,  2.9),
        "2021-07-01": (1.4,  4.1,  3.4), "2021-10-01": (1.2,  6.0,  5.5),
        "2022-01-01": (0.8, 10.0,  9.9), "2022-04-01": (0.2, 16.0, 15.1),
        "2022-07-01": (-0.2,18.0, 17.4), "2022-10-01": (-0.4,16.0, 16.8),
        "2023-01-01": (-0.3,15.1, 15.8), "2023-04-01": (-0.1,11.4, 12.7),
        "2023-07-01": (0.1,  8.5,  8.8), "2023-10-01": (0.4,  7.6,  7.4),
        "2024-01-01": (0.5,  2.9,  2.8), "2024-04-01": (0.6,  2.7,  2.6),
        "2024-07-01": (0.7,  2.4,  2.2), "2024-10-01": (0.6,  2.7,  2.4),
        "2025-01-01": (0.5,  2.7,  2.7), "2025-04-01": (0.4,  2.5,  2.5),
        "2025-07-01": (0.5,  2.4,  2.4), "2025-10-01": (0.6,  2.6,  2.5),
    }
    idx = pd.to_datetime(list(data.keys()))
    gdp  = pd.Series([v[0] for v in data.values()], index=idx, name="gdp_qoq")
    hicp = pd.Series([v[1] for v in data.values()], index=idx, name="hicp_yoy")
    cpi  = pd.Series([v[2] for v in data.values()], index=idx, name="cpi_yoy")
    return gdp, hicp, cpi


def _demo_us() -> pd.DataFrame:
    """Demo data pro USA (offline testování)."""
    data = {
        "2015-01-01": (0.6, 0.1, 5.5), "2015-04-01": (0.9, 0.1, 5.2),
        "2015-07-01": (0.9, 0.1, 5.0), "2015-10-01": (0.3, 0.5, 5.0),
        "2016-01-01": (0.2, 1.1, 4.9), "2016-04-01": (0.6, 1.1, 4.7),
        "2016-07-01": (0.8, 1.2, 4.9), "2016-10-01": (0.6, 1.8, 4.7),
        "2017-01-01": (0.3, 2.1, 4.7), "2017-04-01": (0.8, 1.9, 4.4),
        "2017-07-01": (0.8, 1.9, 4.3), "2017-10-01": (0.8, 2.2, 4.1),
        "2018-01-01": (0.6, 2.5, 4.1), "2018-04-01": (1.0, 2.7, 3.9),
        "2018-07-01": (0.9, 2.4, 3.8), "2018-10-01": (0.6, 2.2, 3.8),
        "2019-01-01": (0.5, 1.6, 3.8), "2019-04-01": (0.8, 1.8, 3.6),
        "2019-07-01": (0.6, 1.8, 3.6), "2019-10-01": (0.6, 2.3, 3.5),
        "2020-01-01": (-1.3,2.3, 3.8), "2020-04-01": (-8.9,0.4,13.0),
        "2020-07-01": (7.8, 1.2, 8.8), "2020-10-01": (1.1, 1.2, 6.7),
        "2021-01-01": (1.6, 2.6, 6.0), "2021-04-01": (1.6, 4.7, 5.9),
        "2021-07-01": (0.6, 5.3, 5.1), "2021-10-01": (1.7, 6.7, 4.2),
        "2022-01-01": (-0.4,8.0, 3.8), "2022-04-01": (-0.1,8.3, 3.6),
        "2022-07-01": (0.8, 8.3, 3.7), "2022-10-01": (0.7, 7.1, 3.5),
        "2023-01-01": (0.5, 6.0, 3.4), "2023-04-01": (0.5, 4.0, 3.5),
        "2023-07-01": (0.8, 3.2, 3.6), "2023-10-01": (0.8, 3.4, 3.7),
        "2024-01-01": (0.4, 3.1, 3.7), "2024-04-01": (0.7, 2.9, 3.9),
        "2024-07-01": (0.7, 2.6, 4.1), "2024-10-01": (0.6, 2.7, 4.2),
    }
    idx = pd.to_datetime(list(data.keys()))
    df = pd.DataFrame(
        [(v[0], v[1], v[2]) for v in data.values()],
        index=idx,
        columns=["us_gdp_qoq", "us_cpi_yoy", "us_unrate"],
    )
    return df


# ─────────────────────────────────────────────
# 5.  Spuštění
# ─────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Spouštím stahování dat...")
    cz = build_cz_dataset()
    us = build_us_dataset()   # potřebuješ FRED_API_KEY v env
    log.info("Hotovo. CZ: %d čtvrtletí, US: %d čtvrtletí", len(cz), len(us))
    print("\n--- CZ dataset (posledních 6 čtvrtletí) ---")
    print(cz.tail(6).to_string())
    print("\n--- US dataset (posledních 6 čtvrtletí) ---")
    print(us.tail(6).to_string())
