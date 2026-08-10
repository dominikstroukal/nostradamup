"""
financial_data.py
=================
Stahování a vizualizace finančních dat:
  - EUR/CZK (kurz ČNB)
  - EUR/USD (ECB)
  - PRIBOR 3M (ČNB ARAD)

Spuštění:
    python financial_data.py

Výstupy:
    data/raw/financial_data.csv
    outputs/charts/fin_EURCZK.png
    outputs/charts/fin_EURUSD.png
    outputs/charts/fin_PRIBOR.png
    outputs/charts/fin_overview.png
"""

import os
import io
import logging
import warnings
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RAW_DIR     = os.path.join(BASE_DIR, "data", "raw")
CHARTS_DIR  = os.path.join(BASE_DIR, "outputs", "charts")
for d in [RAW_DIR, CHARTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── MUP barvy (stejné jako var_model.py) ──────────────────────────────────────
MUP = {
    "navy":      "#00205B",
    "blue":      "#003DA5",
    "blue_mid":  "#2255A4",
    "blue_3":    "#3A6FBF",
    "blue_4":    "#5B8DB8",
    "blue_5":    "#7BAFD4",
    "gray_dark": "#4A4A4A",
    "gray_mid":  "#9E9E9E",
    "gray_bg":   "#EEF2FA",
    "white":     "#FFFFFF",
}

_BLUE = "#003DA5"
FIN_COLORS = {
    "eurczk":    _BLUE,
    "eurusd":    _BLUE,
    "pribor3m":  _BLUE,
    "pribor12m": _BLUE,
    "repo_rate": _BLUE,
    "unempl":    _BLUE,
}

FIN_LABELS = {
    "eurczk":    "EUR/CZK - devizový kurz",
    "eurusd":    "EUR/USD - devizový kurz",
    "pribor3m":  "PRIBOR 3M - mezibankovní sazba (%)",
    "pribor12m": "PRIBOR 12M - mezibankovní sazba (%)",
    "repo_rate": "Repo sazba ČNB (%)",
    "unempl":    "Míra nezaměstnanosti ČR (%)",
}

FIN_YLABEL = {
    "eurczk":    "CZK",
    "eurusd":    "USD",
    "pribor3m":  "%",
    "pribor12m": "%",
    "repo_rate": "%",
    "unempl":    "%",
}

# ── 1.  Stahování dat ──────────────────────────────────────────────────────────

def fetch_eurczk(start: str = "2010-01-01") -> pd.Series:
    """
    EUR/CZK denní kurz z ČNB (datový soubor kurzy.txt).
    Bez API klíče, veřejně dostupné.
    """
    url = "https://www.cnb.cz/cs/financni-trhy/devizovy-trh/kurzy-devizoveho-trhu/kurzy-devizoveho-trhu/rok.txt"
    log.info("ČNB <- EUR/CZK (denní kurzy)")

    # ČNB nabízí data po rocích; stáhneme všechny roky od start do teď
    start_year = int(start[:4])
    end_year   = pd.Timestamp.now().year
    frames = []

    for year in range(start_year, end_year + 1):
        try:
            u = f"https://www.cnb.cz/cs/financni-trhy/devizovy-trh/kurzy-devizoveho-trhu/kurzy-devizoveho-trhu/rok.txt?rok={year}"
            r = requests.get(u, timeout=20)
            r.raise_for_status()
            # Formát: datum|počet|měna|kód|kurz  (odděleno |, desetinná čárka)
            lines = r.text.strip().splitlines()
            if not lines:
                continue
            # První řádek = hlavička, hledáme sloupec EUR
            header = [h.strip() for h in lines[0].split("|")]
            # Datum je vždy první sloupec (název může být "datum" nebo jiný)
            eur_idx = None
            for j, h in enumerate(header):
                if "EUR" in h.upper():
                    eur_idx = j
                    break
            if eur_idx is None:
                continue
            records_y = {}
            for line in lines[1:]:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) <= eur_idx:
                    continue
                try:
                    dt = pd.to_datetime(parts[0], dayfirst=True)
                    val = float(parts[eur_idx].replace(",", "."))
                    records_y[dt] = val
                except Exception:
                    continue
            if records_y:
                frames.append(pd.Series(records_y, name="eurczk"))
        except Exception as e:
            log.debug("EUR/CZK rok %d: %s", year, e)

    fallback = _fallback_eurczk()
    if not frames:
        log.info("EUR/CZK: živé stahování nedostupné, používám záložní data.")
        return fallback
    live = pd.concat(frames).sort_index()
    live = live[live.index >= start]
    live = pd.to_numeric(live, errors="coerce").dropna()
    # Čtvrtletní průměry z živých dat
    live_q = live.resample("QS").mean().dropna()
    combined = fallback.copy()
    combined.update(live_q)
    log.info("EUR/CZK: záložní + %d živých čtvrtletí (poslední: %s)",
             len(live_q), live_q.index[-1].date() if len(live_q) else "n/a")
    return combined.sort_index()


def fetch_eurusd(start: str = "2010-01-01") -> pd.Series:
    """
    EUR/USD denní kurz z ECB Data Portal (free JSON API).
    """
    log.info("ECB <- EUR/USD")
    url = (
        "https://data-api.ecb.europa.eu/service/data/EXR/"
        "D.USD.EUR.SP00.A?format=csvdata&startPeriod={start}&detail=dataonly"
    ).format(start=start)
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        # Sloupce: TIME_PERIOD, OBS_VALUE
        time_col = [c for c in df.columns if "TIME" in c.upper()][0]
        val_col  = [c for c in df.columns if "OBS_VALUE" in c.upper()][0]
        s = pd.Series(
            pd.to_numeric(df[val_col], errors="coerce").values,
            index=pd.to_datetime(df[time_col]),
            name="eurusd",
        ).dropna().sort_index()
        log.info("EUR/USD načteno: %d pozorování", len(s))
        return s
    except Exception as e:
        log.info("EUR/USD: %s – záložní data.", e)
        return _fallback_eurusd()


def fetch_pribor(start: str = "2010-01-01", tenor: str = "3M") -> pd.Series:
    """
    PRIBOR 3M z ČNB - TXT soubory po rocích (bez API klíče).
    URL: https://www.cnb.cz/cs/financni-trhy/penezni-trh/pribor/
         fixing-urokovych-sazeb-na-mezibankovnim-trhu-depozit-pribor/rok.txt?year=YYYY
    Formát: Datum|O/N|1W|1M|2M|3M|6M|9M|1Y  (oddělovač |, decimal ,)
    """
    log.info("ČNB <- PRIBOR %s (TXT po rocích)", tenor)
    _tenor_aliases = {
        "3M": ("3M", "3 M", "3MES", "3 MES"),
        "1Y": ("1Y", "1 Y", "12M", "12 M", "1R", "1 R", "1ROK"),
    }
    _targets = _tenor_aliases.get(tenor.upper(), (tenor.upper(),))
    _colname = "pribor3m" if tenor.upper() == "3M" else "pribor12m"
    start_year = int(start[:4])
    end_year   = pd.Timestamp.now().year
    records = {}

    for year in range(start_year, end_year + 1):
        url = (
            "https://www.cnb.cz/cs/financni-trhy/penezni-trh/pribor/"
            "fixing-urokovych-sazeb-na-mezibankovnim-trhu-depozit-pribor/"
            f"rok.txt?year={year}"
        )
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            lines = r.text.strip().splitlines()
            if not lines:
                continue
            # Najdi řádek s hlavičkou (obsahuje datum a sazby)
            header = []
            data_start = 0
            for i, line in enumerate(lines):
                if "|" in line:
                    header = [h.strip().upper() for h in line.split("|")]
                    data_start = i + 1
                    break
            # Index sloupce podle tenoru - normalizované porovnání
            # (ČNB TXT mívá česky "3 měsíce" / "1 rok", s diakritikou i bez)
            def _norm(x):
                x = x.strip().upper().replace(" ", "")
                for a, b in [("Ě","E"),("É","E"),("Í","I"),("Á","A"),("Ů","U"),("Ú","U")]:
                    x = x.replace(a, b)
                return x
            _norm_targets = {
                "3M": {"3M", "3MES", "3MESICE", "3MESIC", "3MO"},
                "1Y": {"1Y", "12M", "1R", "1ROK", "ROK", "12MES", "12MESICU"},
            }.get(tenor.upper(), {_norm(tenor)})
            col_3m = None
            for j, h in enumerate(header):
                if _norm(h) in _norm_targets:
                    col_3m = j
                    break
            if col_3m is None:
                # Poziční fallback: 3M = 6. sloupec, 1Y = poslední sloupec
                # (formát: Datum|1D|1T|2T|1M|2M|3M|6M|9M|1R - 1Y je vždy poslední)
                if tenor.upper() == "3M" and len(header) >= 6:
                    col_3m = 5
                elif tenor.upper() in ("1Y", "12M") and len(header) >= 9:
                    col_3m = len(header) - 1
            if col_3m is None:
                continue
            for line in lines[data_start:]:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) <= col_3m:
                    continue
                try:
                    dt  = pd.to_datetime(parts[0], dayfirst=True)
                    val = float(parts[col_3m].replace(",", "."))
                    records[dt] = val
                except Exception:
                    continue
        except Exception as e:
            log.debug("PRIBOR rok %d: %s", year, e)

    # Záložní data jako základ – živá data z TXT je doplní/přepíší
    fallback = _fallback_pribor()
    if tenor.upper() == "1Y":
        # 12M nemá vlastní fallback sérii; použij 3M + typický spread ~0.2 pp
        fallback = (fallback + 0.2).rename("pribor12m")
    if records:
        live = pd.Series(records, name=_colname).sort_index()
        live = live[live.index >= start]
        live_q = live.resample("QS").mean().dropna()
        combined = fallback.copy()
        combined.update(live_q)
        log.info("PRIBOR %s: záložní + %d živých čtvrtletí", tenor, len(live_q))
        return combined.sort_index()
    log.info("PRIBOR %s: živé stahování nedostupné, používám záložní data.", tenor)
    return fallback


# ── 2.  Záložní data ───────────────────────────────────────────────────────────

def _fallback_eurczk() -> pd.Series:
    """Čtvrtletní průměry EUR/CZK 2010-2024 (zdroj ČNB)."""
    data = {
        "2010-01-01": 25.90, "2010-04-01": 25.46, "2010-07-01": 24.89, "2010-10-01": 24.55,
        "2011-01-01": 24.35, "2011-04-01": 24.31, "2011-07-01": 24.29, "2011-10-01": 25.00,
        "2012-01-01": 25.20, "2012-04-01": 25.23, "2012-07-01": 25.17, "2012-10-01": 25.16,
        "2013-01-01": 25.60, "2013-04-01": 25.82, "2013-07-01": 25.92, "2013-10-01": 27.20,
        "2014-01-01": 27.43, "2014-04-01": 27.43, "2014-07-01": 27.55, "2014-10-01": 27.64,
        "2015-01-01": 27.73, "2015-04-01": 27.53, "2015-07-01": 27.09, "2015-10-01": 27.19,
        "2016-01-01": 27.02, "2016-04-01": 27.03, "2016-07-01": 27.03, "2016-10-01": 27.02,
        "2017-01-01": 27.02, "2017-04-01": 27.01, "2017-07-01": 26.22, "2017-10-01": 25.68,
        "2018-01-01": 25.38, "2018-04-01": 25.54, "2018-07-01": 25.74, "2018-10-01": 25.84,
        "2019-01-01": 25.73, "2019-04-01": 25.68, "2019-07-01": 25.71, "2019-10-01": 25.55,
        "2020-01-01": 25.10, "2020-04-01": 27.12, "2020-07-01": 26.36, "2020-10-01": 27.01,
        "2021-01-01": 26.02, "2021-04-01": 25.77, "2021-07-01": 25.51, "2021-10-01": 25.40,
        "2022-01-01": 24.46, "2022-04-01": 24.49, "2022-07-01": 24.60, "2022-10-01": 24.63,
        "2023-01-01": 23.87, "2023-04-01": 23.44, "2023-07-01": 23.84, "2023-10-01": 24.39,
        "2024-01-01": 24.82, "2024-04-01": 24.95, "2024-07-01": 25.12, "2024-10-01": 25.38,
        "2025-01-01": 25.05, "2025-04-01": 25.22, "2025-07-01": 25.18, "2025-10-01": 25.28,
    }
    s = pd.Series({pd.Timestamp(k): v for k, v in data.items()}, name="eurczk")
    log.info("Záložní EUR/CZK načteno (%d pozorování)", len(s))
    return s


def _fallback_eurusd() -> pd.Series:
    """Čtvrtletní průměry EUR/USD 2010-2024."""
    data = {
        "2010-01-01": 1.385, "2010-04-01": 1.316, "2010-07-01": 1.291, "2010-10-01": 1.370,
        "2011-01-01": 1.366, "2011-04-01": 1.440, "2011-07-01": 1.421, "2011-10-01": 1.353,
        "2012-01-01": 1.306, "2012-04-01": 1.312, "2012-07-01": 1.237, "2012-10-01": 1.293,
        "2013-01-01": 1.328, "2013-04-01": 1.306, "2013-07-01": 1.318, "2013-10-01": 1.357,
        "2014-01-01": 1.369, "2014-04-01": 1.381, "2014-07-01": 1.353, "2014-10-01": 1.258,
        "2015-01-01": 1.131, "2015-04-01": 1.078, "2015-07-01": 1.101, "2015-10-01": 1.095,
        "2016-01-01": 1.094, "2016-04-01": 1.133, "2016-07-01": 1.112, "2016-10-01": 1.087,
        "2017-01-01": 1.064, "2017-04-01": 1.075, "2017-07-01": 1.134, "2017-10-01": 1.179,
        "2018-01-01": 1.227, "2018-04-01": 1.229, "2018-07-01": 1.168, "2018-10-01": 1.141,
        "2019-01-01": 1.135, "2019-04-01": 1.123, "2019-07-01": 1.112, "2019-10-01": 1.107,
        "2020-01-01": 1.101, "2020-04-01": 1.085, "2020-07-01": 1.149, "2020-10-01": 1.183,
        "2021-01-01": 1.214, "2021-04-01": 1.196, "2021-07-01": 1.181, "2021-10-01": 1.148,
        "2022-01-01": 1.124, "2022-04-01": 1.070, "2022-07-01": 1.017, "2022-10-01": 0.996,
        "2023-01-01": 1.074, "2023-04-01": 1.090, "2023-07-01": 1.096, "2023-10-01": 1.060,
        "2024-01-01": 1.084, "2024-04-01": 1.079, "2024-07-01": 1.088, "2024-10-01": 1.082,
        "2025-01-01": 1.052, "2025-04-01": 1.135, "2025-07-01": 1.120, "2025-10-01": 1.095,
        "2026-01-01": 1.04,
    }
    s = pd.Series({pd.Timestamp(k): v for k, v in data.items()}, name="eurusd")
    log.info("Záložní EUR/USD načteno (%d pozorování)", len(s))
    return s


def _fallback_pribor() -> pd.Series:
    """PRIBOR 3M čtvrtletní průměry 2010-2024 (zdroj ČNB)."""
    data = {
        "2010-01-01": 1.51, "2010-04-01": 1.26, "2010-07-01": 1.21, "2010-10-01": 1.18,
        "2011-01-01": 1.17, "2011-04-01": 1.22, "2011-07-01": 1.20, "2011-10-01": 1.18,
        "2012-01-01": 1.14, "2012-04-01": 1.09, "2012-07-01": 1.03, "2012-10-01": 0.78,
        "2013-01-01": 0.44, "2013-04-01": 0.34, "2013-07-01": 0.32, "2013-10-01": 0.37,
        "2014-01-01": 0.36, "2014-04-01": 0.37, "2014-07-01": 0.37, "2014-10-01": 0.34,
        "2015-01-01": 0.31, "2015-04-01": 0.29, "2015-07-01": 0.29, "2015-10-01": 0.29,
        "2016-01-01": 0.29, "2016-04-01": 0.29, "2016-07-01": 0.29, "2016-10-01": 0.29,
        "2017-01-01": 0.29, "2017-04-01": 0.29, "2017-07-01": 0.30, "2017-10-01": 0.41,
        "2018-01-01": 0.76, "2018-04-01": 1.09, "2018-07-01": 1.22, "2018-10-01": 1.82,
        "2019-01-01": 2.04, "2019-04-01": 2.10, "2019-07-01": 2.18, "2019-10-01": 2.17,
        "2020-01-01": 2.10, "2020-04-01": 0.90, "2020-07-01": 0.42, "2020-10-01": 0.36,
        "2021-01-01": 0.36, "2021-04-01": 0.36, "2021-07-01": 0.72, "2021-10-01": 2.22,
        "2022-01-01": 3.71, "2022-04-01": 5.61, "2022-07-01": 6.99, "2022-10-01": 7.25,
        "2023-01-01": 7.21, "2023-04-01": 7.18, "2023-07-01": 7.16, "2023-10-01": 7.02,
        "2024-01-01": 6.41, "2024-04-01": 5.41, "2024-07-01": 4.57, "2024-10-01": 3.97,
        "2025-01-01": 3.74, "2025-04-01": 3.53, "2025-07-01": 3.50, "2025-10-01": 3.50,
        "2026-01-01": 3.50, "2026-04-01": 3.55, "2026-07-01": 3.78,
    }
    s = pd.Series({pd.Timestamp(k): v for k, v in data.items()}, name="pribor3m")
    log.info("Záložní PRIBOR 3M načteno (%d pozorování)", len(s))
    return s



def fetch_repo_rate(start: str = "2010-01-01") -> pd.Series:
    """Repo sazba CNB - zkusi vice URL variant, fallback na zalozni data."""
    log.info("CNB <- Repo sazba")
    urls = [
        "https://www.cnb.cz/cs/casto-kladene-dotazy/Jak-se-vyvijela-dvoutydenni-repo-sazba-CNB/repo_2T_CZ.txt",
        "https://www.cnb.cz/cs/casto-kladene-dotazy/Jak-se-vyvijela-dvoutydenni-repo-sazba-CNB/repo_CZ.txt",
        "https://www.cnb.cz/cs/casto-kladene-dotazy/Jak-se-vyvijela-dvoutydenni-repo-sazba-CNB/repo_historie.txt",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=15)
            if r.status_code != 200:
                continue
            records = {}
            for line in r.text.strip().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                for sep in (";", "|", "	", ","):
                    parts = line.split(sep)
                    if len(parts) >= 2:
                        try:
                            dt  = pd.to_datetime(parts[0].strip(), dayfirst=True)
                            val = float(parts[1].strip().replace(",", "."))
                            records[dt] = val
                            break
                        except Exception:
                            continue
            if len(records) > 5:
                s = pd.Series(records, name="repo_rate").sort_index()
                s = s[s.index >= start]
                log.info("Repo sazba nactena z %s: %d pozorovani", url, len(s))
                return s
        except Exception as e:
            log.debug("Repo URL %s: %s", url, e)
    log.info("Repo sazba: živé stahování není dostupné (ČNB nevystavuje veřejné API), používám záložní data do Q2 2026.")
    return _fallback_repo_rate()

def _parse_jsonstat_series(js: dict) -> dict:
    """
    Robustně naparsuje JSON-stat odpověď na {time_label: hodnota}.
    Zvládne i víc dimenzí: dekóduje plochý index na souřadnice a vezme
    jen pozorování, kde jsou všechny ne-časové dimenze na indexu 0.
    """
    dim_ids = js["id"]            # pořadí dimenzí
    sizes   = js["size"]          # velikosti dimenzí (stejné pořadí)
    time_pos = dim_ids.index("time")
    time_index = js["dimension"]["time"]["category"]["index"]
    pos_to_time = {pos: label for label, pos in time_index.items()}

    # Strides pro row-major dekódování plochého indexu
    strides = [1] * len(sizes)
    for i in range(len(sizes) - 2, -1, -1):
        strides[i] = strides[i + 1] * sizes[i + 1]

    data = {}
    for idx_str, val in js.get("value", {}).items():
        if val is None:
            continue
        flat = int(idx_str)
        # Dekóduj na souřadnice
        coords = []
        rem = flat
        for st in strides:
            coords.append(rem // st)
            rem = rem % st
        # Vezmi jen pozorování, kde všechny ne-časové dimenze == 0
        if all(c == 0 for i, c in enumerate(coords) if i != time_pos):
            label = pos_to_time.get(coords[time_pos])
            if label:
                data[label] = float(val)
    return data


def fetch_unemployment(start: str = "2010-01-01") -> pd.Series:
    """Míra nezaměstnanosti ČR z Eurostatu - statistics 1.0 API (JSON-stat)."""
    log.info("Eurostat <- Nezaměstnanost ČR (statistics API)")
    base = ("https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/"
            "data/une_rt_q")
    # Zkus víc kombinací věkové skupiny – kód se mezi datasety liší
    attempts = [
        {"freq": "Q", "s_adj": "SA", "age": "TOTAL",  "unit": "PC_ACT", "sex": "T", "geo": "CZ"},
        {"freq": "Q", "s_adj": "SA", "age": "Y15-74", "unit": "PC_ACT", "sex": "T", "geo": "CZ"},
        {"freq": "Q", "s_adj": "SA", "age": "Y_GE15", "unit": "PC_ACT", "sex": "T", "geo": "CZ"},
    ]
    for dims in attempts:
        params = "&".join(f"{k}={v}" for k, v in dims.items())
        url = f"{base}?format=JSON&lang=EN&{params}&sinceTimePeriod=2010"
        try:
            r = requests.get(url, timeout=45, headers={"Accept": "application/json"})
            if r.status_code != 200:
                continue
            js = r.json()
            parsed = _parse_jsonstat_series(js)
            data = {}
            for label, val in parsed.items():
                if "-Q" in label:
                    y, q = label.split("-Q")
                    m = (int(q) - 1) * 3 + 1
                    data[pd.Timestamp(f"{y}-{m:02d}-01")] = val
            if data:
                s = pd.Series(data, name="unempl").sort_index()
                s = s[s.index >= start]
                log.info("Nezaměstnanost načtena: %d pozorování (age=%s)",
                         len(s), dims["age"])
                return s
        except Exception as e:
            log.debug("Nezaměstnanost pokus %s selhal: %s", dims["age"], e)
            continue

    log.info("Nezaměstnanost: Eurostat API nevrátilo data – záložní data.")
    return _fallback_unemployment()

def _fallback_repo_rate() -> pd.Series:
    """Repo sazba ČNB - čtvrtletní průměry 2010-2024 (zdroj ČNB)."""
    data = {
        "2010-01-01": 1.00, "2010-04-01": 0.75, "2010-07-01": 0.75, "2010-10-01": 0.75,
        "2011-01-01": 0.75, "2011-04-01": 0.75, "2011-07-01": 0.75, "2011-10-01": 0.75,
        "2012-01-01": 0.75, "2012-04-01": 0.75, "2012-07-01": 0.50, "2012-10-01": 0.25,
        "2013-01-01": 0.25, "2013-04-01": 0.25, "2013-07-01": 0.05, "2013-10-01": 0.05,
        "2014-01-01": 0.05, "2014-04-01": 0.05, "2014-07-01": 0.05, "2014-10-01": 0.05,
        "2015-01-01": 0.05, "2015-04-01": 0.05, "2015-07-01": 0.05, "2015-10-01": 0.05,
        "2016-01-01": 0.05, "2016-04-01": 0.05, "2016-07-01": 0.05, "2016-10-01": 0.05,
        "2017-01-01": 0.05, "2017-04-01": 0.05, "2017-07-01": 0.25, "2017-10-01": 0.50,
        "2018-01-01": 0.75, "2018-04-01": 1.00, "2018-07-01": 1.25, "2018-10-01": 1.75,
        "2019-01-01": 1.75, "2019-04-01": 2.00, "2019-07-01": 2.00, "2019-10-01": 2.00,
        "2020-01-01": 2.25, "2020-04-01": 0.50, "2020-07-01": 0.25, "2020-10-01": 0.25,
        "2021-01-01": 0.25, "2021-04-01": 0.25, "2021-07-01": 0.75, "2021-10-01": 2.75,
        "2022-01-01": 4.50, "2022-04-01": 6.25, "2022-07-01": 7.00, "2022-10-01": 7.00,
        "2023-01-01": 7.00, "2023-04-01": 7.00, "2023-07-01": 7.00, "2023-10-01": 6.75,
        "2024-01-01": 6.25, "2024-04-01": 5.25, "2024-07-01": 4.50, "2024-10-01": 4.00,
        "2025-01-01": 3.875,"2025-04-01": 3.50, "2025-07-01": 3.50, "2025-10-01": 3.50,
        "2026-01-01": 3.50, "2026-04-01": 3.50, "2026-07-01": 3.75,
    }
    s = pd.Series({pd.Timestamp(k): v for k, v in data.items()}, name="repo_rate")
    log.info("Záložní repo sazba načtena (%d pozorování)", len(s))
    return s


def _fallback_unemployment() -> pd.Series:
    """Míra nezaměstnanosti ČR - čtvrtletní 2010-2024 (zdroj ČSÚ/Eurostat)."""
    data = {
        "2010-01-01": 7.3, "2010-04-01": 7.0, "2010-07-01": 6.9, "2010-10-01": 7.0,
        "2011-01-01": 6.9, "2011-04-01": 6.5, "2011-07-01": 6.4, "2011-10-01": 6.5,
        "2012-01-01": 6.9, "2012-04-01": 6.9, "2012-07-01": 7.0, "2012-10-01": 7.2,
        "2013-01-01": 7.4, "2013-04-01": 7.2, "2013-07-01": 6.9, "2013-10-01": 6.8,
        "2014-01-01": 6.8, "2014-04-01": 6.2, "2014-07-01": 5.9, "2014-10-01": 5.9,
        "2015-01-01": 5.8, "2015-04-01": 5.1, "2015-07-01": 4.8, "2015-10-01": 4.6,
        "2016-01-01": 4.6, "2016-04-01": 4.0, "2016-07-01": 3.8, "2016-10-01": 3.6,
        "2017-01-01": 3.5, "2017-04-01": 2.9, "2017-07-01": 2.8, "2017-10-01": 2.5,
        "2018-01-01": 2.5, "2018-04-01": 2.2, "2018-07-01": 2.2, "2018-10-01": 2.1,
        "2019-01-01": 2.1, "2019-04-01": 2.0, "2019-07-01": 2.0, "2019-10-01": 2.1,
        "2020-01-01": 2.0, "2020-04-01": 2.7, "2020-07-01": 3.1, "2020-10-01": 3.1,
        "2021-01-01": 3.3, "2021-04-01": 3.2, "2021-07-01": 2.8, "2021-10-01": 2.4,
        "2022-01-01": 2.4, "2022-04-01": 2.3, "2022-07-01": 2.3, "2022-10-01": 2.4,
        "2023-01-01": 2.6, "2023-04-01": 2.6, "2023-07-01": 2.7, "2023-10-01": 2.8,
        "2024-01-01": 2.8, "2024-04-01": 2.7, "2024-07-01": 2.8, "2024-10-01": 2.9,
        "2025-01-01": 2.8, "2025-04-01": 2.7, "2025-07-01": 2.7, "2025-10-01": 2.7,
        "2026-01-01": 2.6,
    }
    s = pd.Series({pd.Timestamp(k): v for k, v in data.items()}, name="unempl")
    log.info("Záložní nezaměstnanost načtena (%d pozorování)", len(s))
    return s

# ── 3.  Resample na čtvrtletí ──────────────────────────────────────────────────

def to_quarterly(s: pd.Series) -> pd.Series:
    """Denní -> čtvrtletní průměr."""
    if s.index.freq and "Q" in str(s.index.freq):
        return s
    return s.resample("QS").mean().dropna()



def _current_quarter_start() -> pd.Timestamp:
    """Vrátí začátek aktuálního čtvrtletí (např. 2026-04-01 pro Q2 2026)."""
    today = pd.Timestamp.now()
    q_month = ((today.month - 1) // 3) * 3 + 1
    return pd.Timestamp(today.year, q_month, 1)


def _extend_to_present(series: pd.Series) -> pd.Series:
    """
    Pokud série zaostává za aktuálním čtvrtletím, doplní ji flat-forward
    (posledním známým pozorováním) až do Q současnosti.
    Prognóza pak vždy startuje od dneška, ne z minulosti.
    """
    last = series.dropna().index[-1]
    current_q = _current_quarter_start()
    if last >= current_q:
        return series  # série je aktuální, nic nedoplňovat
    # Doplň chybějící čtvrtletí poslední hodnotou
    fill_idx = pd.date_range(start=last + pd.offsets.QuarterBegin(1),
                             end=current_q, freq="QS")
    fill = pd.Series(series.dropna().iloc[-1], index=fill_idx, name=series.name)
    return pd.concat([series, fill]).sort_index()


# ── 4.  Modely prognózy ───────────────────────────────────────────────────────

def _forecast_rw(series: pd.Series, steps: int = 8, n_sims: int = 2000) -> pd.DataFrame:
    """
    Random walk s drift korekcí - standard pro devizové kurzy.
    Drift = průměrná čtvrtletní změna za posledních 5 let.
    Šířka pásma roste s sqrt(t) - typický FX fan chart.
    """
    series = _extend_to_present(series)
    vals = series.values.astype(float)
    # Drift z posledních 20 čtvrtletí (5 let)
    window = min(20, len(vals) - 1)
    changes = np.diff(vals[-window-1:])
    drift = changes.mean()
    sigma = changes.std()

    all_paths = []
    for _ in range(n_sims):
        path = [vals[-1]]
        for _ in range(steps):
            path.append(path[-1] + drift + np.random.normal(0, sigma))
        all_paths.append(path[1:])

    all_paths = np.array(all_paths)
    fut = pd.date_range(
        start=series.index[-1] + pd.offsets.QuarterBegin(1),
        periods=steps, freq="QS",
    )
    return pd.DataFrame({
        "lower_90": np.percentile(all_paths, 5,  axis=0),
        "lower_50": np.percentile(all_paths, 25, axis=0),
        "median":   np.percentile(all_paths, 50, axis=0),
        "upper_50": np.percentile(all_paths, 75, axis=0),
        "upper_90": np.percentile(all_paths, 95, axis=0),
    }, index=fut)


def _forecast_mean_reversion(
    series: pd.Series,
    steps: int = 8,
    n_sims: int = 2000,
    long_run_mean: float | None = None,
    speed: float = 0.25,
) -> pd.DataFrame:
    """
    AR(1) s mean reversion - vhodný pro úrokové sazby (PRIBOR).
    Ornstein-Uhlenbeck diskretizace:
        x(t+1) = x(t) + speed * (mu - x(t)) + sigma * epsilon

    Parametry
    ---------
    long_run_mean : rovnovážná sazba (pokud None, odhadne se z dat)
    speed         : rychlost návratu k průměru (0-1; 0.25 = pomalý, 0.5 = střední)
    """
    series = _extend_to_present(series)
    vals = series.values.astype(float)
    mu    = long_run_mean if long_run_mean is not None else float(np.mean(vals[-20:]))

    # Sigma z posledních 8 čtvrtletí (2 roky) — vyhne se nafukování
    # z velkých pohybů inflačního cyklu 2021–2024.
    # Minimum 8 pozorování; pokud je série kratší, použij celou.
    window = min(8, len(vals) - 1)
    resid = np.diff(vals[-window-1:]) - speed * (mu - vals[-window-1:-1])
    sigma = max(resid.std(), 0.05)

    all_paths = []
    for _ in range(n_sims):
        path = [vals[-1]]
        for _ in range(steps):
            nxt = path[-1] + speed * (mu - path[-1]) + np.random.normal(0, sigma)
            nxt = max(nxt, 0.0)   # sazba nemůže být záporná
            path.append(nxt)
        all_paths.append(path[1:])

    all_paths = np.array(all_paths)
    fut = pd.date_range(
        start=series.index[-1] + pd.offsets.QuarterBegin(1),
        periods=steps, freq="QS",
    )
    return pd.DataFrame({
        "lower_90": np.percentile(all_paths, 5,  axis=0),
        "lower_50": np.percentile(all_paths, 25, axis=0),
        "median":   np.percentile(all_paths, 50, axis=0),
        "upper_50": np.percentile(all_paths, 75, axis=0),
        "upper_90": np.percentile(all_paths, 95, axis=0),
    }, index=fut)


def _forecast_freeze_repo(
    repo: pd.Series,
    steps: int = 8,
) -> pd.DataFrame:
    """
    Zmrazená repo sazba: konstantní trajektorie na aktuální úrovni.
    Fan chart se nerozevírá (nulový rozptyl) — sazba zůstává fixní.
    Použití: scénář kdy předpokládáme, že ČNB nebude měnit sazby.
    """
    repo = _extend_to_present(repo)
    current = float(repo.dropna().iloc[-1])
    fut = pd.date_range(
        start=repo.index[-1] + pd.offsets.QuarterBegin(1),
        periods=steps, freq="QS",
    )
    vals = [current] * steps
    return pd.DataFrame({
        "lower_90": vals,
        "lower_50": vals,
        "median":   vals,
        "upper_50": vals,
        "upper_90": vals,
    }, index=fut)


def _forecast_pribor_linked(
    pribor: pd.Series,
    repo: pd.Series,
    repo_path: list,
    steps: int = 8,
    n_sims: int = 2000,
    speed: float = 0.30,
    spread_window: int = 8,
) -> pd.DataFrame:
    """
    Prognóza PRIBOR 3M navázaná na repo cestu.

    PRIBOR 3M ≈ očekávaný průměr repo na 3 měsíce + rozpětí (term/likviditní
    prémie). Modelujeme jako Ornstein-Uhlenbeck návrat k časově proměnnému cíli
    target_t = repo_path_t + spread, kde spread je nedávný průměr (PRIBOR − repo).
    PRIBOR tak kopíruje repo hrb místo míření k pevné konstantě.

    speed   - rychlost konvergence k cíli (sdílí --pribor-speed)
    spread  - odhadnut z posledních spread_window čtvrtletí překryvu sérií
    """
    # Spread z překryvu obou sérií (poslední spread_window čtvrtletí)
    joined = pd.concat([pribor.rename("p"), repo.rename("r")], axis=1).dropna()
    if len(joined) >= 1:
        recent = joined.tail(spread_window)
        spread = float((recent["p"] - recent["r"]).mean())
    else:
        spread = 0.0

    current = float(pribor.iloc[-1])

    # Sigma z reziduí PRIBOR vůči vlastnímu posunu (poslední okno),
    # aby fan chart nebyl příliš úzký ani přefouknutý cyklem 2021-2024.
    vals = pribor.values.astype(float)
    window = min(8, len(vals) - 1)
    if window >= 2:
        diffs = np.diff(vals[-window - 1:])
        sigma = max(float(np.std(diffs)), 0.03)
    else:
        sigma = 0.10

    # Cílová cesta = repo + spread
    target_path = [float(repo_path[t]) + spread for t in range(steps)]

    sims = np.zeros((n_sims, steps))
    for s in range(n_sims):
        level = current
        for t in range(steps):
            pull = speed * (target_path[t] - level)
            level = level + pull + np.random.normal(0, sigma)
            sims[s, t] = level

    fut = pd.date_range(start=pribor.index[-1] + pd.offsets.QuarterBegin(1),
                        periods=steps, freq="QS")
    qs = np.percentile(sims, [5, 25, 50, 75, 95], axis=0)
    return pd.DataFrame({
        "lower_90": qs[0], "lower_50": qs[1], "median": qs[2],
        "upper_50": qs[3], "upper_90": qs[4],
    }, index=fut)


def _forecast_unemployment(
    unempl: pd.Series,
    repo_path: list | None = None,
    steps: int = 8,
    n_sims: int = 2000,
    neutral_rate: float = 3.0,
    speed: float = 0.20,
    policy_sensitivity: float = 0.10,   # kalibrováno backtestem: minimum Theil U
                                        # (0.30 bylo 3x silné; český trh práce
                                        # na sazby reaguje jen slabě)
) -> pd.DataFrame:
    """
    Prognóza nezaměstnanosti: návrat k NAIRU + vliv měnové restrikce.

    Trh práce se vrací k strukturální míře (NAIRU), ale restriktivní měnová
    politika (repo nad neutrálem) ekonomiku chladí a tlačí nezaměstnanost
    nad NAIRU. Při napjatém trhu a vysokých sazbách tak u neklesá, ale spíš
    mírně roste k NAIRU - ne mechanická regrese k průměru bez ekonomiky.

    NAIRU = strukturální míra, odhadnutá z dolního okolí nedávných hodnot
    (trh je teď velmi napjatý, takže NAIRU je blízko nedávného minima).
    """
    vals = unempl.values.astype(float)
    current = float(vals[-1])

    # NAIRU jako robustní strukturální podlaha: medián posledních 12Q.
    # (Při velmi napjatém trhu je NAIRU blízko aktuální nízké úrovni.)
    recent = vals[-12:] if len(vals) >= 12 else vals
    nairu = float(np.median(recent))

    # Sigma z nedávných čtvrtletních změn
    window = min(8, len(vals) - 1)
    sigma = max(float(np.std(np.diff(vals[-window - 1:]))), 0.03) if window >= 2 else 0.08

    sims = np.zeros((n_sims, steps))
    for s in range(n_sims):
        level = current
        for t in range(steps):
            # Cíl = NAIRU + efekt restrikce (repo nad neutrálem zvyšuje u)
            if repo_path is not None and t < len(repo_path):
                rate_gap = float(repo_path[t]) - neutral_rate
            else:
                rate_gap = 0.0
            u_target = nairu + policy_sensitivity * max(0.0, rate_gap)
            level = level + speed * (u_target - level) + np.random.normal(0, sigma)
            sims[s, t] = level

    fut = pd.date_range(start=unempl.index[-1] + pd.offsets.QuarterBegin(1),
                        periods=steps, freq="QS")
    qs = np.percentile(sims, [5, 25, 50, 75, 95], axis=0)
    return pd.DataFrame({
        "lower_90": qs[0], "lower_50": qs[1], "median": qs[2],
        "upper_50": qs[3], "upper_90": qs[4],
    }, index=fut)


def _forecast_taylor_repo(
    repo: pd.Series,
    inflation_path: list | None = None,
    gdp_gap_path: list | None = None,
    steps: int = 8,
    n_sims: int = 2000,
    neutral_rate: float = 3.5,
    inflation_target: float = 2.0,
    lambda_pi: float = 1.5,     # citlivost na inflaci (Taylorovo pravidlo)
    lambda_y: float = 0.5,      # citlivost na výstupní mezeru
    step_size: float = 0.25,    # minimální krok ČNB (25 bp)
    change_prob_base: float = 0.45,  # pravděpodobnost změny na zasedání (8×/rok = ~2×/Q)
    anchor_to_neutral: bool = True,  # True: repo konverguje k neutral_rate (baseline);
                                     # False: Taylor reaguje na plnou úroveň inflace (scénáře)
    smoothing: float = 0.0,          # ρ: setrvačnost sazeb (0 = baseline beze změny chování,
                                     # ~0.7 pro scénáře: postupné, vyhlazené reakce ČNB)
) -> pd.DataFrame:
    """
    Prognóza repo sazby ČNB pomocí stochastického Taylorova pravidla.

    Model: v každém čtvrtletí ČNB porovná aktuální sazbu s "Taylorovou sazbou":
        r* = neutrální_sazba + λ_π*(inflace - cíl) + λ_y*výstupní_mezera

    Pokud je rozdíl |r_current - r*| > 0.25 pp, ČNB s pravděpodobností
    change_prob_base sazbu změní o jeden krok 25 bp směrem k r*.
    Fan chart ukazuje distribuci možných trajektorií.
    """
    repo = _extend_to_present(repo)
    vals = repo.values.astype(float)
    current = vals[-1]

    # Defaultní path (pokud není zadán): inflace konverguje k cíli, gap = 0
    if inflation_path is None:
        # Plynulá konvergence k 2 % za 2 roky
        last_infl = 2.7   # přibližná poslední hodnota
        inflation_path = [
            last_infl + (inflation_target - last_infl) * (i / (steps - 1))
            for i in range(steps)
        ]
    if gdp_gap_path is None:
        gdp_gap_path = [0.0] * steps   # neutrální výstup

    fut = pd.date_range(
        start=repo.index[-1] + pd.offsets.QuarterBegin(1),
        periods=steps, freq="QS",
    )

    all_paths = []
    rng = np.random.default_rng(seed=None)

    # Odvoď "skutečnou" neutrální sazbu v Taylor vzorci tak, aby model
    # konvergoval k neutral_rate při inflaci na cíli (pi = inflation_target).
    # Uživatel zadává cílovou steady-state sazbu, ne vstupní r̄ Taylorova vzorce.
    # Při pi = pi*: neutral_rate = r_bar + 0 → r_bar = neutral_rate
    # (Taylor korekce je nulová když inflace = cíl, takže r* = r_bar = neutral_rate)
    # neutral_rate je CÍLOVÁ sazba (kam má repo konvergovat).
    # Taylorovo pravidlo: r_star = r_bar + λπ*(π - π*) + λy*y
    # Aby r_star → neutral_rate při π → průměrné inflaci v horizontu,
    # nastavíme r_bar = neutral_rate - λπ*(π_avg - π*).
    # Tím model konverguje k zadanému neutral_rate jako výsledné sazbě.
    if anchor_to_neutral:
        # Baseline: uživatel zadává CÍLOVOU steady-state sazbu; odečti
        # inflační prémii, aby repo konvergovalo přesně k neutral_rate.
        pi_avg = float(np.mean(inflation_path))
        r_bar = neutral_rate - lambda_pi * (pi_avg - inflation_target)
    else:
        # Scénářový režim: r_bar je skutečná neutrální sazba a Taylor
        # reaguje na plnou úroveň inflace vůči cíli. Inflačnější scénář
        # tak endogenně znamená vyšší sazby.
        r_bar = neutral_rate

    for _ in range(n_sims):
        path = [current]
        r = current
        for t in range(steps):
            pi   = inflation_path[t]
            y    = gdp_gap_path[t]
            # Taylorovo pravidlo: korekce za odchylku inflace OD CÍLE
            # Při pi = pi*: r_star = r_bar = neutral_rate (žádná korekce)
            r_star = r_bar + lambda_pi * (pi - inflation_target) + lambda_y * y
            # Setrvačnost sazeb (inertial Taylor, Clarida-Galí-Gertler 2000):
            # centrální banky vyhlazují, cílová sazba se posouvá postupně.
            r_star = smoothing * r + (1.0 - smoothing) * r_star
            # Přidej náhodnost do r*
            r_star += rng.normal(0, 0.15)

            diff = r_star - r
            # Pravděpodobnost změny roste s velikostí odchylky
            prob = change_prob_base * min(1.0, abs(diff) / 0.5)

            if rng.random() < prob:
                # ČNB změní sazbu o krok (nebo více kroků najednou)
                n_steps = max(1, int(round(abs(diff) / step_size)))
                n_steps = min(n_steps, 4)   # max 100 bp najednou
                direction = 1 if diff > 0 else -1
                r = round(r + direction * n_steps * step_size, 2)
                r = max(0.0, r)   # nulová spodní hranice

            path.append(r)
        all_paths.append(path[1:])

    all_paths = np.array(all_paths)
    return pd.DataFrame({
        "lower_90": np.percentile(all_paths, 5,  axis=0),
        "lower_50": np.percentile(all_paths, 25, axis=0),
        "median":   np.percentile(all_paths, 50, axis=0),
        "upper_50": np.percentile(all_paths, 75, axis=0),
        "upper_90": np.percentile(all_paths, 95, axis=0),
    }, index=fut)


def forecast_financial(
    df: pd.DataFrame,
    steps: int = 8,
    pribor_long_run: float = 3.0,
    pribor_speed: float = 0.30,
    repo_neutral: float = 3.5,
    repo_freeze: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Spustí vhodný model pro každou finanční proměnnou.

    Vrací dict: {proměnná: DataFrame s lower_90/50, median, upper_50/90}

    Parametry
    ---------
    pribor_long_run : dlouhodobá rovnovážná sazba PRIBOR [%]
    pribor_speed    : rychlost konvergence k rovnováze
    repo_neutral    : neutrální repo sazba ČNB (default 3.5 %)
    """
    intervals = {}

    # EUR/CZK a EUR/USD - random walk s driftem
    for var in ["eurczk", "eurusd"]:
        if var in df.columns:
            intervals[var] = _forecast_rw(df[var].dropna(), steps=steps)

    # Repo sazba - POŘADÍ DŮLEŽITÉ: počítá se před PRIBOR, který ji sleduje.
    repo_path = None
    if "repo_rate" in df.columns:
        if repo_freeze:
            intervals["repo_rate"] = _forecast_freeze_repo(
                df["repo_rate"].dropna(),
                steps=steps,
            )
        else:
            intervals["repo_rate"] = _forecast_taylor_repo(
                df["repo_rate"].dropna(),
                steps=steps,
                neutral_rate=repo_neutral,
            )
        repo_path = intervals["repo_rate"]["median"].tolist()

    # PRIBOR 3M - sleduje repo cestu + rozpětí (term/likviditní prémie).
    # PRIBOR 3M je v podstatě průměr očekávané repo na 3M + malé stabilní rozpětí,
    # takže musí kopírovat repo hrb, ne mířit k pevné konstantě.
    # pribor_long_run se použije jen jako záloha, když repo cesta chybí.
    if "pribor3m" in df.columns:
        if repo_path is not None and "repo_rate" in df.columns:
            intervals["pribor3m"] = _forecast_pribor_linked(
                pribor=df["pribor3m"].dropna(),
                repo=df["repo_rate"].dropna(),
                repo_path=repo_path,
                steps=steps,
                speed=pribor_speed,
            )
        else:
            # Záloha: původní mean reversion k pevné rovnováze
            intervals["pribor3m"] = _forecast_mean_reversion(
                df["pribor3m"].dropna(),
                steps=steps,
                long_run_mean=pribor_long_run,
                speed=pribor_speed,
            )

    # PRIBOR 12M - navázaný na repo, spread implikovaný TRHEM (z dat 12M-repo)
    if "pribor12m" in df.columns and repo_path is not None:
        intervals["pribor12m"] = _forecast_pribor_linked(
            pribor=df["pribor12m"].dropna(),
            repo=df["repo_rate"].dropna(),
            repo_path=repo_path,
            steps=steps,
            speed=pribor_speed,
        )

    # Nezaměstnanost - návrat k NAIRU + vliv měnové restrikce (přes repo cestu).
    # Restriktivní politika (repo > neutrál) chladí ekonomiku a tlačí u nahoru;
    # NAIRU je strukturální podlaha, pod kterou trh dlouhodobě neklesá.
    if "unempl" in df.columns:
        intervals["unempl"] = _forecast_unemployment(
            unempl=df["unempl"].dropna(),
            repo_path=repo_path,
            steps=steps,
            neutral_rate=repo_neutral,
        )

    return intervals


# ── 5.  Fan chart pro finanční proměnnou ─────────────────────────────────────

def plot_financial_fan(
    series: pd.Series,
    intervals: pd.DataFrame,
    variable: str,
    quarter_label: str = "",
    history_years: int = 6,
    save_path: str | None = None,
) -> None:
    """
    Fan chart v MUP stylu pro finanční proměnné.
    Stejná vizuální logika jako var_model.plot_fan_chart.
    """
    color  = FIN_COLORS.get(variable, MUP["navy"])
    label  = FIN_LABELS.get(variable, variable.upper())
    ylabel = FIN_YLABEL.get(variable, "")

    s = series.dropna()
    cutoff = s.index[-1] - pd.DateOffset(years=history_years)
    hist_plot = s[s.index >= cutoff]
    ival = intervals

    fig, ax = plt.subplots(figsize=(11, 4.8))
    fig.patch.set_facecolor(MUP["white"])
    ax.set_facecolor(MUP["gray_bg"])

    ax.yaxis.grid(True, color=MUP["white"], linewidth=1.2, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color(MUP["gray_mid"])
        ax.spines[sp].set_linewidth(0.8)

    # Historická linka
    ax.plot(hist_plot.index, hist_plot.values,
            color=MUP["navy"], linewidth=2.2, zorder=5, solid_capstyle="round",
            label="Skutečnost")

    # Fan pásma - napoj od posledního hist bodu
    last_v  = hist_plot.values[-1]
    last_dt = hist_plot.index[-1]

    fan_idx = pd.DatetimeIndex([last_dt]).append(ival.index)
    lo90 = np.concatenate([[last_v], ival["lower_90"].values])
    hi90 = np.concatenate([[last_v], ival["upper_90"].values])
    lo50 = np.concatenate([[last_v], ival["lower_50"].values])
    hi50 = np.concatenate([[last_v], ival["upper_50"].values])
    med  = np.concatenate([[last_v], ival["median"].values])

    ax.fill_between(fan_idx, lo90, hi90, alpha=0.15, color=color, zorder=2)
    ax.fill_between(fan_idx, lo50, hi50, alpha=0.30, color=color, zorder=3)

    # Prognóza (medián)
    ax.plot(fan_idx, med, color=color, linewidth=2.2, linestyle="--",
            zorder=6, solid_capstyle="round", label="Prognóza (medián)")

    # Svislá čára "Nyní"
    ax.axvline(last_dt, color=MUP["gray_mid"], linewidth=1.0, linestyle=":", zorder=4)

    # Anotace
    ax.annotate(f"{last_v:.2f}",
                xy=(last_dt, last_v),
                xytext=(-40, 8), textcoords="offset points",
                fontsize=9, color=MUP["navy"], fontweight="bold")
    fc_last = ival["median"].values[-1]
    ax.annotate(f"{fc_last:.2f}",
                xy=(ival.index[-1], fc_last),
                xytext=(6, 0), textcoords="offset points",
                fontsize=9, color=color, fontweight="bold")

    # COVID pás (pokud v rozsahu)
    covid_s, covid_e = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-09-30")
    if hist_plot.index[0] <= covid_s and covid_e <= hist_plot.index[-1]:
        ax.axvspan(covid_s, covid_e, alpha=0.08, color="#888888", zorder=1)
        ax.text(covid_s + pd.Timedelta(days=45),
                ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.02,
                "COVID", fontsize=7, color=MUP["gray_mid"], va="bottom")

    # Legenda
    legend_elements = [
        Line2D([0], [0], color=MUP["navy"], linewidth=2, label="Skutečnost"),
        Line2D([0], [0], color=color, linewidth=2, linestyle="--", label="Prognóza (medián)"),
        plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.30, label="50% interval"),
        plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.15, label="90% interval"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8.5,
              framealpha=0.9, edgecolor=MUP["gray_mid"], facecolor=MUP["white"])

    ax.set_title(label, fontsize=24, fontweight="bold", color=MUP["navy"], loc="center", pad=14)
    ax.set_ylabel(ylabel, fontsize=10, color=MUP["gray_dark"], rotation=0, labelpad=14)
    ax.tick_params(axis="both", labelsize=9, colors=MUP["gray_dark"])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate(rotation=0, ha="center")

    ql = quarter_label or ""
    fig.text(0.5, 0.01,
             f"Zdroj: ČNB, ECB  |  {('Prognóza ' + ql).strip()}  |  MUP",
             ha="center", va="bottom", fontsize=7.5, color=MUP["gray_mid"])

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=MUP["white"])
        log.info("Fan chart uložen: %s", save_path)
    plt.close(fig)


# ── 6.  Přehledový graf (3 fan charty vedle sebe) ────────────────────────────

def plot_financial_overview(
    df: pd.DataFrame,
    intervals: dict[str, pd.DataFrame],
    quarter_label: str = "",
    history_years: int = 6,
    save_path: str | None = None,
) -> None:
    """Tři fan charty vedle sebe - přehled finančních proměnných."""
    variables = ["eurczk", "eurusd", "pribor3m"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.patch.set_facecolor(MUP["white"])
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=history_years)

    for ax, var in zip(axes, variables):
        if var not in df.columns or var not in intervals:
            ax.set_visible(False)
            continue

        s = df[var].dropna()
        hist = s[s.index >= cutoff]
        ival = intervals[var]
        color  = FIN_COLORS[var]
        label  = FIN_LABELS[var]
        ylabel = FIN_YLABEL[var]

        ax.set_facecolor(MUP["gray_bg"])
        ax.yaxis.grid(True, color=MUP["white"], linewidth=1.0, zorder=0)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["bottom", "left"]:
            ax.spines[sp].set_color(MUP["gray_mid"])
            ax.spines[sp].set_linewidth(0.8)

        # Historická linka
        ax.plot(hist.index, hist.values,
                color=MUP["navy"], linewidth=1.8, zorder=5, solid_capstyle="round")

        # Fan pásma
        last_v  = hist.values[-1]
        last_dt = hist.index[-1]
        fan_idx = pd.DatetimeIndex([last_dt]).append(ival.index)
        lo90 = np.concatenate([[last_v], ival["lower_90"].values])
        hi90 = np.concatenate([[last_v], ival["upper_90"].values])
        lo50 = np.concatenate([[last_v], ival["lower_50"].values])
        hi50 = np.concatenate([[last_v], ival["upper_50"].values])
        med  = np.concatenate([[last_v], ival["median"].values])

        ax.fill_between(fan_idx, lo90, hi90, alpha=0.15, color=color, zorder=2)
        ax.fill_between(fan_idx, lo50, hi50, alpha=0.30, color=color, zorder=3)
        ax.plot(fan_idx, med, color=color, linewidth=1.8, linestyle="--", zorder=6)
        ax.axvline(last_dt, color=MUP["gray_mid"], linewidth=0.8, linestyle=":", zorder=4)

        # Anotace poslední + prognóza
        ax.annotate(f"{last_v:.2f}", xy=(last_dt, last_v),
                    xytext=(-36, 7), textcoords="offset points",
                    fontsize=8, color=MUP["navy"], fontweight="bold")
        ax.annotate(f"{ival['median'].values[-1]:.2f}", xy=(ival.index[-1], ival["median"].values[-1]),
                    xytext=(5, 2), textcoords="offset points",
                    fontsize=8, color=color, fontweight="bold")

        ax.set_title(label, fontsize=20, fontweight="bold",
                     color=MUP["navy"], loc="center", pad=10)
        ax.set_ylabel(ylabel, fontsize=9, color=MUP["gray_dark"], rotation=0, labelpad=14)
        ax.tick_params(axis="both", labelsize=8.5, colors=MUP["gray_dark"])
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))

    fig.autofmt_xdate(rotation=0, ha="center")
    ql = quarter_label or ""
    fig.suptitle(f"Finanční trhy - prognóza  |  {ql}",
                 fontsize=13, fontweight="bold", color=MUP["navy"], y=1.02)
    fig.text(0.99, -0.02,
             f"Zdroj: ČNB, ECB  |  MUP",
             ha="right", fontsize=7.5, color=MUP["gray_mid"])

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=MUP["white"])
        log.info("Přehledový fan chart uložen: %s", save_path)
    plt.close(fig)


# ── 7.  Sestavení datasetu + uložení ──────────────────────────────────────────

def build_financial_dataset(use_cache: bool = False,
                            max_age_hours: float = 12.0) -> pd.DataFrame:
    """Stáhne / načte všechna finanční data a uloží do CSV."""
    if use_cache:
        import time as _time
        _p = os.path.join(RAW_DIR, "financial_data.csv")
        if os.path.exists(_p):
            _age = (_time.time() - os.path.getmtime(_p)) / 3600.0
            if _age <= max_age_hours:
                df = pd.read_csv(_p, index_col=0, parse_dates=True)
                df = pd.DataFrame(
                    {col: _extend_to_present(df[col]) for col in df.columns}
                ).sort_index()
                log.info("Finanční data načtena z cache (%.1f h, do %s)",
                         _age, df.index[-1].date())
                return df
    eurczk = to_quarterly(fetch_eurczk())
    eurusd = to_quarterly(fetch_eurusd())
    pribor = to_quarterly(fetch_pribor())
    pribor12m = to_quarterly(fetch_pribor(tenor="1Y"))

    repo  = to_quarterly(fetch_repo_rate())
    unempl = to_quarterly(fetch_unemployment())

    # Sestav DataFrame – inner join (průnik datumů) zamezí NaN řádkům
    # kdy jedna série (EUR/USD) sahá dál do budoucnosti než ostatní
    series_dict = {
        "eurczk":    eurczk,
        "eurusd":    eurusd,
        "pribor3m":  pribor,
        "pribor12m": pribor12m,
        "repo_rate": repo,
        "unempl":    unempl,
    }

    # Doplň KAŽDOU sérii flat-forward do současnosti PŘED spojením. Jinak by
    # dropna() ořízl dataset na nejkratší sérii a nejnovější data (např. čerstvé
    # zvýšení repo sazby) by se ztratila, protože jiná série tam ještě nesahá.
    series_dict = {k: _extend_to_present(v) for k, v in series_dict.items()}
    df = pd.DataFrame(series_dict).sort_index()
    df = df.ffill(limit=2)   # zaplň drobné mezery
    df = df.dropna()          # odstraní už jen úvodní řádky (kde série ještě nezačala)

    path = os.path.join(RAW_DIR, "financial_data.csv")
    df.to_csv(path)

    log.info("Finanční dataset uložen: %s (%d čtvrtletí, do %s)",
             path, len(df), df.index[-1].date())
    return df


def save_intervals(intervals: dict, path: str) -> None:
    """Uloží prognózní intervaly do JSON pro sdílení mezi skripty."""
    import json
    out = {}
    for var, df in intervals.items():
        out[var] = {
            "index": [str(d.date()) for d in df.index],
            "lower_90": df["lower_90"].tolist(),
            "lower_50": df["lower_50"].tolist(),
            "median":   df["median"].tolist(),
            "upper_50": df["upper_50"].tolist(),
            "upper_90": df["upper_90"].tolist(),
        }
    with open(path, "w") as f:
        json.dump(out, f)
    log.info("Finanční intervaly uloženy: %s", path)


def load_intervals(path: str) -> dict:
    """Načte prognózní intervaly z JSON."""
    import json
    with open(path) as f:
        raw = json.load(f)
    result = {}
    for var, d in raw.items():
        idx = pd.DatetimeIndex(d["index"])
        result[var] = pd.DataFrame({
            "lower_90": d["lower_90"],
            "lower_50": d["lower_50"],
            "median":   d["median"],
            "upper_50": d["upper_50"],
            "upper_90": d["upper_90"],
        }, index=idx)
    return result


# ── 8.  Spuštění ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import datetime
    import argparse

    parser = argparse.ArgumentParser(description="Finanční prognózy - MUP model")
    parser.add_argument("--pribor-eq",    type=float, default=3.0,
                        help="Rovnovážná sazba PRIBOR 3M v %% (default: 3.0)")
    parser.add_argument("--pribor-speed", type=float, default=0.30,
                        help="Rychlost návratu PRIBOR k rovnováze 0-1 (default: 0.30)")
    parser.add_argument("--eurczk-eq",    type=float, default=None,
                        help="Cílová rovnovážná hodnota EUR/CZK (default: random walk bez cíle)")
    parser.add_argument("--eurusd-eq",    type=float, default=None,
                        help="Cílová rovnovážná hodnota EUR/USD (default: random walk bez cíle)")
    parser.add_argument("--fx-speed",     type=float, default=0.15,
                        help="Rychlost návratu FX kurzů k rovnováze, pokud je zadána (default: 0.15)")
    parser.add_argument("--repo-neutral",  type=float, default=3.5,
                        help="Neutrální repo sazba ČNB pro Taylorovo pravidlo (default: 3.5)")
    parser.add_argument("--horizon",      type=int, default=12,
                        help="Prognózní horizont ve čtvrtletích (default: 12 = 3 roky pro ČNB)")
    parser.add_argument("--output-dir",   type=str, default=None,
                        help="Výstupní složka pro grafy (default: outputs/charts vedle skriptu)")
    parser.add_argument("--repo-freeze",   action="store_true", default=False,
                        help="Zmrazit repo sazbu na aktuální úrovni (ignoruje --repo-neutral)")
    args = parser.parse_args()

    now = datetime.date.today()
    ql  = f"{now.year}-Q{(now.month-1)//3+1}"

    log.info("=== Finanční data - fetch + prognózy + grafy ===")
    log.info("Parametry: PRIBOR eq=%.2f%%, speed=%.2f | EUR/CZK eq=%s | EUR/USD eq=%s",
             args.pribor_eq, args.pribor_speed,
             f"{args.eurczk_eq:.3f}" if args.eurczk_eq else "RW",
             f"{args.eurusd_eq:.3f}" if args.eurusd_eq else "RW")

    df = build_financial_dataset()

    print("\nFinanční data (posledních 6 čtvrtletí):")
    print(df.tail(6).round(3).to_string())

    # Prognózy - FX buď čistý RW, nebo mean reversion pokud zadána rovnováha
    intervals_path = os.path.join(RAW_DIR, "fin_intervals.json")
    if args.output_dir:
        import pathlib
        CHARTS_DIR = str(pathlib.Path(args.output_dir).expanduser().resolve())
        os.makedirs(CHARTS_DIR, exist_ok=True)
    log.info("Generuji prognózy...")
    # Použij forecast_financial – zajistí konzistenci se všemi parametry
    intervals = forecast_financial(
        df, steps=args.horizon,
        pribor_long_run=args.pribor_eq,
        pribor_speed=args.pribor_speed,
        repo_neutral=args.repo_neutral,
        repo_freeze=args.repo_freeze,
    )

    # FX: přepis pokud byl zadán cílový kurz (mean reversion místo RW)
    for fx_var, eq_val in [("eurczk", args.eurczk_eq), ("eurusd", args.eurusd_eq)]:
        if fx_var not in df.columns:
            continue
        if eq_val is not None:
            intervals[fx_var] = _forecast_mean_reversion(
                df[fx_var].dropna(), steps=args.horizon,
                long_run_mean=eq_val, speed=args.fx_speed,
            )

    # Tabulka mediánů
    print("\nBodové prognózy (medián):")
    fc_table = pd.DataFrame({var: intervals[var]["median"] for var in intervals})
    fc_table.index = [f"{d.year}-Q{(d.month-1)//3+1}" for d in fc_table.index]
    print(fc_table.round(3).to_string())

    # Individuální fan charty
    # Ulož intervaly pro report_generator a sketch_report
    save_intervals(intervals, intervals_path)

    log.info("Generuji fan charty...")
    for var in ["eurczk", "eurusd", "pribor3m", "repo_rate", "unempl"]:
        if var not in intervals or var not in df.columns:
            log.warning("Přeskakuji graf %s – chybí v intervals nebo df", var)
            continue
        plot_financial_fan(
            series=df[var],
            intervals=intervals[var],
            variable=var,
            quarter_label=ql,
            save_path=os.path.join(CHARTS_DIR, f"fin_{var}.png"),
        )

    # Přehledový graf
    plot_financial_overview(
        df=df,
        intervals=intervals,
        quarter_label=ql,
        save_path=os.path.join(CHARTS_DIR, "fin_overview.png"),
    )

    print(f"\nGrafy uloženy do: {CHARTS_DIR}")
