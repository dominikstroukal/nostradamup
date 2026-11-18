"""
Registr vstupních časových řad a hlídač tichého zastarání.

PROČ TOHLE EXISTUJE
───────────────────
Nejčastější a nejzákeřnější chyba v tomhle projektu není špatný vzorec, ale
mrtvý zdroj dat. Fetcher má zálohu, takže když živý zdroj zmizí nebo se
posune, nic nespadne a čísla vypadají normálně. Stalo se to nejmíň třikrát:
ČSÚ vyřadil sadu WCEN01M a inflace zamrzla na prosinci 2025, EUR/CZK stálo
rok na hodnotě z konce roku 2025, mzdy se tiše flat-forwardovaly.

Tenhle modul z toho dělá měřitelnou podmínku. U každé vstupní řady je
zapsané, jak rychle ji její instituce obvykle zveřejňuje. Z toho se spočítá,
jaké období už MÁ být k dispozici, a porovná se s tím, co jsme skutečně
stáhli. Když je řada pozadu, řekne se to nahlas a `main()` skončí nenulovým
návratovým kódem, takže to zastaví i CI.

Druhá věc, kterou modul hlídá: jestli řada přišla ze živého zdroje, nebo ze
zálohy. Záloha smí zachránit běh, ale nesmí vypadat jako úspěch.

JAK SE POUŽÍVÁ
──────────────
Fetchery po stažení zavolají `record()`:

    data_registry.record("csu_cpi", s.index[-1], live=True)

Souhrn za celý běh vytiskne `report()`. Samostatně:

    python data_registry.py            # tabulka stavu, exit 1 při zpoždění
    python data_registry.py --json     # strojově čitelný výstup
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_STATUS_PATH = os.path.join(_HERE, "data", "raw", "source_status.json")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Registr řad
# ─────────────────────────────────────────────────────────────────────────────
#
# lag_days  = obvyklý počet dní mezi KONCEM sledovaného období a zveřejněním.
# lag_basis = odkud to číslo je:
#               "kalendar"  – z ohlášeného kalendáře instituce,
#               "praxe"     – z dlouhodobě pozorované praxe vydávání,
#               "odhad"     – kvalifikovaný odhad, ověřit.
# grace_days = tolerance navíc, než se řada označí za pozadu (svátky, posuny).
# role      = "primarni" (výchozí) nebo "zaloha". Zastaralá ZÁLOHA není
#             poplach, dokud primární zdroj šlape; teprve když umlknou oba,
#             je to kritické. Alarm, který houká pořád, nikdo nečte.
# backs_up  = klíče primárních řad, které tahle záloha kryje.
#
# Když se ukáže, že nějaké číslo nesedí, oprav ho TADY a napiš proč. Tenhle
# slovník je jediný zdroj pravdy; nikde jinde se lhůty nedrží.

SOURCES: dict[str, dict] = {
    # ── ČSÚ ────────────────────────────────────────────────────────────────
    "csu_cpi": {
        "label": "Index spotřebitelských cen (CPI), měsíčně",
        "institution": "ČSÚ",
        "dataset": "CEN0101E",
        "catalog": "https://data.csu.gov.cz/api/katalog/v1/sady/CEN0101E",
        "freq": "M",
        "lag_days": 12,
        "lag_basis": "praxe",
        "note": "Nástupce vyřazené sady WCEN01M. Ukazatele RR / RM / IZ2015.",
        "feeds": ["cpi_yoy"],
        "fallback": "Eurostat prc_hicp_manr",
    },
    "csu_hicp": {
        "label": "Harmonizovaný index spotřebitelských cen (HICP), měsíčně",
        "institution": "ČSÚ",
        "dataset": "CEN0101M",
        "catalog": "https://data.csu.gov.cz/api/katalog/v1/sady/CEN0101M",
        "freq": "M",
        "lag_days": 12,
        "lag_basis": "praxe",
        "note": "Vychází ve stejné tiskové zprávě jako národní CPI.",
        "feeds": ["hicp_yoy"],
        "fallback": "Eurostat prc_hicp_manr",
    },
    "csu_fuel": {
        "label": "Ceny pohonných hmot, týdenní šetření",
        "institution": "ČSÚ",
        "dataset": "CENPHMT",
        "catalog": "https://data.csu.gov.cz/api/katalog/v1/sady/CENPHMT",
        # Šetření je týdenní, ale fetcher vrací měsíční koše se sloupcem
        # 'weeks'. Hlídá se tedy měsíc; rozběhnutý měsíc se objeví hned.
        "freq": "M",
        "lag_days": 6,
        "lag_basis": "praxe",
        "note": "Hlavní včasný vstup nowcastu inflace, je zhruba dva měsíce "
                "před indexem spotřebitelských cen.",
        "feeds": ["cpi_nowcast"],
        "fallback": None,
    },
    "csu_wages": {
        "label": "Průměrná hrubá měsíční mzda, čtvrtletně",
        "institution": "ČSÚ",
        "dataset": "MZDQ1",
        "catalog": "https://data.csu.gov.cz/api/katalog/v1/sady/MZDQ1",
        "freq": "Q",
        "lag_days": 66,
        "lag_basis": "praxe",
        "note": "Zveřejňuje se zhruba první týden třetího měsíce po konci Q.",
        "feeds": ["wages_yoy"],
        "fallback": "ARAD/ČNB, dále statická záloha (končí 2025Q4)",
    },

    # ── Eurostat ───────────────────────────────────────────────────────────
    "eurostat_gdp": {
        "label": "HDP, sezónně očištěný objem (B1GQ)",
        "institution": "Eurostat",
        "dataset": "namq_10_gdp",
        "catalog": None,
        "freq": "Q",
        "lag_days": 65,
        "lag_basis": "kalendar",
        "note": "Kalendář čtvrtletních národních účtů. Bleskový odhad vychází "
                "dřív, ale my bereme plnou sadu.",
        "feeds": ["gdp_qoq", "gdp_yoy"],
        "fallback": None,
    },
    "eurostat_hicp": {
        "label": "HICP, meziroční míra",
        "institution": "Eurostat",
        "dataset": "prc_hicp_manr",
        "catalog": None,
        "freq": "M",
        "lag_days": 17,
        "lag_basis": "kalendar",
        "note": "Slouží jako záloha ČSÚ. POZOR: řada UŽ JE meziroční míra, "
                "nepočítat z ní pct_change. Od ledna 2026 zaseklá, proto je "
                "primárním zdrojem HICP ČSÚ (CEN0101M).",
        "feeds": ["hicp_yoy", "cpi_yoy"],
        "fallback": None,
        "role": "zaloha",
        "backs_up": ["csu_hicp", "csu_cpi"],
    },
    "eurostat_unempl": {
        "label": "Obecná míra nezaměstnanosti, čtvrtletně",
        "institution": "Eurostat",
        "dataset": "une_rt_q",
        "catalog": None,
        "freq": "Q",
        "lag_days": 50,
        "lag_basis": "odhad",
        "note": "Výběrové šetření pracovních sil. Lhůtu ověřit proti kalendáři.",
        "feeds": ["unempl"],
        "fallback": "statická záloha",
    },

    # ── ČNB ────────────────────────────────────────────────────────────────
    "cnb_eurczk": {
        "label": "Kurz EUR/CZK, denní kurzy devizového trhu",
        "institution": "ČNB",
        "dataset": "kurzy devizového trhu (rok.txt)",
        "catalog": None,
        "freq": "D",
        "lag_days": 1,
        "lag_basis": "kalendar",
        "note": "Vyhlašuje se každý pracovní den po 14:30.",
        "feeds": ["eurczk"],
        "fallback": "statická záloha (končí 2025Q4)",
    },
    "cnb_pribor3m": {
        "label": "PRIBOR 3M, denní fixing",
        "institution": "ČNB",
        "dataset": "PRIBOR (TXT po rocích)",
        "catalog": None,
        "freq": "D",
        "lag_days": 1,
        "lag_basis": "kalendar",
        "note": None,
        "feeds": ["pribor3m"],
        "fallback": "statická záloha (končí 2026Q3)",
    },
    "cnb_pribor12m": {
        "label": "PRIBOR 12M, denní fixing",
        "institution": "ČNB",
        "dataset": "PRIBOR (TXT po rocích)",
        "catalog": None,
        "freq": "D",
        "lag_days": 1,
        "lag_basis": "kalendar",
        "note": None,
        "feeds": ["pribor12m"],
        "fallback": "PRIBOR 3M + spread 0,2 pp",
    },
    "cnb_repo": {
        "label": "Dvoutýdenní repo sazba",
        "institution": "ČNB",
        "dataset": "seznam změn sazeb",
        "catalog": None,
        "freq": "D",
        "lag_days": 1,
        "lag_basis": "kalendar",
        "note": "Mění se jen na měnověpolitickém jednání, 8× ročně. Řada je "
                "'aktuální' i když se dlouho nehýbe, proto se u ní hlídá "
                "dostupnost seznamu, ne stáří poslední změny.",
        "feeds": ["repo_rate"],
        "fallback": "statická záloha",
        "event_driven": True,
    },
    "cnb_mp_meetings": {
        "label": "Kalendář měnověpolitických jednání bankovní rady",
        "institution": "ČNB",
        "dataset": "o_cnb/bankovni-rada",
        "catalog": None,
        "freq": "Y",
        "lag_days": 0,
        "lag_basis": "kalendar",
        "note": "Potřeba pro repo_known_through(). Kalendář na další rok "
                "doplnit do zálohy _MEETINGS_FALLBACK včas.",
        "feeds": ["repo_rate"],
        "fallback": "_MEETINGS_FALLBACK + data/raw/cnb_mp_meetings.json",
        "event_driven": True,
    },

    # ── ECB ────────────────────────────────────────────────────────────────
    "ecb_eurusd": {
        "label": "Kurz EUR/USD, denní referenční kurz",
        "institution": "ECB",
        "dataset": "EXR/D.USD.EUR.SP00.A",
        "catalog": None,
        "freq": "D",
        "lag_days": 1,
        "lag_basis": "kalendar",
        "note": None,
        "feeds": ["eurusd"],
        "fallback": "statická záloha",
    },
}

_FREQ_LABEL = {"D": "denní", "W": "týdenní", "M": "měsíční",
               "Q": "čtvrtletní", "Y": "roční"}

# Tolerance navíc, než řadu označíme za pozadu. Kryje svátky a drobné posuny
# termínů; delší zpoždění už je signál, ne šum.
GRACE_DAYS = 7


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Výpočet očekávaného období
# ─────────────────────────────────────────────────────────────────────────────

def _period_start(ts: pd.Timestamp, freq: str) -> pd.Timestamp:
    """Začátek období dané frekvence, do kterého datum spadá."""
    ts = pd.Timestamp(ts).normalize()
    if freq == "Q":
        return pd.Timestamp(ts.year, 3 * ((ts.month - 1) // 3) + 1, 1)
    if freq == "M":
        return pd.Timestamp(ts.year, ts.month, 1)
    if freq == "W":
        return ts - pd.Timedelta(days=int(ts.dayofweek))
    if freq == "Y":
        return pd.Timestamp(ts.year, 1, 1)
    return ts


def _period_end(start: pd.Timestamp, freq: str) -> pd.Timestamp:
    """Poslední den období, které začíná v `start`."""
    start = pd.Timestamp(start).normalize()
    if freq == "Q":
        # POZOR: pd.offsets.QuarterBegin má výchozí začátky Bře/Čer/Zář/Pro,
        # takže přímá aritmetika lže. Posouvat o tři měsíce.
        return start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
    if freq == "M":
        return start + pd.DateOffset(months=1) - pd.Timedelta(days=1)
    if freq == "W":
        return start + pd.Timedelta(days=6)
    if freq == "Y":
        return start + pd.DateOffset(years=1) - pd.Timedelta(days=1)
    return start


def _step_back(start: pd.Timestamp, freq: str) -> pd.Timestamp:
    """O jedno období zpět."""
    if freq == "Q":
        return start - pd.DateOffset(months=3)
    if freq == "M":
        return start - pd.DateOffset(months=1)
    if freq == "W":
        return start - pd.Timedelta(days=7)
    if freq == "Y":
        return start - pd.DateOffset(years=1)
    return start - pd.Timedelta(days=1)


def expected_latest(key: str, today: pd.Timestamp | None = None) -> pd.Timestamp:
    """Nejnovější období, které už MÁ být podle lhůty zveřejněné.

    Hledá nejmladší období, jehož konec plus obvyklé zpoždění plus tolerance
    už uplynul. U denních řad vrací prostě dnešek zmenšený o lhůtu.
    """
    meta = SOURCES[key]
    freq = meta["freq"]
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    deadline = pd.Timedelta(days=int(meta["lag_days"]) + GRACE_DAYS)

    if freq == "D":
        return today - deadline

    cur = _period_start(today, freq)
    for _ in range(40):
        if _period_end(cur, freq) + deadline <= today:
            return cur
        cur = _step_back(cur, freq)
    return cur


def periods_behind(key: str, observed: pd.Timestamp,
                   today: pd.Timestamp | None = None) -> int:
    """O kolik období je `observed` pozadu za očekáváním (0 = v pořádku)."""
    meta = SOURCES[key]
    freq = meta["freq"]
    exp = expected_latest(key, today)
    obs = pd.Timestamp(observed).normalize()

    if freq == "D":
        return max(0, (exp - obs).days)

    obs_p = _period_start(obs, freq)
    n = 0
    cur = exp
    while cur > obs_p and n < 40:
        cur = _step_back(cur, freq)
        n += 1
    return n


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Sběr stavu za běh
# ─────────────────────────────────────────────────────────────────────────────

_OBSERVED: dict[str, dict] = {}


def record(key: str, last_obs, live: bool = True, detail: str | None = None) -> None:
    """Zapíše, co fetcher skutečně přinesl.

    `key`      klíč do SOURCES
    `last_obs` poslední pozorování v získané řadě (None = nic nepřišlo)
    `live`     True = ze živého zdroje, False = ze zálohy
    """
    if key not in SOURCES:
        raise KeyError(f"Neznámá řada '{key}'. Zapiš ji do SOURCES v data_registry.py.")
    _OBSERVED[key] = {
        "last_obs": None if last_obs is None else pd.Timestamp(last_obs).normalize(),
        "live": bool(live),
        "detail": detail,
    }


def status(today: pd.Timestamp | None = None) -> list[dict]:
    """Stav všech řad registru. Řadí problémy nahoru."""
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    rows = []
    for key, meta in SOURCES.items():
        obs = _OBSERVED.get(key)
        exp = expected_latest(key, today)
        row = {
            "key": key,
            "label": meta["label"],
            "institution": meta["institution"],
            "dataset": meta["dataset"],
            "freq": meta["freq"],
            "lag_days": meta["lag_days"],
            "lag_basis": meta["lag_basis"],
            "feeds": meta["feeds"],
            "expected": exp,
            "observed": None,
            "live": None,
            "behind": None,
            "state": "neznámý",
        }
        if obs is None:
            # Řada se v tomhle běhu vůbec nevolala.
            rows.append(row)
            continue

        row["observed"] = obs["last_obs"]
        row["live"] = obs["live"]
        if obs["last_obs"] is None:
            row["state"] = "chybí"
        elif meta.get("event_driven"):
            # Sazby a kalendáře se nehýbou pravidelně; stáří poslední změny
            # nic neznamená. Hlídá se jen to, jestli zdroj vůbec odpověděl.
            row["behind"] = 0
            row["state"] = "živý" if obs["live"] else "záloha"
        else:
            row["behind"] = periods_behind(key, obs["last_obs"], today)
            if not obs["live"]:
                row["state"] = "záloha"
            elif row["behind"] > 0:
                row["state"] = "pozadu"
            else:
                row["state"] = "aktuální"
        rows.append(row)

    # Zastaralá záloha se zneškodní, dokud její primární zdroj šlape.
    by_key = {r["key"]: r for r in rows}
    for r in rows:
        meta = SOURCES[r["key"]]
        if meta.get("role") != "zaloha":
            continue
        if r["state"] not in ("pozadu", "chybí", "neznámý"):
            continue
        covered = [by_key.get(k) for k in meta.get("backs_up", [])]
        covered = [c for c in covered if c]
        if not (covered and all(c["state"] in ("aktuální", "živý") for c in covered)):
            continue
        # Záloha, kterou primární zdroj nepotřeboval, je dobrá zpráva.
        r["state"] = "nevyužita" if r["state"] == "neznámý" else "záloha zastaralá"
        r["covered_by"] = [c["key"] for c in covered]

    order = {"chybí": 0, "pozadu": 1, "záloha": 2, "neznámý": 3,
             "záloha zastaralá": 4, "nevyužita": 5, "živý": 6, "aktuální": 7}
    rows.sort(key=lambda r: (order.get(r["state"], 9), r["key"]))
    return rows


def problems(today: pd.Timestamp | None = None) -> list[dict]:
    """Jen řady, které si zaslouží pozornost."""
    return [r for r in status(today)
            if r["state"] in ("chybí", "pozadu", "záloha")]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Výstup
# ─────────────────────────────────────────────────────────────────────────────

_MARK = {"aktuální": "ok", "živý": "ok", "záloha": "ZÁLOHA",
         "pozadu": "POZADU", "chybí": "CHYBÍ", "neznámý": "-",
         "záloha zastaralá": "záloha stará", "nevyužita": "nevyužita"}


def _fmt(ts, freq) -> str:
    if ts is None:
        return "-"
    ts = pd.Timestamp(ts)
    if freq == "Q":
        return f"{ts.year}Q{(ts.month - 1) // 3 + 1}"
    if freq == "M":
        return ts.strftime("%Y-%m")
    if freq == "Y":
        return str(ts.year)
    return ts.strftime("%Y-%m-%d")


def report(today: pd.Timestamp | None = None, verbose: bool = True) -> int:
    """Vytiskne tabulku stavu. Vrací počet řad, které jsou v problému."""
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    rows = status(today)
    bad = [r for r in rows if r["state"] in ("chybí", "pozadu")]
    fb = [r for r in rows if r["state"] == "záloha"]

    if verbose:
        print()
        print(f"REGISTR VSTUPNÍCH ŘAD, stav k {today.date()}")
        print("─" * 100)
        print(f"{'ŘADA':<44}{'ZDROJ':<10}{'FREKV':<12}"
              f"{'POSLEDNÍ':<13}{'OČEKÁVÁNO':<13}STAV")
        print("─" * 100)
        for r in rows:
            note = ""
            if r["state"] == "pozadu":
                note = f"  (o {r['behind']}× {_FREQ_LABEL[r['freq']][:4]}.)"
            elif r["state"] == "záloha":
                note = "  (živý zdroj nedostupný)"
            elif r["state"] == "záloha zastaralá":
                note = "  (kryto primárním zdrojem, neblokuje)"
            elif r["state"] == "nevyužita":
                note = "  (primární zdroj stačil)"
            print(f"{r['label'][:43]:<44}{r['institution']:<10}"
                  f"{_FREQ_LABEL[r['freq']]:<12}"
                  f"{_fmt(r['observed'], r['freq']):<13}"
                  f"{('-' if SOURCES[r['key']].get('event_driven') else _fmt(r['expected'], r['freq'])):<13}"
                  f"{_MARK[r['state']]}{note}")
        print("─" * 100)

        if bad:
            print(f"\n  {len(bad)} řad pozadu nebo chybí:")
            for r in bad:
                print(f"    - {r['label']} ({r['institution']} {r['dataset']}), "
                      f"vstupuje do: {', '.join(r['feeds'])}")
                if SOURCES[r['key']].get("catalog"):
                    print(f"      ověř katalog: {SOURCES[r['key']]['catalog']}")
        if fb:
            print(f"\n  {len(fb)} řad ze zálohy (běh prošel, ale zdroj mlčí):")
            for r in fb:
                print(f"    - {r['label']}: {SOURCES[r['key']]['fallback']}")
        if not bad and not fb:
            print("\n  Všechny volané řady jsou aktuální a ze živého zdroje.")
        print()

    return len(bad)


def save_status(today: pd.Timestamp | None = None) -> str:
    """Uloží stav do JSON, aby ho mohl převzít export na web."""
    os.makedirs(os.path.dirname(_STATUS_PATH), exist_ok=True)
    payload = {
        "checked": str(pd.Timestamp(today or pd.Timestamp.today()).date()),
        "grace_days": GRACE_DAYS,
        "sources": [
            {**r,
             "expected": _fmt(r["expected"], r["freq"]),
             "observed": _fmt(r["observed"], r["freq"])}
            for r in status(today)
        ],
    }
    with open(_STATUS_PATH, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    return _STATUS_PATH


def load_status() -> dict:
    """Načte poslední uložený stav, nebo {}."""
    if os.path.exists(_STATUS_PATH):
        try:
            return json.load(open(_STATUS_PATH))
        except Exception:
            return {}
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Samostatný běh
# ─────────────────────────────────────────────────────────────────────────────

def _probe_all() -> None:
    """Zavolá všechny fetchery a zaznamená, co vrátily.

    Tohle je pomalé (stahuje vše), ale je to jediný způsob, jak zkontrolovat
    registr bez plného pipeline.
    """
    import data_fetch as dfh
    import financial_data as fin

    def _try(key, fn, live_check=None):
        """Zavolá fetcher a zapíše výsledek, pokud se fetcher nehlásí sám.

        Fetchery, které volají record() uvnitř, hlásí poslední ŽIVÉ pozorování
        ve zdrojové frekvenci (denní kurzy, denní fixingy). Návratová hodnota
        je u nich už agregovaná na čtvrtletí a spojená se zálohou, takže by
        jejich hlášení přepsala nesmyslem. Proto se tady zapisuje jen to, co
        se nezapsalo samo.
        """
        had = key in _OBSERVED
        try:
            s = fn()
        except Exception as e:
            if not had or key not in _OBSERVED:
                record(key, None, live=False, detail=str(e)[:120])
            return
        if key in _OBSERVED and not had:
            return                      # fetcher se ohlásil sám, nepřepisovat
        if s is None or len(s) == 0:
            record(key, None, live=False)
            return
        live = True if live_check is None else live_check(s)
        record(key, s.index[-1], live=live)

    _try("csu_cpi",         lambda: dfh.fetch_csu_cpi_monthly())
    _try("csu_hicp",        lambda: dfh.fetch_csu_hicp())
    _try("csu_fuel",        lambda: dfh.fetch_csu_fuel())
    _try("csu_wages",       lambda: dfh.fetch_csu_wages())
    _try("eurostat_gdp",    lambda: dfh.fetch_eurostat(
        "namq_10_gdp", "Q.GR.SCA.B1GQ.CZ", "gdp_qoq"))
    _try("eurostat_hicp",   lambda: dfh.fetch_eurostat(
        "prc_hicp_manr", "M.RCH_A.CP00.CZ", "hicp"))
    _try("eurostat_unempl", lambda: fin.fetch_unemployment())
    _try("cnb_eurczk",      lambda: fin.fetch_eurczk())
    _try("cnb_pribor3m",    lambda: fin.fetch_pribor())
    _try("cnb_pribor12m",   lambda: fin.fetch_pribor(tenor="1Y"))
    _try("cnb_repo",        lambda: fin.fetch_repo_rate())
    _try("ecb_eurusd",      lambda: fin.fetch_eurusd())

    try:
        m = fin.fetch_cnb_mp_meetings()
        record("cnb_mp_meetings", max(m) if m else None, live=bool(m))
    except Exception as e:
        record("cnb_mp_meetings", None, live=False, detail=str(e)[:120])


def main() -> int:
    p = argparse.ArgumentParser(
        description="Registr vstupních řad a kontrola tichého zastarání")
    p.add_argument("--json", action="store_true",
                   help="Strojově čitelný výstup místo tabulky")
    p.add_argument("--today", type=str, default=None,
                   help="Simulovat jiné datum (YYYY-MM-DD), pro testy")
    p.add_argument("--no-probe", action="store_true",
                   help="Nestahovat; použít poslední uložený stav")
    args = p.parse_args()

    today = pd.Timestamp(args.today) if args.today else None

    if args.no_probe:
        print(json.dumps(load_status(), ensure_ascii=False, indent=1))
        return 0

    import logging
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(message)s")
    _probe_all()

    if args.json:
        print(json.dumps(
            {**load_status(), "sources": [
                {**r,
                 "expected": _fmt(r["expected"], r["freq"]),
                 "observed": _fmt(r["observed"], r["freq"])}
                for r in status(today)]},
            ensure_ascii=False, indent=1))
        n_bad = len([r for r in status(today)
                     if r["state"] in ("chybí", "pozadu")])
    else:
        n_bad = report(today)

    save_status(today)
    return 1 if n_bad else 0


if __name__ == "__main__":
    # POZOR: spuštěný skript se jmenuje __main__, takže `import data_registry`
    # uvnitř fetcherů by načetl DRUHOU kopii modulu a record() by zapisoval do
    # jiného _OBSERVED, než ze kterého se tiskne. Proto se běh deleguje na
    # importovanou instanci, aby stav existoval jen jednou.
    import data_registry as _single
    sys.exit(_single.main())
