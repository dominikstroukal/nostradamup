"""
external_forecasts.py
=====================
Automatické stažení srovnávací prognózy pro roční tabulku.

Zdroj: MF ČR "Makroekonomická predikce" — XLSX "Tabulky a grafy" (list
Shrnutí_Summary), publikováno 4x ročně (leden/duben/srpen/listopad).

ČNB a ČBA publikují prognózu jen v PDF (žádné strojově čitelné API), proto
NEJSOU zahrnuty — scrapování PDF by bylo nespolehlivé.

VŠE JE GRACEFUL: při jakékoli chybě (nedostupné, změněný layout, chyba parsování)
vrátí None a srovnání se prostě nezobrazí. Raději nic než stará/špatná čísla.
Každá vintage je označena datem vydání MF.

Srovnatelné veličiny (MF summary vs náš model): reálný HDP, průměrná CPI inflace,
míra nezaměstnanosti (LFS), kurz EUR/CZK. Mzdy (MF má jen objem mezd, ne průměr)
a sazby (MF má jen dlouhodobý výnos) se nemapují — nejsou srovnatelné 1:1.
"""

import re
import io
import logging
import requests

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

try:
    from python_calamine import CalamineWorkbook
except ImportError:
    CalamineWorkbook = None

BASE = "https://mf.gov.cz"
LISTING = BASE + "/cs/rozpoctova-politika/makroekonomika/makroekonomicka-predikce"
UA = {"User-Agent": "Mozilla/5.0 (NOSTRADAMUP forecast comparison)"}

MONTHS = {"leden": 1, "unor": 2, "únor": 2, "brezen": 3, "březen": 3, "duben": 4,
          "kveten": 5, "květen": 5, "cerven": 6, "červen": 6, "cervenec": 7,
          "červenec": 7, "srpen": 8, "zari": 9, "září": 9, "rijen": 10, "říjen": 10,
          "listopad": 11, "prosinec": 12}

# naše proměnná -> (klíčová slova v EN/CZ názvu, vyluč slova, klíč jednotky)
MF_VARS = {
    "gdp_yoy": (["gross domestic product"], ["nominal"], "real"),
    "cpi_yoy": (["average inflation"], [], None),
    "unempl":  (["unemployment rate"], [], None),
    "eurczk":  (["exchange rate"], [], None),
}


def _latest_release():
    """(URL nejnovější predikce, 'měsíc rok') z listingu MF; vybírá podle
    roku+měsíce, ne podle pořadí na stránce."""
    h = requests.get(LISTING, timeout=30, headers=UA).text
    links = set(re.findall(
        r'(/cs/[^"\']*makroekonomicka-predikce-([a-zžřščěíáéúůňťď]+)-(\d{4})-\d+)', h, re.I))
    if not links:
        return None, None
    best = max(links, key=lambda l: (int(l[2]), MONTHS.get(l[1].lower(), 0)))
    return BASE + best[0], f"{best[1]} {best[2]}"


def _xlsx_url(release_url):
    h = requests.get(release_url, timeout=30, headers=UA).text
    m = re.findall(r'(/assets/attachments/[^"\']*Tabulky-a-grafy[^"\']*\.xlsx)', h)
    return BASE + m[0] if m else None


def _parse_summary(xlsx_bytes):
    """Vytáhne {var: {rok: hodnota}} z listu Shrnutí_Summary (robustně:
    roky z hlavičky, řádky podle názvu; aktuální predikce = první výskyt roku)."""
    wb = CalamineWorkbook.from_filelike(io.BytesIO(xlsx_bytes))
    rows = wb.get_sheet_by_name("Shrnutí_Summary").to_python()

    def years_in(r):
        out = []
        for j, c in enumerate(r):
            try:
                y = int(float(c))
                if 2015 <= y <= 2035:
                    out.append((j, y))
            except (ValueError, TypeError):
                pass
        return out

    hdr = max(rows, key=lambda r: len(years_in(r)))
    year_col = {}
    for j, y in years_in(hdr):
        year_col.setdefault(y, j)   # první výskyt = aktuální predikce

    result = {}
    for var, (kw, excl, unit) in MF_VARS.items():
        for r in rows:
            txt = " ".join(str(c).lower() for c in r[:4])
            if any(k in txt for k in kw) and not any(e in txt for e in excl):
                if unit and unit not in txt:
                    continue
                vals = {}
                for y, j in year_col.items():
                    try:
                        vals[str(y)] = round(float(r[j]), 2)
                    except (ValueError, TypeError):
                        pass
                result[var] = vals
                break
    return result


def fetch_mf_forecast():
    """Vrátí {source, label, date, values:{var:{rok:val}}} nebo None (graceful)."""
    if CalamineWorkbook is None:
        log.info("python-calamine není nainstalován – srovnání MF přeskočeno.")
        return None
    try:
        rel, label = _latest_release()
        if not rel:
            return None
        xurl = _xlsx_url(rel)
        if not xurl:
            return None
        data = requests.get(xurl, timeout=60, headers=UA).content
        vals = _parse_summary(data)
        if not vals.get("gdp_yoy"):   # sanity: aspoň HDP musí být
            return None
        mdate = re.search(r'(\d{4}-\d{2}-\d{2})', xurl)
        return {"source": "MF ČR", "label": f"MF ČR – predikce {label}",
                "date": mdate.group(1) if mdate else None, "values": vals}
    except Exception as e:
        log.info("Srovnání MF nedostupné (%s) – přeskočeno.", e)
        return None


if __name__ == "__main__":
    import json
    print(json.dumps(fetch_mf_forecast(), ensure_ascii=False, indent=2))
