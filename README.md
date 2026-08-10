# NOSTRADAMUP

Čtvrtletní makroekonomický prognostický model pro ČR — Metropolitní univerzita Praha.
Semi-strukturální novokeynesiánský model (IS křivka + Phillipsova křivka + Taylorovo pravidlo).

**Web:** https://dominikstroukal.github.io/nostradamup/

## Lokální spuštění

```bash
pip install -r requirements.txt
python data_fetch.py
python financial_data.py --repo-neutral 3.5 --output-dir outputs/charts
python report_generator.py --output-dir outputs/charts
python export_web.py            # -> web/data/latest.json
```

Web lokálně: `cd web && python -m http.server 8000` → http://localhost:8000

## Jak to funguje

`export_web.py` vytvoří **sebe-popisný JSON**: obsahuje nejen čísla, ale i
metadata (názvy proměnných, jednotky, skupiny, parametry běhu). Frontend
(`web/index.html`) nic nezná natvrdo — vykreslí, co v JSON najde.

**Důsledek:** když do modelu přidáš proměnnou nebo kanál, stačí ji zaregistrovat
ve slovníku `META` v `export_web.py` a na webu se objeví sama. Do HTML se nesahá.

## Automatizace

`.github/workflows/update.yml` spouští model každé pondělí a po každém pushi do
`main`. Ruční běh s vlastními parametry: záložka **Actions → Aktualizace
prognózy → Run workflow**. Workflow ověří rozumnost výstupu, commitne JSON
a publikuje web na GitHub Pages.

## Soubory

| Soubor | Účel |
|---|---|
| `data_fetch.py` | Eurostat, ČNB — makro data |
| `financial_data.py` | kurzy, sazby, nezaměstnanost + prognózy |
| `report_generator.py` | jádro modelu (`ar_forecast`), report, dekompozice |
| `scenarios.py` | scénářová analýza |
| `cnb_survey.py` | tabulka indikátorů pro dotazník ČNB |
| `backtest.py` | out-of-sample validace (Theil U) |
| `sketch_report.py` | grafy v ručně kresleném stylu |
| `export_web.py` | export do JSON pro web |
| `web/` | statický frontend |
