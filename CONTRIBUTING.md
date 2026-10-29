# Jak přispívat do NOSTRADAMUPu

Díky za zájem. Tenhle dokument je hlavně o tom, **podle čeho se tady pozná dobrá
změna od špatné**, protože u prognostického modelu to není samozřejmé: skoro
každý zásah něco zlepší a něco jiného rozbije, a bez měřítka se to nepozná.

Cíl kvality projektu je srovnatelnost s ČNB, MF ČR a soukromými prognostickými
domy. Tomu odpovídá laťka.

---

## 1. Rozjezd

```bash
git clone https://github.com/dominikstroukal/nostradamup.git
cd nostradamup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Data v repu nejsou** (`data/raw/` je v `.gitignore`), takže první běh musí
začít stažením. Pořadí není libovolné, každý krok čte výstup předchozího:

```bash
python data_fetch.py                                          # ČSÚ, Eurostat → makro
python financial_data.py --repo-neutral 3.5 --output-dir outputs/charts
python report_generator.py --output-dir outputs/charts        # jádro modelu
python export_web.py                                          # → web/data/latest.json
```

Web lokálně:

```bash
cd web && python -m http.server 8000
```

Celý pipeline trvá řádově minuty, většinu z toho stahování dat.

---

## 2. Dvě brány před commitem

### Brána A: pipeline proběhne celý bez chyby

Všechny čtyři kroky výše. Nejen ten, který jsi měnil. Dvakrát se stalo, že
novější skript volal funkci, kterou starší soubor neměl, a **tiše spadl**.

### Brána B: backtest se nezhorší

```bash
python backtest.py                 # všechny proměnné, 12 oken, horizont 8Q
python backtest.py --var cpi_yoy   # jen jedna proměnná, rychlejší
```

Kritérium je **Theil U**: poměr chyby modelu k chybě naivní predikce
(random walk). `U < 1` znamená lepší než náhodná procházka, `U > 1` horší.

Stav k měření z 29. 10. 2026, 12 oken, horizont 4Q:

| Proměnná | Theil U | | Proměnná | Theil U |
|---|---|---|---|---|
| CPI | 0,75 | | HDP QoQ | 0,87 |
| PRIBOR | 0,77 | | Repo | 0,87 |
| HICP | 0,82 | | Nezaměstnanost | 0,92 |
| HDP YoY | 0,86 | | Mzdy | 0,99 |
| | | | EUR/CZK | 1,09 |
| | | | EUR/USD | 1,20 |

**8 z 10 poráží random walk.** Kurzy prohrávají očekávaně, to je
Meese-Rogoffův výsledek z roku 1983 a není to vada tohohle modelu.

> **Tabulku po zásahu do modelu přeměř, nedopisuj.** A když ti Theil U po změně
> vyjde hůř, než je v tabulce, **nejdřív změř stejným backtestem i stav před
> změnou**. Jednou se tu ladil model kvůli „regresi", která byla jen tím, že
> zapsaná čísla byla zastaralá a systematicky optimistická.
>
> Čerstvý příklad z 29. 10. 2026: repo vyskočilo z 0,79 na 0,87 a vypadalo to
> jako regrese po opravě kurzu. Měření stejným backtestem na kódu před opravou
> dalo taky 0,87. Celý rozdíl způsobilo nové čtvrtletí dat, do jehož oken spadlo
> červnové zvýšení sazby. Bez té kontroly by se ladilo něco, co nebylo rozbité.

CI backtest **nepouští**. Je to na tobě.

---

## 3. Čtyři zásady, na kterých tenhle projekt stojí

Tohle není sloh, je to seznam věcí, které se tu už pokazily.

### 3.1 Baseline kanálu vždy z fixní reference, nikdy z průměru prognózní cesty

Tahle chyba se v projektu objevila **třikrát** (kanály B, C, D). Když se mezera
měří vůči průměru vlastní prognózní cesty, kanál se vynuluje sám vůči sobě a
trvale vysoké mzdy pak nevytvoří žádný cost-push.

Správně jsou fixní reference: `wage_norm=4.0`, `potential_gdp_qoq=0.55`, NAIRU
z historie.

### 3.2 Kalibrovat backtestem, ne odhadem

Parametry se hledají gridem přes `backtest.py`, ne citem. Takhle se zjistilo, že
`policy_sensitivity` 0,30 bylo přestřelené (U=1,38) a správná hodnota je 0,10
(U=0,66). Pokud měníš parametr, do PR patří čím.

### 3.3 Backtest musí testovat produkční modely

Jednou testoval zastaralé verze funkcí a tvářil se přitom výborně. Když měníš
model, zkontroluj `_forecast_at()` v `backtest.py`, jestli volá totéž co
produkce, **včetně defaultních hodnot argumentů**.

### 3.4 Poctivost nad hezká čísla

Když backtest ukáže, že tvůj kanál nezlepšuje predikci, napiš to do PR a
zdůvodni, proč ho přesto nechat. Legitimní důvod existuje: IS křivka v tomhle
modelu bodovou přesnost HDP nezlepšuje (IS puzzle), ale je nutná pro koherentní
transmisi a scénáře. Co je nepřijatelné, je zamést to pod koberec.

---

## 4. Pasti, do kterých tu už někdo spadl

**Zdroje mizí a fallback se pak tváří jako úspěch.** Tohle je nejzákeřnější
třída chyb v celém repu, protože nic nespadne a čísla vypadají normálně.

- ČSÚ vyřadil sadu `WCEN01M` i s jejím živým výběrem a inflace tiše zamrzla na
  prosinci 2025. Nástupci jsou sady „od 2026": CPI `CEN0101E`, HICP `CEN0101M`.
- `Series.update()` zapisuje jen do indexů, které v cílové sérii už existují.
  Když je záloha kratší než živá data, novější čtvrtletí **tiše zmizí**.
  Používej `combine_first()`. Kvůli tomuhle bylo EUR/CZK rok zamrzlé.

Když je nějaká řada podezřele stará, ověř `api/katalog/v1/sady/<KÓD>`
(404 = vyřazeno). A **v logu vždycky rozliš živý zdroj od zálohy.**

**Další:**

| Past | Správně |
|---|---|
| `dropna()` ořízne dataset na nejkratší řadu | každou sérii protáhnout `_extend_to_present()` **před** spojením do DataFrame |
| `ax.get_ylim()` před vykreslením dat | volat až po vykreslení, jinak se rozbije umístění popisků |
| PRIBOR sigma z celého vzorku | posledních 8 čtvrtletí (cyklus 2021–24 ji nafoukne) |
| `os.path.realpath(__file__)` | `abspath` (realpath rozbalí symlink Dropboxu) |
| `pd.offsets.QuarterBegin` | `pd.DateOffset(months=3)`; QuarterBegin má výchozí začátky Bře/Čer/Zář/Pro, takže z 1. 4. udělá 1. 6. |
| Zaokrouhlení repa na 1 des. místo | sazby jsou **diskrétní** (mřížka 25 bp); zaokrouhlení patří až frontendu, kvantily přes `method="nearest"` |
| Čtvrtletní **průměr** administrativní sazby | repo se agreguje jako sazba na **konci** čtvrtletí, jak ho vykazuje ČNB |
| Eurostat `prc_hicp_manr` + `pct_change` | ta řada **už je** meziroční míra |

---

## 5. Do čeho nesahat

**`web/index.html` se při změnách modelu nemění.** Frontend je záměrně hloupý:
`web/data/latest.json` je sebe-popisný, nese i metadata (názvy, jednotky,
skupiny). Přidání proměnné na web = **jeden řádek do slovníku `META`** v
`export_web.py`. Pokud se ti zdá, že musíš do HTML, nejspíš chybí metadata.

Karty patří do sekcí definovaných polem `SECTIONS`: `prehled`, `nyni`,
`prognoza`, `rozbor`, `model`. Nová karta = `add('<id sekce>', card)`, nadpis
karty je `<h3>`, protože `<h2>` patří sekci.

---

## 6. Konvence

- **Komunikace, komentáře i dokumentace česky.**
- **Žádné em-dash pomlčky ve větách**, jen v nadpisech a rozsazích.
- Grafy drží identitu MUP: navy `#00205B`, modrá `#003DA5`, pozadí `#EEF2FA`.
- Commit messages česky, v imperativu, vystihující **proč**, ne jen co.
- Ke každému novému ekonomickému kanálu patří odkaz na literaturu, jak to má
  `ar_forecast()` u stávajících (Galí 2015, Benigno-Eggertsson 2023 a dál).

---

## 7. Jak navrhnout změnu

1. Založ branch, ne commit do `main`.
2. Proběhni obě brány z kapitoly 2.
3. V popisu PR uveď:
   - **co** se mění a **proč**,
   - **Theil U před a po**, měřeno stejným backtestem,
   - u parametru doklad kalibrace, ne odhad,
   - pokud se něco zhoršilo, přiznej to a zdůvodni.

PR, který mění model a nemá v popisu backtest, se bude vracet. Není to buzerace,
je to jediná obrana proti tomu, aby se model postupně rozladil hezkými nápady.

### Nesouhlasíš s kalibrací?

To je vítaný druh příspěvku. Parametry jako `anchoring`, `core_persistence`,
`housing_services_pressure` nebo `erpt_coef` jsou názory, ne fakta, a dají se
nastavit jinak (viz signatura `ar_forecast()` v `report_generator.py`).
Backtest rozhodne. Workflow jde pustit i ručně s vlastními parametry:
**Actions → Aktualizace prognózy → Run workflow**.

---

## 8. Licence

Projekt má dvojí licenci a je dobré vědět, pod kterou tvůj příspěvek spadne:

| Co | Licence |
|---|---|
| Kód (`*.py`, `web/index.html`, workflow) | [MIT](LICENSE) |
| Výstupy a dokumentace (prognózy, `web/data/*.json`, `docs_model.html`, grafy a texty) | [CC BY 4.0](LICENSE-CONTENT.md) |

Odesláním PR souhlasíš s tím, že se tvůj příspěvek zveřejní pod odpovídající
licencí. Vstupní data patří ČSÚ, Eurostatu, ČNB a MF ČR a řídí se jejich
podmínkami, ne touhle licencí.

Model je výzkumný a vzdělávací projekt, **ne investiční doporučení**, a je
publikovaný bez záruky. Na to pamatuj u všeho, co se zobrazí na webu.

---

## 9. Kontakt

Dominik Stroukal, Metropolitní univerzita Praha.
Pro věcnou diskuzi nad modelem prosím issue, ať je to dohledatelné.
