# Experimental Anchor Data — Fe-Cr-C / Fe-Cr-Ni-C (extracted 2026-09-30)

All numbers below were read directly from the fetched PDFs (no estimates). Page numbers are
the printed journal/conference pages, with the PDF page in parentheses. Machine-readable rows:
`anchors.csv` (63 data rows, 17 columns).

Fetch status this session: **all 3 target documents fetched successfully** (local copies in
`C:\Users\Lion\AppData\Local\Temp\opencode\anchors\`). Fallbacks therefore not required; their
availability was spot-checked (see §4).

---

## 1. Drozdová, Smetana et al. 2017 (METAL 2017, Brno) — Fe-Cr-C invariant temperatures

**Citation.** DROZDOVÁ Ľ., SMETANA B., ZLÁ S., KALUP A., KAWULOKOVÁ M., ROSYPALOVÁ S.,
ŘEHÁČKOVÁ L., VONTOROVÁ J., STROUHALOVÁ M. "Temperatures of Liquidus, Solidus and Peritectic
Transformation of Fe-C-Cr Based Alloys." *METAL 2017 — 26th Int. Conf. on Metallurgy and
Materials*, Brno, Czech Republic, 24–26 May 2017, pp. 65–70. VSB-Technical University of
Ostrava / TANGER.

**Accessibility.** Open PDF, fetched OK this session:
`https://www.confer.cz/metal/2017/download/1347-phase-diagram-of-the-fe-c-cr-ternary-system.pdf`
(406 128 bytes, 6 pp. = printed pp. 65–70; text layer clean, tables extractable — no
digitization needed). Note: the confer.cz landing page did not respond to a later curl check,
but the direct PDF link worked.

**Alloys (Table 1, p. 66 = PDF p. 2) — 3 alloys, wt%:**

| Alloy | C | Cr | Ni | other |
|---|---|---|---|---|
| A | 0.308 | 1.058 | 0 | Fe balance; Mn, Si, N, … **not reported** |
| B | 0.320 | 1.540 | 0 | idem |
| C | 0.380 | 4.990 | 0 | idem |

**Measured/calculated transformation temperatures (Table 2, p. 67 = PDF p. 3), °C:**

| Alloy | Qty | DTA | TA (heating) | TA (cooling) | Thermo-Calc* | IDS** |
|---|---|---|---|---|---|---|
| A | T_S | 1447 | 1449 | 1451 | 1449 | 1446 |
| A | T_P | 1486 | 1484 | 1458 | 1486 | 1482 |
| A | T_L | 1498 | 1503 | 1499 | 1503 | 1502 |
| B | T_S | 1445 | 1447 | 1437 | 1451 | 1438 |
| B | T_P | 1471 | 1473 | 1449 | 1458 | 1461 |
| B | T_L | 1498 | 1501 | 1495 | 1504 | 1500 |
| C | T_S | 1397 | 1405 | 1410 | 1395 | 1386 |
| C | T_P | 1438 | 1441 | 1416 | 1449 | 1432 |
| C | T_L | 1474 | 1480 | 1476 | 1480 | 1475 |

\*Thermo-Calc 2015b + TCFE8; elements Sn, As, Sb, Pb, Bi not included in calculation;
diamond/graphite phases excluded. \**IDS kinetic solidification package; Sn, B, As, Sb, Pb, Bi
excluded (p. 67, §3 and table footnotes).

**Uncertainty.** No per-point error bars; p. 69 (§4.3) reports the spread of standard deviations
by method: **DTA 0–2, TA heating 0–6, TA cooling 0–16** (°C). Use these as method-level noise
bands in `anchors.csv` (uncertainty column).

**Method (pp. 66–67, §2).**
- DTA: Setaram Setsys 18™ (S-type tri-couple DTA sensor), alumina crucible, **Ar 6N**, sample
  ≈190 mg, **heating 10 °C/min**, 3 measurements/alloy (heating only).
- "Direct" TA: Netzsch STA 449 F3 Jupiter (S-type), alumina crucible, Ar 6N, sample ≈23 g,
  **heating and cooling 5 °C/min**, 2 heating + 2 cooling runs/alloy.
- Temperature calibration vs Ni (4N5) or Pd (5N), corrected for heating rate and sample mass.
- T_P taken as the **start** of the peritectic transformation; T_S start-of-melting;
  T_L extrapolated onset.

**Comparability caveats (Fe-Cr-C anchor).**
- Clean system: **no Mo/Mn/Ni in model** — but the paper reports only C and Cr; the balance is
  assumed Fe and residual elements are simply *unreported* (titled "Fe-C-Cr based").
- Probes only the **low-Cr (1.06–4.99 wt%), Ni-free face** of the Fe-Cr-Ni(−C) domain.
- **Continuous heating/cooling (5–10 °C/min), no isothermal holds** → not strictly equilibrium;
  authors note solidus determination is sensitive to identifying melt onset, and TA-cooling T_P
  values are affected by undercooling/secondary γ nucleation — **the authors explicitly exclude
  TA-cooling T_P from their discussion** (p. 68, §4.2). Those three values are flagged in the CSV.
- Authors' own exp-vs-calc gaps: T_L small (≤6 °C for DTA vs TC), T_S up to 15 °C (alloy C),
  T_P up to 15 °C (alloy B: exp 1471–1473 vs TC 1458) — a realistic acceptance band.
- Largest experiment-method spread: alloy C, T_S 1397–1410 °C (DTA vs TA-cooling);
  tightest: alloy A, T_P 1484–1486 °C (p. 70, §5).

**Figure/table inventory.** Table 1 (compositions), Table 2 (all temperatures) — text tables,
*no digitization required*. Fig. 1 (sensor arrangement), Figs. 2–4 (DTA/heating/cooling curves
with the same transition temperatures marked) — redundant with Table 2; digitizable only if
curve shapes are wanted.

---

## 2. Yamada et al. 1987 (Tetsu-to-Hagané 73(14)) — Fe-Cr-Ni liquidus/solidus, P–E point

**Citation.** YAMADA A., UMEDA T., SUZUKI M., ARAGANE G., KIHARA H., KIMURA K.
"Determination of Liquidus and Solidus Surfaces at Iron-rich Region of Fe-Cr-Ni System"
(Fe-Cr-Ni系鉄高濃度領域における液相面及び固相面の測定), *Tetsu-to-Hagané* (鉄と鋼)
**73**(14): 1676–1683 (1987). DOI 10.2355/tetsutohagane1955.73.14_1676. Received 6 Nov 1986.

**Accessibility.** Open access on J-STAGE; fetched OK this session (after one TLS failure with
Invoke-WebRequest, `curl` succeeded):
`https://www.jstage.jst.go.jp/article/tetsutohagane1955/73/14/73_14_1676/_pdf`
(5 545 444 bytes, 8 pp., scanned PDF **with OCR text layer**). DOI resolves (HTTP 200).
Text is Japanese with an English synopsis (p. 1676).

**Key extracted numbers.**

1. **Peritectic–eutectic transition point = 15 wt% Cr – 10 wt% Ni.**
   - English synopsis p. 1676: "The peritectic-eutectic transition point was found to be at the
     composition **about 15 wt%Cr-10 wt%Ni**"; and "the peritectic-eutectic line … found to shift
     slightly toward the Fe-Cr side **between 10 and 15 wt%Cr**".
   - §3.4, p. 1681: "Fig. 9において遷移点は15 wt%Cr, 10 wt%Niであった" (in Fig. 9 the transition
     point was at 15 wt% Cr, 10 wt% Ni), agrees with Schürmann & Fredriksson.
   - Conclusion p. 1683 repeats: 遷移点は15 wt%Cr, 10 wt%Niの位置に存在する.
   - No uncertainty quoted. Determination criterion: the point where the liquidus–γ tie-line
     slope and the peritectic–eutectic line slope have the same direction (§3.4).

2. **DTA example (only liquidus/solidus temperatures quoted numerically in the text), Fig. 5 /
   §2.5, p. 1679:** alloy **21.0 wt% Cr – 16.6 wt% Ni** (verified from page image; OCR had lost
   the decimal): **T_L = 1441 °C** (measured on cooling), **T_S = 1424 °C** (on reheating).
   The authors state this solidus is much lower than expected from liquidus + tie-lines and
   "**必ずしも信頼性のある結果とは言い難い**" (cannot be called reliable); many solidus readings
   came out low (SiC furnace heat capacity, 7 °C/min, microsegregation). **They therefore built
   the solidus surface from liquidus + tie-lines instead of DTA solidus values** (§2.5, p. 1679).
   → Use T_L = 1441 °C as an anchor; **do not use T_S = 1424 °C**.

3. **Primary-phase identity points (unidirectional solidification, R = 1 mm/s, G ≈ 3 °C/mm):**
   - **24.0 wt% Cr – 18.8 wt% Ni → γ primary (single-phase γ solidification)** — §3.2, p. 1679,
     Photo. 2a.
   - **19.6 wt% Cr – 9.14 wt% Ni → δ primary, then γ by peritectic reaction** — §3.2, p. 1680,
     Photo. 2b.
   These two bracket the peritectic–eutectic line in the 19–24 Cr / 9–19 Ni region.

4. **Specimen basis (§2.1, p. 1677, read from page image):** raw materials electrolytic Fe,
   electrolytic Ni, high-purity low-carbon ferrochromium (64.5 wt% Cr); **~40 compositions**
   selected in **5 wt% < Cr < 30 wt%, 5 wt% < Ni < 20 wt%**, ~1 kg each by high-frequency
   induction melting. Chemical analysis by wet + instrumental methods: **P, S ≈ 0.01 wt%;
   C, Si, Mn ≈ ≤ 0.05 wt%** (verified from page image; OCR line garbled).
   → the alloys are effectively **ternary Fe-Cr-Ni (carbon-free within ≤0.05 wt%)**, no Mo.

5. **Background statement (intro, p. 1676):** the Fe-Cr binary liquidus/solidus minimum lies at
   ~22 wt% Cr at 1505 °C (cited as known fact, not measured here).

**Tables (rendered from page image; the OCR text layer does NOT contain table bodies):**
- **Table 1, p. 1682** — compositions (wt%) of 14 commercial steels used for distribution-
  coefficient work: 304, 308, 309, 310S, W12, W16L, 316L, 317L, W317L, 321, 410, 430, LC430, 12Cr
  (columns C, Si, Mn, P, S, Ni, Cr, Mo, Ti). Example rows: 304 = 0.073 C–0.75 Si–1.52 Mn–0.035 P–
  0.005 S–8.54 Ni–18.42 Cr–0.22 Mo; 316L = 0.017 C–0.98 Si–0.97 Mn–12.12 Ni–17.08 Cr–2.16 Mo;
  430 = 0.067 C–0.30 Si–0.72 Mn–0.10 Ni–16.02 Cr. (These are context, not anchors.)
- **Table 2, p. 1682** — equilibrium distribution coefficients K_Ni, K_Cr, K_Mn, K_Si, K_Mo per
  steel (e.g. 304: 0.79/1.04/0.86/0.93/–; 316L: 0.82/1.01/0.80/0.79/0.85; 410: –/0.99/0.87/–/–).
- Text summary (p. 1681): K_Cr = 0.98–1.02 (δ-primary region), 0.85–0.95 (γ-primary region);
  K_Ni = 0.76–0.83 (δ), 0.85–0.95 (γ); Mn 0.7–0.9, Si 0.7–0.9, Mo 0.75–0.85 (commercial steels,
  δ-primary except 310).

**Figure inventory + digitizability (all figures are scans; axes readable at ≥200 dpi):**

| Fig. | page (PDF) | content | digitizable? |
|---|---|---|---|
| 1, 2 | 1677 (2) | Tammann furnace / stationary-melting schematics | no (schematic) |
| 3 | 1678 (3) | EPMA sweep-step-scan schematic | no |
| 4 | 1678 (3) | example Cr profile vs distance across interface | yes (minor) |
| 5 | 1679 (4) | DTA curve, 21.0Cr–16.6Ni, onsets 1424 / 1441 °C | yes; values already in text |
| 6 | 1679 (4) | **tie-lines (open = liquid, filled = solid), wt% Cr–wt% Ni** | **yes — high value (L/δ, L/γ conjugate pairs)** |
| Photo 1 | 1678 (3) | planar liquid–solid interface micrograph | no |
| Photo 2 | 1680 (5) | dendrites: a = γ-primary (24.0Cr–18.8Ni), b = δ-primary (19.6Cr–9.14Ni) | qualitative |
| 7 | 1680 (5) | **δ/γ primary regions + peritectic–eutectic line (this work vs Schürmann), wt% axes, primary-phase + tie-line symbols** | **yes — high value** |
| 8 | 1680 (5) | **P–E line vs Schürmann/Bain/Jenkins, grid 10–20 wt% Cr, 5–15 wt% Ni** | **yes — best for P–E line digitization** |
| 9 | 1681 (6) | **iso-liquidus (solid) + iso-solidus (dashed) contours 1460–1500 °C, Fe corner** | **yes, with basis caveat below** |
| 10, 11 | 1681 (6) | K_Cr, K_Ni vs wt%Cr/wt%Ni and vs Creq/Nieq | yes (secondary) |

**Digitization caveats.**
- **Fig. 9 axis labels read "at. % Cr" / "at. % Ni" (0–30)** — atomic percent — while Figs. 6–8
  are explicitly **wt%**, and §3.4 quotes the transition point in **wt%**. Basis inconsistency
  inside the paper: verify before digitizing Fig. 9 (convert or restrict to the near-Fe corner
  where the difference is smallest); prefer Figs. 7–8 (wt%) for composition comparisons.
- Fig. 9 isotherm labels legible: 1460, 1470, 1480, 1490, 1500 °C (both liquidus and solidus
  families), plus the δ–γ–L three-phase region boundary.
- Fig. 8 contains 4 P–E lines (1 this work, 2 Schürmann, 3 Bain, 4 Jenkins) — digitize curve "1"
  only for our anchor; symbol sets (primary phase, tie-line endpoints) overlay the curves.
- OCR text layer garbles some characters (e.g. "210wt%cr" for "21.0 wt% Cr"); all quoted numbers
  were re-verified against rendered page images.

**Comparability caveats (Fe-Cr-Ni(−C) anchor).**
- Alloys are **ternary** (C, Si, Mn ≤ 0.05 wt%): compare against the **C ≈ 0 slice** of
  Fe-Cr-Ni-C; P, S ~0.01 wt% only nuisance.
- Tie-lines from stationary-melting method are near-equilibrium (1 h hold at T_L+50 °C, +50 °C
  advance, 1 h hold, oil quench) — the closest-to-equilibrium data in this set.
- DTA at ~7 °C/min, single runs, **no reported uncertainty**; solidus DTA values are explicitly
  unreliable (see above).
- Non-equilibrium only for the unidirectional-solidification primary-phase points (R = 1 mm/s),
  but primary-phase identity is robust to growth rate per the authors (citing Fredriksson).

---

## 3. Forgas Júnior, Otubo & Magnabosco 2016 (JATM 8(3)) — δ-ferrite fraction vs temperature

**Citation.** FORGAS JÚNIOR A., OTUBO J., MAGNABOSCO R. "Ferrite Quantification
Methodologies for Duplex Stainless Steel." *Journal of Aerospace Technology and Management*
**8**(3): 357–362, Jul.–Sep. 2016. DOI 10.5028/jatm.v8i3.653. Received 24 Mar 2016, accepted
5 May 2016.

**Accessibility.** Open access; PDF fetched OK this session:
`https://jatm.com.br/jatm/article/download/653/536/3967` (1 474 294 bytes, 6 pp. = 357–362);
DOI resolves (HTTP 200). Clean text layer, tables extractable — no digitization needed.

**Alloy (Table 1, p. 358 = PDF p. 2), wt% — UNS S31803 duplex stainless steel, hot-rolled
plate 300 × 200 × 3 mm:**

| Cr | Ni | Mo | Mn | N | C | Si | P | S | Fe |
|---|---|---|---|---|---|---|---|---|---|
| 22.07 | 5.68 | 3.20 | 1.38 | 0.17 | 0.017 | 0.34 | 0.02 | 0.001 | balance |

**Heat treatment (p. 358, §Experimental Procedures):** solution treatment **30 min at 1000,
1100 and 1200 °C under nitrogen atmosphere, then water quench**; specimens 10 × 10 mm, polished
to 1 µm, modified Beraha etchant.

**Ferrite volume fractions (Table 3, p. 361 = PDF p. 5), vol%:**

| T (°C) | XRD | Ferritoscope | Quantitative optical metallography | Thermo-Calc (TCFE7) |
|---|---|---|---|---|
| 1000 | 58.4 | 39.8 ± 0.5 | **41.5 ± 0.9** | 40.4 |
| 1100 | 68.5 | 46.2 ± 0.5 | **48.1 ± 0.8** | 51.3 |
| 1200 | 87.2 | 57.0 ± 0.7 | **58.5 ± 1.5** | 66.4 |

The paper does not define the ± (presumably the spread of the 10 fields/10 readings per
condition — flagged as such in the CSV).

**Methods (p. 358).**
- *Quantitative optical metallography*: ASTM E562-02 point counting, image analysis (Leica
  QMetals, DMLM microscope), **10 fields per sample at 500×**. Paper concludes this is "the most
  assertive … direct measurement method" (abstract; Conclusions).
- *Ferritoscope*: FISCHER MP30, magnetic induction, 10 measurements/sample, detection limit
  0.1% ferrite.
- *XRD*: Shimadzu XRD-7000, Cu-Kα, 30–120° 2θ, 1°/min, internal-ratio method (Moser et al. 2014);
  cell parameters a_α = 0.2880 nm, a_γ = 0.3601 nm (p. 360).
- *Thermo-Calc* + **TCFE7** equilibrium over 1000–1200 °C (Fig. 7; values in Table 3).

**Comparability caveats (Fe-Cr-Ni-C model).**
- **Outside model: Mo 3.20 wt%, N 0.17 wt%; also Mn 1.38 and Si 0.34 wt%** — fix as nuisance
  composition or flag exclusion. (P, S negligible.)
- **30-min holds may be short of full equilibrium**; start material is a hot-rolled plate with
  residual rolling texture (Figs. 1–3), which the authors use to explain the XRD bias —
  **use the metallography column only**; XRD is +17, +20, +29 vol% above metallography and
  should not be used.
- Thermo-Calc itself under/over-shoots the measurement by −1.1 (1000 °C), +3.2 (1100 °C),
  +7.9 vol% (1200 °C) — the paper attributes this to texture/grain-size effects not in the
  equilibrium calc; this gap is the natural noise floor for Δf_δ.
- Metallography–ferritoscope agreement within errors at all three temperatures.

**Figure/table inventory.** Table 1 (composition), Table 3 (all numbers above) — text tables;
Table 2 (XRD scattering factors, ASTM E975) not needed. Figs. 1–3 micrographs, Figs. 4–6 XRD
patterns — no anchor data. Fig. 7 (TC ferrite/austenite fraction vs T, 1000–1200 °C) and Fig. 8
(4-method comparison) are digitizable but **fully redundant with Table 3**.

---

## 4. Fallback sources — availability check (only needed if a target fails; none did)

| Fallback | Status this session |
|---|---|
| Nishino & Kagawa 1972, *Tetsu-to-Hagané* 58(1):107–118, DOI 10.2355/tetsutohagane1955.58.1_107 | **Re-verified now:** J-STAGE `_pdf` URL returns HTTP 200, `application/pdf`, 3 156 789 bytes (open). Use for A if Yamada ever becomes unusable (only 25Cr–20Ni cut, C ≤ 1.1 wt%). |
| Smetana et al. 2017 (METAL 2017, pp. 59–64) T_L table for 10 Fe-C-Cr-Ni-Mo steels | Not re-fetched this session (confer.cz landing did not respond to a later curl check; the sibling Drozdová PDF under the same `/metal/2017/download/` path worked). Candidate-list status: open, tabulated T_L; T_P/T_S only in Figs. 4–7. |
| Mundt & Hoffmeister 1983 (Arch. Eisenhüttenwesen), equilibrium δ-fraction 1200–1350 °C | Paywalled (Wiley 403 per candidate list; not re-tested). Cleanest *equilibrium* ternary δ-fraction set — needs library access. |
| Kundrat & Elliott 1986 (Metall. Trans. A) Fe-Cr-Ni-C tie-lines | Paywalled (abstract verified previously). |

---

## 5. Extraction provenance

- PDFs downloaded 2026-09-30 to `C:\Users\Lion\AppData\Local\Temp\opencode\anchors\`
  (`drozdova2017.pdf`, `yamada1987.pdf`, `forgas2016.pdf`) with PyMuPDF text dumps alongside.
- Numbers in `anchors.csv` were cross-checked against word-level extraction (Drozdová Table 2
  layout dump) and/or rendered page images at 200–500 dpi (Yamada Tables 1–2, Figs. 5/7/8/9,
  §2.1 composition ranges, impurity line, Fig. 5 composition line).
- No value was inferred, interpolated, or copied from secondary summaries.
