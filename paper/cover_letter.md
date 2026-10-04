# Cover letter — Computational Materials Science

Date: [submission date]

Dear Editor,

We are pleased to submit our manuscript **"Validated high-throughput
screening with simplex-constrained surrogates for CALPHAD phase-fraction
prediction"** for consideration as a full-length research article in
*Computational Materials Science*.

Machine-learning surrogates are increasingly proposed as replacements for
per-point CALPHAD minimisation, but the literature largely reports
accuracy on a single random split and leaves deployment questions
unanswered. Our manuscript is organised around the three questions a
practitioner must answer before trusting such a surrogate in a screening
workflow, and reports negative results as prominently as positive ones:

1. **How should phase fractions be predicted?** A systematic benchmark of
   six simplex-constraint mechanisms plus a presence-supervised head
   against four classical baselines on five Fe-based ternaries and four
   extension systems (88,542 equilibria) shows that closure to machine
   precision costs no accuracy, that renormalisation alone is sufficient,
   and that presence supervision — not architecture — resolves the
   sharp-boundary cases.
2. **Where can the predictions be trusted?** Contiguous-band and strict
   extrapolation protocols with matched random controls, plus an
   ideal-form control at equal training size, separate spatial shift from
   data-volume effects: interior-band penalties are spatial, not
   volumetric, while one-sided extrapolation is uniformly unreliable.
3. **Does screening actually save cost?** Two lean-nickel screens are
   validated point-by-point under full-set CALPHAD (25,411 confirmation
   solves), giving an end-to-end demonstration rather than an asserted
   speedup.

All claims are re-derived from stored prediction artefacts by an
automated audit (597 checks, 0 failures); the pipeline, data and analysis
code will be archived on Zenodo and are available at
https://github.com/user4072/CALPHAD-based-Phase-Fraction-Prediction.

The work falls squarely within the scope of *Computational Materials
Science*: computational alloy design, surrogate modelling, and the
validation practices that make high-throughput screening trustworthy.
The manuscript is original, is not under consideration elsewhere, and
all authors have approved the submission. We declare no competing
interests.

Thank you for your consideration.

Sincerely,

[Corresponding author] (on behalf of all authors)
[institution]
ss_mousavi@atu.ac.ir
