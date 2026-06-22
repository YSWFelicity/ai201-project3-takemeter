# TakeMeter

A fine-tuned text classifier that measures **discourse quality** in the r/nba
community by sorting comments into three classes based on *how much argued
basketball content* they carry — not whether the opinion is correct.

| Label | Meaning |
|---|---|
| `analysis` | Makes a basketball claim **and supports it** with stats, a mechanism, or a concrete example — specific enough to argue against. |
| `hot_take` | A strong/contrarian/exaggerated claim asserted as fact with **little or no support**. |
| `reaction` | A primarily **emotional or expressive** response (venting, joking, mocking) with no real argumentative claim. |

Full label definitions, edge-case rules, data-collection plan, evaluation
metrics, and the definition of success live in [`planning.md`](planning.md).
This README covers the things that document doesn't: how to run the project, the
baseline reflection, and the AI-usage disclosure.

---

## Repository structure

```
.
├── Copy_of_ai201_project3_takemeter_starter_clean.ipynb  # the Colab notebook (§1–§6) — authoritative results
├── data/
│   └── takemeter_nba_labeled.csv                 # labeled r/nba comments (committed 207-row snapshot)
├── confusion_matrix.png                          # fine-tuned model, test set (§4) — Colab GPU run
├── evaluation_results.json                       # baseline vs fine-tuned summary (§6) — Colab GPU run
├── reproduce_eval.py                             # seed-pinned local re-run of the §2–§4 pipeline (independent check)
├── test_predictions.json                         # per-row preds + softmax confidences (from reproduce_eval.py)
├── planning.md                                   # task design, labels, edge cases, metrics
└── README.md                                     # this file
```

**Dataset** — **210** comments scraped from r/nba game/trade/debate threads,
perfectly balanced **70 / 70 / 70** across `analysis` / `hot_take` / `reaction`
(the dataset the authoritative notebook run was trained and evaluated on). Columns:

| Column | Purpose |
|---|---|
| `text` | the raw comment |
| `label` | gold label (`analysis` / `hot_take` / `reaction`) |
| `notes` | free-text annotation rationale (populated for ~86 borderline rows) |
| `source_permalink` | Reddit URL for traceability |
| `comment_id` | Reddit comment ID |

> **Snapshot note.** The committed `data/takemeter_nba_labeled.csv` is an earlier
> **207-row** version (analysis 70, hot_take 70, reaction 67). The final run added
> 3 more `reaction` comments to reach exact 70/70/70 balance; that 210-row file is
> what the notebook trained on but was not re-exported to `data/`. All split sizes
> and metrics below are from the 210-row run.

The notebook splits this **stratified 70 / 15 / 15** (train **147**, val **31**,
test **32**) with `random_state=42`, so the test set is fixed and never seen during
training or tuning. The test set is 10 `analysis` / 11 `hot_take` / 11 `reaction`.

---

## How to run

The notebook runs on **Google Colab**, not locally — it needs a hosted GPU and a
Groq API key.

1. Open the notebook in Colab and set **Runtime → Change runtime type → T4 GPU**.
   Cell 2 should print `GPU available: True`.
2. **§1 Setup & data** — run the install/import cells, then run the label-map and
   upload cells. Upload `data/takemeter_nba_labeled.csv` when prompted (no manual
   Drive placement needed).
3. **§2 Split & tokenize** — run the split cell and confirm the printed sizes /
   per-class distribution look balanced.
4. **§5 Baseline (run this *before* fine-tuning)** — add your `GROQ_API_KEY` via
   the Colab **Secrets** panel (🔑), then run the prompt, classification, and
   metrics cells. This gives the zero-shot number that fine-tuning must beat.
   > ⚠️ Do **not** run the *Results Comparison* cell yet — it references
   > `ft_accuracy`, which only exists after fine-tuning in §4.
5. **§3–§4 Fine-tune** — run after the baseline is recorded.

> **Security:** never commit your Groq key. Use Colab Secrets, or paste it into a
> cell you do not push. If the session disconnects, re-upload the CSV and re-run
> §1–§2 before continuing.

---

## Baseline reflection & hypothesis (pre-registered)

> The assignment asks for this **before** fine-tuning, so the section below is a
> *prediction*, not a result. It is written so it can be confirmed or refuted
> against the actual baseline confusion matrix once §5 has been run.

**Baseline:** Groq `llama-3.3-70b-versatile`, zero-shot, on the locked 32-example
test set, prompted with the exact §2 definitions and §3 decision rules.

### Where I expect the baseline to struggle

The hard boundaries are designed into the labels (`planning.md` §3), and a general
model with no task-specific training has no reason to apply the tie-break rules
the way the annotation did. I expect errors to cluster on the **two adjacent
boundaries**, in this priority order:

1. **`analysis` ↔ `hot_take` (primary failure mode).** The model keys on the
   *presence* of basketball vocabulary and stats rather than on whether the claim
   is actually *supported well enough to argue against*. I predict it will
   **over-credit confident, stat-flavored `hot_take`s as `analysis`** — the exact
   error §3 Edge case A and §6 warn about ("long/statty comment = analysis"
   shortcut). Direction: more `hot_take → analysis` than the reverse.
2. **`hot_take` ↔ `reaction` (secondary).** Short emotional lines that bury a
   stance ("Bum frauds this is great") sit on §3 Edge case B. Zero-shot, the model
   is unlikely to apply the "takeable position vs. just a feeling?" tie-break
   consistently, so I expect leakage in both directions on one-liners.

### The shortcut to watch for

3. **`analysis` ↔ `reaction` (should be rare).** These are the two *non-adjacent*
   classes. If the baseline confuses them often, it is using a surface heuristic
   (e.g. "long = analysis, short = reaction") rather than the concept. §6 sets the
   bar that these non-adjacent confusions stay **≤ 10% of all errors**. I predict
   they will already be low at baseline.

### Predictions to test after fine-tuning

- Baseline macro-F1 is **decent but capped** by the `analysis`/`hot_take`
  boundary; `analysis` **recall** suffers most relative to its precision (or
  precision suffers from over-crediting — the confusion matrix will say which).
- Fine-tuning should improve **macro-F1 by ≥ 0.05** (the §6 success margin),
  mostly by fixing the `analysis` ↔ `hot_take` cell.
- The error *distribution* should stay on the adjacent boundaries; if fine-tuning
  raises accuracy but introduces `analysis` ↔ `reaction` confusion, that's a
  regression in *what was learned* even if the headline number looks better.

---

## Evaluation (results)

Metrics and rationale are defined in `planning.md` §5–§6. Evaluated on the locked
**32-example** test set (`evaluation_results.json`, `confusion_matrix.png`).

### Overall accuracy (both models)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Zero-shot baseline (Groq `llama-3.3-70b-versatile`) | **0.719** | **0.72** |
| Fine-tuned DistilBERT | **0.500** | **0.46** |

All numbers below are from the authoritative Colab **T4 GPU** run recorded in
[`Copy_of_ai201_project3_takemeter_starter_clean.ipynb`](Copy_of_ai201_project3_takemeter_starter_clean.ipynb)
(§4 fine-tuned report, §6 baseline report), on the locked 32-example test set.

### Per-class metrics — fine-tuned DistilBERT

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `analysis` | 0.43 | **1.00** | 0.61 | 10 |
| `hot_take` | 0.40 | 0.18 | 0.25 | 11 |
| `reaction` | 1.00 | 0.36 | 0.53 | 11 |
| **macro avg** | 0.61 | 0.52 | **0.46** | 32 |

The shape tells the story: `analysis` recall is a perfect **1.00** but its
precision is only **0.43** — the model labels almost everything `analysis`, so it
never misses a real one but is wrong more than half the time it says "analysis."
`hot_take` is the mirror image (it is almost never *predicted*, recall 0.18), and
`reaction` has perfect precision but recall 0.36 (the few times it commits to
`reaction` it is right, but it usually doesn't commit).

### Per-class metrics — zero-shot baseline (Groq `llama-3.3-70b-versatile`)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `analysis` | 0.73 | 0.80 | 0.76 | 10 |
| `hot_take` | 0.86 | 0.55 | 0.67 | 11 |
| `reaction` | 0.64 | 0.82 | 0.72 | 11 |
| **macro avg** | 0.74 | 0.72 | **0.72** | 32 |

The baseline is a **real, balanced classifier**: every class clears F1 0.67, and
its errors are sensible (its weakest cell is `hot_take` recall 0.55 — it sometimes
reads an unsupported take as `analysis`, the same boundary that defeats the
fine-tune, but it never collapses). All 32 responses parsed (0 unparseable). This
is the bar the fine-tune had to beat — and didn't.

**Headline: fine-tuning *regressed*.** Accuracy fell from 71.9% (baseline) to 50.0%
(fine-tuned), an improvement of **−0.219**. This is the opposite of the §6 success
target and is the most important result to explain (below).

### Fine-tuned confusion matrix (test set)

Rows = true label, columns = predicted label. Diagonal = correct.

| true \ pred | analysis | hot_take | reaction |
|---|---|---|---|
| **analysis** | **10** | 0 | 0 |
| **hot_take** | 9 | **2** | 0 |
| **reaction** | 4 | 3 | **4** |

The model **collapsed toward `analysis`**: it predicted `analysis` for 23 of 32
inputs (`analysis` recall = 1.00 but precision = 0.43). `hot_take` is almost never
predicted (recall 0.18). This is a degenerate small-data fine-tune — DistilBERT
latched onto a surface shortcut ("looks statty / basketball-heavy ⇒ analysis")
rather than learning the *supported-vs-unsupported* distinction, which is exactly
the `planning.md` §3 Edge case A / §6 shortcut the labels were designed to stress.

**Success criteria (from §6):**
- [ ] **MISS** — Macro-F1 ≥ 0.75 and every per-class F1 ≥ 0.65. *(macro-F1 = 0.46;
  `hot_take` F1 = 0.25.)*
- [ ] **MISS** — Fine-tuned beats baseline by ≥ 0.05 macro-F1. *(It **lost** by 0.26:
  baseline macro-F1 = 0.72 vs fine-tuned 0.46; −0.219 accuracy. Fine-tuning hurt.)*
- [x] / [ ] **PARTIAL** — ≥ 70% of errors on adjacent boundaries (**75%**, 12/16 ✓),
  but `analysis`↔`reaction` ≤ 10% **MISS** (4/16 = **25%**).
- [ ] **MISS** — `analysis` precision ≥ 0.80 (deployment bar). *(= 0.43.)*

_Unparseable baseline responses:_ 0 / 32 (the §6 baseline cell reports
"32/32 parseable"). _(Revise the prompt if > ~10%.)_

### Confirmed vs. refuted hypotheses (vs. the pre-registered baseline reflection)

- **CONFIRMED (direction):** the primary failure mode is `analysis` ↔ `hot_take`,
  and the dominant direction is `hot_take → analysis` (9 of 16 errors). The model
  over-credits confident, stat-flavored hot takes as analysis — exactly as
  predicted.
- **CONFIRMED (about the baseline):** I predicted the baseline's weakest cell would
  be `hot_take` over-credited as `analysis`. The baseline report bears this out —
  `hot_take` is its lowest-recall class (0.55) while its precision stays high
  (0.86), i.e. when it *does* say `hot_take` it's right, it just misses some. It
  never collapses, though, which is why it lands at macro-F1 0.72.
- **REFUTED (magnitude / outcome):** I predicted fine-tuning would *improve*
  macro-F1 by ≥ 0.05 by fixing this cell. Instead fine-tuning **amplified** the
  shortcut into a near-collapse onto `analysis`, regressing macro-F1 from 0.72 to
  0.46 (−0.26) and accuracy by 22 points.
- **REFUTED:** `analysis` ↔ `reaction` was predicted to stay rare (≤ 10% of
  errors). It was **25%** (4 `reaction → analysis`), so the fine-tuned model is
  also confusing the two *non-adjacent* classes — evidence it is using a
  surface heuristic, not the concept.

### AI-assisted error-pattern analysis (then verified by hand)

Following `planning.md` §7.3, I pasted the notebook's §4 misclassification dump
(text, true label, predicted label, confidence) into an LLM and asked it to
propose candidate failure patterns. It returned four hypotheses. I then verified
each against the §4/§6 reports and the confusion matrix and **kept two, qualified
one, and discarded one** — the unverified suggestions did not make the cut:

| AI-proposed pattern | Verdict | What the data actually says |
|---|---|---|
| "Basketball vocab / stats / player names pull posts to `analysis` regardless of support" | ✅ **Confirmed** | **13 of 22** non-`analysis` test items were predicted `analysis`; **8 of the 9** `hot_take→analysis` errors name players/teams/stats ("Jokic", "50 FTA", "MVP", "Caruso"). |
| "The lexical `hot_take` cue *fraud* should trigger `hot_take`" | ✅ **Confirmed (it fails)** | **Every** misclassified post containing "fraud(s)" (5 of them) was predicted `analysis`, not `hot_take` — the model never learned the single strongest hot-take cue. |
| "Sarcasm is misread as `analysis`" | ⚠️ **Discarded as a standalone cause** | Some errors *are* sarcastic, but non-sarcastic stat posts fail just as often and **every** error sits at confidence **0.34–0.40** (≈ the 0.333 chance floor). The failure is a global `analysis`-collapse, not a sarcasm-specific one — so "sarcasm" over-explains it. |
| "Short posts default to `reaction`" | ❌ **Refuted** | The opposite: short non-`analysis` posts go to **`analysis`**, not `reaction` — e.g. *"A 76ers fan? You most know a lot about 'Fraud MVP's"* and *"I love being a fraud … 50 FTA"* are short and were both predicted `analysis`. The long, well-supported posts are correct only because the always-`analysis` predictor happens to be right on the long `analysis` class (recall 1.00), not because the model learned "long ⇒ analysis." |

The one quantitative pattern that survives all of this and explains the rest:
**the model is barely a classifier at all.** Every one of the 16 errors carries a
top-class softmax probability of just **0.34–0.40** — barely above the three-class
chance floor of 0.333. It is a near-constant "`analysis`, weakly" predictor, so
every non-`analysis` item is at risk — the per-direction error counts are
downstream of that collapse, not separate phenomena.

### Three wrong predictions, analyzed

Texts and confidences are quoted directly from the notebook's §4 misclassification
dump; each illustrates a documented error direction.

**1. `hot_take → analysis`  (the dominant failure, 9 cases — `planning.md` §3 Edge case A)**
> *"6'5\" Caruso clamped the alleged 'the best player in the world' and 'best
> offensive center of all time' Jokic. Let us never forget this."* — pred
> `analysis`, conf **0.38**.

This is a textbook `hot_take`: a gloating, contrarian flex with **zero supporting
reasoning** — no stat, no mechanism, nothing a reader could argue against on the
merits. But it is dense with basketball proper nouns and superlatives, and that
surface is exactly what the fine-tune keyed on. This is the §3 Edge-case-A /
§6 "statty comment ⇒ analysis" shortcut realized: the model learned *presence of
basketball content*, not *whether a claim is supported*. The direction is
consistent (9 of the errors go `hot_take→analysis`, ~zero the other way), which is
the clean directional signature of a learned boundary the model got backwards.

**2. `reaction → analysis`  (the non-adjacent leak, the worst kind of error — §6)**
> *"lol, 2018 LeBron played the 2018 Celtics. That Celtics team gets swept by OkC
> & so do the Cavs."* — pred `analysis`, conf **0.40**.

A dismissive, "lol"-led quip — a reaction venting that *this matchup wouldn't
matter*, with no actual argued claim a reader could evaluate (it asserts two teams
"get swept" but supports neither). By §3 it's a `reaction`. The model says
`analysis`, almost certainly because the text is studded with team/era tokens
("2018", "Celtics", "Cavs", "OkC"). This is an `analysis↔reaction` confusion
between the two *non-adjacent* classes — which `planning.md` §6 names as the
signature of a surface heuristic rather than the concept — and note the leading
"lol" didn't pull it toward `reaction` at all. The model isn't reading function; it
is reading vocabulary.

**3. `reaction → hot_take`  (adjacent boundary, §3 Edge case B)**
> *"There are crazy people in every fan base lmao."* — pred `hot_take`, conf **0.35**.

A throwaway one-liner — a mood, not a stance; there is no basketball position a
reader could disagree with, so the §3 tie-break sends it to `reaction`. The model
promoted it to `hot_take`, reading the sweeping "every fan base" framing as a
contrarian *claim*. This is the genuinely hard, *adjacent* boundary the labels were
designed to stress, and the near-chance confidence (0.35) shows the model has
essentially no grip on it. Note this is one of the rare errors the fine-tune makes
that **isn't** the `analysis`-collapse — and it's still wrong.

### Is this a labeling problem or a data/model problem?

A labeling problem would show up as *inconsistent* gold labels on similar posts. It
doesn't: the three examples above are all labeled by the same §3 rules I'd apply
again, and the gold labels are defensible. The failure is therefore **not
annotation inconsistency** — it is a **data-volume / training problem**. A 66M-param
model fine-tuned on 147 examples for 3 epochs minimizes loss by over-predicting
the easiest, highest-vocabulary class; it never gets enough signal to learn the
*supported-vs-unsupported* distinction that separates `analysis` from `hot_take`.
The fix is more/cleaner training signal and a loss that punishes the collapse — not
re-labeling (see below).

### What I'd change next (given the regression)

The likely cause is fine-tuning a 66M-param model on 147 training examples for 3
epochs: too little signal, so it minimizes loss by over-predicting the easiest
class. Concrete next steps, in priority order:

1. **Fight the collapse directly** — class-weighted cross-entropy (or oversample
   `hot_take`/`reaction`) so the model can't minimize loss by guessing `analysis`.
2. **Stop earlier and gentler** — early stopping on **val macro-F1** (not
   accuracy, which rewards the majority-guess), fewer epochs, smaller LR.
3. **Tighten the hard boundary with data, not definitions** — the §3 rules are
   already consistent (the gold labels hold up), so the lever is *more diverse
   examples that show the hard case explicitly*: more supported-vs-unsupported
   `hot_take`/`analysis` pairs that share vocabulary but differ in whether a claim
   is actually argued, so the model can't lean on basketball nouns as a shortcut.

As it stands, the **zero-shot Groq baseline is the better classifier** on this task.

### Sample classifications (fine-tuned model)

Five test posts run through the fine-tuned model, with the predicted label and its
softmax confidence. ✓/✗ marks whether it matched the gold label.

| # | Post (truncated) | Gold | Predicted | Confidence | |
|---|---|---|---|---|---|
| 1 | "Dallas doesn't need more defense. They need an offensive PG who's a great distributor… 0 volume 3pt shooters…" | `analysis` | `analysis` | ~0.44 † | ✓ |
| 2 | "6'5\" Caruso clamped the alleged 'best player in the world'… Let us never forget this." | `hot_take` | `analysis` | 0.38 | ✗ |
| 3 | "I love being a fraud and fake number one seed… 50 FTA" | `hot_take` | `analysis` | 0.34 | ✗ |
| 4 | "lol, 2018 LeBron played the 2018 Celtics. That Celtics team gets swept by OkC & so do the Cavs" | `reaction` | `analysis` | 0.40 | ✗ |
| 5 | "There are crazy people in every fan base lmao." | `reaction` | `hot_take` | 0.35 | ✗ |

> † Rows 2–5 are quoted verbatim from the notebook's §4 misclassification dump,
> which prints a confidence for each **error**. The notebook does **not** print
> confidences for *correct* rows, so row 1's confidence is taken from the
> independent local re-run ([`reproduce_eval.py`](reproduce_eval.py)); it sits at
> the very top of the same 0.34–0.44 band every prediction occupies.

**Why example 1 is a reasonable prediction.** This is a genuine, well-supported
basketball argument — it makes a claim ("Dallas needs a distributor, not defense")
and *supports it* with a concrete mechanism (losing Klay leaves "0 volume 3pt
shooters," spacing collapses, and they have no Jokić to paper over it). That is
exactly the §2 `analysis` definition: specific enough to argue against — so an
`analysis` prediction here is correct *for the right reason*. The catch is that it
is also about the model's **highest-confidence** prediction in the whole test set
(~0.44), and even then it is barely above the 0.333 chance floor. The model gets
the easy, long, unambiguous `analysis` posts right (recall 1.00 on `analysis`); it
is everything else where the collapse bites (rows 2–5).

### Reflection: what the model captured vs. what I intended

I designed the labels around a single intended axis (`planning.md` §2): **is there
a basketball claim, and is it supported well enough to argue against?** That axis is
about *argumentative structure* — the relationship between a claim and its
evidence.

What the model's decision boundary actually captured is a **lexical proxy**:
*how much basketball-flavored vocabulary is present* — player names, team names,
numbers, superlatives. Those two things correlate in the training data (real
`analysis` posts genuinely are long and stat-dense), so the model achieved low
training loss by learning the correlate instead of the concept. It then
**overfit to that shortcut and over-generalized it**: faced with a vocabulary-rich
but unsupported `hot_take` ("6'5\" Caruso clamped Jokić"), or a "lol"-led quip
stuffed with team/era tokens ("lol, 2018 LeBron played the 2018 Celtics… get swept
by OkC"), it fired `analysis` — because the words were there even though the
*structure* was not.

What it **missed** is the entire thing I cared about: the claim→support
relationship. It has no representation of "asserted vs. argued." The clearest
evidence is direction plus calibration — errors run overwhelmingly *toward*
`analysis` (the high-vocabulary class), and the model is never confident about
anything (max 0.44), which is what an over-parameterized model trained on too few
examples looks like when it has settled on a surface feature rather than a concept.
The gap between intent and outcome is precisely the gap `planning.md` §6 was
written to detect: I wanted "discourse quality," and the model learned "basketball
density." On this data and budget, the labels were learnable in principle but the
*concept* was not learned — only its shadow.

---

## Spec reflection

**Where the spec helped.** The starter notebook's spec fixed the evaluation
contract up front — a **stratified 70/15/15 split with `random_state=42`, a locked
test set, and a baseline-before-fine-tuning ordering** — and that structure did
real work. Because the test set was locked and the baseline was required *first*,
the regression was impossible to hide or rationalize away: I had a pre-registered
zero-shot number (0.719) sitting next to the fine-tuned number (0.500) on the
*identical* 32 rows. The spec's insistence on per-class metrics + a confusion
matrix (not just accuracy) is exactly what surfaced the `analysis`-collapse; had I
only reported accuracy I might have mistaken a degenerate near-constant predictor
for a mediocre-but-real one.

**Where my implementation diverged, and why.** The spec persists only **accuracy**
to `evaluation_results.json` and renders the confusion matrix as a PNG, while the
§4 misclassification dump (texts + confidences) and the §6 per-class reports are
printed to stdout and live only in the notebook's saved cell outputs — not in a
machine-readable committed file. To make the error analysis fully reproducible
*outside* the notebook, I added a seed-pinned [`reproduce_eval.py`](reproduce_eval.py)
that re-runs the identical pipeline and writes **every test row + softmax
confidence** to [`test_predictions.json`](test_predictions.json). The honest
caveat: that script runs on CPU against the committed 207-row dataset snapshot, so
it lands at 0.438 rather than the notebook's GPU 0.500 — **same failure mode, same
collapse onto `analysis`, ±2 examples** — which is exactly why I treat the notebook
run as authoritative for every headline number and use the script only as an
independent confirmation that the collapse isn't an artifact of one GPU seed.

---

## AI-usage disclosure

Per `planning.md` §7, AI tools were used at specific points; the final calls were
always the author's. The two instances below are the substantive, verifiable uses;
the planned uses and an honest gap follow.

**Instance 1 — Error-pattern analysis (after evaluation).** I directed an LLM to
read the full list of misclassified test rows (text, true label, predicted label,
confidence) and propose candidate failure patterns. It produced four hypotheses:
(a) basketball vocabulary pulls posts to `analysis`, (b) the word "fraud" should
trigger `hot_take`, (c) sarcasm is misread, (d) short posts default to `reaction`.
**What I changed/overrode:** I treated all four as hypotheses, not findings, and
verified each against the notebook's §4 misclassification dump and §6 per-class
reports (cross-checked with the local re-run). I **confirmed** (a) and (b),
**discarded** (c) as over-explaining a global collapse, and **refuted** (d)
outright (short posts go to `analysis`, not `reaction`; length is a confound). Only
the verified patterns went into the report — the full kept/discarded table is in
[AI-assisted error-pattern analysis](#ai-assisted-error-pattern-analysis-then-verified-by-hand).

**Instance 2 — Reproduction tooling + evaluation write-up.** I directed an LLM to
write [`reproduce_eval.py`](reproduce_eval.py) (re-run the identical pipeline,
persist per-example predictions + softmax confidences) and to draft this
evaluation section. **What I changed/overrode:** I made the **notebook GPU run the
single authoritative source** for every headline number rather than letting the CPU
re-run's 0.438 stand in for it; I corrected draft figures that came from the wrong
run (e.g. the collapse count is **13 of 22** in the authoritative run, not the 15
of 22 the CPU snapshot showed; I removed example texts like "Braun is not worth
$30m" that exist only in the 207-row split and replaced them with real
notebook-test rows); and I re-verified every quoted number (per-class P/R/F1, the
0.34–0.40 error-confidence band, the confusion matrix) against the notebook's saved
outputs rather than trusting the draft.

**Planned uses (per `planning.md` §7.1–§7.2).** Label stress-testing — an LLM
generated boundary cases to pressure-test the §2/§3 definitions; those synthetic
examples were used only for definition-testing and were **never added to the
labeled dataset**. Pre-labeling — the plan was to AI-pre-label batches and human-review
every row, with the baseline-model/pre-labeler overlap noted as a non-validation
caveat.

**Honest gap.** The committed `takemeter_nba_labeled.csv` does **not** carry the
per-row `ai_prelabeled` / override flags that `planning.md` §7.2 planned, so I
**cannot report exact pre-label or override counts** and have not invented them.
What is verifiable is that 86 rows (of the committed 207-row snapshot) carry a free-text `notes` rationale for
borderline calls. If pre-labeling was used, its provenance was not persisted — a
process gap to fix by adding the provenance columns before the next annotation pass.
