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
├── ai201_project3_takemeter_starter_clean.ipynb   # the Colab notebook (§1–§6)
├── data/
│   └── takemeter_nba_labeled.csv                  # 207 labeled r/nba comments
├── confusion_matrix.png                           # fine-tuned model, test set (§4) — Colab GPU run
├── evaluation_results.json                        # baseline vs fine-tuned summary (§6) — Colab GPU run
├── reproduce_eval.py                              # seed-pinned local re-run of the §2–§4 pipeline
├── test_predictions.json                          # per-row preds + softmax confidences (from reproduce_eval.py)
├── planning.md                                    # task design, labels, edge cases, metrics
└── README.md                                      # this file
```

**Dataset** — 207 comments scraped from r/nba game/trade/debate threads, balanced
to ~1/3 per class (analysis 70, hot_take 70, reaction 67). Columns:

| Column | Purpose |
|---|---|
| `text` | the raw comment |
| `label` | gold label (`analysis` / `hot_take` / `reaction`) |
| `notes` | free-text annotation rationale (populated for ~86 borderline rows) |
| `source_permalink` | Reddit URL for traceability |
| `comment_id` | Reddit comment ID |

The notebook splits this **stratified 70 / 15 / 15** (train ≈ 145, val ≈ 31,
test = 32) with `random_state=42`, so the test set is fixed and never seen during
training or tuning.

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

**Baseline:** Groq `llama-3.3-70b-versatile`, zero-shot, on the locked 31-example
test set (32 examples), prompted with the exact §2 definitions and §3 decision rules.

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
| Zero-shot baseline (Groq `llama-3.3-70b-versatile`) | **0.719** | — *(see note)* |
| Fine-tuned DistilBERT | **0.500** | **0.463** |

### Per-class metrics — fine-tuned DistilBERT

Computed directly from the committed `confusion_matrix.png` (test set, n = 32).

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `analysis` | 0.435 | **1.000** | 0.606 | 10 |
| `hot_take` | 0.400 | 0.182 | 0.250 | 11 |
| `reaction` | 1.000 | 0.364 | 0.533 | 11 |
| **macro avg** | 0.612 | 0.515 | **0.463** | 32 |

The shape tells the story: `analysis` recall is a perfect **1.00** but its
precision is only **0.43** — the model labels almost everything `analysis`, so it
never misses a real one but is wrong more than half the time it says "analysis."
`hot_take` is the mirror image (it is almost never *predicted*, recall 0.18), and
`reaction` has perfect precision but recall 0.36 (the few times it commits to
`reaction` it is right, but it usually doesn't commit).

> **Baseline per-class note (honest gap).** `evaluation_results.json` only
> exported the baseline's **accuracy** (0.719) — the §5 notebook printed the
> baseline `classification_report` to stdout but did not persist it, and the Groq
> baseline requires a `GROQ_API_KEY` to recompute. I have **not** filled in
> baseline per-class precision/recall/F1 rather than fabricate them; to populate
> that row, re-run notebook §5 (one cell) with a key and copy the printed report.

> **Reproduction note (read before trusting the example-level numbers below).**
> The committed `confusion_matrix.png` / `evaluation_results.json` are the
> original **Colab T4 GPU** run (accuracy 0.500). The original run never exported
> the per-example predictions/confidences, so I re-ran the *identical* pipeline
> (same code, same `random_state=42` split, same `seed=42`, same hyper-params)
> locally on CPU via the committed [`reproduce_eval.py`](reproduce_eval.py),
> writing every test row + softmax confidence to
> [`test_predictions.json`](test_predictions.json). The CPU re-run scored **0.438
> (14/32)** — within ±2 examples of the GPU run (CPU vs GPU floating-point
> ordering is nondeterministic even at a fixed seed) and shows the **identical
> failure mode**: a collapse onto `analysis` (predicted for 25 of 32 inputs,
> recall 1.00). **The example texts and confidence scores quoted below come from
> that reproducible CPU run**; the headline 0.500 / confusion-matrix table stay as
> the committed GPU artifacts.

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
- [ ] **MISS** — Fine-tuned beats baseline by ≥ 0.05 macro-F1. *(It lost 21.9 pts of
  accuracy; fine-tuning hurt.)*
- [x] / [ ] **PARTIAL** — ≥ 70% of errors on adjacent boundaries (**75%**, 12/16 ✓),
  but `analysis`↔`reaction` ≤ 10% **MISS** (4/16 = **25%**).
- [ ] **MISS** — `analysis` precision ≥ 0.80 (deployment bar). *(= 0.43.)*

_Unparseable baseline responses:_ 0 / 32 (baseline accuracy 0.719 implies all 32
parsed). _(Revise the prompt if > ~10%.)_

### Confirmed vs. refuted hypotheses

- **CONFIRMED (direction):** the primary failure mode is `analysis` ↔ `hot_take`,
  and the dominant direction is `hot_take → analysis` (9 of 16 errors). The model
  over-credits confident, stat-flavored hot takes as analysis — exactly as
  predicted.
- **REFUTED (magnitude / outcome):** I predicted fine-tuning would *improve*
  macro-F1 by ≥ 0.05 by fixing this cell. Instead fine-tuning **amplified** the
  shortcut into a near-collapse onto `analysis`, regressing accuracy by 22 points.
- **REFUTED:** `analysis` ↔ `reaction` was predicted to stay rare (≤ 10% of
  errors). It was **25%** (4 `reaction → analysis`), so the fine-tuned model is
  also confusing the two *non-adjacent* classes — evidence it is using a
  surface heuristic, not the concept.

### AI-assisted error-pattern analysis (then verified by hand)

Following `planning.md` §7.3, I pasted the full list of misclassified test rows
(text, true label, predicted label, confidence) into an LLM and asked it to
propose candidate failure patterns. It returned four hypotheses. I then verified
each against `test_predictions.json` and the confusion matrix and **kept two,
qualified one, and discarded one** — the unverified suggestions did not make the
cut:

| AI-proposed pattern | Verdict | What the data actually says |
|---|---|---|
| "Basketball vocab / stats / player names pull posts to `analysis` regardless of support" | ✅ **Confirmed** | **15 of 22** non-`analysis` test items were predicted `analysis`; all 9 `hot_take→analysis` errors name players/teams/stats ("Jokic", "2015 Warriors", "50 FTA"). |
| "The lexical `hot_take` cue *fraud* should trigger `hot_take`" | ✅ **Confirmed (it fails)** | **5 of 7** posts containing "fraud(s)" were predicted `analysis` — the model never learned the single strongest hot-take cue. |
| "Sarcasm is misread as `analysis`" | ⚠️ **Discarded as a standalone cause** | Some errors *are* sarcastic, but non-sarcastic stat posts fail just as often and **all** confidences sit at 0.33–0.44 (≈ chance). The failure is a global `analysis`-collapse, not a sarcasm-specific one — so "sarcasm" over-explains it. |
| "Short posts default to `reaction`" | ❌ **Refuted** | The opposite: short non-`analysis` posts mostly go to **`analysis`**, not `reaction`. Length correlates with *correctness* (correct rows avg **331** chars, wrong rows **116**) but only as a confound — the long rows are correct because the always-`analysis` predictor happens to be right on the long `analysis` class, not because the model learned "long ⇒ analysis." |

The one quantitative pattern that survives all of this and explains the rest:
**the model is barely a classifier at all.** Across all 32 test rows the top-class
softmax probability never exceeds **0.442** and averages **0.382** (uniform chance
= 0.333). It is a near-constant "`analysis`, weakly" predictor, so every
non-`analysis` item is at risk — the per-direction error counts are downstream of
that collapse, not separate phenomena.

### Three wrong predictions, analyzed

Texts and confidences are from the reproducible CPU run (`test_predictions.json`);
each illustrates a documented error direction.

**1. `hot_take → analysis`  (the dominant failure, 9 cases — `planning.md` §3 Edge case A)**
> *"6'5\" Caruso clamped the alleged 'the best player in the world' and 'best
> offensive center of all time' Jokic. Let us never forget this."* — pred
> `analysis`, conf **0.39**.

This is a textbook `hot_take`: a gloating, contrarian flex with **zero supporting
reasoning** — no stat, no mechanism, nothing a reader could argue against on the
merits. But it is dense with basketball proper nouns and superlatives, and that
surface is exactly what the fine-tune keyed on. This is the §3 Edge-case-A /
§6 "statty comment ⇒ analysis" shortcut realized: the model learned *presence of
basketball content*, not *whether a claim is supported*. The direction is
consistent (9 of the errors go `hot_take→analysis`, ~zero the other way), which is
the clean directional signature of a learned boundary the model got backwards.

**2. `reaction → analysis`  (the non-adjacent leak, the worst kind of error — §6)**
> *"Braun is not worth $30m lmao"* — pred `analysis`, conf **0.34**.

A four-word dismissive quip with a laugh tag — pure `reaction` by the §3 Edge-case-B
test ("is there a takeable position, or just a feeling?"). There is no argued
claim. The model still says `analysis`, almost certainly because "$30m" reads as a
number/stat. This is an `analysis↔reaction` confusion between the two *non-adjacent*
classes, which `planning.md` §6 names as the signature of a surface heuristic
rather than the concept — and **4 of 6** such errors carry an explicit "lol/lmao"
that didn't even pull them toward `reaction`. The model isn't reading function; it
is reading vocabulary.

**3. `reaction → hot_take`  (adjacent boundary, §3 Edge case B)**
> *"What is this 5 guard lineup lmao?"* — pred `hot_take`, conf **0.34**.

An incredulous one-liner — a mood, not a stance; no position a reader could
disagree with, so the §3 tie-break sends it to `reaction`. The model promoted it to
`hot_take`, reading the incredulous framing ("What is this… lmao?") as a contrarian
*claim*. This is the genuinely hard, *adjacent* boundary the labels were designed
to stress, and the near-chance confidence (0.34) shows the model has essentially no
grip on it. Note this is the rare error the fine-tune makes that **isn't** the
`analysis`-collapse — and it's still wrong.

### Is this a labeling problem or a data/model problem?

A labeling problem would show up as *inconsistent* gold labels on similar posts. It
doesn't: the three examples above are all labeled by the same §3 rules I'd apply
again, and the gold labels are defensible. The failure is therefore **not
annotation inconsistency** — it is a **data-volume / training problem**. A 66M-param
model fine-tuned on ~144 examples for 3 epochs minimizes loss by over-predicting
the easiest, highest-vocabulary class; it never gets enough signal to learn the
*supported-vs-unsupported* distinction that separates `analysis` from `hot_take`.
The fix is more/cleaner training signal and a loss that punishes the collapse — not
re-labeling (see below).

### What I'd change next (given the regression)

The likely cause is fine-tuning a 66M-param model on ~145 training examples for 3
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
softmax confidence (from `test_predictions.json`). ✓/✗ marks whether it matched
the gold label.

| # | Post (truncated) | Gold | Predicted | Confidence | |
|---|---|---|---|---|---|
| 1 | "Dallas doesn't need more defense. They need an offensive PG who's a great distributor… they'd basically have 0 volume 3pt shooters…" | `analysis` | `analysis` | 0.44 | ✓ |
| 2 | "He averaged around 30/13/10 on 66% True Shooting. To put that into…" | `analysis` | `analysis` | 0.43 | ✓ |
| 3 | "Thunder are frauds Wolves in 4!! I am very nervous" | `hot_take` | `hot_take` | 0.35 | ✓ |
| 4 | "6'5\" Caruso clamped the alleged 'best player in the world'… Let us never forget this." | `hot_take` | `analysis` | 0.39 | ✗ |
| 5 | "Braun is not worth $30m lmao" | `reaction` | `analysis` | 0.34 | ✗ |

**Why example 1 is a reasonable prediction.** This is a genuine, well-supported
basketball argument — it makes a claim ("Dallas needs a distributor, not defense")
and *supports it* with a concrete mechanism (losing Klay leaves "0 volume 3pt
shooters," spacing collapses, and they have no Jokić to paper over it). That is
exactly the §2 `analysis` definition: specific enough to argue against. It is also
the model's **highest-confidence** prediction in the whole test set (0.44) — and
revealingly, even when the model is "most sure" and correct, it is still barely
above chance. The model gets the easy, long, unambiguous `analysis` posts right;
it is everything else where the collapse bites (examples 4–5).

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
but unsupported `hot_take` ("6'5\" Caruso clamped Jokić"), or a four-word quip that
merely contains a dollar figure ("Braun is not worth $30m lmao"), it fired
`analysis` — because the words were there even though the *structure* was not.

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

**Where my implementation diverged, and why.** The spec exported only **accuracy**
to `evaluation_results.json` and only rendered the confusion matrix as a PNG,
keeping per-example predictions ephemeral (printed to stdout in §4). For a real
error analysis that was insufficient — I could not quote the misclassified texts or
their confidences from the committed artifacts. So I diverged by adding a committed,
seed-pinned [`reproduce_eval.py`](reproduce_eval.py) that re-runs the identical
pipeline and persists **every test row + softmax confidence** to
[`test_predictions.json`](test_predictions.json). The trade-off is honest and
documented above: my re-run is CPU, not the spec's Colab GPU, so it lands at 0.438
rather than 0.500 — same failure mode, ±2 examples — and I kept the committed GPU
artifacts as the headline rather than overwriting them.

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
verified each against `test_predictions.json`. I **confirmed** (a) and (b),
**discarded** (c) as over-explaining a global collapse, and **refuted** (d)
outright (short posts go to `analysis`, not `reaction`; length is a confound). Only
the verified patterns went into the report — the full kept/discarded table is in
[AI-assisted error-pattern analysis](#ai-assisted-error-pattern-analysis-then-verified-by-hand).

**Instance 2 — Reproduction tooling + evaluation write-up.** I directed an LLM to
write [`reproduce_eval.py`](reproduce_eval.py) (re-run the identical pipeline,
persist per-example predictions + softmax confidences) and to draft this
evaluation section. **What I changed/overrode:** I kept the committed Colab GPU
artifacts (0.500) as the headline rather than letting the CPU re-run's 0.438
silently replace them; I added the explicit provenance/reproduction caveats; and I
verified every quoted number (per-class P/R/F1, the 0.33–0.44 confidence band, the
15-of-22 collapse count, the 331-vs-116 char length figures) against the data
myself rather than trusting the draft.

**Planned uses (per `planning.md` §7.1–§7.2).** Label stress-testing — an LLM
generated boundary cases to pressure-test the §2/§3 definitions; those synthetic
examples were used only for definition-testing and were **never added to the 207-row
dataset**. Pre-labeling — the plan was to AI-pre-label batches and human-review
every row, with the baseline-model/pre-labeler overlap noted as a non-validation
caveat.

**Honest gap.** The committed `takemeter_nba_labeled.csv` does **not** carry the
per-row `ai_prelabeled` / override flags that `planning.md` §7.2 planned, so I
**cannot report exact pre-label or override counts** and have not invented them.
What is verifiable is that 86 of 207 rows carry a free-text `notes` rationale for
borderline calls. If pre-labeling was used, its provenance was not persisted — a
process gap to fix by adding the provenance columns before the next annotation pass.
