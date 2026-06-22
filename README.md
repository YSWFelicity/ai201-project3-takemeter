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
├── confusion_matrix.png                           # fine-tuned model, test set (§4)
├── evaluation_results.json                        # baseline vs fine-tuned summary (§6)
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

| Model | Accuracy | Macro-F1 | `analysis` F1 | `hot_take` F1 | `reaction` F1 |
|---|---|---|---|---|---|
| Zero-shot baseline (Groq `llama-3.3-70b-versatile`) | **0.719** | — | — | — | — |
| Fine-tuned DistilBERT | **0.500** | 0.463 | 0.606 | 0.250 | 0.533 |

> Note: `evaluation_results.json` only exported **accuracy** for the baseline, so
> the baseline per-class F1 / macro-F1 cells above are left blank rather than
> guessed. The fine-tuned per-class numbers are computed directly from
> `confusion_matrix.png`. Re-run §5–§6 if you want the baseline breakdown written
> out as well.

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

### Three wrong predictions to analyze in depth

The committed artifacts don't include the misclassified comment *texts* (they print
in §4 of the notebook). Pull three concrete rows from the §4 misclassification dump
and paste them in, one per error pattern below:

1. **`hot_take → analysis` (9 cases, the main failure).** A strong, unsupported
   claim phrased with basketball vocabulary / a stat reference gets read as
   well-supported. Example: _<paste a §4 row>_.
2. **`reaction → analysis` (4 cases, the non-adjacent leak).** An expressive /
   emotional comment with no real argued claim is still classified `analysis` —
   the model's "basketball words ⇒ analysis" shortcut overriding intent. Example:
   _<paste a §4 row>_.
3. **`reaction → hot_take` (3 cases).** A venting / mocking one-liner that buries a
   faint stance gets promoted to `hot_take` (`planning.md` §3 Edge case B). Example:
   _<paste a §4 row>_.

### What I'd change next (given the regression)

The likely cause is fine-tuning a 66M-param model on ~145 training examples for 3
epochs: too little signal, so it minimizes loss by over-predicting the easiest
class. Concrete next steps: class-weighted loss or oversampling to counter the
`analysis` collapse, fewer epochs with early stopping on val macro-F1, a smaller
learning rate, and more labeled data. As it stands, the **zero-shot Groq baseline
is the better classifier** on this task.

---

## AI-usage disclosure

Per `planning.md` §7, AI tools were used at specific, logged points; the final
calls were always the author's. Fill in the actuals as the project proceeds:

- **Label stress-testing** (before annotation): used an LLM to generate boundary
  cases and pressure-test the §2/§3 definitions. _Synthetic examples were never
  added to the dataset._
- **Pre-labeling** (with human review of every row): _N_ of 207 rows were
  AI-pre-labeled and personally reviewed; overridden on _N_ rows. Because the
  pre-labeler and the baseline model are the same model, their agreement is **not**
  treated as validation.
- **Error-pattern analysis** (after evaluation): the LLM proposes candidate failure
  patterns; each is treated as a hypothesis and only included if confirmed against
  the actual confusion matrix.
- **This README and notebook scaffolding** were drafted with AI assistance and
  reviewed by the author.
