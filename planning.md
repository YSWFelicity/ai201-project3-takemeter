# TakeMeter — Planning Document

A fine-tuned text classifier that measures **discourse quality** in an online NBA
community. This document defines what TakeMeter measures, how the data is
collected and labeled, how edge cases are resolved, and how success is judged.

---

## 1. Community

**Chosen community: r/nba (the main NBA subreddit on Reddit).**

I chose r/nba because it is one of the highest-volume sports discussion forums
online, and the *quality* of its discourse varies enormously within a single
comment thread. The same post about a playoff game will attract:

- detailed breakdowns of lineups, spacing, and matchup math,
- exaggerated "this player is a fraud / the GOAT" provocations posted to start
  arguments, and
- one-line emotional venting at the refs or the score.

This spread is exactly what makes it a good classification target. The community
has **strong, shared, implicit norms** about what counts as a "real take" versus
"just yelling" — regulars instinctively know the difference between someone
arguing in good faith with evidence and someone dropping a rage-bait one-liner.
Those norms are real enough to be learnable but fuzzy enough that the label
design is genuinely hard, which is the point of the project.

The discourse is also **text-first and self-contained**: most comments make
sense without watching the game, the vocabulary is rich (advanced stats, player
names, basketball concepts), and there is a near-endless supply of fresh data
from daily game threads, trade-rumor threads, and "ranking" debate threads.

---

## 2. Labels

TakeMeter classifies a comment by **how it contributes to the discussion**, not
by whether its opinion is correct or which team it favors. There are **three
mutually exclusive labels** ordered roughly by analytical effort:

### `analysis` — reasoned basketball argument
> A comment that makes a claim **and supports it** with basketball reasoning,
> evidence, stats, or concrete examples, such that a reader could evaluate or
> push back on the argument on its merits.

**Examples (from the dataset):**
1. *"Paolo is averaging 28/9/5 on 45/42/72 shooting in the playoffs against
   elite defenses and with shit spacing. Paolo is gonna be a superstar, both are
   good but I lean Paolo."*
2. *"Dallas doesn't need more defense. They need an offensive PG who's a great
   distributor. Losing one of their few remaining shooters would also be a huge
   issue — they'd basically have 0 volume 3pt shooters..."*

### `hot_take` — provocative opinion, little support
> A comment that asserts a **strong, contrarian, or exaggerated opinion** as if
> it were obvious fact, with little or no supporting reasoning — phrased to
> provoke or to stake out a bold position rather than to explain it.

**Examples (from the dataset):**
1. *"FTA is a fraud MVP."*
2. *"This OKC team are such FRAUDS."*

### `reaction` — emotional / low-content response
> A comment that is primarily an **immediate emotional or expressive response**
> — venting, joking, mocking, celebrating, or short interjections — that carries
> no real argumentative claim about basketball.

**Examples (from the dataset):**
1. *"lol. The copium is crazy."*
2. *"74 uncalled fouls smh refs."*

### Why these three are exclusive and exhaustive
The labels separate along a single clean axis — **how much argued basketball
content is present**:

| | Makes a basketball claim? | Supports the claim? |
|---|---|---|
| `analysis` | Yes | Yes |
| `hot_take` | Yes | No / barely |
| `reaction` | No | — |

Because the test "is there a claim, and is it supported?" applies to almost any
comment, >90% of posts fit one bucket without needing an "other" catch-all.
Spam, off-topic, and pure-link comments are excluded at collection time rather
than given a label.

---

## 3. Hard edge cases

Two boundaries are genuinely ambiguous, and the annotation notes confirm these
are where most of the borderline calls happened.

### Edge case A — `hot_take` vs `analysis`
A bold opinion with *a sliver* of reasoning attached.
> *"Paolo a mvp level candidate if he has the spacing Cade had this year"* —
> contrarian claim, but it gestures at a real causal mechanism (spacing).

**Decision rule:** label `analysis` **only if the reasoning is specific enough to
argue against** (names a mechanism, stat, matchup, or example). A vague
justification ("he's just better," "everyone knows") does not upgrade a
`hot_take`. When still tied, default to `hot_take` — the lower-effort label —
because over-crediting borderline posts as `analysis` would inflate the class we
most care about getting right.

### Edge case B — `hot_take` vs `reaction`
A short, emotional line that *also* contains a buried opinion.
> *"Bum frauds this is great"* — emotional venting, but "frauds" is a stance.

**Decision rule:** ask **"is there a takeable position, or just a feeling?"** If a
reader could disagree with a *claim* ("they're frauds" → no they're not), it is a
`hot_take`. If there is nothing to disagree with, only a mood ("lol the copium is
crazy"), it is a `reaction`. Tie-breaks go to `reaction`, since one-liners with
no substantive claim are functionally low-content even when they sound opinionated.

### General principle
- Judge the **dominant function** of the comment, not its loudest word.
- Apply rules **consistently** rather than per-vibe; record the call in the
  `notes` column so the decision is auditable and reproducible.

---

## 4. Data collection plan

**Source.** Comments scraped from r/nba threads (game threads, trade-rumor
threads, and player-ranking/debate threads) via Reddit, retaining the comment
text, permalink, and comment ID for traceability. Threads were chosen to
maximize the natural mix of all three classes (debate threads yield `analysis`
and `hot_take`; live game threads yield `reaction`).

**Target volume.** At least 200 labeled comments, aiming for a **roughly balanced
~1/3 per label** so that no class is starved during fine-tuning and per-class
metrics are meaningful. *(Final dataset used in the authoritative run: **210
comments**, perfectly balanced — analysis 70, hot_take 70, reaction 70. The
committed `data/takemeter_nba_labeled.csv` is an earlier 207-row snapshot with
reaction 67; the last 3 reaction comments were added to reach exact balance.)*

**Splits.** Stratified by label into **train / validation / test (~70 / 15 / 15)**
so every split preserves the class balance and the test set is never seen during
training or hyperparameter tuning.

**If a label is underrepresented after 200 examples:** I will **targeted-collect**
from threads that naturally over-produce the rare class (e.g. live game threads
for `reaction`, hot-button "is X overrated" threads for `hot_take`) rather than
relabel or duplicate existing rows. If a class still lags, I will report the true
distribution honestly and apply **class weighting** in the loss function instead
of faking balance — and I'll flag the imbalance as a limitation in the eval
report rather than hide it.

---

## 5. Evaluation metrics

Accuracy alone is misleading here for two reasons: the classes are only roughly
balanced, and the **cost of errors is not symmetric** (confusing `hot_take` with
`reaction` matters less than misjudging whether something is real `analysis`).
So I report a layered set:

- **Overall accuracy** — headline number, fair to cite only because classes are
  near-balanced; reported for both the fine-tuned model and the zero-shot baseline.
- **Per-class precision, recall, and F1** — the core metrics. They reveal *where*
  the model fails: e.g. high `analysis` precision but low recall means it misses
  real analysis; the opposite means it over-credits weak takes. Macro-averaged F1
  summarizes balanced performance across all three classes.
- **Confusion matrix (3×3)** — shows the *direction* of mistakes and directly
  tests whether the model struggles on the exact boundaries flagged in §3
  (analysis↔hot_take, hot_take↔reaction).
- **Baseline comparison** — the same metrics for Groq `llama-3.3-70b-versatile`
  zero-shot on the identical test set, to quantify what fine-tuning actually bought.
- **Qualitative error analysis** — ≥3 specific misclassifications with my reading
  of *why*, plus a search for systematic patterns (e.g. sarcasm, very short posts).

**Why these are right for this task:** the project's stated goal is measuring the
gap between *what the model learned* and *what I intended*. Per-class F1 plus the
confusion matrix are precisely the tools that expose that gap — they tell me
whether the model learned "discourse quality" or just learned a shortcut like
"long comment = analysis."

---

## 6. Definition of success

**Genuinely useful (the bar I'd be proud of).** Each criterion below is a number I
can compute from the test-set confusion matrix and classification report, so at
the end I can mark each one hit/missed without judgment calls:
- **Macro-F1 ≥ 0.75**, and **every per-class F1 ≥ 0.65** (hard floor, no tilde).
- The fine-tuned model **beats the zero-shot baseline by ≥ 0.05 macro-F1** on the
  identical test set (a margin large enough to not be split noise on ~30 test items).
- Errors **concentrate on the adjacent boundaries** of §3: operationally,
  **≥ 70% of all misclassifications fall in the `analysis↔hot_take` or
  `hot_take↔reaction` off-diagonal cells**, and the `analysis↔reaction`
  confusions (the two non-adjacent classes) are **≤ 10% of all errors**. If the
  model frequently confuses `analysis` with `reaction`, it learned a shortcut, not
  the concept — that fails this criterion regardless of headline accuracy.

**"Good enough" for deployment in a real community tool:** the realistic bar is
lower, because the use case (e.g. surfacing high-quality `analysis` comments, or
softly flagging low-content noise) tolerates error if it's well-calibrated:
- **`analysis` precision ≥ 0.80** — when the tool says "this is quality analysis,"
  it should usually be right, since false promotion is the most damaging error.
- Reaction/hot_take confusion is **acceptable**, because both are "not analysis"
  and the downstream action is similar.
- The model ships with **confidence scores** so low-confidence predictions can be
  deferred to a human instead of acted on automatically.

**What I will *not* accept as success:** beating the baseline on raw accuracy
while failing per-class — that would mean the model learned the class prior, not
the concept. The honest assessment of *what the model learned vs. what I intended*
matters more than any single headline number.

---

## 7. AI Tool Plan

This is a data-and-evaluation project, not an implementation project, so AI tools
are not used to "write the code." They are used at the three points where a second
intelligence genuinely sharpens the work: **stress-testing the labels before
annotating, optionally pre-labeling data, and finding error patterns before I
write them up.** Every use is logged for the AI-usage disclosure in the README.

### 7.1 Label stress-testing (before annotation)
**Goal:** find holes in the label definitions *before* committing to 200 annotations.
**Plan:** give an LLM (Claude / the assignment's Groq model) the §2 definitions and
the §3 edge-case rules, and ask it to generate **8–10 r/nba-style comments that sit
exactly on the `analysis↔hot_take` and `hot_take↔reaction` boundaries** — the two
ambiguous axes. I then try to label each one using only my written rules.
- **If I can classify all of them cleanly** → the definitions are tight enough; proceed.
- **If any are genuinely unclassifiable** → that exposes an ambiguity, and I will
  **tighten the §2/§3 wording now** (e.g. sharpen the "specific enough to argue
  against" test) and re-run until the rules resolve the boundary cases.
- These generated posts are **synthetic and used only for definition-testing — they
  are never added to the 200-example dataset.**

### 7.2 Annotation assistance (pre-labeling)
**Decision: yes, with human review of every row.** I will use an LLM
(Groq `llama-3.3-70b-versatile`, the same model used as the eval baseline) to
**pre-label batches**, then personally review and correct every pre-label — the
final label is always my call, never accepted blindly.
- **Tracking for disclosure:** the dataset already carries a free-text `notes`
  column and a `comment_id`. I will record provenance per row (e.g. an
  `ai_prelabeled` flag plus whether I **agreed with** or **overrode** the model's
  suggestion), so the README can report exactly how many rows were AI-pre-labeled
  and how often I overrode the model.
- **Guardrail against contamination:** because the baseline model and the
  pre-labeler are the same model, I will **not** treat agreement between them as
  validation, and I'll note this shared-model caveat when interpreting the
  baseline's test-set numbers.

### 7.3 Failure-pattern analysis (after evaluation)
**Goal:** move from "here are 3 wrong predictions" to "here is a *systematic* failure mode."
**Plan:** export the full list of misclassified test examples (text, true label,
predicted label, confidence) and give it to an LLM, asking it to **propose
candidate patterns** in the errors.
- **What I'll look for:** sarcasm/irony misread as `analysis`; very short comments
  defaulting to `reaction`; stat-heavy `hot_take`s being over-credited as `analysis`;
  and whether errors cluster on the §3 boundaries as my success criteria predict.
- **Verification (mandatory):** any pattern the AI proposes is a **hypothesis, not a
  finding.** I will verify each one against the actual confusion matrix and by
  re-reading the cited examples — e.g. if it claims "short posts fail," I'll check
  error rate stratified by comment length. **Only patterns I can confirm in the data
  go into the evaluation report**, and unconfirmed AI suggestions are discarded.

> **Note:** This planning.md will be updated *before* starting any stretch feature
> (inter-annotator reliability, confidence calibration, deeper error analysis, or a
> deployed interface), so the plan always precedes the work.
