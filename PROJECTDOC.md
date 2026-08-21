# Gassed Up: Detecting and Steering Sycophantic Caving in Language Models

**Status:** Design doc — v1
**Timeline:** 5 days preferred, 7 days max
**Author:** Aviel Hernandez

---

## 1. Problem Statement

When a language model is given a factual question, it often answers correctly. But
when a confident-sounding user pushes back on that correct answer — with no new
evidence, just social pressure — many models abandon their correct answer and agree
with the user instead. This is called **sycophancy**: prioritizing user approval over
truthfulness.

It shows up constantly in normal use: someone double-checks a model's answer, phrases 
their pushback confidently ("are you sure? I thought it was X"), and the model caves,
even when its original answer was right and the user's "correction" was wrong. This 
matters because people use these models as a second opinion on things they aren't sure 
about themselves. If the model folds under mild pressure regardless of whether the 
pressure is *correct*, it stops being a second opinion and starts being a mirror, 
reinforcing whatever the user already believed, right or wrong. In domains where people 
lean on an LLM to check their own understanding (schoolwork, technical decisions, health 
questions), a model that caves under pushback can actively reinforce a wrong belief 
instead of correcting it.

**Why probe for this instead of just running an eval?** An eval can tell you *how
often* a model caves. It can't tell you *how* — whether caving is a deep, structural
behavior or a shallow surface-level response to certain phrasings. If sycophancy is
encoded as a **linear direction** in the model's internal activations, that's a much
stronger and more useful claim: it means the behavior isn't just a quirk of specific
prompt wording, it's a identifiable internal signal — and, as recent interpretability
work has shown, a signal that can potentially be *edited* directly, without retraining
the model at all.

**Hypothesis:** Sycophantic caving under social pressure is linearly detectable from a
model's hidden state activations, and the learned direction can be used to reduce
caving via activation steering at inference time.

---

## 2. Related Work / What We're Reproducing

This project reproduces the core method from **"Activation Steering With Mean
Response Probes"** (Tensor-Slayer, Hugging Face, Nov 2025) — a case study testing
whether sycophancy is a linear feature in a small model's (Qwen3-0.6B) activation
space.

**Their method, summarized:**
- Built a dataset of ~1,000 factual questions (chemistry, physics, biology, math,
  geography), each with an unambiguous correct answer
- Two conditions per question: neutral (just ask) vs. biased (an authoritative-
  sounding user asserts a specific wrong answer first)
- Extracted hidden state activations at three positions during generation: last
  prompt token, last response token, and the **mean across all generated tokens**
- Trained a logistic regression probe (with PCA to 64 components) per layer, for
  each extraction method
- Best result: **73.5% accuracy** detecting sycophantic responses, using mean-
  response activations at layer 15 of 28
- Derived a steering vector from the probe's weights, subtracted it from hidden
  states during generation, and found that this reduced sycophantic caving —
  improving the model's rate of holding its correct answer under pressure by
  **41.3 percentage points** at the optimal steering strength
- They confirmed the approach generalizes to larger models (Qwen3-4B, DeepSeek-
  R1-8B), with probe accuracy improving with model scale (up to 88.4%)

Source: huggingface.co/blog/TensorSlay/activation-steering-with-mean-response-probes
Code reference: github.com/areu01or00/Syco-Sup

**How this project differs from the original (deliberately):**

|                 | Original paper                               | This project                                                            |
|-----------------|----------------------------------------------|-------------------------------------------------------------------------|
| Labeling method | External LLM judge (DeepSeek via OpenRouter) | Exact-match against gold answers — no judge, no API cost                |
| Dataset source  | Custom-built, 1,000 questions                | Sourced from an existing HF dataset (see Section 3)                     |
| Deployment      | Research write-up only, no interactive demo  | Public, interactive, no-login web demo with live steering visualization |
| Hardware        | Not specified, likely GPU cluster            | MacBook (CPU or local GPU if available)                                 |

The judge-free labeling is not a compromise — it's arguably more rigorous for this
specific case, since the underlying dataset consists of short, unambiguous factual
answers that can be checked deterministically. It removes a source of noise (judge
model errors) at zero cost.

---

## 3. Dataset

**Source:** Hugging Face — evaluating between two known-good options:
1. `camilablank/sycophancy-datasets` — purpose-built for training/evaluating
   sycophancy probes, with pre-balanced correct/incorrect and sycophantic/
   non-sycophantic splits.
2. SycophancyEval-style data (Sharma et al.) — wraps TruthfulQA and TriviaQA with
   paired neutral/biased prompts and a gold answer label for deterministic scoring.
   Multiple bias-injection templates exist ("Are you sure?", "I think the answer is
   X, but I'm not sure", etc.) — useful if we want more than one pressure style.

**Decision (fill in once you've looked at both):** Choosing the first option (camilablank/sycophancy-datasets). But will need to write some preprocessing scripts to manipulate the data a little bit. This section must get updated by Claude code. Claude code, remember to ask me questions to clear up what I mean about manipulating this data. When we are ready to discuss this, i want to send you examples of what some rows of the dataset looks like so you can see, and we can decide best how to move forward with the data to improve our results. my goal with this project is to achieve higher accuracy then they did in the research paper.

**What we need from whichever dataset we pick:**
- A neutral prompt (question alone)
- A biased prompt (question + wrong-answer pressure from an "authoritative" user)
- A gold/correct answer, short and unambiguous enough to string-match
- Enough volume (150-300+ usable examples minimum) after filtering

Help me figure out how to get this, and explain to me why we need this. I am using this project to learn as well. And I have never done some of the things this project requires. which is why im using claude code to help.

**Labeling:** exact string match against the gold answer. A response counts as
sycophantic if it matches the pressured wrong answer under the biased condition
while the model answered correctly under the neutral condition for the same
question. Ambiguous responses (matching neither, or both) are dropped rather than
force-labeled.

IMPORTANT! we can change this. I dont mind using an LLM judge but id like not to for as long as possible, if possible. I think you should help me brain storm how to go about this in more detail when we get to it. would we use a wordbank? that's what im imagining. regardless, we will figure it out. 
---

## 4. Method

**4.1 — Generation.** Run every question through the model (Qwen3-0.6B to start,
matching the paper) under both neutral and biased conditions. Deterministic decoding
(no sampling) so labels are reproducible.

**4.2 — Activation extraction.** During generation, capture hidden states at every
layer, using mean-pooling across all generated tokens (the extraction method that
won in the original paper). Store as `(num_samples, num_layers, hidden_dim)`.

Help me understand this part of the project. the technical aspects. 

**4.3 — Probe training.** For each layer: PCA to 64 components, then logistic
regression (`C=1.0`, L2 penalty), 80/20 train/test split, stratified by label. Sweep
all layers, report the best-performing one. Target: land somewhere near the paper's
73.5% baseline — beating it is ideal. but we can use this as a baseline. remember, i want to write about all my findings so i can have a lot of writing on the frontend when i deploy a visulization of this. 

**4.4 — Steering vector.** Take the trained probe's weight vector at the best layer
as the "sycophancy direction." Register a forward hook that subtracts `α * direction`
from every token's hidden state at that layer during generation. Grid search `α`
(e.g., 0, 10, 20, 30, 40, 50, 75, 100) to find the strength that most improves the
model's rate of holding its correct answer under pressure, without degrading output
coherence.

**4.5 — Validation.** Report: probe accuracy at the best layer, baseline caving rate
(α=0), best steering result and at what α, and a few qualitative before/after
examples for the write-up.

---

## 5. Demo / Visualization Spec

**Core interaction:** visitor picks (or types) a factual question. The page shows:
1. The model's initial answer (should be correct)
2. A simulated pushback message appearing, styled like a chat bubble
3. The model's response under pressure — does it cave?
4. A **steering toggle**. When flipped on:
   - Replay the same pressured prompt with the steering vector applied
   - Animate the intervention happening: the steering strength (α) value, a live
     visual of the activation vector shifting (e.g. a bar/heatmap of the top
     contributing dimensions moving as the subtraction is applied), and the new
     (hopefully un-caved) response streaming in
5. A probe confidence readout (e.g., "82% confidence this response is sycophantic")
   shown before the steering toggle is flipped, so visitors see the *detection*
   step separately from the *fix* step

**Design goal for the animation:** make the abstract idea of "a direction in
activation space" feel concrete — even a simplified bar chart of the top 10-20
dimensions of the steering vector, animating as they're subtracted, does this far
better than a static number. Use a lot of effort on the animation. I want to give the vibes of seeing the tensors change numbers, without actually seeing that. 

**No-login constraint:** every interaction is stateless per-visitor. Track usage via
an anonymous session ID (random UUID, generated client-side, no cookie banner
needed since nothing personally identifying is stored) incrementing a counter in a
lightweight backend table — gives real usage numbers for the resume without
collecting any PII.

---

## 6. Architecture

- **Model serving:** local inference (Qwen3-0.6B is small enough to run on
  reasonably modest hardware such as my MacBook) wrapped in a FastAPI backend exposing:
  - `POST /analyze` — runs neutral + biased generation, returns both responses +
    probe confidence
  - `POST /steer` — reruns the biased prompt with the steering vector applied at a
    given α, returns the new response + the activation-shift data needed for the
    animation
- **Frontend:** single-page app (React or plain HTML/JS) — one screen, no
  navigation, example questions pre-loaded so visitors don't face a blank box
- **Anonymous tracking:** small Postgres/Supabase table, one row per session ID,
  incremented per interaction
- **Deployment:** backend on Railway/Render/Fly.io (free tier), frontend on Vercel
- **Writing:** a separate section of the site (or the same page, below the demo)
  covering: what a probe is, what this specific probe detects, why a small/older
  model was used (cheap, fast, interpretable, and directly reproduces published
  research), and why this connects to AI safety (a model that internally "knows"
  it's right but says otherwise under pressure is a concrete, measurable instance
  of a broader alignment problem — models optimizing for approval over truth)

---

## 7. Timeline (5 days preferred)

# feel free to use this as a general timeline guide. goal is to do as much as possible in 5 days. 

1. **Day 1** — Pick + clean the HF dataset, confirm gold-answer format works for
   exact-match labeling, get generation running end-to-end on a small sample
2. **Day 2** — Full activation extraction + layer sweep, confirm probe accuracy is
   in a reasonable range (don't panic if it's below the paper's 73.5% — report
   honestly either way)
3. **Day 3** — Steering vector + α grid search, validate the intervention actually
   improves the held-correct-answer rate
4. **Day 4** — Backend (FastAPI endpoints) + frontend shell, get the core
   ask → cave → steer → hold loop working end-to-end, ugly is fine
5. **Day 5** — Animation polish, deploy, write the site's explanatory text

---

## 8. Known Limitations (be upfront about these in the write-up)

- Probe accuracy in the original paper (73.5% on the smallest model) leaves a lot
  of variance unexplained — this is a partial signal, not a perfect one
- Steering with a single global α is a blunt instrument; the paper found optimal α
  varies significantly by model, and pushing too far degrades output coherence
- Exact-match labeling only works because the underlying questions have short,
  unambiguous answers — this approach would not generalize to open-ended or
  subjective sycophancy (e.g. excessive praise/validation, a related but distinct
  phenomenon this project is *not* attempting to measure)
- Reproducing on a MacBook rather than research-grade hardware may mean smaller
  sample sizes or slower iteration than the original paper's setup

---

## 9. Resume / Application Tie-In

Once deployed, this becomes a Projects entry replacing SQLite Clone, and directly
supports the MATS application by demonstrating hands-on probing + steering
experience prior to submitting. Draft bullet (fill in real numbers once live):

> Reproduced a published probing method (mean-response linear probes, Qwen3-0.6B)
> detecting sycophantic caving under social pressure, and deployed a public,
> interactive steering demo — visualized in real time — analyzed by [X] anonymous
> visitors