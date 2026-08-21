"""
steer.py

Goal: take the "sycophancy direction" implied by the probe results at the
best layer (layer 17, 74.3% -- see extract_and_train.py's layer sweep), and
use it to STEER Qwen3-0.6B away from caving under pressure, by subtracting
some multiple (alpha) of that direction from the residual stream during
generation.

This is the second half of the project's hypothesis: not just "is caving
linearly detectable" (answered: yes, 74.3% > the paper's 73.5% baseline),
but "can we edit it away without retraining."

Requires activations.npy, labels.npy, samples.json (already produced by
extract_and_train.py).
"""

import json

import numpy as np
import torch
from transformer_lens import HookedTransformer

from data.questions import QUESTION_BANK, build_prompt
from extract_and_train import DEVICE, MODEL_NAME, label_response

BEST_LAYER = 17  
ALPHAS = [0, 10, 20, 30, 40, 50, 75, 100]


def load_direction(method: str = "diff_of_means") -> np.ndarray:
    """
    Returns a single (d_model,) numpy vector: the "sycophancy direction" at
    BEST_LAYER, in the model's real 1024-dim activation space.

    Two ways to derive it -- worth implementing both eventually and
    comparing, since that's a nice extra finding for the write-up.

    method="diff_of_means" (start here -- simpler, no PCA involved):
        Load activations.npy (shape N x num_layers x d_model) and labels.npy
        (values 0/1). Slice out BEST_LAYER's activations -> shape (N, d_model).
        The direction is just:
            mean(activations where label==1) - mean(activations where label==0)
        This is the "Contrastive Activation Addition" approach (Rimsky et
        al., "Steering Llama 2 via Contrastive Activation Addition" -- short,
        readable paper, worth reading before you write this).

    method="probe_weights" (closer to the original paper's stated method,
    trickier -- do this second):
        train_probes() in extract_and_train.py fits logistic regression on
        PCA-reduced (64-dim) activations, not the raw 1024-dim residual
        stream. So the probe's coef_ lives in PCA space -- you can't
        subtract it from the real residual stream directly. You need to
        project it back through the PCA components first.

        Hint on the math (derive/verify this yourself, don't just trust me):
        sklearn's PCA transform is  X_pca = (X - pca.mean_) @ pca.components_.T
        where pca.components_ has shape (n_components, n_features).
        Work out what single matrix multiplication turns a weight vector
        defined in PCA space (shape (n_components,)) into an equivalent
        weight vector in the original (n_features,) space. Search
        "projecting a linear classifier back through PCA" if you get stuck.

        You'll need to actually retrain PCA + LogisticRegression on
        BEST_LAYER's activations here (train_probes() only returns
        accuracies, not the fitted objects) -- use the same random_state=42
        train/test split for consistency with your reported 74.3% number.

    TODO: implement. Consider normalizing the result to a unit vector, so
    that `alpha` alone controls steering strength independent of the raw
    magnitude of the direction you derive.
    """
    raise NotImplementedError


def make_steering_hook(direction: torch.Tensor, alpha: float):
    """
    Returns a function matching the signature TransformerLens hooks expect:
        def hook_fn(resid, hook):
            ...
            return modified_resid

    `resid` is the residual stream tensor at the hook point when this fires,
    shape (batch, seq_len, d_model). Subtract alpha * direction from every
    position and return it.

    This is the SAME hook point name you already used for extraction in
    generate_with_activations() -- f"blocks.{BEST_LAYER}.hook_resid_post".
    Extraction reads from it; steering writes to it. Look at TransformerLens's
    "Main Demo" notebook, hooks section, if the hook_fn signature is unclear.

    TODO: implement and return hook_fn.
    """
    raise NotImplementedError


def generate_steered(model: HookedTransformer, prompt: str, direction: torch.Tensor, alpha: float) -> str:
    """
    Like generate_only() in extract_and_train.py, but with the steering hook
    active for the duration of generation.

    TransformerLens lets you attach hooks temporarily with a context manager:

        with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
            ...call model.generate() in here...

    where hook_name is f"blocks.{BEST_LAYER}.hook_resid_post".

    TODO: implement. Sanity check once it runs: alpha=0 should produce
    output identical to an unsteered generate_only() call (subtracting
    zero times anything is a no-op) -- if it doesn't match, the hook is
    wired up wrong somewhere.
    """
    raise NotImplementedError


def evaluate_alpha(model: HookedTransformer, direction: np.ndarray, alpha: float, caved_samples: list[dict]) -> dict:
    """
    caved_samples: entries from samples.json where is_sycophantic was True
    under the ORIGINAL (unsteered) biased prompt. These are the only ones
    worth re-testing -- steering can't improve on questions that already
    held firm without it.

    For each one:
      1. Look the QAItem back up from QUESTION_BANK by its "id" field, and
         rebuild the biased prompt with build_prompt(item, biased=True).
      2. Regenerate it WITH steering at this alpha via generate_steered().
      3. Re-run label_response() on the new output to see if it now holds
         the correct answer.

    Return a dict summarizing this alpha's result, e.g.:
        {
            "alpha": alpha,
            "held_rate": fraction_that_now_answer_correctly,
            "n": len(caved_samples),
            "examples": [...a few full before/after transcripts for the
                          write-up and the demo's animation...],
        }

    TODO: implement.
    """
    raise NotImplementedError


def main():
    print("Loading steering direction...")
    direction = load_direction(method="diff_of_means")

    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype=torch.float32)
    model.eval()

    with open("samples.json") as f:
        samples = json.load(f)
    caved_samples = [s for s in samples if s["is_sycophantic"]]
    print(f"{len(caved_samples)} samples caved originally -- these are what steering needs to fix.")

    results = []
    for alpha in ALPHAS:
        r = evaluate_alpha(model, direction, alpha, caved_samples)
        print(f"alpha={alpha:>4}  held-correct rate={r['held_rate']:.1%}")
        results.append(r)

    with open("steering_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved steering_results.json")


if __name__ == "__main__":
    main()
