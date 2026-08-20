'''
Hypythesis: GPT-2 will be highly confident (high probibility) on its 
            next word guess for well-known facts, and much less confident
            on obsucure or trick questions

'''

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # some ai generated bs this line is

from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2")

prompts = ["The capital of Dominican Republic is", 
           "Water boils at a temperature of",
           "The 8th president of the United States was",
           "The largest mammal on Earth is",
           "The capital of Australia is",
           ]


with open("output.txt", "w") as f:
    for prompt in prompts:
        logits = model(prompt)
        probs = logits[0, -1].softmax(dim=-1)
        top_prob, top_token_id = probs.max(dim=-1)
        f.write(f"{prompt!r} -> {model.to_string(top_token_id)!r} (probability: {top_prob.item():.4f})\n")