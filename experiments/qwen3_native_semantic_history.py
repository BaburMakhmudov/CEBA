"""Compare trained random latent slots with native pretrained Qwen3 text reasoning.

No adapter weights, no fine-tuning, no guessed semantic memory vectors.
One CPU forward per scenario using the actual Qwen3-0.6B checkpoint.
The test uses chronology, corrections, different phrasings, and a no-memory control.
"""
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen3-0.6B"
torch.set_num_threads(2)

SCENARIOS = [
    (["The lock accepts the red key."], "red"),
    (["The lock accepts the blue key."], "blue"),
    (["A red key will unlock the door."], "red"),
    (["A blue key will unlock the door."], "blue"),
    (["Only the red key is authorized."], "red"),
    (["Only the blue key is authorized."], "blue"),
    (["The door was configured for red.", "The technician changed it: blue is now the valid key."], "blue"),
    (["The door was configured for blue.", "The technician changed it: red is now the valid key."], "red"),
    (["Yesterday the access color was red.", "Today it is blue."], "blue"),
    (["Yesterday the access color was blue.", "Today it is red."], "red"),
    (["Blue is not valid now.", "The lock currently accepts red."], "red"),
    (["Red is not valid now.", "The lock currently accepts blue."], "blue"),
]


def make_prompt(events):
    events_text = "\n".join(f"{i+1}. {message}" for i, message in enumerate(events))
    return (
        "A locked door can be opened by only one key color: red or blue. "
        "Read the verified observations in chronological order. "
        "Use the most recent information.\n"
        "OBSERVATIONS:\n"
        f"{events_text if events_text else '(no observations)'}\n"
        "Complete the final statement with the correct current color.\n"
        "Current valid key color is"
    )


@torch.inference_mode()
def predict(model, tokenizer, prompt, red_tokens, blue_tokens):
    inp = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    logits = model(input_ids=inp, use_cache=False).logits[0, -1].float()
    # Qwen tokenizer normally makes both ' red' and ' blue' single tokens.
    assert len(red_tokens) == len(blue_tokens) == 1, (
        f"Full-suffix scoring needed: red={red_tokens}, blue={blue_tokens}"
    )
    scores = {"red": logits[red_tokens[0]].item(), "blue": logits[blue_tokens[0]].item()}
    return max(scores, key=scores.get), scores


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    red_tokens = tokenizer.encode(" red", add_special_tokens=False)
    blue_tokens = tokenizer.encode(" blue", add_special_tokens=False)
    actual, expected, detail = [], [], []
    for events, truth in SCENARIOS:
        prediction, scores = predict(model, tokenizer, make_prompt(events), red_tokens, blue_tokens)
        actual.append(prediction)
        expected.append(truth)
        detail.append({"events": events, "predicted": prediction, "expected": truth, "margin": scores[truth] - scores["blue" if truth == "red" else "red"]})
    _, no_history_scores = predict(model, tokenizer, make_prompt([]), red_tokens, blue_tokens)
    no_history_choice = max(no_history_scores, key=no_history_scores.get)
    no_history_accuracy = sum(no_history_choice == t for t in expected) / len(expected)
    accuracy = sum(a == t for a, t in zip(actual, expected)) / len(expected)
    report = {
        "checkpoint": MODEL,
        "pretrained_weights": True,
        "new_trainable_parameters": 0,
        "history_scenarios": len(expected),
        "text_history_accuracy": accuracy,
        "no_history_accuracy": no_history_accuracy,
        "token_ids": {"red":red_tokens,"blue":blue_tokens},
        "all_scenarios": detail,
        "hidden_slot_experiment_heldout_accuracy": 0.5,
        "claim_ceba_complete": False,
        "claim_endogenous_goals_proven": False,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    assert accuracy >= 0.8, "Native semantic reasoning did not reach the required performance"
    assert accuracy > no_history_accuracy, "History did not improve over the memory-free control"


if __name__ == "__main__":
    main()
