"""Test a cheap verified-state interface to real pretrained Qwen3 (zero new weights).

World facts arrive as typed CanonicalInput-like events. The reducer controls
versioning and temporal replacement; Qwen sees only the currently committed
facts and suggests a candidate action. Never confuse this limited benchmark
with proof of semantic learning, endogenous goals, or the complete CEBA runtime.
"""
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(2)
MODEL_ID = "Qwen/Qwen3-0.6B"

HISTORIES = [
    ["red"],
    ["blue"],
    ["red", "blue"],
    ["blue", "red"],
    ["red", "red"],
    ["blue", "blue"],
    ["blue", "red", "blue"],
    ["red", "blue", "red"],
    ["red", "red", "blue", "red"],
    ["blue", "blue", "red", "blue"],
    ["red", "blue", "blue", "blue"],
    ["blue", "red", "red", "red"],
]


class VersionedFacts:
    def __init__(self):
        self.version = 0
        self.color = None
        self.events = []

    def commit(self, event, expected_version):
        if expected_version != self.version:
            raise ValueError("Optimistic concurrency conflict")
        if event["sequence"] != self.version + 1:
            raise ValueError("Out-of-order observation")
        if event["key_color"] not in ("red", "blue"):
            raise ValueError("Invalid typed observation")
        before = self.color
        self.events.append(event.copy())
        self.color = event["key_color"]
        self.version += 1
        return {"version": self.version, "changed": before != self.color}

    def replay(self):
        out = VersionedFacts()
        for event in self.events:
            out.commit(event, out.version)
        return out


@torch.inference_mode()
def proposed_action(model, tokenizer, color, red_token, blue_token):
    prompt = (
        "Use the following verified current-state record to select the key. "
        "There is exactly one authorized key.\n"
        f"CURRENT_STATE: valid_key_color = {color}\n"
        "RULE: choose the authorized key from CURRENT_STATE.\n"
        "The correct key color is"
    )
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
    logits = model(ids, use_cache=False).logits[0, -1]
    scores = {"red": logits[red_token].item(), "blue": logits[blue_token].item()}
    return max(scores, key=scores.get), scores


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).eval()
    ids = {c: tokenizer.encode(" " + c, add_special_tokens=False) for c in ("red", "blue")}
    assert all(len(x) == 1 for x in ids.values())
    decisions = []
    for history in HISTORIES:
        state = VersionedFacts()
        for seq, color in enumerate(history, start=1):
            state.commit({"sequence": seq, "key_color": color}, state.version)
        try:
            state.commit({"sequence": state.version + 1, "key_color": "red"}, state.version - 1)
            raise AssertionError("Version mismatch accepted")
        except ValueError:
            pass
        restored = state.replay()
        assert restored.version == state.version and restored.color == state.color
        predicted, scores = proposed_action(model, tokenizer, state.color, ids["red"][0], ids["blue"][0])
        decisions.append({
            "history": history, "committed_version": state.version,
            "committed_state": state.color, "model_proposal": predicted,
            "verified_action": predicted if predicted == state.color else state.color,
            "safety_gate_corrected": predicted != state.color,
            "model_proposal_correct": predicted == state.color,
            "margin_for_truth": scores[state.color] - scores["blue" if state.color == "red" else "red"],
            "replay": "exact",
        })
    model_accuracy = sum(s["model_proposal_correct"] for s in decisions) / len(decisions)
    gate_accuracy = sum(s["verified_action"] == s["committed_state"] for s in decisions) / len(decisions)
    result = {
        "checkpoint": MODEL_ID,
        "actual_pretrained_weights": True,
        "new_trainable_parameters": 0,
        "scenario_count": len(decisions),
        "replay_exact_count": len(decisions),
        "candidate_action_accuracy": model_accuracy,
        "post_validation_action_accuracy": gate_accuracy,
        "corrected_by_gate": sum(s["safety_gate_corrected"] for s in decisions),
        "scope": "typed event -> versioned committed state -> real transformer proposal -> independent validator",
        "full_ceba_runtime_verified": False,
        "learned_endogenous_goals": False,
        "scenarios": decisions,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    assert model_accuracy >= 0.9, "Even with verified typed state Qwen was unreliable"
    assert gate_accuracy == 1.0, "Correctness gate failed"
    assert result["replay_exact_count"] == len(decisions)


if __name__ == "__main__":
    main()
