"""Reproducible state->prediction probe with genuine Qwen3-0.6B weights.

This intentionally tests a necessary neural mechanism, not CEBA semantics.
It fails if the real checkpoint cannot be downloaded or loaded.
"""
import json
import platform

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
SLOTS = 4
torch.set_num_threads(2)


@torch.inference_mode()
def updated_state(model, tokenizer, old_state, observation):
    ids = tokenizer(observation, return_tensors="pt", add_special_tokens=False).input_ids
    emb = model.get_input_embeddings()(ids)
    probes = torch.zeros_like(old_state)
    inputs = torch.cat([old_state, emb, probes], dim=1)
    out = model.model(inputs_embeds=inputs, use_cache=False, return_dict=True)
    update = torch.tanh(out.last_hidden_state[:, -SLOTS:, :])
    return (0.75 * old_state + 0.25 * update).detach().clone()


@torch.inference_mode()
def future_logits(model, tokenizer, state, question):
    ids = tokenizer(question, return_tensors="pt", add_special_tokens=False).input_ids
    emb = model.get_input_embeddings()(ids)
    out = model(inputs_embeds=torch.cat([state, emb], dim=1), use_cache=False, return_dict=True)
    return out.logits[0, -1, :].float()


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).eval()
    assert model.config.model_type == "qwen3"
    assert model.config.hidden_size == 1024
    d = model.config.hidden_size
    initial = torch.zeros(1, SLOTS, d, dtype=torch.float32)

    # The same question is preceded by different causal histories.
    red = updated_state(model, tokenizer, initial, "The access key is red.")
    blue = updated_state(model, tokenizer, initial, "The access key is blue.")
    red_replay = updated_state(model, tokenizer, initial, "The access key is red.")

    question = "The access key is"
    logits_initial = future_logits(model, tokenizer, initial, question)
    logits_red = future_logits(model, tokenizer, red, question)
    logits_blue = future_logits(model, tokenizer, blue, question)

    state_history_l2 = torch.linalg.vector_norm(red - blue).item()
    logits_history_l2 = torch.linalg.vector_norm(logits_red - logits_blue).item()
    logits_control_l2 = torch.linalg.vector_norm(logits_red - logits_initial).item()
    replay_exact = torch.equal(red, red_replay)

    # Both directions must work: history must affect the carried state AND
    # carried state must affect the frozen pretrained model's next-token logits.
    assert torch.isfinite(red).all() and torch.isfinite(blue).all()
    assert torch.isfinite(logits_red).all() and torch.isfinite(logits_blue).all()
    assert state_history_l2 > 1e-6, "Different histories produced identical states"
    assert logits_history_l2 > 1e-6, "History had no effect on future LM logits"
    assert logits_control_l2 > 1e-6, "Nonempty state had no effect on LM logits"
    assert replay_exact, "Identical replay did not reproduce the state bit-for-bit"

    result = {
        "status": "PASS",
        "checkpoint": MODEL_ID,
        "model_type": model.config.model_type,
        "num_layers": model.config.num_hidden_layers,
        "hidden_size": d,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "state_history_l2": state_history_l2,
        "future_logits_history_l2": logits_history_l2,
        "future_logits_control_l2": logits_control_l2,
        "replay_exact": replay_exact,
        "semantics_trained": False,
        "ceba_runtime_verified": False,
        "scope": "real pretrained neural state->future logits causal wiring",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
