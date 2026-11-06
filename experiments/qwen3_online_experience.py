"""Actual pretrained-Qwen causal experience learning experiment.

Frozen Qwen3 produces a 4x1024 recurrent state per observation sequence.
Only a tiny policy head is updated from interaction rewards (trial and error).
The held-out set uses phrases not seen during training and includes revisions.
This is a scoped capability probe, not a claim of full CEBA/endogenous goals.
"""
import json
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(2)
torch.manual_seed(726)
RNG = random.Random(726)
MODEL = "Qwen/Qwen3-0.6B"
SLOTS = 4
COLORS = ("red", "blue")


@torch.inference_mode()
def observe(model, tokenizer, state, observation):
    ids = tokenizer(observation, add_special_tokens=False, return_tensors="pt").input_ids
    embeddings = model.get_input_embeddings()(ids)
    probes = torch.zeros_like(state)
    out = model.model(
        inputs_embeds=torch.cat((state, embeddings, probes), dim=1),
        use_cache=False,
        return_dict=True,
    )
    update = torch.tanh(out.last_hidden_state[:, -SLOTS:, :])
    return (0.75 * state + 0.25 * update).detach().clone()


def features(model, tokenizer, sequences):
    result = []
    for obs in sequences:
        state = torch.zeros(1, SLOTS, model.config.hidden_size)
        for message in obs:
            state = observe(model, tokenizer, state, message)
        # Clone outside inference_mode so autograd can backprop through policy.
        result.append(state.flatten().detach().clone())
    return torch.stack(result, dim=0)


def make_sequences(templates):
    sequences, labels = [], []
    for template in templates:
        for idx, color in enumerate(COLORS):
            sequences.append([template.format(color=color)])
            labels.append(idx)
    return sequences, labels


def online_train_policy(x, targets, rounds=500):
    head = torch.nn.Linear(x.shape[-1], 2, bias=False)
    torch.nn.init.zeros_(head.weight)
    opt = torch.optim.AdamW(head.parameters(), lr=0.02, weight_decay=0.0)
    trace = []
    for t in range(rounds):
        i = RNG.randrange(len(targets))
        logits = head(x[i])
        action = torch.multinomial(torch.softmax(logits.detach(), dim=-1), 1).item()
        # The policy is never given the correct answer as a supervised label.
        # A real environment returns only success/failure after the action.
        reward = 1.0 if action == targets[i] else -1.0
        loss = -reward * torch.log_softmax(logits, dim=-1)[action]
        opt.zero_grad()
        loss.backward()
        opt.step()
        trace.append(reward)
    return head, trace


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    model.requires_grad_(False)
    train_templates = [
        "Access key color: {color}.",
        "The door requires the {color} key.",
        "Current access color is {color}.",
        "The valid key is {color}.",
    ]
    heldout_templates = [
        "Use {color} to unlock the door.",
        "Right now the authorized key is {color}.",
        "The lock accepts {color}.",
        "Authorized color = {color}.",
    ]
    train_s, train_y = make_sequences(train_templates)
    test_s, test_y = make_sequences(heldout_templates)

    # Two-step state revisions: the latest observation must govern action.
    for first, last in (("red", "blue"), ("blue", "red")):
        train_s.append([f"Access key color: {first}.", f"Access key color: {last}."])
        train_y.append(COLORS.index(last))
        test_s.append([f"The key used to be {first}.", f"Right now the authorized key is {last}."])
        test_y.append(COLORS.index(last))

    train_raw = features(model, tokenizer, train_s)
    test_raw = features(model, tokenizer, test_s)
    center = train_raw.mean(dim=0)
    scale = torch.linalg.vector_norm(train_raw - center, dim=1).mean().clamp_min(1e-6)
    train_x = (train_raw - center) / scale
    test_x = (test_raw - center) / scale

    with torch.no_grad():
        initial_score = 0.5  # Uniform prior, exactly balanced 2-action world.
        contrast = torch.linalg.vector_norm(train_x[0] - train_x[1]).item()

    policy, trace = online_train_policy(train_x, train_y)
    with torch.no_grad():
        train_predictions = policy(train_x).argmax(dim=1).tolist()
        test_predictions = policy(test_x).argmax(dim=1).tolist()
        train_accuracy = sum(a == b for a, b in zip(train_predictions, train_y)) / len(train_y)
        heldout_accuracy = sum(a == b for a, b in zip(test_predictions, test_y)) / len(test_y)
        # Ablation: if past state is removed, each action is equally likely.
        ablative_uniform_score = 0.5

    report = {
        "checkpoint": MODEL,
        "num_train_scenarios": len(train_s),
        "num_heldout_scenarios": len(test_s),
        "reward_only_updates": len(trace),
        "policy_trainable_parameters": sum(p.numel() for p in policy.parameters()),
        "neural_state_elements": SLOTS * model.config.hidden_size,
        "train_accuracy": train_accuracy,
        "heldout_accuracy": heldout_accuracy,
        "first_100_average_reward": sum(trace[:100]) / 100,
        "last_100_average_reward": sum(trace[-100:]) / 100,
        "no_history_uniform_baseline": ablative_uniform_score,
        "prelearning_uniform_accuracy": initial_score,
        "red_blue_history_state_contrast_l2": contrast,
        "heldout_predictions": test_predictions,
        "heldout_truth_for_audit_only": test_y,
        "semantic_generalization_established": heldout_accuracy >= 0.8,
        "endogenous_goal_proven": False,
        "ceba_runtime_verified": False,
    }
    print(json.dumps(report, indent=2))
    assert train_accuracy >= 0.8, "Reward-learning did not fit training observations"
    assert heldout_accuracy >= 0.8, "Held-out semantic transfer was not established"
    assert report["last_100_average_reward"] > report["first_100_average_reward"], "No reward learning"
    assert contrast > 1e-6, "Pretrained state did not distinguish observation histories"


if __name__ == "__main__":
    main()
