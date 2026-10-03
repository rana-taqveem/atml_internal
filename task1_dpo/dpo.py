from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
):
    """Return scalar DPO loss plus lightweight diagnostics.

    L = -E[log sigma(beta * m)], where the preference margin is
    m = [log pi(y+|x) - log ref(y+|x)] - [log pi(y-|x) - log ref(y-|x)]
      = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected).

    Fix vs. the starter: the starter used `policy_margin + ref_margin`, which rewards the policy
    for the reference's own preference instead of measuring the change relative to the reference.
    With the sign fixed, m == 0 at initialization (policy == reference) and the loss is log(2).
    Preference accuracy also follows the manual definition (m > 0), not the raw policy margin.
    """
    policy_margin = policy_chosen_logp - policy_rejected_logp
    ref_margin = ref_chosen_logp - ref_rejected_logp
    margin = policy_margin - ref_margin
    logits = beta * margin

    loss = -F.logsigmoid(logits).mean()
    chosen_reward = beta * (policy_chosen_logp - ref_chosen_logp)
    rejected_reward = beta * (policy_rejected_logp - ref_rejected_logp)
    return loss, {
        "logit_mean": logits.detach().mean(),
        "margin_mean": margin.detach().mean(),
        "policy_margin_mean": policy_margin.detach().mean(),
        "chosen_reward_mean": chosen_reward.detach().mean(),
        "rejected_reward_mean": rejected_reward.detach().mean(),
        "preference_accuracy": (margin > 0).float().mean().detach(),
    }
