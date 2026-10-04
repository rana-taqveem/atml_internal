from __future__ import annotations

import torch

from common.metrics import masked_mean, sampled_kl, sample_entropy


def group_relative_advantages(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float = 1e-6):
    """Return one scalar advantage per sampled completion.

    `group_ids[i]` identifies which prompt produced reward `rewards[i]`.

    A_k = (r_k - mean_g) / (std_g + eps), with mean/std taken *within* each prompt group g.

    Fix vs. the starter: the starter normalised with the mean/std of the whole batch, ignoring
    `group_ids`. That compares completions of *different* prompts (an easy prompt's average answer
    gets a positive advantage just because the prompt is easy) and gives non-zero advantages to a
    group whose completions all received the same reward.
    """
    adv = torch.zeros_like(rewards)
    for g in torch.unique(group_ids):
        idx = group_ids == g
        r = rewards[idx]
        adv[idx] = (r - r.mean()) / (r.std(unbiased=False) + eps)
    return adv


def group_reward_stats(rewards: torch.Tensor, group_ids: torch.Tensor, tol: float = 1e-6):
    """Per-group population std and whether the group is informative (std > tol)."""
    stds = torch.stack([rewards[group_ids == g].std(unbiased=False) for g in torch.unique(group_ids)])
    return stds, stds > tol


def grpo_policy_loss(
    new_logp,
    old_logp,
    seq_adv,
    token_mask,
    ref_logp,
    eps,
    beta,
    loss_type="grpo",
    max_completion_length: int | None = None,
):
    """PPO-style clipped GRPO loss for already-sampled completions.

    `token_mask` may be all-zero for a completion that was deliberately masked because it hit the
    maximum generation length.
    """
    ratio = torch.exp(new_logp - old_logp)
    adv = seq_adv[:, None]
    s1 = ratio * adv
    s2 = ratio.clamp(1.0 - eps, 1.0 + eps) * adv
    objective = torch.minimum(s1, s2)

    token_sum = (objective * token_mask).sum(-1)
    if loss_type == "grpo":
        denom = token_mask.sum(-1).clamp_min(1.0)
        per_sequence = token_sum / denom
        policy_term = -per_sequence.mean()
    elif loss_type == "dr_grpo":
        if max_completion_length is None:
            raise ValueError("dr_grpo requires max_completion_length")
        # Constant normalization rather than dividing by each response's realized length.
        per_sequence = token_sum / float(max_completion_length)
        policy_term = -per_sequence.mean()
    else:
        raise ValueError(f"Unknown loss_type={loss_type!r}")

    log_ratio_ref_over_policy = ref_logp - new_logp
    per_token_kl = torch.exp(log_ratio_ref_over_policy) - log_ratio_ref_over_policy - 1.0
    kl = masked_mean(per_token_kl, token_mask)
    loss = policy_term + float(beta) * kl
    affected = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    return loss, {
        "policy_term": policy_term.detach(),
        "sampled_kl": kl.detach(),
        "clip_fraction": masked_mean(affected, token_mask).detach(),
        "ratio_mean": masked_mean(ratio.detach(), token_mask),
        "sample_entropy": sample_entropy(new_logp.detach(), token_mask),
    }


def per_sequence_loss_terms(new_logp, old_logp, seq_adv, token_mask, ref_logp, eps, beta, loss_type="grpo",
                            max_completion_length: int | None = None, n_sequences: int | None = None,
                            total_tokens=None):
    """Split `grpo_policy_loss` into one additive term per completion.

    With `n_sequences`/`total_tokens` set to the full batch's values this can be called on a
    subset of rows; summing the terms of all rows equals grpo_policy_loss(...)[0], so
    backpropagating each term separately gives the gradient contributed by each completion
    (used for the length-normalisation study).
    """
    ratio = torch.exp(new_logp - old_logp)
    adv = seq_adv[:, None]
    objective = torch.minimum(ratio * adv, ratio.clamp(1.0 - eps, 1.0 + eps) * adv)
    token_sum = (objective * token_mask).sum(-1)
    n = token_mask.shape[0] if n_sequences is None else n_sequences
    if total_tokens is None:
        total_tokens = token_mask.sum()
    if loss_type == "grpo":
        policy = -(token_sum / token_mask.sum(-1).clamp_min(1.0)) / n
    elif loss_type == "dr_grpo":
        policy = -(token_sum / float(max_completion_length)) / n
    else:
        raise ValueError(f"Unknown loss_type={loss_type!r}")
    log_r = ref_logp - new_logp
    kl_sum = ((torch.exp(log_r) - log_r - 1.0) * token_mask).sum(-1)
    return policy + float(beta) * kl_sum / torch.as_tensor(total_tokens).clamp_min(1.0)


def mask_truncated_sequences(token_mask: torch.Tensor, truncated: list[bool] | torch.Tensor):
    truncated = torch.as_tensor(truncated, device=token_mask.device, dtype=torch.bool)
    keep = (~truncated).to(token_mask.dtype)[:, None]
    return token_mask * keep
