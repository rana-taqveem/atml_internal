"""Small mathematical counterexamples; no downloaded models or GPU required."""
import math
import unittest

import torch

from common.evidence import validate_cached_generation, validate_ids
from task1_dpo.dpo import dpo_loss
from task2_ppo.ppo import clip_diagnostics, compute_gae, ppo_policy_loss, shaped_rewards
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences, per_sequence_loss_terms
from task5_feedback.rlvr import exact_reward


class Objectives(unittest.TestCase):
    def test_dpo_reference_identity_and_preference_direction(self):
        c, r = torch.tensor([-3.0]), torch.tensor([-5.0])
        loss, stats = dpo_loss(c, r, c, r, 0.1)
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        self.assertEqual(stats["preference_accuracy"].item(), 0)
        new = c.clone().requires_grad_()
        improved, _ = dpo_loss(new + 1, r, c, r, 0.1)
        self.assertLess(improved.item(), loss.item())
        improved.backward()
        self.assertLess(new.grad.item(), 0)

    def test_ppo_clipping_sign_and_masked_gradient(self):
        new = torch.tensor([[math.log(1.5), math.log(0.5), math.log(1.5), math.log(0.5), 10.]], requires_grad=True)
        old, adv, mask = torch.zeros_like(new), torch.tensor([[1., -1., -1., 1., 1.]]), torch.tensor([[1., 1., 1., 1., 0.]])
        loss, ratio, fraction = ppo_policy_loss(new, old, adv, mask, 0.2)
        self.assertAlmostEqual(loss.item(), 0.15, places=6)
        self.assertEqual(fraction.item(), 1)
        self.assertEqual(clip_diagnostics(ratio, adv, mask, .2)["affected_fraction"], .5)
        loss.backward()
        self.assertEqual(new.grad[0, 0].item(), 0)
        self.assertEqual(new.grad[0, 1].item(), 0)
        self.assertEqual(new.grad[0, 4].item(), 0)
        self.assertNotEqual(new.grad[0, 2].item(), 0)

    def test_gae_terminal_and_padding(self):
        rewards, values, mask = torch.tensor([[0., 2., 999.]]), torch.tensor([[.5, 1., 999.]]), torch.tensor([[1., 1., 0.]])
        advantage, _ = compute_gae(rewards, values, mask, gamma=1, lam=1)
        torch.testing.assert_close(advantage, torch.tensor([[1.5, 1., 0.]]))
        shaped = shaped_rewards(torch.tensor([2.]), torch.ones_like(mask), torch.zeros_like(mask), mask, .1)
        torch.testing.assert_close(shaped, torch.tensor([[-.1, 1.9, 0.]]))

    def test_grpo_groups_do_not_share_baselines(self):
        rewards = torch.tensor([1., 3., 100., 100.])
        adv = group_relative_advantages(rewards, torch.tensor([0, 0, 1, 1]))
        torch.testing.assert_close(adv, torch.tensor([-1., 1., 0., 0.]), atol=2e-6, rtol=0)

    def test_grpo_decomposition_and_length_normalizer_gradients(self):
        mask = torch.tensor([[1., 0., 0., 0.], [1., 1., 1., 1.], [0., 0., 0., 0.]])
        adv = torch.tensor([1., -1., 2.])
        for kind in ["grpo", "dr_grpo"]:
            new = torch.zeros_like(mask, requires_grad=True)
            old, ref = torch.zeros_like(mask), torch.full_like(mask, -.2)
            loss, _ = grpo_policy_loss(new, old, adv, mask, ref, .2, .1, kind, 4)
            terms = per_sequence_loss_terms(new, old, adv, mask, ref, .2, .1, kind, 4)
            torch.testing.assert_close(terms.sum(), loss)
            loss.backward()
            full_grad = new.grad.clone()
            new.grad = None
            terms = per_sequence_loss_terms(new, old, adv, mask, ref, .2, .1, kind, 4)
            terms.sum().backward()
            torch.testing.assert_close(new.grad, full_grad)
            self.assertEqual(full_grad[2].abs().sum().item(), 0)
        kept = mask_truncated_sequences(torch.ones(2, 4), [False, True])
        self.assertEqual(kept[1].sum().item(), 0)

    def test_verifier_uses_last_designated_final(self):
        self.assertEqual(exact_reward("Gold might be 42; #### 7", "42"), 0)
        self.assertEqual(exact_reward("#### 42 then corrected: #### 7", "42"), 0)
        self.assertEqual(exact_reward("#### 1,234", "1234"), 1)

    def test_partial_or_stale_generation_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_ids([{"id": "a"}], ["a", "b"], "id", context="test")
        with self.assertRaises(ValueError):
            validate_ids([{"id": "a"}, {"id": "a"}], ["a", "b"], "id", context="test")
        with self.assertRaises(ValueError):
            validate_cached_generation([{"id": "a", "generation_fingerprint": "old"}], ["a"], "id", "new", context="test")


if __name__ == "__main__":
    unittest.main()
