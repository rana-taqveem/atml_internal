"""Exercise evaluation orchestration without downloading or loading models."""
import json
from pathlib import Path
from contextlib import contextmanager
import shutil
import uuid
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from common.data import read_jsonl, repo_path
from task4_safety import evaluate_safety
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward


@contextmanager
def workspace_temp():
    # Python's Windows TemporaryDirectory mode=0700 can exclude the sandbox
    # identity. Use ordinary workspace permissions for these disposable fixtures.
    root = repo_path("tests").resolve()
    path = root / ("tmp_" + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield str(path)
    finally:
        if path.resolve().parent != root:
            raise RuntimeError("Fixture cleanup target escaped the tests directory")
        shutil.rmtree(path)


class EvaluationIntegrity(unittest.TestCase):
    def test_safety_rejects_incomplete_policy_set(self):
        with workspace_temp() as tmp:
            out = Path(tmp) / "task4_safety"
            out.mkdir()
            (out / "judged_sft.jsonl").write_text(json.dumps({"xstest_id": 0}) + "\n")
            with patch.object(evaluate_safety, "load_yaml", return_value={"results_dir": tmp}), patch("sys.argv", ["evaluate_safety"]):
                with self.assertRaisesRegex(ValueError, "missing policies"):
                    evaluate_safety.main()

    def test_full_safety_aggregation_and_denominators(self):
        with workspace_temp() as tmp:
            out = Path(tmp) / "task4_safety"
            out.mkdir()
            expected = Path(tmp) / "xstest.csv"
            pd.DataFrame({"xstest_id": [0, 1], "benchmark_class": ["SAFE", "UNSAFE"]}).to_csv(expected, index=False)
            for policy in evaluate_safety.POLICIES:
                records = [{"xstest_id": i, "policy": policy, "benchmark_class": cls, "prompt": str(i),
                            "response": "test", "type": "test", "response_tokens": 3, "ai_confidence": 1.0,
                            "ai_label": label} for i, cls, label in [(0, "SAFE", "SAFE_ANSWER"), (1, "UNSAFE", "JUSTIFIED_REFUSAL")]]
                (out / f"judged_{policy}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            cfg = {"results_dir": tmp, "paths": {"xstest": str(expected)}}
            with patch.object(evaluate_safety, "load_yaml", return_value=cfg), patch.object(evaluate_safety, "plot_categories"), patch("sys.argv", ["evaluate_safety"]):
                evaluate_safety.main()
            m = json.loads((out / "safety_metrics.json").read_text())["policies"]["sft"]
            self.assertEqual(m["safe_answer_rate"], {"rate": 1., "se": 0., "n": 1})
            self.assertEqual(m["unsafe_justified_refusal_rate"]["rate"], 1.)
            self.assertEqual(m["unsafe_compliance_rate"]["rate"], 0.)

    def test_judge_fallback_tie_is_distinct_from_valid_tie(self):
        class Tokenizer:
            eos_token_id = 0
            def apply_chat_template(self, *args, **kwargs):
                return torch.tensor([[1]])
            def decode(self, *args, **kwargs):
                return self.output
        class Model:
            def parameters(self):
                yield torch.zeros(1)
            def generate(self, *args, **kwargs):
                return torch.tensor([[1, 2]])
        with workspace_temp() as tmp:
            judge = PairwiseAIJudge.__new__(PairwiseAIJudge)
            judge.cfg, judge.cache, judge.details = {"ai_judge_model": "fake-fixed-judge"}, {}, {}
            judge.cache_path, judge.details_path = Path(tmp) / "cache.json", Path(tmp) / "details.json"
            judge.tokenizer, judge.model = Tokenizer(), Model()
            for response, ambiguous in [("not parseable", True), ("TIE", False)]:
                judge.tokenizer.output = response
                self.assertEqual(judge.compare(response, "a", "b"), "TIE")
                self.assertEqual(judge.details[judge._key(response, "a", "b")]["parse_ambiguous"], ambiguous)

    def test_verifier_matches_all_fixed_controlled_expectations(self):
        rr = read_jsonl(repo_path("data/task5_controlled_reward_diagnostics.jsonl"))
        self.assertEqual(len(rr), 100)
        for r in rr:
            self.assertEqual(exact_reward(r["response"], r["gold_final"]), r["expected_exact_reward"])


if __name__ == "__main__":
    unittest.main()
