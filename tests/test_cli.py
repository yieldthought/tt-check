from __future__ import annotations

import argparse
import unittest

from tt_check import cli


class CliHelpersTest(unittest.TestCase):
    def test_validate_runs(self) -> None:
        args = argparse.Namespace(
            runs=0,
            pcc_threshold=0.99,
            activation_width_per_device=1024,
            prefill_rows=1024,
            decode_rows=1,
            intermediate_multiplier=4,
        )

        with self.assertRaisesRegex(cli.CheckError, "--runs must be >= 1"):
            cli._validate_args(args)

    def test_system_summary_from_blackhole_snapshot(self) -> None:
        snapshot = {
            "device_info": [
                {
                    "arch": "blackhole",
                    "board_type": "p150b",
                    "board_number": "0000041100000000",
                }
            ]
        }

        summary = cli.summarize_system(snapshot)

        self.assertEqual(summary["architecture"], ["blackhole"])
        self.assertEqual(summary["board_types"], ["p150b"])
        self.assertEqual(summary["card_count"], 1)
        self.assertIn("single-card", summary["mesh_topology"])

    def test_human_result_mentions_trace(self) -> None:
        result = {
            "system": {
                "architecture": ["blackhole"],
                "device_series": ["p150b"],
                "card_count": 1,
                "mesh_topology": "single-card/non-mesh inferred from card count",
            },
            "mlp": [
                {"mode": "prefill", "runs": 100, "pcc": 0.9998160161456406},
                {"mode": "decode", "runs": 100, "pcc": 0.999871423156057},
            ],
        }

        text = cli._format_human_result(result)

        self.assertIn("trace x100", text)
        self.assertIn("ready.", text)


if __name__ == "__main__":
    unittest.main()
