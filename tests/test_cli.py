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
                {
                    "mode": "prefill",
                    "runs": 100,
                    "pcc": 0.9998160161456406,
                    "tensor_parallel_degree": 2,
                    "mesh_shape": [1, 2],
                    "ccl": "all_reduce",
                },
                {
                    "mode": "decode",
                    "runs": 100,
                    "pcc": 0.999871423156057,
                    "tensor_parallel_degree": 2,
                    "mesh_shape": [1, 2],
                    "ccl": "all_reduce",
                },
            ],
        }

        text = cli._format_human_result(result)

        self.assertEqual(
            text,
            "tt-check: passed | prefill pcc 0.99981602 | decode pcc 0.99987142",
        )
        self.assertTrue(text.startswith("tt-check: passed"))

    def test_runtime_system_summary_mentions_mesh_shape(self) -> None:
        system = {
            "architecture": ["blackhole"],
            "device_series": ["p300a"],
            "card_count": 1,
            "mesh_topology": "single-card/non-mesh inferred from card count",
        }

        text = cli._format_runtime_system_summary(system, (1, 2))

        self.assertEqual(text, "1x2 mesh (1x p300a | blackhole)")

    def test_multi_device_open_lets_ttnn_choose_placement(self) -> None:
        class FakeMeshShape:
            def __init__(self, *shape: int) -> None:
                self.shape = shape

        class FakeTtnn:
            MeshShape = FakeMeshShape

            class FabricConfig:
                FABRIC_1D = "fabric-1d"

            def __init__(self) -> None:
                self.fabric_config = None
                self.open_mesh_kwargs = None

            def get_num_devices(self) -> int:
                return 4

            def set_fabric_config(self, fabric_config: str) -> None:
                self.fabric_config = fabric_config

            def open_mesh_device(self, **kwargs: object) -> object:
                self.open_mesh_kwargs = kwargs
                return object()

        fake_ttnn = FakeTtnn()

        context = cli._open_ttnn_device_context(fake_ttnn, device_id=0)

        self.assertTrue(context.is_mesh)
        self.assertEqual(context.mesh_shape, (1, 4))
        self.assertEqual(context.tensor_parallel_degree, 4)
        self.assertEqual(fake_ttnn.fabric_config, fake_ttnn.FabricConfig.FABRIC_1D)
        self.assertIsNotNone(fake_ttnn.open_mesh_kwargs)
        self.assertEqual(fake_ttnn.open_mesh_kwargs["mesh_shape"].shape, (1, 4))
        self.assertEqual(fake_ttnn.open_mesh_kwargs["trace_region_size"], 0)
        self.assertNotIn("physical_device_ids", fake_ttnn.open_mesh_kwargs)

    def test_format_failure_keeps_error_context(self) -> None:
        stderr = "\n".join(
            [
                "debug noise",
                "ERROR: running TTNN MLP readiness check failed: prefill tensor-parallel MLP failed: RuntimeError: boom",
                "",
                "Traceback:",
                "  File \"cli.py\", line 1, in _run_mlp_mode",
                "RuntimeError: boom",
            ]
        )

        text = cli._format_failure("", stderr)

        self.assertIn("prefill tensor-parallel MLP failed", text)
        self.assertIn("Traceback:", text)
        self.assertIn("RuntimeError: boom", text)


if __name__ == "__main__":
    unittest.main()
