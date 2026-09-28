from __future__ import annotations

import argparse
import contextlib
import io
import unittest
from unittest.mock import Mock, patch

from tt_check import cli


class CliHelpersTest(unittest.TestCase):
    def test_validate_runs(self) -> None:
        args = argparse.Namespace(
            runs=0,
            time_seconds=None,
            pcc_threshold=0.99,
            activation_width_per_device=1024,
            prefill_rows=1024,
            decode_rows=1,
            intermediate_multiplier=4,
        )

        with self.assertRaisesRegex(cli.CheckError, "--runs must be >= 1"):
            cli._validate_args(args)

    def test_run_limits(self) -> None:
        default = cli._parse_args([])
        self.assertEqual(default.runs, 100)
        self.assertIsNone(default.time_seconds)
        timed = cli._parse_args(["--time", "0.25"])
        cli._validate_args(timed)
        self.assertIsNone(timed.runs)
        self.assertEqual(timed.time_seconds, 0.25)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli._parse_args(["--runs", "2", "--time", "1"])

    def test_invalid_time(self) -> None:
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value), self.assertRaisesRegex(cli.CheckError, "--time"):
                cli._validate_args(cli._parse_args(["--time", value]))

    def test_rows_validation(self) -> None:
        for mode in ("prefill", "decode"):
            cli._validate_args(cli._parse_args([f"--{mode}-rows", "0"]))
            with self.assertRaisesRegex(cli.CheckError, "must be >= 0"):
                cli._validate_args(cli._parse_args([f"--{mode}-rows", "-1"]))
        with self.assertRaisesRegex(cli.CheckError, "at least one"):
            cli._validate_args(cli._parse_args(["--prefill-rows", "0", "--decode-rows", "0"]))

    def test_skipped_modes_are_not_run(self) -> None:
        for prefill_rows, decode_rows, modes in ((1024, 0, ["prefill"]), (0, 1, ["decode"]), (1024, 1, ["prefill", "decode"])):
            with self.subTest(modes=modes):
                fake_ttnn = Mock()
                context = cli.TtnnDeviceContext(object(), 1, False, None)
                with (
                    patch.dict("sys.modules", {"torch": Mock(), "ttnn": fake_ttnn}),
                    patch.object(cli, "_open_ttnn_device_context", return_value=context),
                    patch.object(cli, "_run_mlp_mode", side_effect=lambda **kw: {"mode": kw["mode"]}) as run,
                ):
                    results = cli.run_ttnn_mlp_check(
                        device_id=0, runs=None, time_seconds=2.5, pcc_threshold=0.99,
                        activation_width=1024, prefill_rows=prefill_rows, decode_rows=decode_rows,
                        intermediate_multiplier=4, seed=0,
                    )
                self.assertEqual([r["mode"] for r in results], modes)
                self.assertTrue(all(c.kwargs["time_seconds"] == 2.5 for c in run.call_args_list))
                fake_ttnn.close_device.assert_called_once_with(context.device)

    def test_fixed_replay_count_does_not_use_deadline(self) -> None:
        with patch.object(cli.time, "perf_counter", side_effect=AssertionError("unexpected clock read")):
            self.assertEqual(list(cli._replay_indices(3, None, 0)), [0, 1, 2])

    def test_timed_replays_stop_at_deadline(self) -> None:
        with patch.object(cli.time, "perf_counter", side_effect=[100.1, 100.9, 101.0]):
            self.assertEqual(list(cli._replay_indices(None, 1, 100)), [0, 1, 2])

    def test_short_time_still_completes_one_replay(self) -> None:
        with patch.object(cli.time, "perf_counter", return_value=102):
            self.assertEqual(list(cli._replay_indices(None, 0.001, 100)), [0])

    def test_timed_progress_preserves_fractional_seconds(self) -> None:
        with patch.object(cli._ProgressRenderer, "_load_tqdm") as load:
            renderer = cli._ProgressRenderer(enabled=True)
            renderer.handle({"event": "bar_start", "label": "prefill mlp", "total": 0.5, "unit": "s"})
            renderer.handle({"event": "bar_update", "advance": 0.25})
            self.assertEqual(load.return_value.call_args.kwargs["total"], 0.5)
            self.assertEqual(load.return_value.call_args.kwargs["unit"], "s")
            load.return_value.return_value.update.assert_called_once_with(0.25)

    def test_human_result_with_only_one_mode(self) -> None:
        for mode in ("prefill", "decode"):
            result = {"elapsed_s": 2.5, "mlp": [{"mode": mode, "pcc": 0.9999}]}
            self.assertEqual(cli._format_human_result(result), f"tt-check: passed in 2.5 seconds | {mode} pcc 0.99990000")

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
