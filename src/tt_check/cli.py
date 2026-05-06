from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CheckError(RuntimeError):
    """Expected readiness-check failure."""


@dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _parse_args(raw_argv)
    if args._worker_json:
        return _main_worker(args)
    return _main_parent(raw_argv, args)


def _main_worker(args: argparse.Namespace) -> int:
    try:
        result = run_check(args)
        Path(args._worker_json).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    except CheckError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("ERROR: interrupted", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: unexpected failure: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


def _main_parent(raw_argv: list[str], args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="tt-check-") as tmp_dir:
        result_path = Path(tmp_dir) / "result.json"
        stdout_path = Path(tmp_dir) / "stdout.txt"
        stderr_path = Path(tmp_dir) / "stderr.txt"
        worker_argv = [*raw_argv, "--_worker-json", str(result_path)]
        env = os.environ.copy()
        env.setdefault("TT_LOGGER_LEVEL", "FATAL")
        env.setdefault("LOGURU_LEVEL", "ERROR")

        with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_file:
            process = subprocess.Popen(
                [sys.executable, "-m", "tt_check.cli", *worker_argv],
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                env=env,
            )
            try:
                _wait_with_progress(process, enabled=not args.json)
            except KeyboardInterrupt:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                print("ERROR: interrupted", file=sys.stderr)
                return 1

        stdout = _read_text(stdout_path)
        stderr = _read_text(stderr_path)
        if process.returncode != 0:
            print(_format_failure(stdout, stderr), file=sys.stderr)
            return 1
        if not result_path.exists():
            print(_format_failure(stdout, stderr, fallback="ERROR: tt-check did not produce a result"), file=sys.stderr)
            return 1

        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"ERROR: invalid result JSON: {exc}", file=sys.stderr)
            return 1

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(_format_human_result(result))
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small TTNN readiness check.")
    parser.add_argument("--device-id", type=int, default=0, help="TTNN device id to open.")
    parser.add_argument("--runs", type=int, default=100, help="Number of identical MLP runs per mode.")
    parser.add_argument("--pcc-threshold", type=float, default=0.99, help="Minimum PCC against PyTorch reference.")
    parser.add_argument(
        "--activation-width-per-device",
        type=int,
        default=1024,
        help="Activation shard width used for the simulated tensor-parallel MLP.",
    )
    parser.add_argument("--prefill-rows", type=int, default=1024, help="Input rows for prefill mode.")
    parser.add_argument("--decode-rows", type=int, default=1, help="Input rows for decode mode.")
    parser.add_argument(
        "--intermediate-multiplier",
        type=int,
        default=4,
        help="MLP intermediate width multiplier relative to the activation width.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic inputs and weights.")
    parser.add_argument("--tt-smi-timeout", type=float, default=120.0, help="Timeout for each tt-smi command.")
    parser.add_argument("--json", action="store_true", help="Print final result as JSON.")
    parser.add_argument("--_worker-json", dest="_worker_json", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    _require_executable("tt-smi")

    reset_result = _run_command(["tt-smi", "-r"], timeout=args.tt_smi_timeout)
    snapshot = collect_tt_smi_snapshot(timeout=args.tt_smi_timeout)
    system_info = summarize_system(snapshot)

    mlp_results = run_ttnn_mlp_check(
        device_id=args.device_id,
        runs=args.runs,
        pcc_threshold=args.pcc_threshold,
        activation_width=args.activation_width_per_device,
        prefill_rows=args.prefill_rows,
        decode_rows=args.decode_rows,
        intermediate_multiplier=args.intermediate_multiplier,
        seed=args.seed,
    )

    return {
        "status": "pass",
        "reset": {"stdout": reset_result.stdout.strip(), "stderr": reset_result.stderr.strip()},
        "system": system_info,
        "mlp": mlp_results,
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.runs < 1:
        raise CheckError("--runs must be >= 1")
    if not 0.0 <= args.pcc_threshold <= 1.0:
        raise CheckError("--pcc-threshold must be between 0 and 1")
    for name in ("activation_width_per_device", "prefill_rows", "decode_rows", "intermediate_multiplier"):
        if getattr(args, name) < 1:
            raise CheckError(f"--{name.replace('_', '-')} must be >= 1")
    if args.activation_width_per_device % 32 != 0:
        raise CheckError("--activation-width-per-device must be divisible by 32 for tiled matmuls")
    if (args.activation_width_per_device * args.intermediate_multiplier) % 32 != 0:
        raise CheckError("intermediate width must be divisible by 32 for tiled matmuls")


def _require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise CheckError(f"{name} not found on PATH")


def _run_command(command: list[str], *, timeout: float) -> CommandResult:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError:
        raise CheckError(f"{command[0]} not found on PATH") from None
    except subprocess.TimeoutExpired as exc:
        raise CheckError(f"{' '.join(command)} timed out after {timeout:g}s") from exc
    if result.returncode != 0:
        details = _command_output_summary(result.stdout, result.stderr)
        raise CheckError(f"{' '.join(command)} failed with exit code {result.returncode}{details}")
    return CommandResult(stdout=result.stdout, stderr=result.stderr)


def _command_output_summary(stdout: str, stderr: str) -> str:
    combined = "\n".join(part.strip() for part in (stdout, stderr) if part.strip())
    if not combined:
        return ""
    return f": {combined[-2000:]}"


def _wait_with_progress(process: subprocess.Popen[str], *, enabled: bool) -> None:
    start = time.monotonic()
    drew_progress = False
    while process.poll() is None:
        elapsed = time.monotonic() - start
        if enabled and sys.stdout.isatty() and elapsed >= 10:
            drew_progress = True
            sys.stdout.write(f"\rtt-check {_moving_bar(elapsed)} {elapsed:4.0f}s")
            sys.stdout.flush()
        time.sleep(0.2)
    if drew_progress:
        sys.stdout.write("\r" + " " * 48 + "\r")
        sys.stdout.flush()


def _moving_bar(elapsed: float, *, width: int = 24) -> str:
    position = int(elapsed * 8) % (width * 2)
    if position >= width:
        position = width * 2 - position - 1
    cells = ["-"] * width
    for offset in range(5):
        index = position - offset
        if 0 <= index < width:
            cells[index] = "#"
    return "[" + "".join(cells) + "]"


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _format_failure(stdout: str, stderr: str, *, fallback: str = "ERROR: tt-check failed") -> str:
    combined = "\n".join(part.strip() for part in (stderr, stdout) if part.strip())
    if not combined:
        return fallback

    lines = [line.rstrip() for line in combined.splitlines() if line.strip()]
    error_lines = [line for line in lines if line.startswith("ERROR:")]
    if error_lines:
        return error_lines[-1]

    tail = "\n".join(lines[-20:])
    return f"{fallback}\n\nLast diagnostics:\n{tail}"


def collect_tt_smi_snapshot(*, timeout: float) -> dict[str, Any]:
    for command in (["tt-smi", "-s", "--snapshot_no_tty"], ["tt-smi", "-s"]):
        try:
            result = _run_command(command, timeout=timeout)
            return _parse_json_from_output(result.stdout)
        except CheckError:
            if command[-1] != "-s":
                continue
            raise
    raise CheckError("unable to collect tt-smi snapshot")


def _parse_json_from_output(output: str) -> dict[str, Any]:
    text = output.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise CheckError("tt-smi snapshot did not contain JSON")
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise CheckError(f"failed to parse tt-smi JSON snapshot: {exc}") from exc
    if not isinstance(parsed, dict):
        raise CheckError("tt-smi snapshot JSON root was not an object")
    return parsed


def summarize_system(snapshot: dict[str, Any]) -> dict[str, Any]:
    devices = _devices_from_snapshot(snapshot)
    if not devices:
        raise CheckError("tt-smi snapshot reported no devices")

    architectures = sorted(_compact(_architecture_for_device(d) for d in devices))
    board_types = sorted(
        _compact(
            _value_from_paths(d, ("board_type", "type"))
            or _value_from_paths(d.get("board_info", {}), ("board_type", "type"))
            for d in devices
        )
    )
    device_series = sorted(
        _compact(
            _value_from_paths(d, ("device_series", "series", "board_name", "product_name"))
            or _value_from_paths(d.get("board_info", {}), ("device_series", "series", "board_name", "product_name"))
            for d in devices
        )
    )
    board_numbers = sorted(_compact(_board_identity(d) for d in devices))
    coordinates = sorted(
        _compact(
            _value_from_paths(d, ("coords", "coordinate", "coordinates"))
            or _value_from_paths(d.get("board_info", {}), ("coords", "coordinate", "coordinates"))
            for d in devices
        )
    )

    return {
        "architecture": architectures or ["unknown"],
        "board_types": board_types or ["unknown"],
        "device_series": device_series or board_types or ["unknown"],
        "device_count": len(devices),
        "card_count": len(board_numbers) if board_numbers else len(devices),
        "board_numbers": board_numbers,
        "device_coordinates": coordinates,
        "mesh_topology": _infer_mesh_topology(snapshot, len(devices), len(board_numbers) if board_numbers else len(devices)),
    }


def _devices_from_snapshot(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    raw = snapshot.get("device_info", [])
    if isinstance(raw, dict):
        values = raw.values()
    elif isinstance(raw, list):
        values = raw
    else:
        values = []
    return [item for item in values if isinstance(item, dict)]


def _value_from_paths(data: Any, names: Iterable[str]) -> str | None:
    if not isinstance(data, dict):
        return None
    lowered = {str(key).lower(): value for key, value in data.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is not None:
            text = str(value).strip()
            if text and text.upper() != "N/A":
                return text
    return None


def _board_identity(device: dict[str, Any]) -> str | None:
    names = (
        "board_number",
        "board_id",
        "board_serial",
        "serial_number",
        "serial",
        "card_id",
        "card_serial",
    )
    return _value_from_paths(device, names) or _value_from_paths(device.get("board_info", {}), names)


def _architecture_for_device(device: dict[str, Any]) -> str | None:
    explicit = _value_from_paths(device, ("arch", "architecture", "device_arch"))
    if explicit:
        return explicit

    text = " ".join(
        str(value).lower()
        for key, value in _walk_key_values(device)
        if any(token in key.lower() for token in ("arch", "board_type", "device_series", "product", "name"))
    )
    if any(token in text for token in ("wormhole", "n150", "n300", "nebula")):
        return "wormhole_b0"
    if any(token in text for token in ("blackhole", "p100", "p150", "p300", "p150a", "p150b")):
        return "blackhole"
    return None


def _compact(values: Iterable[str | None]) -> set[str]:
    return {value for value in values if value}


def _infer_mesh_topology(snapshot: dict[str, Any], device_count: int, card_count: int) -> str:
    reported = []
    for key, value in _walk_key_values(snapshot):
        lowered = key.lower()
        if any(token in lowered for token in ("mesh", "topology", "cluster")):
            text = str(value).strip()
            if text and text.upper() != "N/A" and len(text) < 160:
                reported.append(f"{key}={text}")
    if reported:
        return "; ".join(sorted(set(reported))[:6])

    series_text = " ".join(
        str(value).lower()
        for key, value in _walk_key_values(snapshot)
        if key.lower() in {"device_series", "series", "board_type", "board_name", "product_name"}
    )
    if any(token in series_text for token in ("quietbox", "loudbox", " t3000", "qb", "lb")):
        return "mesh system inferred from board series"
    if card_count >= 32 or device_count >= 32:
        return "galaxy-scale system inferred from device/card count"
    if card_count in {4, 8}:
        return f"{card_count}-card mesh inferred from card count; explicit topology not reported by tt-smi snapshot"
    if card_count <= 1:
        return "single-card/non-mesh inferred from card count"
    return "unknown; not reported by tt-smi snapshot"


def _walk_key_values(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, nested in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from _walk_key_values(nested, name)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            yield from _walk_key_values(nested, f"{prefix}[{index}]")
    else:
        yield prefix, value


def run_ttnn_mlp_check(
    *,
    device_id: int,
    runs: int,
    pcc_threshold: float,
    activation_width: int,
    prefill_rows: int,
    decode_rows: int,
    intermediate_multiplier: int,
    seed: int,
) -> list[dict[str, Any]]:
    try:
        import torch
        import ttnn
    except ModuleNotFoundError as exc:
        raise CheckError(f"missing Python dependency: {exc.name}") from exc

    torch.manual_seed(seed)
    device = None
    try:
        device = ttnn.open_device(device_id=device_id, trace_region_size=0)
        results = []
        for mode, rows in (("prefill", prefill_rows), ("decode", decode_rows)):
            results.append(
                _run_mlp_mode(
                    torch=torch,
                    ttnn=ttnn,
                    device=device,
                    mode=mode,
                    rows=rows,
                    activation_width=activation_width,
                    intermediate_width=activation_width * intermediate_multiplier,
                    runs=runs,
                    pcc_threshold=pcc_threshold,
                )
            )
        return results
    finally:
        if device is not None:
            try:
                ttnn.close_device(device)
            except Exception as exc:
                raise CheckError(f"ttnn.close_device failed: {type(exc).__name__}: {exc}") from exc


def _run_mlp_mode(
    *,
    torch: Any,
    ttnn: Any,
    device: Any,
    mode: str,
    rows: int,
    activation_width: int,
    intermediate_width: int,
    runs: int,
    pcc_threshold: float,
) -> dict[str, Any]:
    shape = (1, 1, rows, activation_width)
    x = (torch.randn(shape, dtype=torch.float32) * 0.5).to(torch.bfloat16)
    w1 = torch.randn((activation_width, intermediate_width), dtype=torch.float32) / math.sqrt(activation_width)
    w3 = torch.randn((activation_width, intermediate_width), dtype=torch.float32) / math.sqrt(activation_width)
    w2 = torch.randn((intermediate_width, activation_width), dtype=torch.float32) / math.sqrt(intermediate_width)

    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_w1 = ttnn.from_torch(w1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    tt_w3 = ttnn.from_torch(w3, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    tt_w2 = ttnn.from_torch(w2, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

    try:
        qw1 = ttnn.to_torch(tt_w1).to(torch.float32)
        qw3 = ttnn.to_torch(tt_w3).to(torch.float32)
        qw2 = ttnn.to_torch(tt_w2).to(torch.float32)
        reference = torch.matmul(torch.nn.functional.silu(torch.matmul(x.to(torch.float32), qw1)) * torch.matmul(x.to(torch.float32), qw3), qw2)

        mode_start = time.perf_counter()
        pcc_values = []

        warmup_start = time.perf_counter()
        warmup_output, _ = _ttnn_mlp_forward(ttnn, tt_x, tt_w1, tt_w3, tt_w2)
        warmup_torch = ttnn.to_torch(warmup_output).to(torch.float32)
        _deallocate(ttnn, warmup_output)
        warmup_elapsed = time.perf_counter() - warmup_start
        first_output = warmup_torch
        pcc_values.append(_validate_mlp_output(torch, reference, warmup_torch, pcc_threshold, mode, "warmup", None))

        _synchronize_device(ttnn, device)
        trace_id = None
        trace_output = None
        trace_intermediates = ()
        try:
            capture_start = time.perf_counter()
            trace_id = ttnn.begin_trace_capture(device, cq_id=0)
            trace_output, trace_intermediates = _ttnn_mlp_forward(
                ttnn, tt_x, tt_w1, tt_w3, tt_w2, keep_intermediates=True
            )
            ttnn.end_trace_capture(device, trace_id, cq_id=0)
            capture_elapsed = time.perf_counter() - capture_start
            _synchronize_device(ttnn, device)

            capture_torch = ttnn.to_torch(trace_output).to(torch.float32)
            pcc_values.append(
                _validate_mlp_output(torch, reference, capture_torch, pcc_threshold, mode, "capture", first_output)
            )

            replay_start = time.perf_counter()
            for run_index in range(runs):
                ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
                output_torch = ttnn.to_torch(trace_output).to(torch.float32)
                pcc_values.append(
                    _validate_mlp_output(
                        torch, reference, output_torch, pcc_threshold, mode, f"trace replay {run_index}", first_output
                    )
                )
            replay_elapsed = time.perf_counter() - replay_start
        finally:
            if trace_id is not None:
                _release_trace(ttnn, device, trace_id)
            for tensor in (trace_output, *trace_intermediates):
                _deallocate(ttnn, tensor)

        unique_pcc_values = {f"{pcc:.12g}" for pcc in pcc_values}
        if len(unique_pcc_values) != 1:
            raise CheckError(f"{mode}: PCC values were not identical across runs: {sorted(unique_pcc_values)}")

        return {
            "mode": mode,
            "shape": list(shape),
            "runs": runs,
            "execution": "trace",
            "warmup_runs": 1,
            "captured_runs": 1,
            "pcc": pcc_values[0],
            "pcc_threshold": pcc_threshold,
            "outputs_identical": True,
            "timing": {
                "warmup_s": warmup_elapsed,
                "capture_s": capture_elapsed,
                "replay_s": replay_elapsed,
                "total_s": time.perf_counter() - mode_start,
            },
        }
    finally:
        for tensor in (tt_x, tt_w1, tt_w3, tt_w2):
            _deallocate(ttnn, tensor)


def _ttnn_mlp_forward(
    ttnn: Any, x: Any, w1: Any, w3: Any, w2: Any, *, keep_intermediates: bool = False
) -> tuple[Any, tuple[Any, ...]]:
    w1_out = ttnn.linear(x, w1, dtype=ttnn.bfloat16)
    w3_out = ttnn.linear(x, w3, dtype=ttnn.bfloat16)
    activated = ttnn.silu(w1_out)
    hidden = ttnn.mul(activated, w3_out, dtype=ttnn.bfloat16)
    output = ttnn.linear(hidden, w2, dtype=ttnn.bfloat16)

    intermediates = (hidden, activated, w3_out, w1_out)
    if keep_intermediates:
        return output, intermediates
    for tensor in intermediates:
        _deallocate(ttnn, tensor)
    return output, ()


def _validate_mlp_output(
    torch: Any,
    reference: Any,
    actual: Any,
    pcc_threshold: float,
    mode: str,
    run_label: str,
    expected_identical: Any | None,
) -> float:
    if not torch.isfinite(actual).all():
        raise CheckError(f"{mode} {run_label}: TTNN output contains non-finite values")

    pcc = _pearson_corrcoef(torch, reference, actual)
    if pcc < pcc_threshold:
        raise CheckError(f"{mode} {run_label}: PCC {pcc:.8f} < {pcc_threshold:.8f}")

    if expected_identical is not None and not torch.equal(expected_identical, actual):
        max_abs = torch.max(torch.abs(expected_identical - actual)).item()
        differing = torch.count_nonzero(expected_identical != actual).item()
        raise CheckError(
            f"{mode} {run_label}: output differs from warmup "
            f"({differing} elements differ, max_abs_diff={max_abs:.8g})"
        )
    return pcc


def _synchronize_device(ttnn: Any, device: Any) -> None:
    try:
        ttnn.synchronize_device(device)
    except AttributeError:
        return


def _release_trace(ttnn: Any, device: Any, trace_id: Any) -> None:
    try:
        ttnn.release_trace(device, trace_id)
    except Exception:
        pass


def _deallocate(ttnn: Any, tensor: Any) -> None:
    if tensor is None:
        return
    try:
        ttnn.deallocate(tensor)
    except Exception:
        pass


def _pearson_corrcoef(torch: Any, expected: Any, actual: Any) -> float:
    expected_flat = expected.reshape(-1).to(torch.float64)
    actual_flat = actual.reshape(-1).to(torch.float64)
    if expected_flat.numel() != actual_flat.numel():
        raise CheckError(f"shape mismatch: expected {tuple(expected.shape)}, actual {tuple(actual.shape)}")

    expected_centered = expected_flat - torch.mean(expected_flat)
    actual_centered = actual_flat - torch.mean(actual_flat)
    denominator = torch.linalg.norm(expected_centered) * torch.linalg.norm(actual_centered)
    if denominator.item() == 0.0:
        return 1.0 if torch.equal(expected_flat, actual_flat) else 0.0
    return (torch.dot(expected_centered, actual_centered) / denominator).item()


def _format_human_result(result: dict[str, Any]) -> str:
    system = result["system"]
    mlp_by_mode = {item["mode"]: item for item in result["mlp"]}
    prefill = mlp_by_mode["prefill"]
    decode = mlp_by_mode["decode"]
    runs = max(item["runs"] for item in result["mlp"])
    cards = system["card_count"]
    lines = [
        "tt-check passed",
        "",
        "reset   ok",
        f"system  {_join_short(system['architecture'])} | {_join_short(system['device_series'])} | {cards} {_plural(cards, 'card')} | {_short_topology(system['mesh_topology'])}",
        f"mlp     prefill pcc {prefill['pcc']:.8f} | decode pcc {decode['pcc']:.8f} | trace x{runs}",
        "ready.",
    ]
    return "\n".join(lines)


def _join_short(values: list[str]) -> str:
    return ",".join(values) if values else "unknown"


def _plural(count: int, noun: str) -> str:
    return noun if count == 1 else f"{noun}s"


def _short_topology(topology: str) -> str:
    lowered = topology.lower()
    if lowered.startswith("single-card"):
        return "single-card"
    if "galaxy" in lowered:
        return "galaxy"
    if "4-card mesh" in lowered:
        return "4-card mesh"
    if "8-card mesh" in lowered:
        return "8-card mesh"
    if "mesh system" in lowered:
        return "mesh"
    if "unknown" in lowered:
        return "topology unknown"
    return topology


if __name__ == "__main__":
    sys.exit(main())
