from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROGRESS_PREFIX = "TT_CHECK_PROGRESS "


class CheckError(RuntimeError):
    """Expected readiness-check failure."""


@dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str


@dataclass(frozen=True)
class TtnnDeviceContext:
    device: Any
    tensor_parallel_degree: int
    is_mesh: bool
    mesh_shape: tuple[int, int] | None


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
        print(
            f"ERROR: unexpected failure: {type(exc).__name__}: {exc}\n\nTraceback:\n{traceback.format_exc()}",
            file=sys.stderr,
        )
        return 1
    return 0


@contextmanager
def _phase(name: str) -> Iterable[None]:
    try:
        yield
    except CheckError as exc:
        raise CheckError(f"{name} failed: {exc}") from exc
    except Exception as exc:
        raise CheckError(
            f"{name} failed: {type(exc).__name__}: {exc}\n\nTraceback:\n{traceback.format_exc()}"
        ) from exc


def _main_parent(raw_argv: list[str], args: argparse.Namespace) -> int:
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="tt-check-") as tmp_dir:
        result_path = Path(tmp_dir) / "result.json"
        stdout_path = Path(tmp_dir) / "stdout.txt"
        stderr_path = Path(tmp_dir) / "stderr.txt"
        worker_argv = [*raw_argv, "--_worker-json", str(result_path)]
        env = os.environ.copy()
        env.setdefault("TT_LOGGER_LEVEL", "FATAL")
        env.setdefault("LOGURU_LEVEL", "ERROR")
        if not args.json:
            env["TT_CHECK_PROGRESS"] = "1"

        with (
            stdout_path.open("w", encoding="utf-8") as stdout_file,
            stderr_path.open("w", encoding="utf-8") as stderr_file,
        ):
            progress = _ProgressRenderer(enabled=not args.json)
            process = subprocess.Popen(
                [sys.executable, "-m", "tt_check.cli", *worker_argv],
                stdout=stdout_file,
                stderr=subprocess.PIPE if not args.json else stderr_file,
                text=True,
                env=env,
            )
            stderr_thread = None
            if process.stderr is not None:
                stderr_thread = threading.Thread(
                    target=_drain_worker_stderr,
                    args=(process.stderr, stderr_file, progress),
                    daemon=True,
                )
                stderr_thread.start()
            try:
                process.wait()
            except KeyboardInterrupt:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                progress.close()
                print("ERROR: interrupted", file=sys.stderr)
                return 1
            finally:
                if stderr_thread is not None:
                    stderr_thread.join(timeout=5)
                progress.close()

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

    result["elapsed_s"] = time.monotonic() - start
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

    _emit_progress("step_start", label="resetting device")
    with _phase("resetting device with tt-smi -r"):
        reset_result = _run_command(["tt-smi", "-r"], timeout=args.tt_smi_timeout)
    _emit_progress("step_done", label="resetting device")
    with _phase("detecting system with tt-smi -s"):
        snapshot = collect_tt_smi_snapshot(timeout=args.tt_smi_timeout)
        system_info = summarize_system(snapshot)

    with _phase("running TTNN MLP readiness check"):
        mlp_results = run_ttnn_mlp_check(
            system_info=system_info,
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


class _ProgressRenderer:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self.current_bar = None
        self.current_label = ""
        self.current_step = False
        self.tqdm = self._load_tqdm() if enabled else None

    def handle(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        event_type = event.get("event")
        if event_type == "step_start":
            self.start_step(str(event.get("label", "work")))
        elif event_type == "step_done":
            self.finish_step(str(event.get("status", "ok")))
        elif event_type == "message":
            self.write(str(event.get("text", "")))
        elif event_type == "bar_start":
            self.start_bar(str(event.get("label", "work")), int(event.get("total", 0)))
        elif event_type == "bar_update":
            self.update_bar(int(event.get("advance", 1)))
        elif event_type == "bar_done":
            self.finish_bar(str(event.get("label", self.current_label)))

    def start_step(self, label: str) -> None:
        self.close()
        self.current_step = True
        print(f"tt-check: {label}... ", end="", flush=True)

    def finish_step(self, status: str) -> None:
        if self.current_step:
            print(status, flush=True)
            self.current_step = False
        else:
            print(f"tt-check: {status}", flush=True)

    def start_bar(self, label: str, total: int) -> None:
        self.close()
        self.current_label = label
        if self.tqdm is None:
            print(f"tt-check: {label}...", flush=True)
            return
        self.current_bar = self.tqdm(
            total=total,
            desc=f"tt-check: {label}",
            unit="run",
            dynamic_ncols=True,
            leave=True,
            file=sys.stdout,
        )

    def update_bar(self, advance: int) -> None:
        if self.current_bar is not None:
            self.current_bar.update(advance)

    def finish_bar(self, label: str) -> None:
        if self.current_bar is not None:
            self.current_bar.close()
            self.current_bar = None
        elif self.tqdm is None and label:
            print(f"tt-check: {label} done", flush=True)
        self.current_label = ""

    def write(self, text: str) -> None:
        if self.current_step:
            print(flush=True)
            self.current_step = False
        if self.current_bar is not None:
            self.current_bar.write(f"tt-check: {text}")
        else:
            print(f"tt-check: {text}", flush=True)

    def close(self) -> None:
        if self.current_bar is not None:
            self.current_bar.close()
            self.current_bar = None
        if self.current_step:
            print("failed", flush=True)
            self.current_step = False

    @staticmethod
    def _load_tqdm() -> Any | None:
        try:
            from tqdm import tqdm
        except ModuleNotFoundError:
            return None
        return tqdm


def _drain_worker_stderr(stderr_pipe: Any, stderr_file: Any, progress: _ProgressRenderer) -> None:
    for line in iter(stderr_pipe.readline, ""):
        if line.startswith(PROGRESS_PREFIX):
            try:
                progress.handle(json.loads(line[len(PROGRESS_PREFIX) :]))
            except Exception:
                stderr_file.write(line)
                stderr_file.flush()
        else:
            stderr_file.write(line)
            stderr_file.flush()


def _emit_progress(event: str, **payload: Any) -> None:
    if os.environ.get("TT_CHECK_PROGRESS") != "1":
        return
    print(
        PROGRESS_PREFIX + json.dumps({"event": event, **payload}, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


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
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].startswith("ERROR:"):
            return "\n".join(lines[index:])

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
    system_info: dict[str, Any] | None = None,
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
    context: TtnnDeviceContext | None = None
    try:
        with _phase("opening TTNN device/mesh"):
            context = _open_ttnn_device_context(ttnn, device_id=device_id)
        if system_info is not None:
            _emit_progress("message", text=_format_runtime_system_summary(system_info, context.mesh_shape))
        results = []
        for mode, rows in (("prefill", prefill_rows), ("decode", decode_rows)):
            with _phase(f"{mode} tensor-parallel MLP"):
                results.append(
                    _run_mlp_mode(
                        torch=torch,
                        ttnn=ttnn,
                        device=context.device,
                        is_mesh=context.is_mesh,
                        tensor_parallel_degree=context.tensor_parallel_degree,
                        mesh_shape=context.mesh_shape,
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
        if context is not None:
            try:
                if context.is_mesh:
                    ttnn.close_mesh_device(context.device)
                else:
                    ttnn.close_device(context.device)
            except Exception as exc:
                raise CheckError(f"ttnn device close failed: {type(exc).__name__}: {exc}") from exc


def _open_ttnn_device_context(ttnn: Any, *, device_id: int) -> TtnnDeviceContext:
    device_count = _available_ttnn_devices(ttnn)
    if device_id != 0 or device_count <= 1:
        device = ttnn.open_device(device_id=device_id, trace_region_size=0)
        return TtnnDeviceContext(device=device, tensor_parallel_degree=1, is_mesh=False, mesh_shape=None)

    _enable_1d_fabric(ttnn)
    mesh_shape = (1, device_count)
    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*mesh_shape),
        trace_region_size=0,
    )
    return TtnnDeviceContext(device=device, tensor_parallel_degree=device_count, is_mesh=True, mesh_shape=mesh_shape)


def _available_ttnn_devices(ttnn: Any) -> int:
    errors = []
    for name in ("get_num_devices", "GetNumAvailableDevices"):
        query = getattr(ttnn, name, None)
        if query is None:
            continue
        try:
            count = int(query())
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if count < 1:
            raise CheckError(f"{name} reported no available TTNN devices")
        return count
    if errors:
        raise CheckError("failed to query TTNN device count: " + "; ".join(errors))
    return 1


def _enable_1d_fabric(ttnn: Any) -> None:
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    except Exception as exc:
        raise CheckError(f"failed to enable TTNN fabric for tensor-parallel CCLs: {type(exc).__name__}: {exc}") from exc


def _run_mlp_mode(
    *,
    torch: Any,
    ttnn: Any,
    device: Any,
    is_mesh: bool,
    tensor_parallel_degree: int,
    mesh_shape: tuple[int, int] | None,
    mode: str,
    rows: int,
    activation_width: int,
    intermediate_width: int,
    runs: int,
    pcc_threshold: float,
) -> dict[str, Any]:
    shape = (1, 1, rows, activation_width)
    global_intermediate_width = intermediate_width * tensor_parallel_degree
    x = (torch.randn(shape, dtype=torch.float32) * 0.5).to(torch.bfloat16)
    w1 = torch.randn((activation_width, global_intermediate_width), dtype=torch.float32) / math.sqrt(activation_width)
    w3 = torch.randn((activation_width, global_intermediate_width), dtype=torch.float32) / math.sqrt(activation_width)
    w2 = torch.randn((global_intermediate_width, activation_width), dtype=torch.float32) / math.sqrt(
        global_intermediate_width
    )

    if is_mesh:
        tt_x = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        tt_w1 = ttnn.from_torch(
            w1,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=1),
        )
        tt_w3 = ttnn.from_torch(
            w3,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=1),
        )
        tt_w2 = ttnn.from_torch(
            w2,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0),
        )
        weight_composers = (
            ttnn.ConcatMeshToTensor(device, dim=1),
            ttnn.ConcatMeshToTensor(device, dim=1),
            ttnn.ConcatMeshToTensor(device, dim=0),
        )
    else:
        tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tt_w1 = ttnn.from_torch(w1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        tt_w3 = ttnn.from_torch(w3, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        tt_w2 = ttnn.from_torch(w2, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        weight_composers = (None, None, None)

    try:
        qw1 = _to_torch_tensor(ttnn, tt_w1, mesh_composer=weight_composers[0]).to(torch.float32)
        qw3 = _to_torch_tensor(ttnn, tt_w3, mesh_composer=weight_composers[1]).to(torch.float32)
        qw2 = _to_torch_tensor(ttnn, tt_w2, mesh_composer=weight_composers[2]).to(torch.float32)
        reference = torch.matmul(
            torch.nn.functional.silu(torch.matmul(x.to(torch.float32), qw1)) * torch.matmul(x.to(torch.float32), qw3),
            qw2,
        )

        mode_start = time.perf_counter()
        pcc_values = []

        warmup_start = time.perf_counter()
        warmup_output, _ = _ttnn_mlp_forward(
            ttnn,
            tt_x,
            tt_w1,
            tt_w3,
            tt_w2,
            tensor_parallel_degree=tensor_parallel_degree,
            ccl_cluster_axis=1,
        )
        warmup_torch = _to_torch_replicated(ttnn, torch, warmup_output, tensor_parallel_degree).to(torch.float32)
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
                ttnn,
                tt_x,
                tt_w1,
                tt_w3,
                tt_w2,
                tensor_parallel_degree=tensor_parallel_degree,
                ccl_cluster_axis=1,
                keep_intermediates=True,
            )
            ttnn.end_trace_capture(device, trace_id, cq_id=0)
            capture_elapsed = time.perf_counter() - capture_start
            _synchronize_device(ttnn, device)

            capture_torch = _to_torch_replicated(ttnn, torch, trace_output, tensor_parallel_degree).to(torch.float32)
            pcc_values.append(
                _validate_mlp_output(torch, reference, capture_torch, pcc_threshold, mode, "capture", first_output)
            )

            _emit_progress("bar_start", label=f"{mode} mlp", total=runs)
            replay_start = time.perf_counter()
            try:
                for run_index in range(runs):
                    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
                    output_torch = _to_torch_replicated(ttnn, torch, trace_output, tensor_parallel_degree).to(
                        torch.float32
                    )
                    pcc_values.append(
                        _validate_mlp_output(
                            torch,
                            reference,
                            output_torch,
                            pcc_threshold,
                            mode,
                            f"trace replay {run_index}",
                            first_output,
                        )
                    )
                    _emit_progress("bar_update", advance=1)
            finally:
                _emit_progress("bar_done", label=f"{mode} mlp")
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
            "global_intermediate_width": global_intermediate_width,
            "runs": runs,
            "execution": "trace",
            "tensor_parallel_degree": tensor_parallel_degree,
            "mesh_shape": list(mesh_shape) if mesh_shape is not None else None,
            "ccl": "all_reduce" if tensor_parallel_degree > 1 else "none",
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
    ttnn: Any,
    x: Any,
    w1: Any,
    w3: Any,
    w2: Any,
    *,
    tensor_parallel_degree: int,
    ccl_cluster_axis: int,
    keep_intermediates: bool = False,
) -> tuple[Any, tuple[Any, ...]]:
    w1_out = ttnn.linear(x, w1, dtype=ttnn.bfloat16)
    w3_out = ttnn.linear(x, w3, dtype=ttnn.bfloat16)
    activated = ttnn.silu(w1_out)
    hidden = ttnn.mul(activated, w3_out, dtype=ttnn.bfloat16)
    partial_output = ttnn.linear(hidden, w2, dtype=ttnn.bfloat16)
    output = (
        ttnn.all_reduce(partial_output, cluster_axis=ccl_cluster_axis, topology=ttnn.Topology.Linear)
        if tensor_parallel_degree > 1
        else partial_output
    )

    intermediates = (hidden, activated, w3_out, w1_out)
    if output is not partial_output:
        intermediates = (partial_output, *intermediates)
    if keep_intermediates:
        return output, intermediates
    for tensor in intermediates:
        _deallocate(ttnn, tensor)
    return output, ()


def _to_torch_tensor(ttnn: Any, tensor: Any, *, mesh_composer: Any | None = None) -> Any:
    if mesh_composer is None:
        return ttnn.to_torch(tensor)
    return ttnn.to_torch(tensor, mesh_composer=mesh_composer)


def _to_torch_replicated(ttnn: Any, torch: Any, tensor: Any, tensor_parallel_degree: int) -> Any:
    if tensor_parallel_degree <= 1:
        return ttnn.to_torch(tensor)

    shards = [ttnn.to_torch(device_tensor) for device_tensor in ttnn.get_device_tensors(tensor)]
    first = shards[0]
    for index, shard in enumerate(shards[1:], start=1):
        if not torch.equal(first, shard):
            max_abs = torch.max(torch.abs(first.to(torch.float32) - shard.to(torch.float32))).item()
            differing = torch.count_nonzero(first != shard).item()
            raise CheckError(
                f"tensor-parallel all-reduce output differs on device shard {index} "
                f"({differing} elements differ, max_abs_diff={max_abs:.8g})"
            )
    return first


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
    mlp_by_mode = {item["mode"]: item for item in result["mlp"]}
    prefill = mlp_by_mode["prefill"]
    decode = mlp_by_mode["decode"]
    elapsed = result.get("elapsed_s")
    elapsed_text = f" in {elapsed:.1f} seconds" if isinstance(elapsed, (int, float)) else ""
    return (
        f"tt-check: passed{elapsed_text} | "
        f"prefill pcc {prefill['pcc']:.8f} | decode pcc {decode['pcc']:.8f}"
    )


def _format_runtime_system_summary(system: dict[str, Any], mesh_shape: tuple[int, int] | None) -> str:
    cards = system["card_count"]
    series = _join_short(system["device_series"])
    card_text = f"{cards}x {series}" if series != "unknown" else f"{cards} {_plural(cards, 'card')}"
    arch_text = _join_short(system["architecture"])
    topology_text = _format_mesh_shape(mesh_shape) or _short_topology(system["mesh_topology"])
    return f"{topology_text} ({card_text} | {arch_text})"


def _format_mesh_shape(mesh_shape: Any) -> str:
    if not mesh_shape:
        return ""
    try:
        rows, columns = mesh_shape
    except (TypeError, ValueError):
        return ""
    return f"{rows}x{columns} mesh"


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
