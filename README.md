# tt-check

`tt-check` is a small command-line readiness test for a Tenstorrent system with
`tt-smi` and the public `ttnn` Python package installed.

It performs the following checks:

1. Resets devices with `tt-smi -r`.
2. Collects system information from `tt-smi -s --snapshot_no_tty`.
3. Opens a TTNN device.
4. Runs a tensor-parallel three-weight gated MLP 100 times in both
   prefill and decode shapes:
   - BF16 activations
   - BFP8 weights
   - 1024 activation width per device shard
   - prefill rows: 1024
   - decode rows: 1
5. On multi-device systems, replicates activations, column-shards `w1`/`w3`,
   row-shards `w2`, and all-reduces the output activations across the mesh.
6. Warms each MLP shape once, captures one dynamic TTNN trace per shape, then
   executes each trace for the requested run count.
7. Compares each trace replay against a PyTorch reference with PCC >= 0.99 and
   requires TTNN output tensors to be identical across all replays.

## Install

Run from PyPI with `uv`:

```bash
uvx --python 3.10 tt-check
```

Or install from a checkout:

```bash
uv tool install --python 3.10 .
```

Or run directly from a checkout:

```bash
uv run tt-check
```

`uv` uses the PyTorch CPU wheel index for this project; the check only needs
Torch for reference math. The project pins a Python version compatible with the
current public `ttnn` wheels.

Pip also works:

```bash
python3 -m pip install .
```

## Run

```bash
tt-check
```

The command exits `0` if all checks pass. It exits `1` and writes the failure to
stderr otherwise.

Example output:

```text
tt-check: resetting device... ok
tt-check: 1x2 mesh (1x p300a | blackhole)
tt-check: prefill mlp: 100%|██████████| 100/100 [00:00<00:00, 206.20run/s]
tt-check: decode mlp: 100%|██████████| 100/100 [00:00<00:00, 2660.53run/s]
tt-check: passed in 44.9 seconds | prefill pcc 0.99981364 | decode pcc 0.99986366
```

Useful options:

```bash
tt-check --device-id 0 --runs 100 --pcc-threshold 0.99
tt-check --prefill-rows 1024 --decode-rows 1 --activation-width-per-device 1024
```
