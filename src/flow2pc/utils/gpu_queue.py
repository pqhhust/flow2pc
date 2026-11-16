"""Proven multi-GPU queue helpers recovered from the successful real-data sweeps.

See docs/SCHEDULER_SOURCE.json. The controller uses nvidia-smi, never a CUDA
memory query. These helpers do not import Torch or start workers at import time.
"""

import csv
import os
import signal
import subprocess
import time
from collections import Counter


def choose_device(devices, active, free_bytes, workers_per_gpu, min_free_gb, now):
    """Balance workers across GPUs, reserving memory while new workers warm up."""
    counts = Counter(worker["device"] for worker in active)
    required = min_free_gb * 2**30
    candidates = []
    for device in devices:
        # Avoid launching a burst before CUDA contexts/allocations become visible.
        warming = sum(
            worker["device"] == device and now - worker["started"] < 30
            for worker in active
        )
        available = free_bytes[device] - warming * required
        if counts[device] < workers_per_gpu and available >= required:
            candidates.append((counts[device], -available, device))
    return min(candidates)[2] if candidates else None


def next_task(datasets):
    """One task per dataset per round; finished datasets do not strand GPUs."""
    for _ in range(len(datasets)):
        dataset = datasets[0]
        datasets.rotate(-1)
        if dataset.pending:
            return dataset.pending.popleft()
    return None


def stop_workers(active):
    """Signal every process group before waiting, including SDK child services."""
    for worker in active:
        try:
            os.killpg(worker["process"].pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 15
    for worker in active:
        try:
            worker["process"].wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(worker["process"].pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            worker["process"].wait()
        worker["log"].close()


def gpu_inventory():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"nvidia-smi query failed: {error}") from error
    inventory = {}
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) != 3:
            continue
        index, uuid, free = [value.strip() for value in row]
        if not index.isdigit():
            continue
        inventory[index] = {
            "uuid": uuid,
            "free": max(0, int(free)) * 2**20 if free.isdigit() else 0,
        }
    return inventory


def selected_devices(value, inventory):
    def resolve(token):
        token = token.strip()
        if token in inventory:
            return token
        matches = [
            index
            for index, row in inventory.items()
            if token and row["uuid"].startswith(token)
        ]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(f"Unknown/ambiguous physical GPU index or UUID: {token}")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    allowed = (
        list(inventory)
        if visible is None
        else [resolve(token) for token in visible.split(",") if token.strip()]
    )
    devices = (
        allowed if value == "auto" else [resolve(token) for token in value.split(",")]
    )
    if len(devices) < 2 or len(set(devices)) != len(devices):
        raise ValueError(
            "Select at least two distinct GPUs; no CPU or single-GPU fallback"
        )
    if set(devices) - set(allowed):
        raise ValueError(
            "Requested GPUs are outside the inherited CUDA_VISIBLE_DEVICES allocation"
        )
    return devices
