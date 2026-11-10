"""Paths independent of the depth of an implementation module."""
import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("FLOW2PC_REPO_ROOT", Path(__file__).resolve().parents[3]))
MFM_LOCAL_ROOT = Path(__file__).resolve().parents[1] / "pretrain/models/mfm/_source"
PRETRAIN_ROOT = REPO_ROOT.parent / "pretrain"
MFM_PRETRAIN_ROOT = Path(
    os.environ.get("FLOW2PC_MFM_PRETRAIN_ROOT", PRETRAIN_ROOT / "metric-flow-matching")
)
MFM_ROOT = Path(os.environ.get("FLOW2PC_MFM_ROOT", (
    REPO_ROOT.parent / "metric-flow-matching"
    if (REPO_ROOT.parent / "metric-flow-matching/mfm").is_dir()
    else MFM_PRETRAIN_ROOT
)))

MMFM_ROOT = Path(os.environ.get("FLOW2PC_MMFM_ROOT", PRETRAIN_ROOT / "MMFM"))
