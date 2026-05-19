"""Batch configuration for the full OpenFOAM Federico re-extraction pipeline.

Defines the 17 extractable batches (7 new + 10 old) that will be processed
into the dirty/clean/strict consolidated dataset.

Excluded batches:
  - OPENFOAM_020: no CASES_v3 directory
  - OPENFOAM_007/011/014/015/016: no CASES_v3 directory
  - OPENFOAM_017: 0 VTK-ready cases (simulations never completed)
  - old_simulations/CASES_v3: standalone folder, excluded by user spec

Usage:
    from openfoam_batch_config import BATCHES, BASE_DIR, DIRTY_DIR, CLEAN_DIR, STRICT_DIR
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
#  Root paths
# ---------------------------------------------------------------------------

BASE_DIR = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM_federico")
OUTPUT_BASE = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/openfoam_merged")

DIRTY_DIR = OUTPUT_BASE / "dirty"
CLEAN_DIR = OUTPUT_BASE / "clean"
STRICT_DIR = OUTPUT_BASE / "strict"


@dataclass(frozen=True)
class BatchConfig:
    """Configuration for one extraction batch.

    Attributes:
        batch_short: Short identifier used as directory prefix in consolidated outputs
            (e.g. ``of021``, ``old001``).
        cases_dir: Absolute path to the CASES_v3 directory for this batch.
        source_dataset: Value written into ``data.source_dataset`` on every .pt file.
    """

    batch_short: str
    cases_dir: Path
    source_dataset: str

    @property
    def dirty_output_dir(self) -> Path:
        """Per-batch subdirectory inside dirty/."""
        return DIRTY_DIR / self.batch_short


# ---------------------------------------------------------------------------
#  Batch registry — 17 batches total
# ---------------------------------------------------------------------------

_OF_BASE = BASE_DIR
_OLD_BASE = BASE_DIR / "old_simulations"

BATCHES: list[BatchConfig] = [
    # ---- New batches (021-027) ----
    BatchConfig(
        batch_short="of021",
        cases_dir=_OF_BASE / "OPENFOAM_021" / "CASES_v3",
        source_dataset="openfoam_federico_021",
    ),
    BatchConfig(
        batch_short="of022",
        cases_dir=_OF_BASE / "OPENFOAM_022" / "CASES_v3",
        source_dataset="openfoam_federico_022",
    ),
    BatchConfig(
        batch_short="of023",
        cases_dir=_OF_BASE / "OPENFOAM_023" / "CASES_v3",
        source_dataset="openfoam_federico_023",
    ),
    BatchConfig(
        batch_short="of024",
        cases_dir=_OF_BASE / "OPENFOAM_024" / "CASES_v3",
        source_dataset="openfoam_federico_024",
    ),
    BatchConfig(
        batch_short="of025",
        cases_dir=_OF_BASE / "OPENFOAM_025" / "CASES_v3",
        source_dataset="openfoam_federico_025",
    ),
    BatchConfig(
        batch_short="of026",
        cases_dir=_OF_BASE / "OPENFOAM_026" / "CASES_v3",
        source_dataset="openfoam_federico_026",
    ),
    BatchConfig(
        batch_short="of027",
        cases_dir=_OF_BASE / "OPENFOAM_027" / "CASES_v3",
        source_dataset="openfoam_federico_027",
    ),
    # ---- Old batches (001-013, skip 007/010/011/014-017) ----
    BatchConfig(
        batch_short="old001",
        cases_dir=_OLD_BASE / "OPENFOAM_001" / "CASES_v3",
        source_dataset="openfoam_federico_old_001",
    ),
    BatchConfig(
        batch_short="old002",
        cases_dir=_OLD_BASE / "OPENFOAM_002" / "CASES_v3",
        source_dataset="openfoam_federico_old_002",
    ),
    BatchConfig(
        batch_short="old003",
        cases_dir=_OLD_BASE / "OPENFOAM_003" / "CASES_v3",
        source_dataset="openfoam_federico_old_003",
    ),
    BatchConfig(
        batch_short="old004",
        cases_dir=_OLD_BASE / "OPENFOAM_004" / "CASES_v3",
        source_dataset="openfoam_federico_old_004",
    ),
    BatchConfig(
        batch_short="old005",
        cases_dir=_OLD_BASE / "OPENFOAM_005" / "CASES_v3",
        source_dataset="openfoam_federico_old_005",
    ),
    BatchConfig(
        batch_short="old006",
        cases_dir=_OLD_BASE / "OPENFOAM_006" / "CASES_v3",
        source_dataset="openfoam_federico_old_006",
    ),
    BatchConfig(
        batch_short="old008",
        cases_dir=_OLD_BASE / "OPENFOAM_008" / "CASES_v3",
        source_dataset="openfoam_federico_old_008",
    ),
    BatchConfig(
        batch_short="old009",
        cases_dir=_OLD_BASE / "OPENFOAM_009" / "CASES_v3",
        source_dataset="openfoam_federico_old_009",
    ),
    BatchConfig(
        batch_short="old012",
        cases_dir=_OLD_BASE / "OPENFOAM_012" / "CASES_v3",
        source_dataset="openfoam_federico_old_012",
    ),
    BatchConfig(
        batch_short="old013",
        cases_dir=_OLD_BASE / "OPENFOAM_013" / "CASES_v3",
        source_dataset="openfoam_federico_old_013",
    ),
]

# Convenience mapping for SLURM array indexing (0-based)
BATCH_BY_INDEX: dict[int, BatchConfig] = dict(enumerate(BATCHES))
