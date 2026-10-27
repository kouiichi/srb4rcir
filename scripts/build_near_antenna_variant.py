#!/usr/bin/env python3
"""Build the reproducible NEAR antenna-replacement USD asset."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from srb.nmpc.near_variant import build_near_variant


VISUAL = ROOT/'assets/nasa_candidates/near_shoemaker/visual'


def main():
    manifest = build_near_variant(
        VISUAL/'near_shoemaker_original.usdc',
        VISUAL/'near_shoemaker_antenna_boss.usdc',
    )
    print(f"Created {manifest['derived_usd']}")
    print(f"Removed antenna components: {manifest['removed_component_count']}")
    print(f"Boss dimensions (source units): {manifest['boss_dimensions_source_units']}")
    print(f"Source SHA-256: {manifest['source_sha256']}")


if __name__ == '__main__':
    main()
