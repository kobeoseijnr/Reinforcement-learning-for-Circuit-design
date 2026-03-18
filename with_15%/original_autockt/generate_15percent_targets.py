"""
Generate target_specs_15percent.json from target_specs_original.json.
Each target is increased by 15%: original * 1.15
Run: python generate_15percent_targets.py
"""

import json
import math
from pathlib import Path


def main():
    BASE_DIR = Path(__file__).parent
    ORIGINAL_PATH = BASE_DIR / 'data' / 'target_specs_original.json'
    OUTPUT_PATH = BASE_DIR / 'data' / 'target_specs_15percent.json'

    with open(ORIGINAL_PATH, 'r') as f:
        original = json.load(f)

    targets_15percent = {}
    for spec_id, spec in original.items():
        gain_linear = spec['target_gain_linear'] * 1.15
        ugbw_mhz = spec['target_ugbw_mhz'] * 1.15
        pm_deg = spec['target_pm_deg']
        ibias_ma = spec['target_ibias_ma'] * 1.15

        gain_db = 20 * math.log10(gain_linear) if gain_linear > 0 else 0

        targets_15percent[spec_id] = {
            'target_gain_linear': gain_linear,
            'target_gain_db': gain_db,
            'target_ugbw_mhz': ugbw_mhz,
            'target_pm_deg': pm_deg,
            'target_ibias_ma': ibias_ma
        }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(targets_15percent, f, indent=2)

    print(f"Generated {len(targets_15percent)} specs with 15% increased targets")
    print(f"Example spec 0: gain {original['0']['target_gain_linear']} -> {targets_15percent['0']['target_gain_linear']:.2f}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == '__main__':
    import sys
    print("Use main.py as the only entry point: python main.py")
    sys.exit(1)
