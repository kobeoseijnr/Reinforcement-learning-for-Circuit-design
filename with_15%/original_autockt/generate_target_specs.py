"""
Regenerate target_specs_original.json and gen_specs pickle from the original YAML ranges.
Uses the same sampling logic as gen_specs.py - no 15% increase applied.
Run from with_15%/original_autockt: python generate_target_specs.py --num_specs 1000
"""

import json
import pickle
import random
import yaml
import argparse
from pathlib import Path
from collections import OrderedDict


class OrderedDictYAMLLoader(yaml.Loader):
    def __init__(self, *args, **kwargs):
        yaml.Loader.__init__(self, *args, **kwargs)
        self.add_constructor(u'tag:yaml.org,2002:map', type(self).construct_yaml_map)
        self.add_constructor(u'tag:yaml.org,2002:omap', type(self).construct_yaml_map)

    def construct_yaml_map(self, node):
        data = OrderedDict()
        yield data
        value = self.construct_mapping(node)
        data.update(value)

    def construct_mapping(self, node, deep=False):
        if isinstance(node, yaml.MappingNode):
            self.flatten_mapping(node)
        else:
            raise yaml.constructor.ConstructorError(None, None,
                'expected a mapping node, but found %s' % node.id, node.start_mark)
        mapping = OrderedDict()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping


def generate_target_specs(yaml_path, num_specs, output_dir, seed=42):
    """Generate target specs from YAML ranges and save as JSON."""
    random.seed(seed)
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.load(f, OrderedDictYAMLLoader)

    specs_range = yaml_data['target_specs']
    specs_range_vals = list(specs_range.values())
    specs_valid = []
    for spec in specs_range_vals:
        if isinstance(spec[0], int):
            list_val = [random.randint(int(spec[0]), int(spec[1])) for _ in range(num_specs)]
        else:
            list_val = [random.uniform(float(spec[0]), float(spec[1])) for _ in range(num_specs)]
        specs_valid.append(list_val)

    targets = {}
    for i in range(num_specs):
        gain_linear = float(specs_valid[0][i])
        ugbw_hz = float(specs_valid[1][i])
        phm_deg = float(specs_valid[2][i])
        ibias_A = float(specs_valid[3][i])

        gain_db = 20 * (__import__('math').log10(gain_linear)) if gain_linear > 0 else 0
        ugbw_mhz = ugbw_hz / 1e6
        ibias_ma = ibias_A * 1000.0

        targets[str(i)] = {
            'target_gain_linear': gain_linear,
            'target_gain_db': gain_db,
            'target_ugbw_mhz': ugbw_mhz,
            'target_pm_deg': phm_deg,
            'target_ibias_ma': ibias_ma
        }

    output_path = Path(output_dir) / 'target_specs_original.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(targets, f, indent=2)

    specs_range_dict = OrderedDict([
        ('gain_min', tuple(specs_valid[0])),
        ('ugbw_min', tuple(specs_valid[1])),
        ('phm_min', tuple(specs_valid[2])),
        ('ibias_max', tuple(specs_valid[3]))
    ])
    pickle_path = Path(output_dir).parent / 'autockt' / 'gen_specs' / 'ngspice_specs_gen_two_stage_opamp'
    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pickle_path, 'wb') as f:
        pickle.dump(specs_range_dict, f)

    print(f"Generated {num_specs} target specs from original YAML ranges")
    print(f"Ranges: gain [200,400], ugbw [1e6, 2.5e7] Hz, pm [75, 75.0000001] deg, ibias [0.0001, 0.01] A")
    print(f"Saved: {output_path}")
    print(f"Saved: {pickle_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Regenerate target_specs_original.json from YAML ranges')
    parser.add_argument('--num_specs', type=int, default=1000, help='Number of specs to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    BASE_DIR = Path(__file__).parent
    YAML_PATH = BASE_DIR / 'eval_engines' / 'ngspice' / 'ngspice_inputs' / 'yaml_files' / 'two_stage_opamp.yaml'
    OUTPUT_DIR = BASE_DIR / 'data'

    if not YAML_PATH.exists():
        print(f"ERROR: YAML not found: {YAML_PATH}")
        return 1

    generate_target_specs(YAML_PATH, args.num_specs, OUTPUT_DIR, args.seed)
    return 0


if __name__ == '__main__':
    import sys
    print("Use main.py as the only entry point: python main.py")
    sys.exit(1)
