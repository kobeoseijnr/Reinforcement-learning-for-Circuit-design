"""
Generate NW (Normalized Weighted Sum) scalarization version of morl_autockt_results_original.csv
using the same raw data and reward rule. Then compare NW vs Cosine results.
"""
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
RAW_RESULTS = BASE_DIR / "results" / "morl_autockt_results_raw.json"
GEN_SPECS = BASE_DIR / "autockt" / "gen_specs" / "ngspice_specs_gen_two_stage_opamp"
OUTPUT_DIR = BASE_DIR / "results"

# ---------------------------------------------------------------------------
# Same functions from main.py
# ---------------------------------------------------------------------------
def lookup(spec, goal_spec):
    spec = np.array([float(e) for e in spec])
    goal_spec = np.array([float(e) for e in goal_spec])
    return (spec - goal_spec) / (goal_spec + spec)

def reward(spec, goal_spec, specs_id):
    rel_specs = lookup(spec, goal_spec)
    reward_val = 0.0
    for i, rel_spec in enumerate(rel_specs):
        if specs_id[i] == 'ibias_max':
            rel_spec = rel_spec * -1.0
        if rel_spec < 0:
            reward_val += rel_spec
    return reward_val if reward_val < -0.02 else 10

def nw_scalarization(reward_vector, preference):
    """Normalized Weighted Sum: dot product of reward and preference."""
    return float(np.dot(reward_vector, preference))

def cosine_similarity_scalarization(reward_vector, preference):
    """Cosine similarity scalarization (for comparison)."""
    reward_norm = np.linalg.norm(reward_vector)
    pref_norm = np.linalg.norm(preference)
    if reward_norm == 0 or pref_norm == 0:
        return 0.0
    cosine_sim = np.dot(reward_vector, preference) / (reward_norm * pref_norm)
    return float(cosine_sim * reward_norm)

def _fom(g_out, u_out, p_out, i_out, g_tgt, u_tgt, p_tgt, i_tgt):
    def _safe(x):
        return max(float(x), 1e-9) if x is not None and not (isinstance(x, float) and np.isnan(x)) else 1e-9
    gt, ut, pt, it = _safe(g_tgt), _safe(u_tgt), _safe(p_tgt), _safe(i_tgt)
    go, uo, po, io = float(g_out), float(u_out), float(p_out), float(i_out)
    if any(np.isnan(v) for v in [go, uo, po, io]):
        return np.nan
    return (go - gt) / gt + (uo - ut) / ut + (po - pt) / pt - (io - it) / it

def check_target_reached_strict(actual_specs, target_specs, specs_id):
    actual_dict = dict(zip(specs_id, actual_specs))
    target_dict = dict(zip(specs_id, target_specs))
    gain_actual = actual_dict.get('gain_min', actual_specs[0])
    ugbw_actual = actual_dict.get('ugbw_min', actual_specs[3])
    phm_actual = actual_dict.get('phm_min', actual_specs[2])
    ibias_actual = actual_dict.get('ibias_max', actual_specs[1])
    gain_target = target_dict.get('gain_min', target_specs[0])
    ugbw_target = target_dict.get('ugbw_min', target_specs[3])
    phm_target = target_dict.get('phm_min', target_specs[2])
    ibias_target = target_dict.get('ibias_max', target_specs[1])
    if gain_actual < 100:
        gain_actual = 10 ** (gain_actual / 20)
    if gain_target < 100:
        gain_target = 10 ** (gain_target / 20)
    spec_array = np.array([gain_actual, ibias_actual, phm_actual, ugbw_actual])
    goal_array = np.array([gain_target, ibias_target, phm_target, ugbw_target])
    reward_val = reward(spec_array, goal_array, specs_id)
    return reward_val >= 10

def load_targets_from_gen_specs(gen_specs_path):
    with open(gen_specs_path, 'rb') as f:
        specs = pickle.load(f)
    n = len(list(specs.values())[0])
    target_specs = {}
    for i in range(n):
        target_specs[str(i)] = {
            'target_gain_linear': float(specs['gain_min'][i]),
            'target_ugbw_mhz': float(specs['ugbw_min'][i]) / 1e6,
            'target_pm_deg': float(specs['phm_min'][i]),
            'target_ibias_ma': float(specs['ibias_max'][i]) * 1000.0,
        }
    return target_specs

# ---------------------------------------------------------------------------
# Process with NW scalarization
# ---------------------------------------------------------------------------
def process_with_scalarization(raw_path, target_specs, scal_name, scal_fn):
    with open(raw_path) as f:
        raw = json.load(f)

    specs_id = ['gain_min', 'ibias_max', 'phm_min', 'ugbw_min']
    results = []

    for sol in raw['all_solutions']:
        spec_num = sol['spec']
        target_key = str(spec_num - 1)
        td = target_specs.get(target_key, {})

        g = sol.get('output_gain_linear', 0)
        u = sol.get('output_ugbw_mhz', 0)
        p = sol.get('output_pm_deg', 0)
        i = sol.get('output_ibias_ma', 0)

        actual = [g, i / 1000.0, p, u * 1e6]
        target = [
            td.get('target_gain_linear', 0),
            td.get('target_ibias_ma', 0) / 1000.0,
            td.get('target_pm_deg', 0),
            td.get('target_ugbw_mhz', 0) * 1e6,
        ]

        # target_reached using strict reward rule
        reached = check_target_reached_strict(actual, target, specs_id)

        # Compute reward vector for scalarization
        gt = td.get('target_gain_linear', 1e-9)
        ut = td.get('target_ugbw_mhz', 1e-9)
        pt = td.get('target_pm_deg', 1e-9)
        it = td.get('target_ibias_ma', 1e-9)
        gain_obj = (g - gt) / gt if gt > 0 else 0
        ugbw_obj = (u - ut) / ut if ut > 0 else 0
        phm_obj = (p - pt) / pt if pt > 0 else 0
        ibias_obj = -(i - it) / it if it > 0 else 0
        reward_vector = np.array([gain_obj, ugbw_obj, phm_obj, ibias_obj], dtype=np.float32)

        preference = sol.get('preference', [0.25, 0.25, 0.25, 0.25])
        pref_arr = np.array(preference, dtype=np.float32)
        pref_norm = np.linalg.norm(pref_arr)
        if pref_norm > 0:
            pref_arr = pref_arr / pref_norm

        scal_val = scal_fn(reward_vector, pref_arr)
        fom_val = _fom(g, u, p, i, gt, ut, pt, it)

        # Per-objective pass
        g_ok = g >= gt
        u_ok = u >= ut
        p_ok = p >= pt
        i_ok = i <= it

        results.append({
            'spec': spec_num,
            'solution': sol.get('solution', 1),
            'target_gain_linear': gt,
            'target_ugbw_mhz': ut,
            'target_pm_deg': pt,
            'target_ibias_ma': it,
            'output_gain_linear': g,
            'output_gain_db': sol.get('output_gain_db', 20 * np.log10(g) if g > 0 else 0),
            'output_ugbw_mhz': u,
            'output_pm_deg': p,
            'output_ibias_ma': i,
            'fom': round(float(fom_val), 6) if not np.isnan(fom_val) else None,
            'gain_pass': 'Yes' if g_ok else 'No',
            'ugbw_pass': 'Yes' if u_ok else 'No',
            'pm_pass': 'Yes' if p_ok else 'No',
            'ibias_pass': 'Yes' if i_ok else 'No',
            'complete_pass': 'Yes' if (g_ok and u_ok and p_ok and i_ok) else 'No',
            'target_reached': 'Yes' if reached else 'No',
            'scalarized_value': round(scal_val, 6),
            'preference': str(preference),
            'scalarization': scal_name,
        })

    return results


def results_to_csv(results, out_path, scal_name):
    df = pd.DataFrame(results)
    n = len(df)
    ns = df['spec'].nunique()
    passed = (df['target_reached'] == 'Yes').sum()
    avg_fom = df['fom'].astype(float).mean()
    best = df.groupby('spec')['fom'].max()
    avg_best = best.mean()
    top20 = best.nlargest(20).mean()

    summaries = pd.DataFrame([
        {'spec': 'summary', 'target_reached': f'{passed}/{n} solutions; {ns}/{ns} specs'},
        {'spec': 'summary_avg_fom_' + str(n) + '_solutions', 'fom': round(avg_fom, 6)},
        {'spec': 'summary_avg_best_fom_' + str(ns) + '_specs', 'fom': round(avg_best, 6)},
        {'spec': 'summary_avg_top20_best_fom', 'fom': round(top20, 6)},
    ])
    df = pd.concat([df, summaries], ignore_index=True)
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    print(f"  {scal_name}: {passed}/{n} pass, avg FOM={avg_fom:.6f}, avg best={avg_best:.6f}, top20={top20:.6f}")
    return avg_fom, avg_best


def generate_comparison(cosine_results, nw_results):
    def best_per_spec(results):
        by_spec = {}
        for sol in results:
            spec = sol['spec']
            f = sol['fom']
            if f is None:
                continue
            if spec not in by_spec or f > by_spec[spec]['fom']:
                by_spec[spec] = sol
        return by_spec

    cos_best = best_per_spec(cosine_results)
    nw_best = best_per_spec(nw_results)
    all_specs = sorted(set(cos_best.keys()) & set(nw_best.keys()))

    rows = []
    wins_cos = wins_nw = ties = 0
    cos_foms = []
    nw_foms = []

    for spec in all_specs:
        cf = cos_best[spec]['fom']
        nf = nw_best[spec]['fom']
        cos_foms.append(cf)
        nw_foms.append(nf)
        if abs(cf - nf) < 1e-9:
            w = 'tie'; ties += 1
        elif cf > nf:
            w = 'cosine'; wins_cos += 1
        else:
            w = 'nw'; wins_nw += 1
        rows.append({
            'spec': spec,
            'fom_cosine': round(cf, 6),
            'fom_nw': round(nf, 6),
            'scal_cosine': round(cos_best[spec]['scalarized_value'], 6),
            'scal_nw': round(nw_best[spec]['scalarized_value'], 6),
            'winner': w,
        })

    df = pd.DataFrame(rows)
    summaries = pd.DataFrame([
        {'spec': 'summary_specs', 'fom_cosine': len(rows)},
        {'spec': 'summary_wins_cosine', 'fom_cosine': wins_cos},
        {'spec': 'summary_wins_nw', 'fom_nw': wins_nw},
        {'spec': 'summary_ties', 'fom_cosine': ties},
        {'spec': 'summary_avg_fom_cosine', 'fom_cosine': round(np.mean(cos_foms), 6)},
        {'spec': 'summary_avg_fom_nw', 'fom_nw': round(np.mean(nw_foms), 6)},
    ])
    df = pd.concat([df, summaries], ignore_index=True)
    out = OUTPUT_DIR / "morl_compare_original_cosine_vs_nw.csv"
    df.to_csv(out, index=False)
    print(f"\n  Comparison saved: {out}")
    print(f"  Cosine wins: {wins_cos}, NW wins: {wins_nw}, Ties: {ties}")
    print(f"  Cosine avg best FOM: {np.mean(cos_foms):.6f}")
    print(f"  NW avg best FOM:     {np.mean(nw_foms):.6f}")
    print(f"  Difference (NW-Cos): {np.mean(nw_foms) - np.mean(cos_foms):.6f}")


# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  NW vs COSINE SCALARIZATION — ORIGINAL DATA")
    print("=" * 60)

    targets = load_targets_from_gen_specs(GEN_SPECS)

    print("\nProcessing with COSINE scalarization...")
    cosine_results = process_with_scalarization(
        RAW_RESULTS, targets, "cosine", cosine_similarity_scalarization
    )
    results_to_csv(cosine_results, OUTPUT_DIR / "morl_autockt_results_original_cosine.csv", "Cosine")

    print("\nProcessing with NW scalarization...")
    nw_results = process_with_scalarization(
        RAW_RESULTS, targets, "nw", nw_scalarization
    )
    results_to_csv(nw_results, OUTPUT_DIR / "morl_autockt_results_original_nw.csv", "NW")

    print("\nGenerating comparison...")
    generate_comparison(cosine_results, nw_results)

    print("\nDONE")
