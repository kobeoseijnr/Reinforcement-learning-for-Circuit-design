"""
Merge LLM-masked results into the original cosine CSV as new columns.
Matches by spec+solution, adds llm_ prefixed output columns, FOM, and summary.
"""
import pandas as pd
import numpy as np
import io

# Load original cosine CSV (skip summary rows)
cos_path = r"C:\Users\kobeo\OneDrive\Desktop\trail\with_15%_with_20\with_15%\morl_autockt\results\morl_autockt_results_original_cosine.csv"
lines = open(cos_path).readlines()
header = lines[0]
data_lines = [header] + [l for l in lines[1:] if not l.startswith('summary')]
df_cos = pd.read_csv(io.StringIO(''.join(data_lines)))

# Load LLM-masked CSV (skip summary rows)
llm_path = r"C:\Users\kobeo\OneDrive\Desktop\trail\OSEI_MORL_Aucrt (1)\OSEI_MORL_Aucrt\results\morl_autockt_results_llm_masked.csv"
lines2 = open(llm_path).readlines()
header2 = lines2[0]
data_lines2 = [header2] + [l for l in lines2[1:] if not l.startswith('summary')]
df_llm = pd.read_csv(io.StringIO(''.join(data_lines2)))

# Ensure spec/solution are ints
df_cos['spec'] = df_cos['spec'].astype(int)
df_cos['solution'] = df_cos['solution'].astype(int)
df_llm['spec'] = df_llm['spec'].astype(int)
df_llm['solution'] = df_llm['solution'].astype(int)

# Rename LLM columns with llm_ prefix
llm_cols = {
    'output_gain_linear': 'llm_output_gain_linear',
    'output_gain_db': 'llm_output_gain_db',
    'output_ugbw_mhz': 'llm_output_ugbw_mhz',
    'output_pm_deg': 'llm_output_pm_deg',
    'output_ibias_ma': 'llm_output_ibias_ma',
    'fom': 'llm_fom',
    'scalarization': 'llm_scalarization',
}
df_llm_renamed = df_llm[['spec', 'solution'] + list(llm_cols.keys())].rename(columns=llm_cols)

# Merge
merged = df_cos.merge(df_llm_renamed, on=['spec', 'solution'], how='left')

# Add pass/fail columns for LLM (strict)
merged['llm_gain_pass'] = merged.apply(lambda r: 'Yes' if r['llm_output_gain_linear'] >= r['target_gain_linear'] else 'No', axis=1)
merged['llm_ugbw_pass'] = merged.apply(lambda r: 'Yes' if r['llm_output_ugbw_mhz'] >= r['target_ugbw_mhz'] else 'No', axis=1)
merged['llm_pm_pass'] = merged.apply(lambda r: 'Yes' if r['llm_output_pm_deg'] >= r['target_pm_deg'] else 'No', axis=1)
merged['llm_ibias_pass'] = merged.apply(lambda r: 'Yes' if r['llm_output_ibias_ma'] <= r['target_ibias_ma'] else 'No', axis=1)
merged['llm_complete_pass'] = merged.apply(
    lambda r: 'Yes' if (r['llm_gain_pass'] == 'Yes' and r['llm_ugbw_pass'] == 'Yes' and
                        r['llm_pm_pass'] == 'Yes' and r['llm_ibias_pass'] == 'Yes') else 'No', axis=1)

# FOM winner per row
merged['fom_winner'] = merged.apply(
    lambda r: 'tie' if abs(r['fom'] - r['llm_fom']) < 1e-9
    else ('cosine' if r['fom'] > r['llm_fom'] else 'llm_masked'), axis=1)

# Summary stats
n = len(merged)
ns = merged['spec'].nunique()

cos_avg_fom = merged['fom'].mean()
llm_avg_fom = merged['llm_fom'].mean()

cos_best = merged.groupby('spec')['fom'].max()
llm_best = merged.groupby('spec')['llm_fom'].max()
cos_avg_best = cos_best.mean()
llm_avg_best = llm_best.mean()
cos_top20 = cos_best.nlargest(20).mean()
llm_top20 = llm_best.nlargest(20).mean()

# Best-per-spec winner counts
best_merged = pd.DataFrame({'cos_best': cos_best, 'llm_best': llm_best})
wins_cos = (best_merged['cos_best'] > best_merged['llm_best']).sum()
wins_llm = (best_merged['llm_best'] > best_merged['cos_best']).sum()
ties_best = (abs(best_merged['cos_best'] - best_merged['llm_best']) < 1e-9).sum()

cos_pass = (merged['complete_pass'] == 'Yes').sum()
llm_pass = (merged['llm_complete_pass'] == 'Yes').sum()

# Per-row winner counts
row_wins_cos = (merged['fom_winner'] == 'cosine').sum()
row_wins_llm = (merged['fom_winner'] == 'llm_masked').sum()
row_ties = (merged['fom_winner'] == 'tie').sum()

# Build summary rows
summary_data = [
    {'spec': 'summary_cosine', 'fom': f'avg_fom={cos_avg_fom:.6f}; avg_best={cos_avg_best:.6f}; top20={cos_top20:.6f}; pass={cos_pass}/{n}'},
    {'spec': 'summary_llm_masked', 'llm_fom': f'avg_fom={llm_avg_fom:.6f}; avg_best={llm_avg_best:.6f}; top20={llm_top20:.6f}; pass={llm_pass}/{n}'},
    {'spec': 'summary_best_per_spec_wins', 'fom': f'cosine={wins_cos}', 'llm_fom': f'llm={wins_llm}', 'fom_winner': f'ties={ties_best}'},
    {'spec': 'summary_per_row_wins', 'fom': f'cosine={row_wins_cos}', 'llm_fom': f'llm={row_wins_llm}', 'fom_winner': f'ties={row_ties}'},
    {'spec': 'summary_avg_fom_cosine', 'fom': round(cos_avg_fom, 6)},
    {'spec': 'summary_avg_fom_llm', 'llm_fom': round(llm_avg_fom, 6)},
    {'spec': 'summary_avg_best_fom_cosine', 'fom': round(cos_avg_best, 6)},
    {'spec': 'summary_avg_best_fom_llm', 'llm_fom': round(llm_avg_best, 6)},
    {'spec': 'summary_top20_best_fom_cosine', 'fom': round(cos_top20, 6)},
    {'spec': 'summary_top20_best_fom_llm', 'llm_fom': round(llm_top20, 6)},
]
summary_df = pd.DataFrame(summary_data)
out = pd.concat([merged, summary_df], ignore_index=True)

out_path = r"C:\Users\kobeo\OneDrive\Desktop\trail\with_15%_with_20\with_15%\morl_autockt\results\morl_autockt_results_original_cosine_with_llm.csv"
out.to_csv(out_path, index=False)

print(f"Merged: {n} rows, {ns} specs")
print(f"\n=== COSINE (original 1.488) ===")
print(f"  Avg FOM:      {cos_avg_fom:.6f}")
print(f"  Avg best FOM: {cos_avg_best:.6f}")
print(f"  Top 20 FOM:   {cos_top20:.6f}")
print(f"  Pass rate:    {cos_pass}/{n}")
print(f"\n=== LLM-MASKED ===")
print(f"  Avg FOM:      {llm_avg_fom:.6f}")
print(f"  Avg best FOM: {llm_avg_best:.6f}")
print(f"  Top 20 FOM:   {llm_top20:.6f}")
print(f"  Pass rate:    {llm_pass}/{n}")
print(f"\n=== BEST-PER-SPEC WINS ===")
print(f"  Cosine: {wins_cos}, LLM: {wins_llm}, Ties: {ties_best}")
print(f"\n=== PER-ROW WINS ===")
print(f"  Cosine: {row_wins_cos}, LLM: {row_wins_llm}, Ties: {row_ties}")
print(f"\nWINNER: {'LLM-MASKED' if llm_avg_best > cos_avg_best else 'COSINE'}")
print(f"\nSaved: {out_path}")
