# MORL AutoCkt — Multi-Objective Reinforcement Learning for Analog Circuit Design

This repository contains experiments comparing **Multi-Objective Reinforcement Learning (MORL)** agents against the **Original AutoCkt** baseline for analog circuit (two-stage op-amp) optimization with a **15% tolerance** threshold.

---

## Project Structure

```
with_15%/
├── morl_autockt/             # MORL experiment code and results
│   ├── autockt/              # AutoCkt environment (gym-based RL env)
│   ├── methodology/          # MORL agent implementations
│   │   ├── autockt/
│   │   │   ├── models/       # Neural network architectures (DDQN, etc.)
│   │   │   ├── evaluation/   # Multi-objective evaluator (hypervolume, sparsity)
│   │   │   └── utils/        # MO utilities
│   │   └── eval_engines/     # NGSpice simulation engine & surrogate wrapper
│   ├── data/                 # Target specifications (JSON)
│   ├── results/              # All experiment results (CSV, JSON)
│   ├── main.py               # Main training entry point
│   ├── evaluate.py           # Evaluation script
│   ├── train_nw_vs_cosine.py # Train NW agent vs Cosine agent
│   ├── gen_nw_original.py    # Generate NW results on original specs
│   └── merge_llm_to_cosine.py# Merge LLM-guided results into cosine CSV
│
├── original_autockt/         # Original AutoCkt baseline (single-objective RL)
│   ├── autockt/              # Original AutoCkt environment
│   ├── eval_engines/         # NGSpice simulation engine
│   ├── results/              # Baseline results
│   │   ├── original_autockt_results_original.csv
│   │   └── original_autockt_results_15percent.csv
│   ├── main.py               # Original AutoCkt training
│   └── evaluate.py           # Original AutoCkt evaluation
│
├── best_20/                  # Analysis & graph generation
│   ├── generate_std_vs_llm_graphs.py   # 4-group comparison graphs
│   ├── generate_all_report_graphs.py   # Full report graph generation
│   ├── Comprehensive_Comparison_Report.md
│   └── std_vs_llm_figures/             # Output figures
│
└── create_best_comparison.py # Best-of-1000 comparison script
```

---

## Agents Compared

| Agent | Description | Reward Function |
|-------|-------------|-----------------|
| **Original AutoCkt** | Single-objective RL baseline (DQN) | Original scalar reward |
| **Std MORL (cosine)** | Multi-objective DDQN with cosine-similarity reward | Cosine similarity to preference vector |
| **LLM MORL (cosine+llm)** | MORL with LLM-guided preference shaping | Cosine + LLM-based reward adjustment |
| **NW MORL** | MORL with neural-network-based reward shaping | Cosine + learned NW reward |

---

## Key Results Files

### Raw Training Results
- `morl_autockt_results_original.csv` — Std MORL on original specs
- `morl_autockt_results_llm_cosine.csv` — LLM-guided MORL agent
- `morl_autockt_results_nw.csv` — NW MORL agent
- `morl_autockt_results_original_cosine_with_llm.csv` — Combined cosine + LLM results

### Best-per-Spec (1000 specs)
- `morl_best_per_spec_1000.csv` — Best FOM per spec from Std MORL
- `morl_best_per_spec_llm_cosine.csv` — Best FOM per spec from LLM agent
- `morl_best_per_spec_nw.csv` — Best FOM per spec from NW agent

### Top 10 Specs
- `morl_top10_llm_cosine.csv` — Top 10 specs by FOM (LLM agent)
- `morl_top10_nw.csv` — Top 10 specs by FOM (NW agent)
- `morl_top10_std_cosine.csv` — Top 10 specs by FOM (Std MORL)

### Metrics
- `hypervolume_sparsity_comparison.csv` — Per-spec hypervolume and sparsity for all agents

---

## Hypervolume & Sparsity Summary

| Metric | Original AutoCkt | Std MORL (cosine) | LLM MORL (cosine+llm) |
|--------|-----------------|-------------------|----------------------|
| Mean Hypervolume | 1.40e+05 | 3.83e+09 | 2.62e+09 |
| Median Hypervolume | 9.44e+04 | 3.82e+09 | 2.61e+09 |
| Mean Sparsity | 0.00 | 3.60e+05 | 1.35e+05 |
| Mean Pareto Front Size | 1.00 | 20.00 | 20.00 |

- MORL agents produce Pareto fronts of ~20 solutions per spec vs 1 for Original AutoCkt.
- LLM-guided MORL achieves lower sparsity (tighter Pareto front coverage).

---

## Average FOM Comparison

| Agent | Avg FOM (1000 specs) |
|-------|---------------------|
| Original AutoCkt (15%) | 0.4335 |

---

## Graph Generation

### 4-Group Comparison Graphs (6-subplot pairwise objectives)

Run `generate_std_vs_llm_graphs.py` to produce:
- `top10_6subplots_std_vs_llm_FINAL.png` — LLM top 10 specs, all 4 agents
- `top10_6subplots_std_vs_nw_FINAL.png` — NW top 10 specs, all 4 agents
- `top10_6subplots_4groups_auckt.png` — AutoCkt top 10 specs, all 4 agents

**Marker legend:**
- ○ Circle → Target
- ⬡ Hexagon → Original AutoCkt
- ■ Square → Std MORL (cosine)
- ▲ Triangle → LLM MORL or NW

```bash
cd with_15%/best_20
python generate_std_vs_llm_graphs.py
```

### Full Report Graphs

```bash
cd with_15%/best_20
python generate_all_report_graphs.py
```

---

## Environment

- **Circuit:** Two-stage operational amplifier (45nm bulk CMOS)
- **Simulator:** NGSpice (via surrogate wrapper)
- **Objectives:** Gain (dB), UGBW (MHz), Phase Margin (deg), Bias Current (mA)
- **Tolerance:** 15% above target specifications
- **Seed:** 42

---

## Dependencies

- Python 3.8+
- NumPy, Pandas, Matplotlib
- PyTorch (for MORL agent training)
- NGSpice (for circuit simulation)
