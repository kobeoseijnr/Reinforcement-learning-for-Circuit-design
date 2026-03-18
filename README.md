# RLMAG: A Reinforcement Learning-based Multi-Objective Analog Circuit Design Optimizer with Large Language Models Guidance

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-red.svg)](https://pytorch.org/)
[![NGSpice](https://img.shields.io/badge/Simulator-NGSpice-green.svg)](http://ngspice.sourceforge.net/)

## Overview

Analog circuit design automation using Reinforcement Learning (RL) can reduce manual effort and speed up the design process. However, most existing RL methods optimize a single objective, and even those that claim multi-objective (MO) support collapse all design specs into one scalar reward. This hides the real trade-offs between objectives, produces sub-optimal designs, and forces retraining whenever specs change.

**RLMAG** solves these problems. It is an RL framework for multi-objective analog circuit optimization that uses **vector-valued learning** and **preference-aware conditioning** instead of a scalar reward. A preference vector controls the objective weights, so one trained model can produce designs for different trade-offs without retraining. Two preference guidance strategies, **Normalized Weight** and **Cosine-aligned guidance**, help the agent converge to high-quality Pareto fronts. An **LLM-guided action masking** step filters out actions that would lead to bad designs or waste runtime.

### Highlights

- **13x runtime speedup** over state-of-the-art methods
- **Meets all 1,000 target specs** (6.6% improvement over baselines)
- **300x better figure of merit** on output specifications
- **No retraining needed** when preferences change

## Motivation

Designing analog circuits means balancing competing goals. Increasing gain often costs bandwidth or power. Standard RL approaches like AutoCkt use a single weighted reward, which has three problems:

1. You have to manually tune objective weights
2. You only get one solution per run
3. You cannot see the full trade-off landscape

RLMAG fixes all three by learning a **Pareto front** of solutions per spec, using **cosine similarity** to a preference vector as the reward signal, and adding **LLM-guided** and **neural-network-based** reward shaping to improve quality and diversity.

## Circuit Under Test

| Parameter | Value |
|-----------|-------|
| Circuit | Two-stage operational amplifier |
| Technology | 45nm bulk CMOS |
| Simulator | NGSpice (via surrogate model) |
| Design Variables | Transistor widths, lengths, bias currents |

### Optimization Objectives

| Objective | Unit | Goal |
|-----------|------|------|
| Gain | dB | Maximize |
| Unity-Gain Bandwidth (UGBW) | MHz | Maximize |
| Phase Margin (PM) | degrees | Maximize |
| Bias Current (Ibias) | mA | Minimize |

## Agents Compared

| Agent | Architecture | Reward | What It Does |
|-------|-------------|--------|--------------|
| Original AutoCkt | DQN | Scalar weighted sum | Baseline, produces 1 solution per spec |
| Std MORL (cosine) | Double DQN | Cosine similarity | Multi-objective agent, produces ~20 solutions per spec |
| LLM MORL (cosine+llm) | Double DQN | Cosine + LLM adjustment | Uses an LLM to guide preference shaping for better Pareto coverage |
| NW MORL | Double DQN | Cosine + learned NN reward | Uses a neural network to adaptively shape the reward |

All MORL agents use a **Double Deep Q-Network (DDQN)** with experience replay, target network updates, epsilon-greedy exploration, and cosine similarity reward decomposition. The LLM variant adds preference adjustments from a language model. The NW variant co-trains an auxiliary reward network alongside the policy.

## Key Results

### Hypervolume and Sparsity (1,000 specs)

| Metric | Original AutoCkt | Std MORL | LLM MORL |
|--------|:---------------:|:--------:|:--------:|
| Mean Hypervolume | 1.40e+05 | **3.83e+09** | 2.62e+09 |
| Median Hypervolume | 9.44e+04 | **3.82e+09** | 2.61e+09 |
| Std Hypervolume | 1.33e+05 | 2.49e+08 | 9.13e+07 |
| Mean Sparsity | 0.00 | 3.60e+05 | **1.35e+05** |
| Mean Pareto Front Size | 1 | **20** | **20** |

**Hypervolume** = volume of objective space covered by the Pareto front (higher is better).
**Sparsity** = how spread out solutions are along the front (lower means more uniform coverage).

### What This Means

1. **MORL produces 27,000x higher hypervolume** than AutoCkt, showing vastly better multi-objective coverage.
2. **MORL gives ~20 solutions per spec** instead of just 1, providing designers with real trade-off options.
3. **LLM guidance improves coverage uniformity** with lower sparsity (1.35e+05 vs 3.60e+05).
4. **AutoCkt sparsity is zero** because it only returns one solution, so there is no front to measure.

### Average Figure of Merit

| Agent | Avg FOM (1,000 specs) |
|-------|:--------------------:|
| Original AutoCkt | 0.4335 |

## Project Structure

```
morl_experiments/
├── morl_autockt/                          # MORL code and results
│   ├── autockt/                           # OpenAI Gym environment for the op-amp
│   │   ├── envs/                          # Environment definitions
│   │   └── gen_specs/                     # Target spec generator
│   ├── methodology/                       # Agent implementations
│   │   ├── autockt/
│   │   │   ├── models/                    # DDQN architectures
│   │   │   ├── evaluation/                # Hypervolume, sparsity evaluators
│   │   │   └── utils/                     # Utility functions
│   │   └── eval_engines/                  # NGSpice and surrogate wrapper
│   │       └── ngspice/
│   │           ├── ngspice_inputs/        # Netlists, SPICE models, configs
│   │           ├── ngspice_wrapper.py     # Direct NGSpice interface
│   │           └── surrogate_wrapper.py   # Fast surrogate evaluator
│   ├── data/                              # Target spec files (JSON)
│   ├── results/                           # All outputs (CSV, JSON, models)
│   ├── main.py                            # Training entry point
│   ├── evaluate.py                        # Evaluation script
│   ├── train_nw_vs_cosine.py              # NW vs Cosine agent comparison
│   ├── gen_nw_original.py                 # NW results on original specs
│   └── merge_llm_to_cosine.py             # Merge LLM results into cosine CSV
│
├── original_autockt/                      # AutoCkt baseline
│   ├── autockt/                           # Original environment
│   ├── eval_engines/                      # Original NGSpice engine
│   ├── results/                           # Baseline results
│   ├── graphs/                            # Baseline plots
│   ├── main.py                            # Training script
│   └── evaluate.py                        # Evaluation script
│
├── best_20/                               # Analysis and visualization
│   ├── generate_std_vs_llm_graphs.py      # 4-group comparison graphs
│   ├── generate_all_report_graphs.py      # Full report figures
│   ├── Comprehensive_Comparison_Report.md # Written comparison
│   └── std_vs_llm_figures/                # Output figures
│
└── create_best_comparison.py              # Best-of-1000 comparison
```

## Results Files

### Training Results (all 1,000 specs)

| File | Description |
|------|-------------|
| `morl_autockt_results_original.csv` | Std MORL outputs |
| `morl_autockt_results_llm_cosine.csv` | LLM MORL outputs |
| `morl_autockt_results_nw.csv` | NW MORL outputs |
| `morl_autockt_results_original_cosine_with_llm.csv` | Combined cosine + LLM |

### Best Per Spec

| File | Description |
|------|-------------|
| `morl_best_per_spec_1000.csv` | Best FOM per spec, Std MORL |
| `morl_best_per_spec_llm_cosine.csv` | Best FOM per spec, LLM agent |
| `morl_best_per_spec_nw.csv` | Best FOM per spec, NW agent |

### Top 10 Specs

| File | Description |
|------|-------------|
| `morl_top10_llm_cosine.csv` | Top 10 by FOM, LLM agent |
| `morl_top10_nw.csv` | Top 10 by FOM, NW agent |
| `morl_top10_std_cosine.csv` | Top 10 by FOM, Std MORL |

### Metrics

| File | Description |
|------|-------------|
| `hypervolume_sparsity_comparison.csv` | Per-spec hypervolume and sparsity |
| `training_history_cosine.json` | Training curves, cosine agent |
| `training_history_nw.json` | Training curves, NW agent |

## Visualization

### Comparison Graphs

Generate 6-subplot pairwise objective plots comparing all four agents:

```bash
cd morl_experiments/best_20
python generate_std_vs_llm_graphs.py
```

Outputs:

| File | Description |
|------|-------------|
| `top10_6subplots_std_vs_llm_FINAL.png` | LLM top 10 specs, all 4 agents |
| `top10_6subplots_std_vs_nw_FINAL.png` | NW top 10 specs, all 4 agents |
| `top10_6subplots_4groups_auckt.png` | AutoCkt top 10 specs, all 4 agents |

Marker legend: Circle = Target, Hexagon = Original AutoCkt, Square = Std MORL, Triangle = LLM/NW MORL.

### Full Report Graphs

```bash
cd morl_experiments/best_20
python generate_all_report_graphs.py
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- NGSpice (or use the included surrogate model)
- NumPy, Pandas, Matplotlib

### Installation

```bash
git clone https://github.com/kobeoseijnr/Reinforcement-learning-for-Circuit-design.git
cd Reinforcement-learning-for-Circuit-design
pip install numpy pandas matplotlib torch
```

### Train

```bash
cd morl_experiments/morl_autockt
python main.py
```

### Evaluate

```bash
cd morl_experiments/morl_autockt
python evaluate.py
```

### Generate Graphs

```bash
cd morl_experiments/best_20
python generate_std_vs_llm_graphs.py
python generate_all_report_graphs.py
```

## Methodology

### Multi-Objective Reward

Instead of a scalar reward, agents get a **cosine similarity** score between their achieved objective vector and the target preference vector:

```
reward = cos(achieved_objectives, target_preference)
```

This pushes the agent toward the desired region of objective space while keeping solutions diverse across the Pareto front.

### LLM-Guided Action Masking

The LLM variant queries a language model to suggest preference adjustments based on the current Pareto front. This focuses exploration on under-represented regions, improves solution spacing, and adapts preference vectors during training.

### Neural Network Reward Shaping

The NW variant trains an auxiliary neural network alongside the policy. This network takes the current state and objective values as input and outputs a reward adjustment that is co-trained to maximize Pareto front quality.

## Configuration

| Parameter | Value |
|-----------|-------|
| Random Seed | 42 |
| Number of Specs | 1,000 |
| Pareto Front Size | 20 solutions per spec |
| Training Episodes | 5,000 (NW agent) |
| Exploration | Epsilon-greedy with decay |

## License

This project is for academic and research purposes.

## Acknowledgments

- Original [AutoCkt](https://github.com/ksettaluri6/AutoCkt) framework by Settaluri et al.
- NGSpice open-source circuit simulator
- PyTorch deep learning framework
