# RLMAG: A Reinforcement Learning-based Multi-Objective Analog Circuit Design Optimizer with Large Language Models Guidance

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-red.svg)](https://pytorch.org/)
[![NGSpice](https://img.shields.io/badge/Simulator-NGSpice-green.svg)](http://ngspice.sourceforge.net/)

## Overview

This project extends the [AutoCkt](https://arxiv.org/abs/2001.01808) framework — an RL-based approach to analog circuit parameter optimization — into a **multi-objective** setting. Instead of collapsing multiple circuit performance metrics into a single scalar reward, our **Multi-Objective Reinforcement Learning (MORL)** agents learn to produce diverse **Pareto-optimal** solution sets that simultaneously optimize competing objectives.

We compare three MORL agent variants against the original single-objective AutoCkt baseline, evaluating them across **1,000 target specifications** for a two-stage operational amplifier designed in **45nm bulk CMOS** technology.

---

## Motivation

Analog circuit design involves balancing multiple conflicting objectives — for example, increasing gain often comes at the cost of bandwidth or power consumption. Traditional RL approaches like AutoCkt use a weighted scalar reward, which:

- Requires manual tuning of objective weights
- Produces only a single solution per optimization run
- Cannot capture the full trade-off landscape

Our MORL approach addresses these limitations by:

- Learning a **Pareto front** of non-dominated solutions per specification
- Using **cosine similarity** to a preference vector as a multi-objective reward signal
- Incorporating **LLM-guided** and **neural-network-based** reward shaping to improve solution quality and diversity

---

## Circuit Under Test

| Parameter | Value |
|-----------|-------|
| **Circuit** | Two-stage operational amplifier |
| **Technology** | 45nm bulk CMOS |
| **Simulator** | NGSpice (via surrogate model) |
| **Design Variables** | Transistor widths, lengths, bias currents |

### Optimization Objectives

| Objective | Unit | Direction |
|-----------|------|-----------|
| **Gain** | dB | Maximize |
| **Unity-Gain Bandwidth (UGBW)** | MHz | Maximize |
| **Phase Margin (PM)** | degrees | Maximize |
| **Bias Current (Ibias)** | mA | Minimize |

---

## Agents Compared

| Agent | Architecture | Reward Function | Description |
|-------|-------------|-----------------|-------------|
| **Original AutoCkt** | DQN | Scalar weighted sum | Single-objective baseline — produces 1 solution per spec |
| **Std MORL (cosine)** | Double DQN | Cosine similarity to preference vector | Standard multi-objective agent — produces a Pareto front of ~20 solutions |
| **LLM MORL (cosine+llm)** | Double DQN | Cosine + LLM-based reward adjustment | Uses a Large Language Model to guide preference shaping for improved Pareto coverage |
| **NW MORL** | Double DQN | Cosine + learned neural-network reward | Uses a trained neural network to adaptively shape the reward signal |

### Agent Architecture Details

All MORL agents use a **Double Deep Q-Network (DDQN)** with:
- Experience replay buffer
- Target network with periodic updates
- Epsilon-greedy exploration with decay
- Multi-objective reward decomposition via cosine similarity

The **LLM-guided** variant augments the cosine reward with preference adjustments suggested by a language model, while the **NW variant** learns an auxiliary reward network that co-trains with the policy.

---

## Key Results

### Hypervolume & Sparsity Comparison (1,000 specs)

| Metric | Original AutoCkt | Std MORL (cosine) | LLM MORL (cosine+llm) |
|--------|:---------------:|:-----------------:|:--------------------:|
| **Mean Hypervolume** | 1.40e+05 | **3.83e+09** | 2.62e+09 |
| **Median Hypervolume** | 9.44e+04 | **3.82e+09** | 2.61e+09 |
| **Std Hypervolume** | 1.33e+05 | 2.49e+08 | 9.13e+07 |
| **Mean Sparsity** | 0.00 | 3.60e+05 | **1.35e+05** |
| **Mean Pareto Front Size** | 1.00 | **20.00** | **20.00** |

> **Hypervolume** measures the volume of objective space dominated by the Pareto front — higher is better.
> **Sparsity** measures how spread out solutions are along the front — lower means tighter, more uniform coverage.

### Key Findings

1. **MORL massively outperforms AutoCkt on hypervolume** — Std MORL achieves ~27,000x higher hypervolume than Original AutoCkt, demonstrating vastly superior multi-objective coverage.

2. **Pareto front diversity** — MORL agents produce fronts of ~20 non-dominated solutions per spec, compared to just 1 for AutoCkt. This gives designers a rich set of trade-off options.

3. **LLM-guided reward shaping improves Pareto coverage** — The LLM variant achieves lower sparsity (1.35e+05 vs 3.60e+05), meaning its solutions are more evenly distributed along the Pareto front.

4. **Original AutoCkt has zero sparsity** — because it only produces a single solution (front size = 1), there is no spread to measure.

### Average Figure of Merit (FOM)

| Agent | Avg FOM (1,000 specs) |
|-------|:--------------------:|
| Original AutoCkt | 0.4335 |

---

## Project Structure

```
morl_experiments/
├── morl_autockt/                          # MORL experiment code and results
│   ├── autockt/                           # OpenAI Gym environment for the op-amp
│   │   ├── envs/                          # Gym environment definitions
│   │   └── gen_specs/                     # Target specification generator
│   ├── methodology/                       # MORL agent implementations
│   │   ├── autockt/
│   │   │   ├── models/                    # DDQN neural network architectures
│   │   │   ├── evaluation/                # Multi-objective evaluators (HV, sparsity)
│   │   │   └── utils/                     # MORL utility functions
│   │   └── eval_engines/                  # NGSpice simulation & surrogate wrapper
│   │       └── ngspice/
│   │           ├── ngspice_inputs/        # Netlists, SPICE models, YAML configs
│   │           ├── ngspice_wrapper.py     # Direct NGSpice interface
│   │           └── surrogate_wrapper.py   # Surrogate model for fast evaluation
│   ├── data/                              # Target specification files (JSON)
│   ├── results/                           # All experiment outputs
│   │   ├── morl_best_per_spec_*.csv       # Best FOM per spec for each agent
│   │   ├── morl_top10_*.csv               # Top 10 specs by FOM per agent
│   │   ├── hypervolume_sparsity_comparison.csv
│   │   ├── training_history_*.json        # Training curves
│   │   └── models_*/                      # Saved model checkpoints
│   ├── main.py                            # Main MORL training entry point
│   ├── evaluate.py                        # Post-training evaluation
│   ├── train_nw_vs_cosine.py              # NW agent vs Cosine agent comparison
│   ├── gen_nw_original.py                 # Generate NW results on original specs
│   └── merge_llm_to_cosine.py             # Merge LLM results into cosine CSV
│
├── original_autockt/                      # Original AutoCkt baseline
│   ├── autockt/                           # Original Gym environment
│   ├── eval_engines/                      # Original NGSpice engine
│   ├── results/                           # Baseline CSV results
│   ├── graphs/                            # Baseline performance plots
│   ├── main.py                            # Original training script
│   └── evaluate.py                        # Original evaluation script
│
├── best_20/                               # Analysis & visualization
│   ├── generate_std_vs_llm_graphs.py      # 4-group comparison graphs (6 subplots)
│   ├── generate_all_report_graphs.py      # Full report figure generation
│   ├── Comprehensive_Comparison_Report.md # Detailed written comparison
│   └── std_vs_llm_figures/                # Output figure directory
│
└── create_best_comparison.py              # Best-of-1000 comparison script
```

---

## Results Files Reference

### Raw Training Results (per episode, all 1,000 specs)
| File | Description |
|------|-------------|
| `morl_autockt_results_original.csv` | Std MORL agent outputs |
| `morl_autockt_results_llm_cosine.csv` | LLM-guided MORL agent outputs |
| `morl_autockt_results_nw.csv` | NW MORL agent outputs |
| `morl_autockt_results_original_cosine_with_llm.csv` | Combined cosine + LLM results |

### Aggregated Results (best per spec)
| File | Description |
|------|-------------|
| `morl_best_per_spec_1000.csv` | Best FOM per spec — Std MORL |
| `morl_best_per_spec_llm_cosine.csv` | Best FOM per spec — LLM agent |
| `morl_best_per_spec_nw.csv` | Best FOM per spec — NW agent |

### Top Performing Specs
| File | Description |
|------|-------------|
| `morl_top10_llm_cosine.csv` | Top 10 specs by FOM — LLM agent |
| `morl_top10_nw.csv` | Top 10 specs by FOM — NW agent |
| `morl_top10_std_cosine.csv` | Top 10 specs by FOM — Std MORL |

### Multi-Objective Metrics
| File | Description |
|------|-------------|
| `hypervolume_sparsity_comparison.csv` | Per-spec hypervolume and sparsity for all agents |
| `training_history_cosine.json` | Training reward curves — cosine agent |
| `training_history_nw.json` | Training reward curves — NW agent |

---

## Visualization

### 4-Group Comparison Graphs

6-subplot pairwise objective plots comparing all four agents on the same specifications:

```bash
cd morl_experiments/best_20
python generate_std_vs_llm_graphs.py
```

**Outputs:**
| File | Description |
|------|-------------|
| `top10_6subplots_std_vs_llm_FINAL.png` | LLM top 10 specs — all 4 agents |
| `top10_6subplots_std_vs_nw_FINAL.png` | NW top 10 specs — all 4 agents |
| `top10_6subplots_4groups_auckt.png` | AutoCkt top 10 specs — all 4 agents |

**Marker Legend:**

| Marker | Group |
|--------|-------|
| ○ Circle | Target specification |
| ⬡ Hexagon | Original AutoCkt |
| ■ Square | Std MORL (cosine) |
| ▲ Triangle | LLM MORL or NW |

### Full Report Graphs

```bash
cd morl_experiments/best_20
python generate_all_report_graphs.py
```

---

## Getting Started

### Prerequisites

- **Python** 3.8+
- **PyTorch** (for DDQN agent training)
- **NGSpice** (for circuit simulation, or use the included surrogate model)
- **NumPy**, **Pandas**, **Matplotlib** (data processing and visualization)

### Installation

```bash
git clone https://github.com/kobeoseijnr/Reinforcement-learning-for-Circuit-design.git
cd Reinforcement-learning-for-Circuit-design
pip install numpy pandas matplotlib torch
```

### Training a MORL Agent

```bash
cd morl_experiments/morl_autockt
python main.py
```

### Evaluating Results

```bash
cd morl_experiments/morl_autockt
python evaluate.py
```

### Generating Comparison Graphs

```bash
cd morl_experiments/best_20
python generate_std_vs_llm_graphs.py
python generate_all_report_graphs.py
```

---

## Methodology

### Multi-Objective Reward Design

Instead of a single scalar reward, our agents receive a **cosine similarity** score measuring alignment between the achieved objective vector and a target preference vector:

```
reward = cos(achieved_objectives, target_preference)
```

This encourages the agent to move toward the desired region of objective space while maintaining diversity across the Pareto front.

### LLM-Guided Reward Shaping

The LLM variant queries a language model to suggest **preference adjustments** based on the current state of the Pareto front. This helps:
- Focus exploration on under-represented regions of the front
- Improve uniformity of solution spacing
- Adapt preference vectors dynamically during training

### Neural Network Reward Shaping

The NW variant trains an auxiliary neural network alongside the policy to learn an adaptive reward bonus. This network:
- Takes the current state and objective values as input
- Outputs a reward adjustment term
- Is co-trained to maximize Pareto front quality metrics

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Random Seed | 42 |
| Number of Specs | 1,000 |
| Pareto Front Size | 20 solutions per spec |
| Training Episodes | 5,000 (NW agent) |
| Exploration | Epsilon-greedy with decay |

---

## License

This project is for academic and research purposes.

---

## Acknowledgments

- Original [AutoCkt](https://github.com/ksettaluri6/AutoCkt) framework by Settaluri et al.
- NGSpice open-source circuit simulator
- PyTorch deep learning framework
