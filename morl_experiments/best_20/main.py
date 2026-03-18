"""
Best 20 graphs - Single entry point.
Generates all report graphs: best20_pca_scatter, best20_both_4subplots,
best1000_original_morl_4subplots, random20_pca_scatter.
Uses seed=20 by default. Run with --random for random seed.
"""

from generate_all_report_graphs import main


if __name__ == "__main__":
    main()
