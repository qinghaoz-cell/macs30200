# Remote Work and Salary Inequality in the AI Labor Market

This repository contains data, analysis code, results, and poster materials for my MACS 30200 Computational Research Design project.

## Research Question

Does remote work reduce or reproduce salary inequality in AI- and data-related jobs?

The project uses salary data for AI, machine-learning, and data-related jobs to compare salary outcomes across on-site, hybrid, and remote work categories. The main preliminary finding is that remote work does not show a clear salary premium in this dataset; after comparing more similar jobs, remote positions remain slightly lower paid than on-site positions.

## Repository Structure

- `data/salaries.csv`: salary dataset used for the analysis.
- `scripts/extended_analysis.py`: main Python script for cleaning data, producing figures, matching comparisons, and adjusted salary estimates.
- `figures/`: figures used in the draft paper and poster.
- `results/`: generated CSV and LaTeX result tables.
- `poster/Poster_Draft.tex`: current Overleaf poster source.

## How to Reproduce

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the analysis:

```bash
python scripts/extended_analysis.py
```

The script reads `data/salaries.csv`, writes figures to `figures/`, and writes result tables to `results/`.

## Preliminary Results

- Full-time analytic sample: 150,541 observations.
- Raw mean salaries:
  - On-site: $159,716
  - Remote: $151,887
  - Hybrid: $82,814
- Matched comparison within the same experience level, company location, company size, and job title:
  - On-site mean: about $162,788
  - Remote mean: about $153,989
- Adjusted model:
  - Remote jobs are associated with about 5.9% lower salary than on-site jobs after controls.
  - Hybrid estimates are less stable because hybrid observations are rare.

These results are observational and should not be interpreted as causal.

## Data Source

The dataset is based on public AI/data job salary data from ai-jobs.net, distributed through Kaggle as a salaries for data science jobs dataset.
