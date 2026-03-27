# Silver Truth Project Documentation

## Introduction

Welcome to the documentation for the Silver Truth project. This project focuses on generating and evaluating "silver-standard" corpora for cell tracking challenges.

A **Silver-standard corpus (silver truth)** is defined as computer-generated reference annotations, obtained as the majority opinion over the results of several competitive algorithms submitted by former challenge participants. This corpus serves as a robust benchmark for evaluating new cell tracking algorithms.

## Project Goal

This project aims to create a new, better silver truth using more advanced techniques of Quality Assurance and Ensemble methods.

## Key Components & Workflow

The Silver Truth project is structured around a series of command-line tools that facilitate the entire process from data preparation to evaluation.

### High-Level Workflow Diagram

Raw Data -> Synchronization -> DataFrame -> Job Files -> Fusion -> Evaluation

### Core Modules

*   **`silver-preprocessing` / `src/silver_truth/cli/preprocessing.py`**: Synchronization, dataframe generation, and segmentation size checks.
*   **`silver-qa` / `src/silver_truth/cli/qa.py`**: QA crop generation and QA CNN training.
*   **`silver-fusion` / `src/silver_truth/cli/fusion.py`**: Job-file generation, Java fusion, and crop-level fusion orchestration.
*   **`silver-evaluation` / `src/silver_truth/cli/evaluation.py`**: Competitor evaluation, QA evaluation, fusion reconstruction scoring, and parquet filtering.
*   **`silver-ensemble` / `src/silver_truth/cli/ensemble.py`**: Databank building, ensemble training, and checkpoint evaluation.
*   **`scripts/run_ablation.py`**: Config-driven orchestration for Phase A/B/C paper experiments.

## Getting Started

For detailed installation instructions and basic usage examples, please refer to the main [README.md](../README.md) file in the project repository.

## Further Documentation

*   [Current Pipeline Handoff](current_pipeline_handoff.md): Current-state summary of what the repository does, what QA and ensemble consume/produce, and which metric is the final comparable one.
*   [Repository Workflow Map](repository_workflow_map.md): Canonical map of what currently runs, what each CLI does, and where ablations are orchestrated.
*   [Executable Index](executable_index.md): Task-oriented list of what agents should try first, including maintained CLI entrypoints, scripts, and reusable functions.
*   [DVC Guide](dvc_guide.md): Complete guide to setting up DVC and downloading datasets.
*   [Paper Protocol (Fold-Locked)](paper_protocol.md): Exact paper run order from fold prep to ablation/sweep.
*   [Ensemble C3 Experiment Log](ensemble_c3_experiment_log.md): Negative-result log for the richer consensus-channel ensemble input experiment on HSC and MuSC.
*   [Ensemble Transfer Plan](ensemble_transfer_plan.md): Fold-locked plan for validating MuSC first, then testing MuSC -> HSC transfer.
*   [Function Index](function_index.md): Fast map of key functions/CLI commands by workflow.
*   [API Reference (Generated)](api_reference_generated.md): Signatures and one-line summaries from docstrings.
*   [Label Synchronization Process](label_synchronizer.md): Detailed explanation of how labels are synchronized.
*   [Silver Truth Generation Algorithm](Silver-truth-generation.md): Information on how the silver truth is computationally derived.
*   [Evaluation Strategy](Evaluations.md): Details on the metrics and approach used for evaluating results.
*   [Jupyter Notebooks Overview](notebooks.MD): A guide to the various analytical and utility notebooks.
*   [Project Roadmap](Roadmap.md): High-level overview of project phases, goals, and progress.
*   [References and Related Publications](References.md): Context for key research papers relevant to the project.

## Contact

Issues and questions can be raised on GitHub.
