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

*   **`cli_preprocessing.py`**: Handles initial data preparation, including synchronizing datasets and creating structured dataframes.
*   **`cli_fusion.py`**: Manages the generation of job files and the execution of the cell segmentation fusion process to create the silver truth.
*   **`cli_evaluation.py`**: Provides tools for evaluating competitor algorithms against ground truth or the generated silver truth.

## Getting Started

For detailed installation instructions and basic usage examples, please refer to the main [README.md](../README.md) file in the project repository.

## Further Documentation

*   [Label Synchronization Process](label_synchronizer.md): Detailed explanation of how labels are synchronized.
*   [Silver Truth Generation Algorithm](Silver-truth-generation.md): Information on how the silver truth is computationally derived.
*   [Evaluation Strategy](Evaluations.md): Details on the metrics and approach used for evaluating results.
*   [Jupyter Notebooks Overview](notebooks.MD): A guide to the various analytical and utility notebooks.
*   [Project Roadmap](Roadmap.md): High-level overview of project phases, goals, and progress.
*   [References and Related Publications](References.md): Context for key research papers relevant to the project.

## Contact

Issues and questions can be raised on GitHub.