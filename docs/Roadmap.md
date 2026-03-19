# Project Roadmap: Silver Truth - Beyond Recreation

This roadmap outlines the strategic progression of the "Silver Truth" project, moving from the current recreation phase towards advanced research and development in Quality Assurance (QA) and Dynamic Ensemble methods.

Important:
- this page is a high-level research backlog, not the canonical status tracker
- for current implementation status and runnable protocol, use `PAPER_STATUS.md`, `docs/paper_protocol.md`, and `docs/repository_workflow_map.md`

**Status Key:**
*   `[ ] TODO`: Not yet started.
*   `[x] DONE`: Completed.
*   `[ ] IN PROGRESS`: Currently being worked on.
*   `[ ] ON HOLD`: Temporarily paused.

---

## Phase 0: Current State - Silver Truth Recreation

**Goal:** To accurately reproduce the existing silver truth generation process using the provided tools and data.

*   **[x] DONE** Data synchronization (`silver-preprocessing synchronize-datasets`)
    *   _Status_: Completed.
    *   _Details_: Initial setup and verification of synchronization process.
*   **[x] DONE** Dataset DataFrame creation (`silver-preprocessing create-dataset-dataframe`)
    *   _Status_: Completed.
    *   _Details_: Generation of `.parquet` files for downstream use.
*   **[x] DONE** Job file generation (`silver-fusion generate-jobfiles`)
    *   _Status_: Implemented and used in DVC stages.
*   **[x] DONE** Fusion execution (`silver-fusion run-fusion` / `silver-fusion run-fusion-crops`)
    *   _Status_: Implemented and used for both direct fusion and crop-oriented experiments.
*   **[x] DONE** Evaluation of recreated silver truth against known benchmarks (`silver-evaluation evaluate-competitor`)
    *   _Status_: Implemented and used in current experiment runs.

---

## Phase 1: Novel QA-Enhanced Fusion (Core Research & Development)

**Goal:** To develop and integrate a novel Quality Assurance (QA) network and a Dynamic Ensemble Architecture to improve the quality of fused segmentations by proactively filtering low-quality inputs.

*   **[ ] TODO** Develop QA Network
    *   **[ ] TODO** Implement Frame-by-Frame QA
        *   _Status_: Research & Design.
        *   _Challenges_: [Link to QA Network Challenges (TODO)]
    *   **[ ] TODO** Implement Temporal QA (e.g., Temporal Consistency Scoring based on IoU between consecutive frames)
        *   _Status_: Research & Design.
        *   _Challenges_: [Link to Temporal QA Design (TODO)]
    *   **[ ] TODO** Develop Adaptive Thresholding for QA scores
        *   _Status_: Research & Design.
        *   _Challenges_: [Link to Adaptive Thresholding Strategy (TODO)]
*   **[ ] TODO** Develop Dynamic Ensemble Architecture
    *   **[ ] TODO** Initial implementation with fixed-size inputs, exploring weight sharing and ignore masks for variable input handling
        *   _Status_: Research & Design.
        *   _Challenges_: [Link to Dynamic Ensemble Design (TODO)]
    *   **[ ] TODO** Investigate alternative approaches (e.g., recurrent networks, transformer-based, feature pooling) for handling variable input sizes and order invariance
        *   _Status_: Research & Design.
        *   _Challenges_: [Link to Variable Input Handling Research (TODO)]
*   **[ ] TODO** Integrate QA and Dynamic Ensemble
    *   **[ ] TODO** Design and implement the multi-stage pipeline: Base Learners -> Synchronization -> QA Filter -> Ensemble -> Improved Segmentation
        *   _Status_: High-level design complete, detailed design pending.
        *   _Details_: [Link to Integrated Pipeline Architecture (TODO)]
    *   **[ ] TODO** Determine how to effectively pass QA scores as additional context to the ensemble
        *   _Status_: Research & Design.
        *   _Details_: [Link to QA Score Integration (TODO)]

---

## Phase 2: Comprehensive Evaluation and Validation

**Goal:** To rigorously assess the performance of the novel QA-enhanced fusion method against state-of-the-art individual models and static ensemble methods.

*   **[ ] TODO** Compare proposed method against CTC Baseline and best individual models using standard metrics (IoU, Dice Coefficient, F1 score)
*   **[ ] TODO** Conduct evaluation on a separate, unseen test set
*   **[ ] TODO** Perform ablation studies: Test each component (Frame-by-Frame QA, Temporal QA, Dynamic Ensemble) individually to quantify its specific impact on overall performance

---

## Phase 3: Application and Generalization

**Goal:** To leverage the improved silver truth for training new models and explore the applicability of the developed methodologies to other computer vision domains.

*   **[ ] TODO** Utilize the higher-quality silver truth data generated in Phase 1 to train new cell segmentation models
*   **[ ] TODO** Evaluate the performance of models trained on the improved silver truth
*   **[ ] TODO** Explore the applicability of the QA and dynamic ensemble techniques to other image segmentation tasks (e.g., medical imaging, autonomous driving)
*   **[ ] TODO** Investigate potential applications beyond computer vision (e.g., dynamic simulation reweighting)
