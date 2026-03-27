# API Reference (Generated)

This page is generated from function signatures and docstrings.

## `silver_truth.metrics.evaluation_logic`

| Function | Signature | Summary |
|---|---|---|
| `evaluate_by_split` | `(parquet_path: pathlib.Path, segmentation_column: str, gt_column: str = 'gt_image') -> Dict[str, Dict[str, float]]` | Evaluate segmentation results broken down by train/val/test split. |
| `run_evaluation` | `(dataset_dataframe_path: pathlib.Path, competitor: Optional[str] = None, output: Optional[pathlib.Path] = None, visualize: bool = False, campaign_col: str = 'campaign_number') -> Dict[str, Any]` | Evaluates competitor segmentation results against ground truth using Jaccard index. |

## `silver_truth.metrics.metrics`

| Function | Signature | Summary |
|---|---|---|
| `calculate_jaccard_scores` | `(gt_image, mask_image)` |  |
| `calculate_qa_jaccard_score` | `(gt_image, predicted_mask, target_label, original_image_key, campaign, qa_row)` | Calculate Jaccard score for QA cropped images. |

## `silver_truth.fusion.crops_experiment`

| Function | Signature | Summary |
|---|---|---|
| `_binary_mask` | `(layer: 'np.ndarray') -> 'np.ndarray'` |  |
| `_build_cell_level_eval_df` | `(qa_df_with_paths: 'pd.DataFrame', fused_column: 'str') -> 'pd.DataFrame'` |  |
| `_build_key_path_lookup` | `(mapping: 'Dict[int, CellKey]', base_dir: 'Path', file_prefix: 'str') -> 'CellPathLookup'` |  |
| `_compute_metrics` | `(fused_path: 'Path', gt_path: 'Path') -> 'Tuple[float, float]'` |  |
| `_crop_to_shape` | `(source: 'np.ndarray', target_shape: 'Tuple[int, int]') -> 'np.ndarray'` |  |
| `_enrich_qa_with_paths` | `(qa_df: 'pd.DataFrame', fused_lookup: 'CellPathLookup', gt_lookup: 'CellPathLookup', fused_column: 'str') -> 'pd.DataFrame'` |  |
| `_extract_competitor_weights` | `(df: 'pd.DataFrame', competitors: 'Sequence[str]', weights_column: 'Optional[str]') -> 'Optional[Dict[str, float]]'` |  |
| `_extract_split_core_metrics` | `(eval_metrics_by_split: 'Dict[str, Dict[str, float]]') -> 'Dict[str, float]'` |  |
| `_first_non_null_value` | `(values: 'pd.Series') -> 'Any'` |  |
| `_gt_mask_from_metadata` | `(metadata: 'Dict[str, Any]', label: 'CellLabel', reference_shape: 'Tuple[int, int]') -> 'Optional[np.ndarray]'` |  |
| `_is_finite_number` | `(value: 'Any') -> 'bool'` |  |
| `_label_specific_mask` | `(segmentation: 'np.ndarray', label: 'CellLabel') -> 'np.ndarray'` |  |
| `_normalize_campaign` | `(value: 'Any') -> 'str'` |  |
| `_normalize_label` | `(value: 'Any') -> 'CellLabel'` |  |
| `_persist_gt_masks` | `(source_gt_dir: 'Path', destination_dir: 'Path', mapping: 'Dict[int, CellKey]') -> 'None'` |  |
| `_resolve_output_path` | `(output_dir: 'Union[Path, str]') -> 'Path'` |  |
| `_resolve_stacked_path` | `(path_value: 'Any', qa_parquet: 'Path') -> 'Optional[Path]'` |  |
| `_resolve_tracking_path` | `(tracking_path: 'Union[Path, str]') -> 'Path'` |  |
| `_resolve_weight_column` | `(df: 'pd.DataFrame', weights_column: 'Optional[str]') -> 'Optional[str]'` |  |
| `_safe_metric_name` | `(value: 'str') -> 'str'` |  |
| `_safe_run_token` | `(value: 'str') -> 'str'` |  |
| `_select_ranking_split` | `(split_metrics: 'Dict[str, float]') -> 'str'` |  |
| `build_cell_groups` | `(df: 'pd.DataFrame', qa_parquet: 'Path') -> 'Tuple[CellGroups, CellSplits, Dict[CellKey, Dict[str, Any]]]'` | Group rows by logical cell key and collect stacked paths per competitor. |
| `build_competitor_dir_names` | `(competitors: 'Sequence[str]') -> 'Dict[str, str]'` |  |
| `chunk_indices` | `(indices: 'Sequence[int]', chunk_size: 'int') -> 'List[List[int]]'` |  |
| `evaluate_model_results` | `(fusion_out_dir: 'Path', gt_dir: 'Path', mapping: 'Dict[int, CellKey]', cell_splits: 'CellSplits', model: 'str') -> 'pd.DataFrame'` |  |
| `fusion_job_dir` | `(output_base_dir: 'Path', keep_job_dir: 'bool', explicit_job_dir: 'Optional[Path]') -> 'Iterator[Path]'` |  |
| `inspect_fused_outputs` | `(fusion_out_dir: 'Path', mapping: 'Dict[int, CellKey]') -> 'Dict[str, float]'` |  |
| `prepare_fusion_job` | `(job_dir: 'Path', cell_groups: 'CellGroups', competitors: 'Sequence[str]', cell_metadata: 'Optional[Dict[CellKey, Dict[str, Any]]]' = None, competitor_dir_names: 'Optional[Dict[str, str]]' = None) -> 'Dict[int, CellKey]'` | Build a fusion job directory in which each logical cell is represented as a |
| `run_crops_fusion_experiment` | `(qa_parquet: 'Union[Path, str]', output_dir: 'Union[Path, str]' = 'fusion_results_crops', models: 'Sequence[str]' = (), all_models: 'bool' = False, flat_models_only: 'bool' = False, num_threads: 'int' = 4, mlflow_experiment: 'str' = 'fusion-crops-baseline', mlflow_tracking_path: 'Union[Path, str]' = 'data/fusion_experiments/mlruns', weights_column: 'Optional[str]' = None, skip_fusion: 'bool' = False, keep_job_dir: 'bool' = False, job_dir: 'Optional[Union[Path, str]]' = None, chunk_size: 'int' = 0, debug: 'bool' = False) -> 'Dict[str, Any]'` | Run fusion on QA crops for multiple models with MLflow tracking. |
| `select_models` | `(models: 'Sequence[str]', all_models: 'bool' = False, flat_models_only: 'bool' = False) -> 'List[str]'` | Resolve model list from CLI flags. |
| `summarize_results` | `(results_df: 'pd.DataFrame') -> 'Dict[str, float]'` |  |
| `validate_columns` | `(df: 'pd.DataFrame') -> 'None'` |  |
| `write_job_file` | `(job_dir: 'Path', competitors: 'Sequence[str]', weighted: 'bool', competitor_dir_names: 'Optional[Dict[str, str]]' = None, competitor_weights: 'Optional[Dict[str, float]]' = None) -> 'Path'` |  |
