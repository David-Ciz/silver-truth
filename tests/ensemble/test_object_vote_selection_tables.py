import pandas as pd

from scripts.build_object_vote_selection_tables import (
    select_by_boundary_margin,
    select_by_global_validation,
    select_by_leave_one_dataset_boundary_margin,
    select_by_leave_one_dataset_validation,
    select_by_modality_rule,
    select_by_validation,
    summarize_selected_outputs,
)


def test_object_vote_selection_uses_validation_and_modality_rules() -> None:
    rows = pd.DataFrame(
        [
            {
                "run_id": "uniform_thr050",
                "dataset": "BF-C2DL-HSC",
                "crop_tag": "sz64",
                "variant": "baseline",
                "split": "validation",
                "mean_iou": 0.9,
                "mean_f1": 0.95,
            },
            {
                "run_id": "boundary_full_floor025",
                "dataset": "BF-C2DL-HSC",
                "crop_tag": "sz64",
                "variant": "baseline",
                "split": "validation",
                "mean_iou": 0.8,
                "mean_f1": 0.9,
            },
            {
                "run_id": "uniform_thr050",
                "dataset": "BF-C2DL-HSC",
                "crop_tag": "sz64",
                "variant": "baseline",
                "split": "test",
                "mean_iou": 0.7,
                "mean_f1": 0.8,
            },
            {
                "run_id": "boundary_full_floor025",
                "dataset": "BF-C2DL-HSC",
                "crop_tag": "sz64",
                "variant": "baseline",
                "split": "test",
                "mean_iou": 0.95,
                "mean_f1": 0.98,
            },
            {
                "run_id": "uniform_thr050",
                "dataset": "Fluo-C2DL-MSC",
                "crop_tag": "sz1024",
                "variant": "msc_crop1024",
                "split": "validation",
                "mean_iou": 0.5,
                "mean_f1": 0.6,
            },
            {
                "run_id": "boundary_full_floor025",
                "dataset": "Fluo-C2DL-MSC",
                "crop_tag": "sz1024",
                "variant": "msc_crop1024",
                "split": "validation",
                "mean_iou": 0.6,
                "mean_f1": 0.7,
            },
            {
                "run_id": "uniform_thr050",
                "dataset": "Fluo-C2DL-MSC",
                "crop_tag": "sz1024",
                "variant": "msc_crop1024",
                "split": "test",
                "mean_iou": 0.55,
                "mean_f1": 0.65,
            },
            {
                "run_id": "boundary_full_floor025",
                "dataset": "Fluo-C2DL-MSC",
                "crop_tag": "sz1024",
                "variant": "msc_crop1024",
                "split": "test",
                "mean_iou": 0.65,
                "mean_f1": 0.75,
            },
        ]
    )

    validation_selected = select_by_validation(rows)
    global_validation_selected = select_by_global_validation(rows)
    leave_one_dataset_selected = select_by_leave_one_dataset_validation(rows)
    modality_rule = select_by_modality_rule(rows)
    overall = summarize_selected_outputs(
        {
            "validation_selected_by_dataset": validation_selected,
            "global_validation_selected": global_validation_selected,
            "leave_one_dataset_validation_selected": leave_one_dataset_selected,
            "modality_rule": modality_rule,
        },
        fixed_runs=rows,
    )

    selected = validation_selected.set_index("dataset")["run_id"].to_dict()
    global_selected = global_validation_selected.set_index("dataset")[
        "run_id"
    ].to_dict()
    lodo_selected = leave_one_dataset_selected.set_index("dataset")[
        "run_id"
    ].to_dict()
    modality = modality_rule.set_index("dataset")["run_id"].to_dict()

    assert selected == {
        "BF-C2DL-HSC": "uniform_thr050",
        "Fluo-C2DL-MSC": "boundary_full_floor025",
    }
    assert global_selected == {
        "BF-C2DL-HSC": "boundary_full_floor025",
        "Fluo-C2DL-MSC": "boundary_full_floor025",
    }
    assert lodo_selected == {
        "BF-C2DL-HSC": "boundary_full_floor025",
        "Fluo-C2DL-MSC": "uniform_thr050",
    }
    assert modality == selected
    assert float(
        overall.set_index("selection_rule").loc[
            "validation_selected_by_dataset", "mean_iou"
        ]
    ) == 0.675
    assert float(
        overall.set_index("selection_rule").loc[
            "leave_one_dataset_validation_selected", "mean_iou"
        ]
    ) == 0.75


def test_object_vote_boundary_margin_selector_uses_baseline_when_gain_is_small() -> None:
    rows = pd.DataFrame(
        [
            {
                "run_id": "set_thr055",
                "dataset": "Toy-A",
                "split": "validation",
                "mean_iou": 0.90,
                "mean_f1": 0.95,
            },
            {
                "run_id": "set_boundary075",
                "dataset": "Toy-A",
                "split": "validation",
                "mean_iou": 0.904,
                "mean_f1": 0.96,
            },
            {
                "run_id": "set_thr055",
                "dataset": "Toy-A",
                "split": "test",
                "mean_iou": 0.88,
                "mean_f1": 0.93,
            },
            {
                "run_id": "set_boundary075",
                "dataset": "Toy-A",
                "split": "test",
                "mean_iou": 0.84,
                "mean_f1": 0.90,
            },
            {
                "run_id": "set_thr055",
                "dataset": "Toy-B",
                "split": "validation",
                "mean_iou": 0.80,
                "mean_f1": 0.88,
            },
            {
                "run_id": "set_boundary075",
                "dataset": "Toy-B",
                "split": "validation",
                "mean_iou": 0.83,
                "mean_f1": 0.90,
            },
            {
                "run_id": "set_thr055",
                "dataset": "Toy-B",
                "split": "test",
                "mean_iou": 0.81,
                "mean_f1": 0.89,
            },
            {
                "run_id": "set_boundary075",
                "dataset": "Toy-B",
                "split": "test",
                "mean_iou": 0.86,
                "mean_f1": 0.92,
            },
        ]
    )

    selected = select_by_boundary_margin(
        rows,
        baseline_run_id="set_thr055",
        validation_margin=0.01,
    )

    assert selected.set_index("dataset")["run_id"].to_dict() == {
        "Toy-A": "set_thr055",
        "Toy-B": "set_boundary075",
    }
    assert selected["selection_rule"].unique().tolist() == [
        "boundary_margin_selected_0p01"
    ]


def test_leave_one_dataset_boundary_margin_selects_margin_without_target_dataset() -> None:
    rows = pd.DataFrame(
        [
            {
                "run_id": "set_thr055",
                "dataset": "Toy-A",
                "split": "validation",
                "mean_iou": 0.80,
                "mean_f1": 0.88,
            },
            {
                "run_id": "set_boundary075",
                "dataset": "Toy-A",
                "split": "validation",
                "mean_iou": 0.81,
                "mean_f1": 0.89,
            },
            {
                "run_id": "set_thr055",
                "dataset": "Toy-A",
                "split": "test",
                "mean_iou": 0.70,
                "mean_f1": 0.80,
            },
            {
                "run_id": "set_boundary075",
                "dataset": "Toy-A",
                "split": "test",
                "mean_iou": 0.90,
                "mean_f1": 0.95,
            },
            {
                "run_id": "set_thr055",
                "dataset": "Toy-B",
                "split": "validation",
                "mean_iou": 0.80,
                "mean_f1": 0.88,
            },
            {
                "run_id": "set_boundary075",
                "dataset": "Toy-B",
                "split": "validation",
                "mean_iou": 0.80,
                "mean_f1": 0.88,
            },
            {
                "run_id": "set_thr055",
                "dataset": "Toy-B",
                "split": "test",
                "mean_iou": 0.75,
                "mean_f1": 0.82,
            },
            {
                "run_id": "set_boundary075",
                "dataset": "Toy-B",
                "split": "test",
                "mean_iou": 0.76,
                "mean_f1": 0.83,
            },
        ]
    )

    selected = select_by_leave_one_dataset_boundary_margin(
        rows,
        baseline_run_id="set_thr055",
        validation_margins=(0.0, 0.02),
    )

    by_dataset = selected.set_index("dataset")
    assert by_dataset.loc["Toy-A", "boundary_validation_margin"] == 0.02
    assert by_dataset.loc["Toy-A", "run_id"] == "set_thr055"
    assert by_dataset.loc["Toy-A", "mean_iou"] == 0.70
    assert selected["selection_rule"].unique().tolist() == [
        "leave_one_dataset_boundary_margin_selected"
    ]
