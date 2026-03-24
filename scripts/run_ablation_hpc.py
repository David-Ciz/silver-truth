#!/usr/bin/env python3
"""
Submit the ablation pipeline to Slurm and execute it from a scratch worktree.

The job copies the current repository state to a persistent scratch workspace for
the selected dataset/variant/fold, stages the heavy input data there, and keeps
durable outputs on project storage via symlinks:

- scratch: repo worktree, staged input data, local checkpoint state
- project storage: MLflow runs, paper_runs outputs, generated batch script

Typical usage on Karolina:

    python scripts/run_ablation_hpc.py \
        --config experiments/variants/baseline.yaml \
        --fold 1 \
        --submit
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from textwrap import dedent

import click

from run_ablation import PROJECT_ROOT, load_config

DEFAULT_ACCOUNT = "eu-25-40"
DEFAULT_PARTITION = "qgpu"
DEFAULT_TIME = "24:00:00"
DEFAULT_GPUS = 1
DEFAULT_CPUS_PER_TASK = 16
DEFAULT_PROJECT_STORAGE_ROOT = Path("/mnt/proj1/eu-25-40/innovaite")
DEFAULT_VENV_DIRNAME = ".venv"
DEFAULT_DURABLE_DIRNAME = "silver-truth-hpc"
DEFAULT_SCRATCH_ROOT = Path("/scratch/project/eu-25-40/silver-truth/ablation")
DEFAULT_MODULES = (
    "CUDA/12.8.0",
    "Python/3.11.3-GCCcore-12.3.0",
    "libjpeg-turbo/2.1.5.1-GCCcore-12.3.0",
)
RSYNC_EXCLUDES = (
    ".git/",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    "__pycache__/",
    ".DS_Store",
    ".venv/",
    ".state/",
    "data/mlflow/",
    "data/paper_runs/",
    "data/dataframes/",
    "data/qa_crops/",
    "data/synchronized_data/*/",
    "data/job_files/",
)


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _ensure_repo_relative(path: Path) -> Path:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise click.BadParameter(
            f"Path must be inside the repository: {resolved}"
        ) from exc


def _build_job_script(
    *,
    config: Path,
    fold: str,
    phase: str,
    reset: bool,
    dry_run: bool,
    source_repo: Path,
    durable_root: Path,
    venv_dir: Path,
    scratch_root: Path,
    log_dir: Path,
    account: str,
    partition: str,
    time_limit: str,
    gpus: int,
    cpus_per_task: int,
    modules: tuple[str, ...],
    job_name: str,
) -> str:
    cfg = load_config(config, fold)
    variant = cfg["variant"]
    dataset = cfg["dataset"]
    split_name = cfg["split_name"]
    config_rel = _ensure_repo_relative(config)

    scratch_workspace = scratch_root / dataset / variant / split_name
    scratch_repo = scratch_workspace / "repo"
    durable_paper_runs = durable_root / "paper_runs"
    durable_mlflow = durable_root / "mlflow"

    module_lines = "\n".join(f"ml {shlex.quote(module)}" for module in modules)
    rsync_excludes = " ".join(
        f"--exclude={shlex.quote(pattern)}" for pattern in RSYNC_EXCLUDES
    )

    ablation_cmd = [
        "python",
        "scripts/run_ablation.py",
        "--config",
        str(scratch_repo / config_rel),
        "--fold",
        fold,
        "--phase",
        phase,
    ]
    if reset:
        ablation_cmd.append("--reset")
    if dry_run:
        ablation_cmd.append("--dry-run")

    cmd_literal = _shell_join(ablation_cmd)

    output_path = log_dir / f"{job_name}_%j.out"
    error_path = log_dir / f"{job_name}_%j.err"

    return dedent(
        f"""\
        #!/bin/bash -l
        #SBATCH --job-name={job_name}
        #SBATCH --account={account}
        #SBATCH --partition={partition}
        #SBATCH --nodes=1
        #SBATCH --gpus={gpus}
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task={cpus_per_task}
        #SBATCH --time={time_limit}
        #SBATCH --output={output_path}
        #SBATCH --error={error_path}

        set -euo pipefail

        SOURCE_REPO={shlex.quote(str(source_repo))}
        SCRATCH_WORKSPACE={shlex.quote(str(scratch_workspace))}
        SCRATCH_REPO={shlex.quote(str(scratch_repo))}
        DURABLE_ROOT={shlex.quote(str(durable_root))}
        DURABLE_PAPER_RUNS={shlex.quote(str(durable_paper_runs))}
        DURABLE_MLFLOW={shlex.quote(str(durable_mlflow))}
        VENV_DIR={shlex.quote(str(venv_dir))}
        DATASET={shlex.quote(dataset)}

        echo "=== Ablation HPC job started at $(date) ==="
        echo "Job ID: ${{SLURM_JOB_ID:-unknown}}"
        echo "Node: ${{SLURMD_NODENAME:-$(hostname)}}"
        echo "Dataset: $DATASET"
        echo "Variant: {variant}"
        echo "Fold: {split_name}"
        echo "Scratch workspace: $SCRATCH_WORKSPACE"
        echo "Durable outputs: $DURABLE_ROOT"

        command -v ml >/dev/null 2>&1 && ml purge >/dev/null 2>&1 || true
        {module_lines}

        source "$VENV_DIR/bin/activate"

        mkdir -p "$SCRATCH_WORKSPACE" "$DURABLE_PAPER_RUNS" "$DURABLE_MLFLOW/mlruns"
        mkdir -p "$SCRATCH_REPO"

        echo "Syncing repository to scratch..."
        rsync -a --delete {rsync_excludes} "$SOURCE_REPO/" "$SCRATCH_REPO/"

        stage_dir() {{
            local rel="$1"
            local delete_mode="${{2:-delete}}"
            local src="$SOURCE_REPO/$rel"
            local dst="$SCRATCH_REPO/$rel"

            if [ ! -e "$src" ]; then
                echo "ERROR: Required input path is missing: $src"
                exit 1
            fi

            mkdir -p "$(dirname "$dst")"
            if [ -d "$src" ]; then
                if [ "$delete_mode" = "delete" ]; then
                    rsync -aL --delete "$src/" "$dst/"
                else
                    rsync -aL "$src/" "$dst/"
                fi
            else
                rsync -aL "$src" "$dst"
            fi
        }}

        mkdir -p "$SCRATCH_REPO/data"
        rm -rf "$SCRATCH_REPO/data/paper_runs" "$SCRATCH_REPO/data/mlflow"
        ln -sfn "$DURABLE_PAPER_RUNS" "$SCRATCH_REPO/data/paper_runs"
        ln -sfn "$DURABLE_MLFLOW" "$SCRATCH_REPO/data/mlflow"
        mkdir -p "$SCRATCH_REPO/data/job_files/$DATASET"

        if [ ! -d "$SCRATCH_REPO/data/synchronized_data/$DATASET" ] && [ -f "$SCRATCH_REPO/data/synchronized_data/$DATASET.dvc" ]; then
            echo "Checking out DVC data for $DATASET inside scratch worktree..."
            (
                cd "$SCRATCH_REPO"
                dvc checkout "data/synchronized_data/$DATASET.dvc"
            )
        fi

        echo "Staging input data to scratch..."
        stage_dir "data/dataframes/$DATASET" preserve
        stage_dir "data/qa_crops/$DATASET"
        if [ -d "$SOURCE_REPO/data/synchronized_data/$DATASET" ]; then
            stage_dir "data/synchronized_data/$DATASET"
        elif [ ! -d "$SCRATCH_REPO/data/synchronized_data/$DATASET" ]; then
            echo "ERROR: Missing synchronized data for $DATASET in both source and scratch worktrees."
            exit 1
        fi

        export PYTHONPATH="$SCRATCH_REPO/src${{PYTHONPATH:+:$PYTHONPATH}}"
        export PYTHONUNBUFFERED=1
        export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-{cpus_per_task}}}"

        cd "$SCRATCH_REPO"

        echo "Running ablation pipeline..."
        echo "Command: {cmd_literal}"
        {cmd_literal}

        echo "=== Ablation HPC job completed at $(date) ==="
        echo "Paper runs: $DURABLE_PAPER_RUNS"
        echo "MLflow: $DURABLE_MLFLOW/mlruns"
        """
    )


@click.command()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to variant YAML (for example experiments/variants/baseline.yaml).",
)
@click.option(
    "--fold",
    "-f",
    required=True,
    type=click.Choice(["1", "2", "mixed"], case_sensitive=False),
    help="Split to run: 1, 2, or mixed.",
)
@click.option(
    "--phase",
    type=click.Choice(["A", "B", "C", "all"]),
    default="all",
    show_default=True,
    help="Run only steps belonging to the specified phase.",
)
@click.option(
    "--reset",
    is_flag=True,
    default=False,
    help="Pass --reset through to scripts/run_ablation.py inside the Slurm job.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Run scripts/run_ablation.py in dry-run mode inside the Slurm job.",
)
@click.option(
    "--submit/--no-submit",
    default=False,
    show_default=True,
    help="Submit the generated batch script with sbatch.",
)
@click.option(
    "--job-name",
    type=str,
    default=None,
    help="Override the Slurm job name. Defaults to ablation_<variant>_<fold>.",
)
@click.option(
    "--account",
    default=DEFAULT_ACCOUNT,
    show_default=True,
    help="Slurm account.",
)
@click.option(
    "--partition",
    default=DEFAULT_PARTITION,
    show_default=True,
    help="Slurm partition.",
)
@click.option(
    "--time",
    "time_limit",
    default=DEFAULT_TIME,
    show_default=True,
    help="Slurm wall time limit.",
)
@click.option(
    "--gpus",
    type=int,
    default=DEFAULT_GPUS,
    show_default=True,
    help="Number of GPUs to request.",
)
@click.option(
    "--cpus-per-task",
    type=int,
    default=DEFAULT_CPUS_PER_TASK,
    show_default=True,
    help="Number of CPU cores for the single Slurm task.",
)
@click.option(
    "--module",
    "modules",
    multiple=True,
    default=DEFAULT_MODULES,
    show_default=True,
    help="Environment module to load. Pass multiple times to override the defaults.",
)
@click.option(
    "--source-repo",
    type=click.Path(file_okay=False, path_type=Path),
    default=PROJECT_ROOT,
    show_default=True,
    help="Persistent repository checkout visible on the HPC filesystem.",
)
@click.option(
    "--project-storage-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_PROJECT_STORAGE_ROOT,
    show_default=True,
    help="Durable project storage root used for the venv and persisted outputs.",
)
@click.option(
    "--durable-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override the durable output root. Defaults to <project-storage-root>/silver-truth-hpc.",
)
@click.option(
    "--venv-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override the virtualenv path. Defaults to <project-storage-root>/silver-truth-venv.",
)
@click.option(
    "--scratch-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_SCRATCH_ROOT,
    show_default=True,
    help="Scratch root used for the staged worktrees.",
)
def main(
    config: Path,
    fold: str,
    phase: str,
    reset: bool,
    dry_run: bool,
    submit: bool,
    job_name: str | None,
    account: str,
    partition: str,
    time_limit: str,
    gpus: int,
    cpus_per_task: int,
    modules: tuple[str, ...],
    source_repo: Path,
    project_storage_root: Path,
    durable_root: Path | None,
    venv_dir: Path | None,
    scratch_root: Path,
) -> None:
    source_repo = source_repo.resolve()
    project_storage_root = project_storage_root.resolve()
    durable_root = (
        durable_root.resolve()
        if durable_root is not None
        else (project_storage_root / DEFAULT_DURABLE_DIRNAME)
    )
    venv_dir = (
        venv_dir.resolve()
        if venv_dir is not None
        else (project_storage_root / DEFAULT_VENV_DIRNAME)
    )
    scratch_root = scratch_root.resolve()

    cfg = load_config(config, fold)
    default_job_name = f"ablation_{cfg['variant']}_{cfg['split_name']}".replace(
        "-", "_"
    )
    effective_job_name = job_name or default_job_name

    log_dir = source_repo / "logs" / "ablation_hpc"
    log_dir.mkdir(parents=True, exist_ok=True)

    job_script = _build_job_script(
        config=config.resolve(),
        fold=fold,
        phase=phase,
        reset=reset,
        dry_run=dry_run,
        source_repo=source_repo,
        durable_root=durable_root,
        venv_dir=venv_dir,
        scratch_root=scratch_root,
        log_dir=log_dir,
        account=account,
        partition=partition,
        time_limit=time_limit,
        gpus=gpus,
        cpus_per_task=cpus_per_task,
        modules=modules,
        job_name=effective_job_name,
    )

    job_script_path = (
        log_dir / f"{effective_job_name}_{cfg['dataset']}_{cfg['split_name']}.sbatch"
    )
    job_script_path.write_text(job_script)

    click.echo(f"Wrote batch script: {job_script_path}")
    click.echo(
        f"Scratch workspace: {scratch_root / cfg['dataset'] / cfg['variant'] / cfg['split_name']}"
    )
    click.echo(f"Durable outputs: {durable_root}")

    if not submit:
        click.echo(f"Submit with: sbatch {job_script_path}")
        return

    result = subprocess.run(
        ["sbatch", str(job_script_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if stdout:
        click.echo(stdout)
    if stderr:
        click.echo(stderr, err=True)


if __name__ == "__main__":
    main()
