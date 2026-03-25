#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/migrate_mlflow_filestore_to_sqlite.sh [options]

Migrates an MLflow file-store directory (mlruns) into a SQLite database, keeps
timestamped backups, runs integrity checks, and writes a stable export copy that
is safe to download with scp.

Options:
  --source-dir PATH      Source MLflow file store. Default:
                         /mnt/proj1/eu-25-40/innovaite/silver-truth-hpc/mlflow/mlruns
  --target-db PATH       Target live SQLite DB. Default:
                         /mnt/proj1/eu-25-40/innovaite/silver-truth-hpc/mlflow/mlflow.db
  --export-db PATH       Stable backup/export SQLite DB. Default:
                         <target-db stem>_export.db
  --backup-dir PATH      Backup directory. Default:
                         <target-db dir>/backups
  --skip-source-backup   Do not create a tar.gz backup of the source mlruns dir.
  --keep-target-db       Reuse existing target DB instead of removing it first.
  --help                 Show this help.

Notes:
  - This is intended for one-off migration/export, not as a live MLflow backend
    on shared HPC storage.
  - The export DB is produced via sqlite .backup and should be the file you scp.
EOF
}

SOURCE_DIR="/mnt/proj1/eu-25-40/innovaite/silver-truth-hpc/mlflow/mlruns"
TARGET_DB="/mnt/proj1/eu-25-40/innovaite/silver-truth-hpc/mlflow/mlflow.db"
EXPORT_DB=""
BACKUP_DIR=""
SKIP_SOURCE_BACKUP=0
KEEP_TARGET_DB=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-dir)
      SOURCE_DIR="$2"
      shift 2
      ;;
    --target-db)
      TARGET_DB="$2"
      shift 2
      ;;
    --export-db)
      EXPORT_DB="$2"
      shift 2
      ;;
    --backup-dir)
      BACKUP_DIR="$2"
      shift 2
      ;;
    --skip-source-backup)
      SKIP_SOURCE_BACKUP=1
      shift
      ;;
    --keep-target-db)
      KEEP_TARGET_DB=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v mlflow >/dev/null 2>&1; then
  echo "mlflow not found in PATH" >&2
  exit 1
fi

if ! command -v sqlite3 >/dev/null 2>&1; then
  echo "sqlite3 not found in PATH" >&2
  exit 1
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Source MLflow file-store directory not found: $SOURCE_DIR" >&2
  exit 1
fi

TARGET_DB_DIR="$(dirname "$TARGET_DB")"
TARGET_DB_BASE="$(basename "$TARGET_DB")"
TARGET_DB_STEM="${TARGET_DB_BASE%.db}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

if [[ -z "$EXPORT_DB" ]]; then
  EXPORT_DB="${TARGET_DB_DIR}/${TARGET_DB_STEM}_export.db"
fi

if [[ -z "$BACKUP_DIR" ]]; then
  BACKUP_DIR="${TARGET_DB_DIR}/backups"
fi

mkdir -p "$TARGET_DB_DIR" "$BACKUP_DIR"

echo "=== MLflow Migration ==="
echo "Source dir : $SOURCE_DIR"
echo "Target DB  : $TARGET_DB"
echo "Export DB  : $EXPORT_DB"
echo "Backup dir : $BACKUP_DIR"
echo "Timestamp  : $TIMESTAMP"

if [[ $SKIP_SOURCE_BACKUP -eq 0 ]]; then
  SOURCE_PARENT="$(dirname "$SOURCE_DIR")"
  SOURCE_NAME="$(basename "$SOURCE_DIR")"
  SOURCE_BACKUP="${BACKUP_DIR}/${SOURCE_NAME}_${TIMESTAMP}.tar.gz"
  echo "Creating source backup: $SOURCE_BACKUP"
  tar -C "$SOURCE_PARENT" -czf "$SOURCE_BACKUP" "$SOURCE_NAME"
fi

if [[ -f "$TARGET_DB" ]]; then
  TARGET_DB_BACKUP="${BACKUP_DIR}/${TARGET_DB_STEM}_${TIMESTAMP}.pre_migration.db"
  echo "Backing up existing target DB: $TARGET_DB_BACKUP"
  sqlite3 "$TARGET_DB" ".backup '$TARGET_DB_BACKUP'" || cp -f "$TARGET_DB" "$TARGET_DB_BACKUP"
  if [[ $KEEP_TARGET_DB -eq 0 ]]; then
    echo "Removing existing target DB before migration"
    rm -f "$TARGET_DB"
  fi
fi

TMP_EXPORT="${EXPORT_DB}.tmp"
rm -f "$TMP_EXPORT"

echo "Running mlflow migrate-filestore"
mlflow migrate-filestore \
  --source "$SOURCE_DIR" \
  --target "sqlite:///$TARGET_DB"

echo "Checking target DB integrity"
sqlite3 "$TARGET_DB" "PRAGMA integrity_check;" | tee /tmp/mlflow_integrity_check.txt
if ! grep -qx 'ok' /tmp/mlflow_integrity_check.txt; then
  echo "Target DB integrity check failed" >&2
  exit 1
fi

echo "Creating stable export DB with sqlite backup"
sqlite3 "$TARGET_DB" ".backup '$TMP_EXPORT'"

echo "Checking export DB integrity"
sqlite3 "$TMP_EXPORT" "PRAGMA integrity_check;" | tee /tmp/mlflow_export_integrity_check.txt
if ! grep -qx 'ok' /tmp/mlflow_export_integrity_check.txt; then
  echo "Export DB integrity check failed" >&2
  rm -f "$TMP_EXPORT"
  exit 1
fi

mv -f "$TMP_EXPORT" "$EXPORT_DB"

echo
echo "Migration completed."
echo "Live DB   : $TARGET_DB"
echo "Export DB : $EXPORT_DB"
echo
echo "Use the export DB for scp:"
echo "  $EXPORT_DB"
