import os
from datetime import datetime, timezone

import mlflow

DEFAULT_TRACKING_URI = "http://mlflow:5000"
CANONICAL_EXPERIMENT_NAME = "ThaiFood_Training"


def init_mlflow(tracking_uri=None):
    """Initialize MLflow tracking URI with environment fallback."""
    resolved_tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(resolved_tracking_uri)
    return resolved_tracking_uri


def _normalize_scalar(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def normalize_tags(tags):
    normalized = {}
    for key, value in (tags or {}).items():
        normalized_value = _normalize_scalar(value)
        if normalized_value is not None:
            normalized[str(key)] = normalized_value
    return normalized


def apply_run_metadata(tags=None, description=None):
    normalized = normalize_tags(tags)
    if normalized:
        mlflow.set_tags(normalized)
    if description:
        mlflow.set_tag("mlflow.note.content", description)


def infer_data_version(data_dir):
    if not data_dir:
        return None
    return os.path.basename(os.path.normpath(data_dir))


def build_model_version_description(context):
    candidate_acc = context.get("candidate_acc")
    production_acc = context.get("production_acc")

    delta_pct = None
    if candidate_acc is not None and production_acc is not None:
        delta_pct = (candidate_acc - production_acc) * 100.0

    lines = [
        f"Phase: {context.get('phase', 'unknown')}",
        f"Source run_id: {context.get('source_run_id', 'n/a')}",
        f"Data version: {context.get('data_version', 'unknown')}",
        f"Trigger source: {context.get('trigger_source', 'unknown')}",
        f"DAG: {context.get('dag_id', 'n/a')}",
        f"Task: {context.get('task_id', 'n/a')}",
        f"Airflow run_id: {context.get('airflow_run_id', 'n/a')}",
        f"Drift triggered: {_normalize_scalar(context.get('drift_triggered')) or 'unknown'}",
        f"Base model: {context.get('base_model', 'none')}",
    ]

    if candidate_acc is not None:
        lines.append(f"Candidate test_acc: {candidate_acc:.6f}")
    if production_acc is not None:
        lines.append(f"Production test_acc: {production_acc:.6f}")
    if delta_pct is not None:
        lines.append(f"Accuracy delta (%): {delta_pct:.4f}")

    note = context.get("note")
    if note:
        lines.append(f"Note: {note}")

    lines.append(f"Promoted at (UTC): {datetime.now(timezone.utc).isoformat()}")
    return "\n".join(lines)
