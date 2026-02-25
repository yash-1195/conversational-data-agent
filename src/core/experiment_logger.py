import mlflow
from pathlib import Path


class ExperimentLogger:
    def __init__(self, experiment_name: str, tracking_uri: str = "file:./mlruns"):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.run = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()
        return False  # do not suppress exceptions

    def start_run(self, run_name: str | None = None):
        if mlflow.active_run():
            mlflow.end_run()
        self.run = mlflow.start_run(run_name=run_name)  # type: ignore[arg-type]

    def log_params(self, params: dict):
        safe = {k: str(v) for k, v in params.items()}
        mlflow.log_params(safe)

    def log_metrics(self, metrics: dict):
        numeric = {str(k): float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        if numeric:
            mlflow.log_metrics(numeric)  # type: ignore[arg-type]

    def log_artifact(self, file_path: str):
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {file_path}")
        mlflow.log_artifact(str(path))

    def log_text(self, text: str, filename: str):
        mlflow.log_text(text, filename)

    def log_dict(self, data: dict, filename: str):
        mlflow.log_dict(data, filename)

    def log_tags(self, tags: dict):
        mlflow.set_tags({k: str(v) for k, v in tags.items()})

    def end_run(self):
        if mlflow.active_run():
            mlflow.end_run()

    def start_question_run(
        self,
        question_id: str,
        question_text: str,
        dataset_name: str,
        row_count: int,
        col_count: int,
    ):
        self.start_run(run_name=question_id)
        self.log_params({
            "question_id": question_id,
            "question": question_text,
            "dataset_name": dataset_name,
            "row_count": row_count,
            "col_count": col_count,
        })

    def log_attempt(
        self,
        attempt_index: int,
        error_type: str,
        error_message: str,
        code: str,
    ):
        self.log_dict({
            "attempt_index": attempt_index,
            "error_type": error_type,
            "error_message": error_message,
            "code": code,
        }, filename=f"attempt_{attempt_index}.json")

    def finalize(
        self,
        success: bool,
        answer_type: str,   # "table" | "plot" | "text" | "error"
        latency_ms: float,
        total_attempts: int,
    ):
        self.log_metrics({
            "latency_ms": latency_ms,
            "total_attempts": total_attempts,
        })
        self.log_tags({
            "success": str(success),
            "answer_type": answer_type,
        })
        self.end_run()
