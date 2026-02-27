import pytest
from src.core.experiment_logger import ExperimentLogger


def test_logger(tmp_path):
    logger = ExperimentLogger(
        experiment_name="test-experiment",
        tracking_uri=f"file:{tmp_path}",
    )
    logger.start_run(run_name="test-run")
    logger.log_params({"model": "gpt-4o", "temperature": 0.0})
    logger.log_metrics({"latency": 1.23, "retries": 0})
    logger.log_tags({"status": "success"})
    logger.end_run()


def test_question_run_schema(tmp_path):
    logger = ExperimentLogger(
        experiment_name="test-schema",
        tracking_uri=f"file:{tmp_path}",
    )
    logger.start_question_run(
        question_id="q_001",
        question_text="What is the average sales by region?",
        dataset_name="sales_data.csv",
        row_count=1200,
        col_count=8,
    )
    logger.log_attempt(
        attempt_index=1,
        error_type="missing_result",
        error_message="Variable 'result' not defined",
        code="df.groupby('region').mean()",
    )
    logger.finalize(
        success=True,
        answer_type="table",
        latency_ms=843.5,
        total_attempts=2,
    )