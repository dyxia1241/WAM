import json
import subprocess
import sys

from mvp0.config import apply_overrides, load_config
from mvp0.reports import collect_runs, write_report
from mvp0.run_ablation import run_ablation
from mvp0.train import train


def _fast_config(tmp_path):
    return apply_overrides(
        load_config("mvp0/configs/debug.yaml"),
        [
            "train.max_epochs=1",
            "data.num_samples=12",
            "data.batch_size=6",
            "model.hidden_dim=32",
            "model.transformer_layers=1",
            "model.transformer_heads=4",
            f"output_dir={tmp_path}",
        ],
    )


def test_train_writes_manifest(tmp_path):
    config = _fast_config(tmp_path)
    config["experiment"] = "time_prior"

    train(config)
    manifest = json.loads((tmp_path / "time_prior" / "manifest.json").read_text())

    assert manifest["kind"] == "train"
    assert manifest["experiment"] == "time_prior"
    assert "runtime" in manifest
    assert "metrics" in manifest


def test_run_ablation_writes_multiple_runs(tmp_path):
    config = _fast_config(tmp_path)

    results = run_ablation(config, ["time_prior", "obs_stage"], tmp_path)

    assert set(results) == {"time_prior", "obs_stage"}
    assert (tmp_path / "time_prior" / "metrics.json").exists()
    assert (tmp_path / "obs_stage" / "metrics.json").exists()


def test_reports_collect_and_write_summary(tmp_path):
    config = _fast_config(tmp_path)
    run_ablation(config, ["time_prior", "obs_stage"], tmp_path)

    rows = collect_runs(tmp_path)
    report_dir = tmp_path / "report"
    write_report(rows, report_dir)

    assert {row["experiment"] for row in rows} == {"time_prior", "obs_stage"}
    assert (report_dir / "summary.csv").exists()
    assert (report_dir / "summary.json").exists()


def test_run_ablation_and_reports_cli(tmp_path):
    output_dir = tmp_path / "outputs"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mvp0.run_ablation",
            "--config",
            "mvp0/configs/debug.yaml",
            "--experiments",
            "time_prior,obs_stage",
            "--output-dir",
            str(output_dir),
            "train.max_epochs=1",
            "data.num_samples=12",
            "data.batch_size=6",
            "model.hidden_dim=32",
            "model.transformer_layers=1",
            "model.transformer_heads=4",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "time_prior" in result.stdout
    report = subprocess.run(
        [
            sys.executable,
            "-m",
            "mvp0.reports",
            "--outputs",
            str(output_dir),
            "--output",
            str(output_dir / "report"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "wrote report" in report.stdout
    assert (output_dir / "report" / "summary.csv").exists()

