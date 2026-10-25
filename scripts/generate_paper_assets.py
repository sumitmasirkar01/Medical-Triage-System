"""Generate deterministic manuscript figures, tables, and method equations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "stat-triage-paper-assets-v1"
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "paper_assets"
COLORS = {
    "ct": "#2F6B9A",
    "cxr": "#2A9D8F",
    "mri": "#8E5AA9",
    "confirmatory": "#D1495B",
    "neutral": "#51606B",
    "light": "#F3F6F8",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _save(figure: plt.Figure, output: Path, stem: str) -> None:
    figure.savefig(
        output / f"{stem}.svg", bbox_inches="tight", metadata={"Date": None}
    )
    figure.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _box(
    axes: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    color: str,
    fontsize: float = 9,
) -> None:
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.2, edgecolor=color, facecolor="white",
    )
    axes.add_patch(patch)
    axes.text(
        x + width / 2, y + height / 2, text,
        ha="center", va="center", fontsize=fontsize, color="#1E2A32",
    )


def _arrow(
    axes: plt.Axes, start: tuple[float, float], end: tuple[float, float]
) -> None:
    axes.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=13,
            linewidth=1.2, color="#78848C",
        )
    )


def architecture_figure(output: Path) -> None:
    figure, axes = plt.subplots(figsize=(13.5, 6.2))
    axes.set_xlim(-0.005, 1.01)
    axes.set_ylim(0, 1)
    axes.axis("off")
    axes.set_title(
        "STAT-Triage: modality-routed research workflow",
        loc="left", fontsize=17, fontweight="bold", pad=16,
    )
    axes.text(
        0, 0.965,
        "Independent modality models retain their own labels, calibration, and thresholds; scores are never fused across modalities.",
        fontsize=10.5, color=COLORS["neutral"], va="top",
    )

    rows = [
        (0.72, "Head CT DICOM\nslice or series", "EfficientNet-B0\n6 hemorrhage outputs", "Locked ‘any’\nthreshold", COLORS["ct"]),
        (0.45, "Frontal chest\nradiograph", "DenseNet-121\n4 binary targets", "Affine calibration +\nper-target thresholds", COLORS["cxr"]),
        (0.18, "Three-plane\nknee MRI exam", "Shared ResNet-18\n3 binary targets", "Affine calibration +\nper-target thresholds", COLORS["mri"]),
    ]
    for y, source, model, decision, color in rows:
        _box(axes, 0.01, y, 0.17, 0.14, source, color=color)
        _box(axes, 0.245, y, 0.18, 0.14, model, color=color)
        _box(axes, 0.48, y, 0.18, 0.14, decision, color=color)
        _arrow(axes, (0.18, y + 0.07), (0.245, y + 0.07))
        _arrow(axes, (0.425, y + 0.07), (0.48, y + 0.07))
        _arrow(axes, (0.66, y + 0.07), (0.72, 0.52))

    _box(
        axes, 0.72, 0.43, 0.14, 0.18,
        "Transparent\ntriage policy\nRED / YELLOW / GREEN",
        color=COLORS["confirmatory"], fontsize=8.3,
    )
    _box(
        axes, 0.88, 0.43, 0.10, 0.18,
        "Audited\nworklist +\nhuman review",
        color=COLORS["confirmatory"], fontsize=9,
    )
    _arrow(axes, (0.86, 0.52), (0.88, 0.52))
    axes.text(
        0.72, 0.35,
        "Research demonstration only\nEvery outcome requires human review",
        fontsize=9, color=COLORS["confirmatory"], ha="left", va="top",
    )
    _save(figure, output, "figure_1_system_architecture")


def protocol_figure(output: Path) -> None:
    figure, axes = plt.subplots(figsize=(13.5, 4.8))
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.axis("off")
    axes.set_title(
        "Leakage-controlled model development and locked evaluation",
        loc="left", fontsize=17, fontweight="bold", pad=16,
    )
    stages = [
        (0.01, "Training", "Fit model\nparameters", "No test access"),
        (0.21, "Validation", "Select model and\nCT pooling method", "Development only"),
        (0.41, "Calibration", "Fit calibration and\nlock thresholds", "Target sensitivity"),
        (0.61, "Protocol lock", "Freeze checkpoint,\npooling, threshold", "Hash artifacts"),
        (0.81, "Held-out test", "Evaluate once +\ngrouped bootstrap", "No reselection"),
    ]
    for index, (x, title, action, guard) in enumerate(stages):
        color = COLORS["confirmatory"] if index >= 3 else COLORS["ct"]
        _box(axes, x, 0.43, 0.16, 0.31, f"{title}\n\n{action}\n\n{guard}", color=color)
        if index < len(stages) - 1:
            _arrow(axes, (x + 0.16, 0.585), (stages[index + 1][0], 0.585))
    axes.text(
        0.01, 0.27,
        "Confirmatory component: CT Phase 2 study endpoint (240 patients).",
        fontsize=10.5, fontweight="bold", color=COLORS["confirmatory"],
    )
    axes.text(
        0.01, 0.17,
        "Retrospective locked baselines: CT slice pilot, NIH ChestX-ray14, and Stanford MRNet. Post-test analyses are labelled exploratory.",
        fontsize=10, color=COLORS["neutral"],
    )
    _save(figure, output, "figure_2_evaluation_protocol")


def _record_label(record: dict[str, Any]) -> str:
    if record["modality"] == "head_ct":
        return f"Head CT — {record['evaluation_level']} — {record['target']}"
    if record["modality"] == "chest_xray":
        return f"Chest X-ray — {record['target'].replace('_', ' ')}"
    return f"Knee MRI — {record['target']}"


def auc_forest_figure(output: Path, records: list[dict[str, Any]]) -> None:
    ordered = list(reversed(records))
    labels = [_record_label(record) for record in ordered]
    y = np.arange(len(ordered))
    values = np.array([record["roc_auc"] for record in ordered])
    lower = np.array([record["roc_auc_ci_lower"] for record in ordered])
    upper = np.array([record["roc_auc_ci_upper"] for record in ordered])
    colors = [
        COLORS["confirmatory"]
        if record["evidence_classification"] == "confirmatory_prespecified_locked"
        else COLORS[{"head_ct": "ct", "chest_xray": "cxr", "knee_mri": "mri"}[record["modality"]]]
        for record in ordered
    ]

    figure, axes = plt.subplots(figsize=(11.5, 7.2))
    for index, (value, lo, hi, color) in enumerate(zip(values, lower, upper, colors)):
        axes.errorbar(
            value, index, xerr=[[value - lo], [hi - value]], fmt="o",
            markersize=7, color=color, ecolor=color, elinewidth=2, capsize=4,
        )
    axes.axvline(0.5, color="#9AA4AA", linestyle="--", linewidth=1)
    axes.set_yticks(y, labels)
    axes.set_xlim(0.45, 1.01)
    axes.set_xlabel("ROC-AUC with grouped-bootstrap 95% confidence interval")
    axes.set_title("Locked discrimination results", loc="left", fontsize=16, fontweight="bold")
    axes.text(
        0.45, len(ordered) + 0.15,
        "Rows use different cohorts, endpoints, prevalence, and evaluation units; they are not direct model comparisons.",
        fontsize=9.5, color=COLORS["neutral"], va="bottom",
    )
    axes.grid(axis="x", color="#E2E7EA", linewidth=0.8)
    axes.spines[["top", "right", "left"]].set_visible(False)
    axes.tick_params(axis="y", length=0)
    figure.subplots_adjust(left=0.31, top=0.87)
    _save(figure, output, "figure_3_locked_auc_forest")


def operating_figure(output: Path, records: list[dict[str, Any]]) -> None:
    labels = [_record_label(record) for record in records]
    y = np.arange(len(records))
    sensitivity = [record["sensitivity"] for record in records]
    specificity = [record["specificity"] for record in records]
    height = 0.34
    figure, axes = plt.subplots(figsize=(11.5, 7.4))
    axes.barh(y + height / 2, sensitivity, height, color="#D1495B", label="Sensitivity")
    axes.barh(y - height / 2, specificity, height, color="#457B9D", label="Specificity")
    axes.set_yticks(y, labels)
    axes.invert_yaxis()
    axes.set_xlim(0, 1.02)
    axes.set_xlabel("Locked-threshold operating characteristic")
    axes.set_title(
        "Locked-threshold operating points",
        loc="left", fontsize=16, fontweight="bold",
    )
    axes.grid(axis="x", color="#E2E7EA", linewidth=0.8)
    axes.legend(frameon=False, ncols=2, loc="upper right")
    axes.spines[["top", "right", "left"]].set_visible(False)
    axes.tick_params(axis="y", length=0)
    figure.subplots_adjust(left=0.31, top=0.89)
    _save(figure, output, "figure_4_locked_operating_points")


def _format(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def write_tables(output: Path, summary: dict[str, Any]) -> None:
    records = summary["records"]
    columns = [
        "modality", "evaluation_level", "evidence_classification", "target",
        "test_units", "positive_units", "roc_auc", "roc_auc_ci_lower",
        "roc_auc_ci_upper", "average_precision", "sensitivity", "specificity",
        "accuracy",
    ]
    with (output / "table_1_locked_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: record.get(key) for key in columns} for record in records)

    lines = [
        "# Table 1. Locked model results",
        "",
        "Results are not direct cross-modality comparisons.",
        "",
        "| Modality | Unit | Evidence class | Target | N (+) | ROC-AUC (95% CI) | AP | Sensitivity | Specificity | Accuracy |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        interval = (
            f"{_format(row['roc_auc'])} "
            f"({_format(row['roc_auc_ci_lower'])}–{_format(row['roc_auc_ci_upper'])})"
        )
        lines.append(
            f"| {row['modality']} | {row['evaluation_level']} | "
            f"{row['evidence_classification']} | {row['target']} | "
            f"{row['test_units']} ({row['positive_units']}) | {interval} | "
            f"{_format(row['average_precision'])} | {_format(row['sensitivity'])} | "
            f"{_format(row['specificity'])} | {_format(row['accuracy'])} |"
        )
    (output / "table_1_locked_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )

    ct = next(
        row for row in records
        if row["modality"] == "head_ct" and row["evaluation_level"] == "study"
    )
    ct_lines = [
        "# Table 2. Primary confirmatory CT study-level result",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Held-out patients/studies | {ct['test_units']} |",
        f"| Positive studies | {ct['positive_units']} |",
        f"| Negative studies | {ct['negative_units']} |",
        f"| Locked threshold | {_format(ct['threshold'], 6)} |",
        f"| ROC-AUC (95% CI) | {_format(ct['roc_auc'])} ({_format(ct['roc_auc_ci_lower'])}–{_format(ct['roc_auc_ci_upper'])}) |",
        f"| Average precision | {_format(ct['average_precision'])} |",
        f"| Sensitivity | {_format(ct['sensitivity'])} |",
        f"| Specificity | {_format(ct['specificity'])} |",
        f"| PPV | {_format(ct['ppv'])} |",
        f"| NPV | {_format(ct['npv'])} |",
        f"| Accuracy | {_format(ct['accuracy'])} |",
        "| Confusion counts | TP 88; TN 96; FP 52; FN 4 |",
    ]
    (output / "table_2_primary_ct_result.md").write_text(
        "\n".join(ct_lines) + "\n", encoding="utf-8", newline="\n"
    )


def write_equations(output: Path) -> None:
    content = r"""# Methods equations

These equations describe the implemented pipeline. Symbols must be defined again
in the manuscript because the modalities have different endpoints and evaluation
units.

## 1. Independent binary probabilities

For target (k), a model logit (z_k) is converted to a probability by

$$
p_k = \sigma(z_k) = \frac{1}{1 + e^{-z_k}}.
$$

The targets are independent binary outputs; they are not a mutually exclusive
softmax.

## 2. Masked class-weighted binary cross-entropy

The primary multi-label training objective is

$$
\mathcal{L}_{\mathrm{WBCE}}
= \frac{1}{\sum_{i,k} m_{ik}}
\sum_{i,k} m_{ik}
\left[-w_k y_{ik}\log \sigma(z_{ik})
- (1-y_{ik})\log\left(1-\sigma(z_{ik})\right)\right],
$$

where (m_{ik}\in\{0,1\}) marks an observed label and (w_k>0) is computed from
the training partition only. The implementation uses the numerically stable
`BCEWithLogits` form.

## 3. Affine logistic calibration

Chest X-ray and knee MRI targets use held-out affine logistic scaling:

$$
\tilde p_k = \sigma(a_k z_k + b_k), \qquad a_k > 0.
$$

Parameters (a_k,b_k) minimize calibration negative log-likelihood. Identity
calibration is retained when the prespecified minimum improvement is not achieved.
The CT baseline does not claim population probability calibration.

## 4. CT study aggregation

For a study with (n) slice-level `any` scores (p_1,\ldots,p_n), the
validation-selected mean pooling rule is

$$
s_{\mathrm{study}} = \frac{1}{n}\sum_{j=1}^{n} p_j.
$$

Maximum, mean, top-(k) mean, and noisy-OR were candidate rules on validation;
mean pooling was locked before final test evaluation.

## 5. Sensitivity-constrained threshold selection

On the calibration partition, the locked operating threshold is

$$
\tau^{*} = \max\left\{\tau:\operatorname{Sensitivity}_{\mathrm{cal}}(\tau)
\geq s_{0}\right\}, \qquad s_0=0.90,
$$

and the binary decision is

$$
\hat y = \mathbb{1}[\tilde p \geq \tau^{*}].
$$

## 6. Operating characteristics

$$
\operatorname{Sensitivity}=\frac{TP}{TP+FN},\qquad
\operatorname{Specificity}=\frac{TN}{TN+FP},
$$

$$
\operatorname{PPV}=\frac{TP}{TP+FP},\qquad
\operatorname{NPV}=\frac{TN}{TN+FN},
$$

$$
\operatorname{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}.
$$

ROC-AUC is threshold-independent discrimination and must not be described as
accuracy.

## 7. Calibration diagnostics

For (N) observations, the Brier score and negative log-likelihood are

$$
\operatorname{Brier}=\frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2,
$$

$$
\operatorname{NLL}=-\frac{1}{N}\sum_{i=1}^{N}
\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right].
$$

For equal-width bins (B_m), expected calibration error is

$$
\operatorname{ECE}=\sum_{m=1}^{M}\frac{|B_m|}{N}
\left|\operatorname{acc}(B_m)-\operatorname{conf}(B_m)\right|.
$$

## 8. Grouped bootstrap uncertainty

Patients are sampled with replacement for CT and chest X-ray; MRNet uses exams
because patient identifiers are unavailable. For metric (\theta), the reported
95% percentile interval is

$$
CI_{0.95}(\theta)=
\left[Q_{0.025}(\theta^{*(1)},\ldots,\theta^{*(B)}),
Q_{0.975}(\theta^{*(1)},\ldots,\theta^{*(B)})\right],
\qquad B=2000.
$$

## 9. Transparent workflow priority

The deterministic within-zone priority score is

$$
P(z,w)=b_z + \lambda w,
$$

where (b_{\mathrm{GREEN}}=0), (b_{\mathrm{YELLOW}}=10),
(b_{\mathrm{RED}}=20), (w) is waiting time in minutes, and
(\lambda=0.002). This is a workflow ordering score, not a calibrated clinical
risk. Cases are first ordered by zone and then FIFO; authorized human overrides
are retained in the audit trail.
"""
    (output / "methods_equations.md").write_text(
        content, encoding="utf-8", newline="\n"
    )


def write_index(output: Path) -> None:
    content = """# Manuscript asset pack

Generated from locked repository artifacts with:

```powershell
python -m scripts.generate_paper_assets
```

## Figures

1. `figure_1_system_architecture`: modality-routed system architecture.
2. `figure_2_evaluation_protocol`: leakage-controlled evaluation protocol.
3. `figure_3_locked_auc_forest`: ROC-AUC estimates and grouped-bootstrap intervals.
4. `figure_4_locked_operating_points`: sensitivity-specificity trade-offs.

Each figure is supplied as editable SVG and 300-dpi PNG. The forest plot is a
compact summary, not a statistical comparison across modalities.

## Tables and equations

- `table_1_locked_results`: all locked endpoints in Markdown and CSV.
- `table_2_primary_ct_result.md`: primary confirmatory CT result.
- `methods_equations.md`: implemented loss, calibration, aggregation, threshold,
  metric, uncertainty, and workflow equations.

Use the CT Phase 2 study row as the primary confirmatory evidence. Label the CT
slice, chest X-ray, and knee MRI results as retrospective locked baselines. Do not
state that ROC-AUC is accuracy or calculate a single multimodal accuracy.
"""
    (output / "README.md").write_text(content, encoding="utf-8", newline="\n")


def generate(output: Path, project_root: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    summary = _read(
        project_root / "artifacts/consolidated_results/locked_multimodal_results.json"
    )
    records = summary.get("records")
    if not isinstance(records, list) or len(records) != 9:
        raise ValueError("Expected nine locked result records")
    architecture_figure(output)
    protocol_figure(output)
    auc_forest_figure(output, records)
    operating_figure(output, records)
    write_tables(output, summary)
    write_equations(output)
    write_index(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(args.output_dir.resolve(), args.project_root.resolve())
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
