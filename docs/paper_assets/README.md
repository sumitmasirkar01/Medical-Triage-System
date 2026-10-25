# Manuscript asset pack

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
