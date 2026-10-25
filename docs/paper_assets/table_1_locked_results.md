# Table 1. Locked model results

Results are not direct cross-modality comparisons.

| Modality | Unit | Evidence class | Target | N (+) | ROC-AUC (95% CI) | AP | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| head_ct | slice | retrospective_locked_baseline | any | 242 (77) | 0.842 (0.786–0.891) | 0.777 | 0.883 | 0.509 | 0.628 |
| head_ct | study | confirmatory_prespecified_locked | any | 240 (92) | 0.908 (0.868–0.941) | 0.850 | 0.957 | 0.649 | 0.767 |
| chest_xray | image | retrospective_locked_baseline | pneumothorax | 25596 (2665) | 0.813 (0.799–0.826) | 0.319 | 0.960 | 0.343 | 0.407 |
| chest_xray | image | retrospective_locked_baseline | pleural_effusion | 25596 (4658) | 0.782 (0.770–0.794) | 0.438 | 0.917 | 0.436 | 0.523 |
| chest_xray | image | retrospective_locked_baseline | consolidation | 25596 (1815) | 0.667 (0.652–0.683) | 0.117 | 0.939 | 0.225 | 0.276 |
| chest_xray | image | retrospective_locked_baseline | atelectasis | 25596 (3279) | 0.720 (0.706–0.734) | 0.265 | 0.896 | 0.356 | 0.425 |
| knee_mri | exam | retrospective_locked_baseline | abnormal | 120 (95) | 0.739 (0.604–0.851) | 0.901 | 0.979 | 0.280 | 0.833 |
| knee_mri | exam | retrospective_locked_baseline | acl | 120 (54) | 0.857 (0.783–0.924) | 0.772 | 1.000 | 0.152 | 0.533 |
| knee_mri | exam | retrospective_locked_baseline | meniscus | 120 (52) | 0.771 (0.686–0.852) | 0.685 | 0.962 | 0.235 | 0.550 |
