# Dataset contracts and evaluation scope

Recorded: 2026-09-30. Status: implementation specification, not completed validation.

The user reports using the datasets below. Reproduction requires their actual
paths/notebook mounts, release names, cohort manifests, checkpoints, and prediction
artifacts. The repository's current unified benchmark generates synthetic results;
it cannot by itself verify previous real-data runs elsewhere.

## RSNA Intracranial Hemorrhage Detection (2019)

Source: [official competition data description](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/data).

The competition supplies image-level multiple binary labels: five hemorrhage
subtypes and an additional `any` label. Preserve all six in an explicit named order:

```text
epidural, intraparenchymal, intraventricular, subarachnoid, subdural, any
```

Implementation decisions:

- Parse image IDs and label suffixes carefully; join labels by exact image ID.
- Group slices using DICOM study/series metadata; preserve patient grouping across
  train, tuning, calibration, and held-out partitions where identifiers are available.
  If patient identity cannot be established, disclose that limitation rather than
  claiming patient-disjoint evaluation.
- Store rescale/windowing and model normalization parameters in the run configuration.
- Use independent binary outputs for coexisting findings. A trained six-output head
  needs matching checkpoint metadata and a suitable multi-label loss.
- An explicit `any` output represents hemorrhage presence, not general clinical
  urgency. Do not sum subtype probabilities and call the result calibrated risk.
- For a study-level endpoint, document how slice labels and predictions are aggregated;
  report that separately from slice-level competition evaluation. A positive slice
  can define a positive study label, but incomplete/missing labels require an explicit
  exclusion or missingness policy.

Current mismatch: `HeadCTAdapter` uses Normal plus four subtypes, omits IVH, and
applies a categorical softmax. Training shapes are also hardcoded to five outputs.
Changing names alone cannot repair the checkpoint, training, and calibration semantics.

## Chest X-ray: NIH ChestX-ray14 and CheXpert

Sources: [NIH dataset hosting documentation](https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest)
and [official CheXpert documentation](https://stanfordmlgroup.github.io/competitions/chexpert/).

Treat these as separate datasets with their own patient IDs, splits, and label
provenance. The proposed project tasks are pneumothorax, pleural effusion,
consolidation, and atelectasis. They can coexist.

Implementation decisions:

- Explicitly map NIH `Effusion` to the project's `pleural_effusion` name while
  preserving the original source label in metadata. Do not silently equate
  consolidation, pneumonia, and infiltration.
- CheXpert has positive, negative, uncertain, and unmentioned labels. Predeclare
  training/evaluation handling; retain an observation mask rather than coercing
  every missing or uncertain entry into a confirmed negative.
- Select frontal views from metadata if reporting a frontal-only experiment.
  Resizing to 224 by 224 is project/model preprocessing, not the original data format.
- Split by patient and report NIH and CheXpert metrics separately. Do not use an
  external-test cohort to tune prompts, thresholds, or calibration.
- Use per-finding image/text scoring for a zero-shot baseline, with frozen prompts
  and explicit score interpretation. Do not represent a softmax over mutually
  competing disease names as independent disease probabilities.
- The project's task selection differs from CheXpert's official five competition
  tasks; identify it as a custom task subset rather than an official leaderboard result.

Current mismatch: the chest adapter uses a five-way disease/normal softmax, and
the synthetic benchmark compares text embeddings instead of encoding radiographs.

Current execution status: NIH ChestX-ray14 is the selected first CXR dataset. The
read-only inventory implementation is in `stat_triage/datasets/nih_cxr14.py`, with
the Kaggle entry point `scripts/inventory_nih_cxr14.py`. It validates exact source
label mapping, metadata and image coverage, duplicate filenames, view counts, and
patient overlap in supplied split lists before a training manifest can be built.

## Stanford MRNet

Source: [official MRNet dataset and competition documentation](https://stanfordmlgroup.github.io/competitions/mrnet/).

Model three binary exam-level targets: `abnormal`, `acl`, and `meniscus`.
The distributed series are NumPy arrays grouped by axial, coronal, and sagittal
planes, with label CSVs. Build a dedicated loader for those arrays instead of
assuming the CT DICOM parser can ingest the release.

Implementation decisions:

- Join all available planes and labels by exam ID; validate missing-plane handling
  and preserve the supplied split information.
- Keep all views of an exam together. Record whether stronger patient-level
  grouping is possible from the available metadata.
- Use three binary outputs with a trained model and explicit plane aggregation.
  Normal is not a mutually exclusive fourth training label.
- Evaluate each target separately. ACL and meniscal scores are not directly an
  emergency-triage probability; queue policy needs a separately stated rationale.

Current mismatch: `KneeMRIAdapter` has a randomly initialized four-class softmax
head and no implemented three-plane MRNet loader. This remains outside the required
one-week delivery scope unless suitable prior model/run artifacts can be recovered.

## BiomedCLIP backbone provenance

Source: [Microsoft's model card](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224).

The model card documents PMC-15M pretraining on biomedical image/text pairs. Keep
this separate from task-specific training and project evaluation. Use the matching
image transforms and tokenizer, and record the checkpoint identifier/revision.
Do not count PMC-15M as this project's held-out clinical evaluation cohort.

## Artifact checklist for the next execution step

For each dataset actually available, record:

1. Local root or Kaggle input path, release/subset, and label-file paths.
2. Counts of images, studies/exams, and patients where available; per-label counts.
3. Split manifests and exclusion rules, including missing/uncertain labels.
4. Checkpoint source/hash, output-label order, training data, and preprocessing.
5. Previous notebooks, logs, and prediction files, if already run.

Do not download all four datasets anew before checking existing copies. The first
execution target is a reproducible RSNA cohort; retain the other modalities as
documented extensions while the one-week deadline applies.
