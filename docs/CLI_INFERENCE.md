# Safe multimodal inference CLI

`scripts.infer_multimodal` is the supported non-graphical inference boundary. It
routes every request through `UnifiedTriageService` and emits one JSON schema for
successful, unsupported, and failed requests. It is a research demonstration and
every result requires human review.

## Commands

Run commands from the repository root:

```powershell
python -m scripts.infer_multimodal head-ct --dicom CT_SLICE.dcm
python -m scripts.infer_multimodal head-ct-series --dicom SLICE_1.dcm SLICE_2.dcm
python -m scripts.infer_multimodal chest-xray --image CXR.png --exam-id CXR-001
python -m scripts.infer_multimodal knee-mri --axial AXIAL.npy --coronal CORONAL.npy --sagittal SAGITTAL.npy --exam-id MRI-001
```

Use the global `--device cpu` or `--device cuda` option before the subcommand. Use
`--output RESULT.json` before the subcommand to write the same JSON that is emitted
to standard output. Chest X-ray and knee MRI exam identifiers must be supplied by
the caller so filenames are not reused as identifiers.

## Output and exit codes

Schema `2.0` contains an `outcome` for every request and includes `evidence` only
when inference succeeded. Evidence input hashes use positional aliases such as
`input_0001`; local filenames and paths are not emitted. Failure reasons are stable
machine-readable codes, while internal decoder and runtime details remain private.

| Exit code | Meaning |
| --- | --- |
| `0` | A `RED`, `YELLOW`, or `GREEN` research triage outcome was produced. |
| `2` | Processing failed safely and the disposition is `FAILED`. |
| `3` | The input is outside the supported contract and the disposition is `UNSUPPORTED`. |

The CLI enforces the resource budgets documented in `INPUT_RESOURCE_LIMITS.md`
through the modality adapters. A `GREEN` result only means that no supported target
crossed its locked threshold; it does not establish that the examination is normal.
