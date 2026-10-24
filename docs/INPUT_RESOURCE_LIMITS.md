# Input resource limits

The public preprocessing boundary rejects oversized chest radiographs and knee MRI
arrays before model inference. These limits bound memory use for untrusted local
files; they do not validate clinical acquisition quality or establish suitability
for clinical use.

## Chest radiograph

`decode_nih_radiograph` checks the compressed file size before Pillow opens it. It
then reads only image metadata and rejects dimensions above 8192 by 8192, more than
32 Mi pixels, or an estimated decoded working set above 128 MiB before calling
`convert("L")`. The compressed-file limit is 32 MiB. The decoded grayscale array is
checked again after conversion. Resource violations use the stable public code
`input_resource_limit`; corrupt or unsupported encodings retain
`image_decode_failed`.

## Knee MRI

MRNet `.npy` inputs are first opened with `numpy.load(..., mmap_mode="r",
allow_pickle=False)`. Shape and dtype come from the bounded NumPy header path, so the
volume is not copied into memory until it has passed all checks. Each plane is
restricted to uint8 `[slices,height,width]`, at most 96 slices, dimensions no larger
than 1024 by 1024, 80 MiB on disk, 64 MiB decoded, and a 64 MiB normalized output
estimate at the requested resize.

Before any plane tensor is created, the axial, coronal, and sagittal headers are
validated together. Their aggregate limits are 240 MiB on disk, 192 MiB decoded,
and 192 MiB for normalized tensors. Each plane is checked again when loaded to close
ordinary caller errors between exam validation and preprocessing. Resource
violations use `input_resource_limit`; malformed shape, dtype, or missing-plane
errors retain `input_contract_violation`.

The limits exceed the observed locked MRNet cohort maxima (61 axial, 58 coronal,
and 51 sagittal slices) and the 1024 by 1024 NIH source-image dimensions, so they do
not alter the frozen evaluation transforms or the reported model results.
