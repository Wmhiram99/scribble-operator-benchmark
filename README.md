# Scribble Operator Benchmark (Private Research Repository)

A compact, private research repository for the four scribble perturbation operators used in our VOC2012 and Cityscapes experiments, with a lightweight visual walkthrough.

## Contents

- `operators/voc2012/` and `operators/cityscapes/` contain the dataset-specific Class, Offset, Direction, and Shrink generator scripts.
- `demo/operator_overview.svg` is a synthetic visual walkthrough of the four intended changes. It contains no dataset image, annotation, prediction, or model output and is not a benchmark result.
- `requirements.txt` lists the Python packages used by the generator scripts.

## Data and local paths

Dataset files are not included. Obtain VOC2012 and Cityscapes through their official sources and follow their access terms. Configure paths using the environment variables listed below, or edit the script settings. Keep data, outputs, checkpoints, logs, and credentials outside this repository.

| Operator | VOC2012 path variables | Cityscapes path variables |
|---|---|---|
| Class | `VOC_SCRIBBLE_DIR`, `VOC_GT_DIR`, `VOC_CLASS_OUT_ROOT` | `CITYSCAPES_SCRIBBLE_DIR`, `CITYSCAPES_GT_DIR`, `CITYSCAPES_CLASS_OUT_ROOT` |
| Direction | `VOC_SCRIBBLE_DIR`, `VOC_GT_DIR`, `VOC_DIRECTION_OUT_ROOT` | `CITYSCAPES_SCRIBBLE_DIR`, `CITYSCAPES_GT_DIR`, `CITYSCAPES_DIRECTION_OUT_ROOT` |
| Offset | `VOC_DATA_ROOT`, `VOC_OFFSET_OUT_ROOT` | `OFFSET_BASE_DIR`, `OFFSET_OUT_ROOT` |
| Shrink | Use command-line arguments shown by `python operators/<dataset>/shrink.py --help` | Same |

The dataset-specific scripts preserve the research implementations. They were written as experiment scripts and may need local configuration and dataset-specific preprocessing before use. In particular, Cityscapes label IDs/trainIds and ignore values must match the chosen input annotation format.

## Install

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
```

Some OpenCV, SciPy, or scikit-image versions may require a platform-appropriate wheel. The full experiments require the corresponding datasets and can be computationally expensive.

## Demo

Open `demo/operator_overview.svg` in a browser or image viewer. It illustrates the operator concepts using synthetic marks only. It does not run the benchmark generators or reproduce reported numerical results.

## Provenance and scope

The operator scripts are selected copies of the project's dataset-specific generator code. Path defaults were changed from workstation-specific paths to relative paths or environment-variable settings. The operator logic is otherwise retained in these copies. This repository deliberately excludes source datasets, derived full-dataset annotations, model weights, logs, analysis tables, and manuscript files.

