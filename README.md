# Scribble Operator Benchmark

A research code bundle for the Class, Offset, Direction, Shrink, Random Drop, and Fragmentation scribble perturbation operators used with VOC2012 and Cityscapes.

## Repository contents

- `operators/voc2012/` and `operators/cityscapes/` contain the dataset-specific operator scripts. Fragmentation has a generator (`fragmentation_noise.py`) and its matching score module (`fragmentation_score.py`), which should remain together in each dataset folder.
- `demo/operator_overview.svg` is a synthetic conceptual illustration of the four operators. It is not a real sample or an experimental result.
- `requirements.txt` lists Python dependencies.
- `PROVENANCE.csv` maps each included script to its source path in the research workspace.

The intended shareable contents are code, documentation, and a synthetic demonstration. The repository does not contain source images, ground-truth masks, scribble annotations, perturbed dataset outputs, predictions, model weights, experiment logs, or manuscript files.

## Obtain the source datasets

### PASCAL VOC2012

Download the training/validation data and development kit from the [official VOC2012 page](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/). The official development kit includes `JPEGImages`, `SegmentationClass`, and split lists under `VOCdevkit/VOC2012`. The VOC2012 test set does not provide public pixel-level segmentation labels.

The automatic scribble annotations used in the experiments are not part of the official VOC download and are not included here. Obtain them from an authorized project source. Prepare the same image basenames in the ground-truth and scribble directories before running the scripts.

### Cityscapes

Create an account and download the needed image and fine-annotation packages from the [official Cityscapes portal](https://www.cityscapes-dataset.com/downloads/). Access requires registration and approval. Read the [current Cityscapes terms](https://www.cityscapes-dataset.com/license/) before downloading or using the data. The terms prohibit redistribution of the dataset and modified versions. Do not place Cityscapes images, masks, scribbles, or recoverable derivatives in this repository or another shared location unless the proposed sharing is explicitly permitted.

The official package uses nested city directories such as `leftImg8bit/train/<city>/` and `gtFine/train/<city>/`. The experiment scripts do not all consume this raw layout directly. Prepare the paired, same-name PNG masks expected by the selected script, and check its label encoding before running it.

## Suggested local data layout

The following layout is a guide for local files only. The `datasets/` directory is ignored by Git.

```text
datasets/
├── voc2012/
│   ├── VOCdevkit/VOC2012/
│   │   ├── JPEGImages/
│   │   ├── SegmentationClass/
│   │   └── ImageSets/Segmentation/
│   ├── GtMask/                 # prepared per-image GT masks used by the scripts
│   └── auto_scribbles_idx/     # authorized scribble masks, paired by PNG basename
└── cityscapes/
    ├── leftImg8bit/            # official nested image archive, local only
    ├── gtFine/                 # official nested annotations, local only
    ├── GtMask/                 # prepared per-image masks for the selected script
    └── auto_scribbles_idx/     # prepared scribbles paired by PNG basename
```

For scripts that read a flat directory, put one PNG per image directly in `GtMask/` and `auto_scribbles_idx/`. Use matching basenames and dimensions. The Cityscapes Direction script can pair nested files recursively, but the Cityscapes Class and Offset scripts enumerate PNG files at the top level, so use a flat prepared directory for those scripts. Avoid basename collisions when flattening.

## Label encoding and background values

- **VOC2012 semantic masks:** class IDs are 0–20, with 0 as the valid background class and 255 as void/ignore. Do not convert 0 to ignore.
- **VOC scribble canvases:** the blank canvas is commonly encoded as 255. This is an annotation-canvas convention and must not be confused with the meaning of 0 in the dense VOC ground truth.
- **Cityscapes:** train IDs 0–18 are valid semantic classes and 255 is ignore. In this encoding, train ID 0 is `road`, not background. Raw label IDs use a different value space, where 0 denotes an unlabeled/void value. Do not mix label IDs and train IDs in one pair.
- The Cityscapes Direction script includes a label-ID-to-train-ID mapping. Other scripts may assume already prepared index masks and do not necessarily perform that conversion. Inspect the selected script and verify the mask values before processing.
- The Cityscapes Fragmentation scripts require single-channel train-ID masks with values 0–18 and ignore 255. The score treats all valid train IDs, including road (0), as foreground. This avoids inventing a semantic background class, but measures connectivity over the union of valid classes. Its score is not directly comparable to the VOC foreground/background score.
- Palette, grayscale, RGB, and indexed masks are not interchangeable. Use the representation expected by the selected script. Ensure paired images have the same width and height and that every label value is valid for that dataset.

## Install dependencies

Use a clean Python environment. From the repository directory:

```bash
python -m venv .venv
```

Activate the environment, then install:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The scripts use NumPy, Pillow, OpenCV, pandas, Matplotlib, tqdm, SciPy, and scikit-image. Install a wheel compatible with your operating system and Python version. The full image sweeps can take substantial time and memory.

## Configure paths

Paths can be set per terminal session. The defaults are relative to the current working directory.

### PowerShell

```powershell
$env:VOC_DATA_ROOT = "D:\data\voc2012"
$env:VOC_GT_DIR = "D:\data\voc2012\GtMask"
$env:VOC_SCRIBBLE_DIR = "D:\data\voc2012\auto_scribbles_idx"
$env:VOC_CLASS_OUT_ROOT = "D:\results\voc2012\class"
$env:VOC_DIRECTION_OUT_ROOT = "D:\results\voc2012\direction"
$env:VOC_OFFSET_OUT_ROOT = "D:\results\voc2012\offset"
$env:VOC_FRAGMENT_GT_DIR = "D:\data\voc2012\GtMask"
$env:VOC_FRAGMENT_SCRIBBLE_DIR = "D:\data\voc2012\auto_scribbles_idx"
$env:VOC_FRAGMENT_OUT_ROOT = "D:\results\voc2012\fragmentation"
$env:VOC_FRAGMENT_SCORE_OUT_DIR = "D:\results\voc2012\fragmentation_score"

$env:CITYSCAPES_GT_DIR = "D:\data\cityscapes\GtMask"
$env:CITYSCAPES_SCRIBBLE_DIR = "D:\data\cityscapes\auto_scribbles_idx"
$env:CITYSCAPES_CLASS_OUT_ROOT = "D:\results\cityscapes\class"
$env:CITYSCAPES_DIRECTION_OUT_ROOT = "D:\results\cityscapes\direction"
$env:OFFSET_BASE_DIR = "D:\data\cityscapes"
$env:OFFSET_OUT_ROOT = "D:\results\cityscapes\offset"
$env:CITYSCAPES_FRAGMENT_GT_DIR = "D:\data\cityscapes\GtMask"
$env:CITYSCAPES_FRAGMENT_SCRIBBLE_DIR = "D:\data\cityscapes\auto_scribbles_idx"
$env:CITYSCAPES_FRAGMENT_OUT_ROOT = "D:\results\cityscapes\fragmentation"
$env:CITYSCAPES_FRAGMENT_SCORE_OUT_DIR = "D:\results\cityscapes\fragmentation_score"
```

`OFFSET_BASE_DIR` is shared by the two Offset scripts. Set it to the appropriate dataset root before running each one. The Offset scripts then expect `GtMask/` and `auto_scribbles_idx/` beneath that root.

### macOS/Linux

Use the same variable names with `export NAME=/path/to/directory`. For example:

```bash
export VOC_DATA_ROOT=/data/voc2012
export VOC_GT_DIR=/data/voc2012/GtMask
export VOC_SCRIBBLE_DIR=/data/voc2012/auto_scribbles_idx
```

## Run an operator

Run commands from the repository root after configuring the paths. Class, Direction, Offset, and Fragmentation scripts run their configured dataset sweep when invoked. Shrink and Random Drop accept input and output directories and expose their options through `--help`.

```bash
python operators/voc2012/class_noise_idx.py
python operators/voc2012/direction_noise.py
python operators/voc2012/offset_noise.py
python operators/voc2012/shrink.py --help
python operators/voc2012/random_drop.py --help
python operators/voc2012/fragmentation_noise.py
python operators/voc2012/fragmentation_score.py
```

For Cityscapes, substitute `cityscapes` for `voc2012` and set the Cityscapes variables above. Fragmentation's score script is also used internally by its generator. The included Fragmentation generator currently defaults to `RATIOS = [0.1]`; edit that constant to run other levels. The standalone score script defaults to `LIMIT_N = 200`; set it to `None` to score every paired image. Before a full sweep, inspect the ratio list, image limit, output path, and label assumptions. Outputs should be directed outside the repository.

Random Drop samples a seeded subset of skeleton points and optionally thickens those points within the original stroke support. It is not independent per-pixel dropout. Example for a single-channel VOC scribble mask with blank value 255:

```bash
python operators/voc2012/random_drop.py \
  --in_dir /data/voc2012/auto_scribbles_idx \
  --out_dir /results/voc2012/random_drop_keep05 \
  --target_keep 0.5 \
  --bg_val 255 \
  --ignore_val 255 \
  --seed 0 \
  --keep_width
```

Adjust `--bg_val` and `--ignore_val` to the actual input encoding. For Cityscapes train-ID scribbles, preserve valid labels 0–18 and use 255 for the blank canvas.

Example Shrink invocation for a mask directory whose blank canvas is 255:

```bash
python operators/voc2012/shrink.py \
  --in_dir /data/voc2012/auto_scribbles_idx \
  --out_dir /results/voc2012/shrink_r05 \
  --target_keep 0.5 \
  --bg_val 255 \
  --ignore_val -1
```

Check the selected script's `--help` output and label semantics before applying this example to other encodings.

## Demo

Open `demo/operator_overview.svg` in a browser or image viewer. It uses synthetic marks only and does not execute the dataset generators or reproduce reported numerical results.

## Citation and licensing

Cite the [VOC2012 challenge](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/) and the [Cityscapes dataset](https://www.cityscapes-dataset.com/) when using them. Cityscapes has specific registration, use, and non-redistribution terms; consult the linked license directly. Dataset access and compliance are the user's responsibility.

The included scripts are selected copies of research code. Workstation-specific path defaults were changed to environment-variable or relative-path settings; the perturbation logic was retained. The Cityscapes Fragmentation score settings treat train-ID 0 as a valid class and omit a synthetic background class. See `NOTICE.md` and `PROVENANCE.csv`. No open-source license has been assigned to the code in this repository. Public visibility alone does not grant permission to reuse, modify, or redistribute it. The repository does not grant rights to the datasets or third-party code.
