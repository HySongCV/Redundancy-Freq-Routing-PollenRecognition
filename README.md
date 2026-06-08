# Redundancy-Aware Frequency and Routing Optimization for Efficient Fine-Grained Pollen Recognition

This repository provides the implementation and reproducibility materials for the manuscript:

**Redundancy-Aware Frequency and Routing Optimization for Efficient Fine-Grained Pollen Recognition**

The paper studies efficient fine-grained pollen recognition based on the heat-conduction-based vision backbone vHeat. We focus on the Heat Conduction Operator (HCO) and investigate two aspects of exploitable redundancy inside HCO: **channel-path redundancy** and **frequency-response redundancy**. Based on this analysis, we propose an HCO-oriented redundancy-aware optimization framework consisting of:

* **Partial Heat Routing (PHR)**: routes part of the channels from the complete HCO / Heat2D branch to a lightweight local branch to reduce redundant full-channel heat computation.
* **Frequency-domain Regularization (FDR)**: introduces a controlled frequency-retention constraint in the DCT domain to reduce dependence on weakly informative or redundant frequency responses.
* **Stage-wise deployment strategy**: coordinates the routing ratios of PHR and the frequency keep ratios of FDR across different stages of vHeat.

The released code is intended to support the main training, evaluation, and profiling procedures reported in the manuscript.

---

## Repository Structure

```text
.
├── configs/
│   └── vHeat/
│       └── vHeat_tiny_224.yaml
│
├── data/
│   ├── __init__.py
│   ├── build.py
│   ├── pollen149.py
│   ├── new_zealand_pollen.py
│   └── cpd1.py
│
├── evaluate/
│   ├── eval_models_speed_flops.py
│   ├── eval_test.py
│   └── variants.json
│
├── models/
│   ├── __init__.py
│   └── vHeat.py
│
├── utils/
│   ├── config.py
│   ├── logger.py
│   ├── lr_scheduler.py
│   ├── optimizer.py
│   └── utils.py
│
└── main.py
```

---

## Naming Note

For compatibility with the original experimental implementation, some configuration keys keep their original names:

```text
MODEL.PARTIAL_HEAT  -> Partial Heat Routing (PHR)
MODEL.FP            -> Frequency-domain Regularization (FDR)
```

In the manuscript, we refer to `PARTIAL_HEAT` as **PHR** and `FP` as **FDR**.

---

## Environment

The experiments in the manuscript were conducted with PyTorch 2.1.0 on a single NVIDIA RTX 3090 GPU.

A typical environment can be prepared as follows:

```bash
conda create -n phr_fdr python=3.10 -y
conda activate phr_fdr

pip install torch==2.1.0 torchvision
pip install timm yacs pyyaml numpy pandas tqdm scikit-learn pillow opencv-python thop
```

CUDA versions should be adjusted according to the local GPU and PyTorch installation.

---

## Datasets

This repository supports the following datasets used in the manuscript:

* **Pollen149**
* **New Zealand Pollen Dataset**
* **CPD-1**

### Dataset names used in code

```text
Pollen149                 -> pollen149
New Zealand Pollen Dataset -> new_zealand_pollen
CPD-1                     -> cpd1
```

### Expected directory format

For folder-based datasets, the expected structure is:

```text
dataset_root/
├── train/
│   ├── class_001/
│   │   ├── image_0001.jpg
│   │   └── ...
│   └── class_002/
│       └── ...
│
├── val/
│   ├── class_001/
│   └── class_002/
│
└── test/
    ├── class_001/
    └── class_002/
```

If CSV split files are used, the expected format is:

```csv
image_path,label
train/class_001/image_0001.jpg,0
train/class_002/image_0002.jpg,1
```

For datasets that cannot be redistributed, this repository provides the expected data organization format and reproduction commands rather than releasing restricted image data.

---

## Configuration

The main configuration file is:

```text
configs/vHeat/vHeat_tiny_224.yaml
```

The final PHR+FDR configuration used in the manuscript is:

```yaml
MODEL:
  PARTIAL_HEAT:
    ENABLE: True
    MODE: stage
    HEAT_RATIO: 1.0
    STAGE_HEAT_RATIOS: [0.5, 0.5, 0.625, 0.75]
    LIGHT_BRANCH: dwconv

  FP:
    ENABLE: True
    MODE: stage
    KEEP_RATIO: 1.0
    STAGE_RATIOS: [1.0, 1.0, 0.8, 0.6]
```

The baseline vHeat setting disables both components:

```yaml
MODEL:
  PARTIAL_HEAT:
    ENABLE: False

  FP:
    ENABLE: False
```

---

## Training

### Train original vHeat baseline

```bash
python main.py \
  --cfg configs/vHeat/vHeat_tiny_224.yaml \
  --data-path /path/to/pollen149 \
  --data-set pollen149 \
  --batch-size 128 \
  --output outputs/pollen149 \
  --tag vheat_baseline \
  --opts \
    MODEL.PARTIAL_HEAT.ENABLE False \
    MODEL.FP.ENABLE False
```

### Train PHR-only

```bash
python main.py \
  --cfg configs/vHeat/vHeat_tiny_224.yaml \
  --data-path /path/to/pollen149 \
  --data-set pollen149 \
  --batch-size 128 \
  --output outputs/pollen149 \
  --tag phr_only \
  --opts \
    MODEL.PARTIAL_HEAT.ENABLE True \
    MODEL.PARTIAL_HEAT.MODE stage \
    MODEL.PARTIAL_HEAT.STAGE_HEAT_RATIOS "[0.5,0.5,0.625,0.75]" \
    MODEL.FP.ENABLE False
```

### Train FDR-only

```bash
python main.py \
  --cfg configs/vHeat/vHeat_tiny_224.yaml \
  --data-path /path/to/pollen149 \
  --data-set pollen149 \
  --batch-size 128 \
  --output outputs/pollen149 \
  --tag fdr_only \
  --opts \
    MODEL.PARTIAL_HEAT.ENABLE False \
    MODEL.FP.ENABLE True \
    MODEL.FP.MODE stage \
    MODEL.FP.STAGE_RATIOS "[1.0,1.0,0.8,0.6]"
```

### Train PHR+FDR

```bash
python main.py \
  --cfg configs/vHeat/vHeat_tiny_224.yaml \
  --data-path /path/to/pollen149 \
  --data-set pollen149 \
  --batch-size 128 \
  --output outputs/pollen149 \
  --tag phr_fdr \
  --opts \
    MODEL.PARTIAL_HEAT.ENABLE True \
    MODEL.PARTIAL_HEAT.MODE stage \
    MODEL.PARTIAL_HEAT.STAGE_HEAT_RATIOS "[0.5,0.5,0.625,0.75]" \
    MODEL.FP.ENABLE True \
    MODEL.FP.MODE stage \
    MODEL.FP.STAGE_RATIOS "[1.0,1.0,0.8,0.6]"
```

---

## Evaluation

Evaluate a trained checkpoint:

```bash
python main.py \
  --cfg configs/vHeat/vHeat_tiny_224.yaml \
  --data-path /path/to/pollen149 \
  --data-set pollen149 \
  --batch-size 128 \
  --resume /path/to/checkpoint.pth \
  --eval \
  --output outputs/eval \
  --tag eval_phr_fdr
```

A variant-based evaluation example is also provided in:

```text
evaluate/variants.json
evaluate/eval_test.py
```

Please edit the checkpoint paths in `variants.json` before running the script.

---

## Model Complexity and Inference Profiling

The profiling protocol used in the manuscript is implemented in:

```text
evaluate/eval_models_speed_flops.py
```

The script measures:

* number of parameters,
* FLOPs using `thop`,
* throughput,
* average latency per image under batched inference.

The profiling protocol uses:

```text
FLOPs input:   1 × 3 × 224 × 224
Speed input:   B × 3 × 224 × 224
Batch size B:  256
Warm-up:       50 iterations
Timed runs:    200 iterations
```

Example command:

```bash
python evaluate/eval_models_speed_flops.py \
  --variants-json evaluate/variants.json \
  --batch-size 256 \
  --img-size 224 \
  --device cuda \
  --measure-flops \
  --warmup-iters 50 \
  --speed-iters 200 \
  --output-csv outputs/profiling_summary.csv
```

---

## Example `variants.json`

```json
[
  {
    "name": "vHeat_baseline",
    "cfg": "configs/vHeat/vHeat_tiny_224.yaml",
    "resume": "/path/to/checkpoints/vheat_baseline/ckpt_epoch_best.pth",
    "opts": [
      "MODEL.PARTIAL_HEAT.ENABLE", "False",
      "MODEL.FP.ENABLE", "False"
    ]
  },
  {
    "name": "PHR_FDR",
    "cfg": "configs/vHeat/vHeat_tiny_224.yaml",
    "resume": "/path/to/checkpoints/phr_fdr/ckpt_epoch_best.pth",
    "opts": [
      "MODEL.PARTIAL_HEAT.ENABLE", "True",
      "MODEL.PARTIAL_HEAT.MODE", "stage",
      "MODEL.PARTIAL_HEAT.STAGE_HEAT_RATIOS", "[0.5,0.5,0.625,0.75]",
      "MODEL.FP.ENABLE", "True",
      "MODEL.FP.MODE", "stage",
      "MODEL.FP.STAGE_RATIOS", "[1.0,1.0,0.8,0.6]"
    ]
  }
]
```

---

## Main Results

On Pollen149, the final stage-wise PHR+FDR configuration achieves the following accuracy–efficiency trade-off compared with the original vHeat baseline:

| Method         | Top-1 Acc. | Params | FLOPs |    Throughput |   Latency |
| -------------- | ---------: | -----: | ----: | ------------: | --------: |
| Original vHeat |      96.20 | 30.20M | 4.94G |  924.15 img/s | 1.0821 ms |
| PHR+FDR        |      96.01 | 25.69M | 3.97G | 1015.27 img/s | 0.9850 ms |

The reported speed values are measured under the unified profiling protocol described above.

---

## Code and Data Availability

This repository releases the core implementation, main configuration files, evaluation/profiling scripts, and data organization instructions required to reproduce the main experimental workflow.

Pollen149 cannot be directly redistributed due to data-use restrictions. For restricted datasets, we provide the expected directory structure and split-file format instead of releasing the original image data.

Trained checkpoints may be released where technically and ethically feasible. If some checkpoints cannot be released, the provided configurations, commands, and evaluation protocol can be used to reproduce the experiments with access to the corresponding datasets.

---

## Acknowledgement

Parts of the training utilities are adapted from the Swin Transformer / vHeat-style training framework under the original open-source license. We retain the original copyright notices where applicable.

The main contribution of this repository is the HCO-oriented PHR/FDR implementation, configuration files, and reproducibility materials for efficient fine-grained pollen recognition.

---

## Citation

If you use this code, please cite our manuscript:

```bibtex
@article{song2026redundancy,
  title={Redundancy-Aware Frequency and Routing Optimization for Efficient Fine-Grained Pollen Recognition},
  author={Song, Hongyu and Shi, Bao and Yang, Jingping and Zhang, XinRu and Xue, Yuan and Wang, Mengfei},
  journal={The Visual Computer},
  year={2026},
  note={Manuscript under review}
}
```
