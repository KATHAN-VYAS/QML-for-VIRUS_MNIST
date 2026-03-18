# QML-for-VIRUS_MNIST

Hybrid **Quantum Machine Learning (QML)** experiments on the **VirusMNIST** image dataset using **PyTorch + PennyLane**.

This repository contains Jupyter notebooks and saved model checkpoints exploring:
- A **hybrid quantum-classical classifier** (CNN feature reducer → variational quantum circuit → classifier head)
- A **purely classical baseline** using the same feature-reduction backbone for a fair comparison
- Training logs, plots (confusion matrix / per-class F1 / epsilon sweep figures), and `.pth` checkpoints

> Note: Most of the runnable code is currently inside notebooks under `Kathan2/`.

---

## Project structure (high level)

- `Kathan2/`
  - `exp2.ipynb` — **Hybrid QNN** training + evaluation
  - `exp2 - classical.ipynb` — **Classical baseline** training + evaluation
  - `exp1.*.ipynb` — additional experiments
  - `*.pth` — saved model weights / checkpoints
  - `*.png` — generated figures (confusion matrix, F1 plots, sweep plots, etc.)
  - `timm-1.0.25-*.whl` — a wheel file included in the repo (optional dependency)

Other folders exist (e.g., `using RESNET/`, `other testing/`) depending on what you’re exploring.

---

## What the hybrid model does

The hybrid notebook (`Kathan2/exp2.ipynb`) builds a pipeline:

1. **Image loading** via `torchvision.datasets.ImageFolder`
2. **Transforms**: grayscale → tensor → normalize
3. **FeatureReduce CNN** compresses images into a small vector of size `n_qubits` (default: `6`)
4. **Quantum layer** (PennyLane `qnode` wrapped as a `pennylane.qnn.TorchLayer`)
   - Applies RX/RY encodings
   - Has trainable RY layers + entangling CNOT ring
   - Returns expectation values of Pauli-Z measurements (one per qubit)
5. **Classifier head** maps quantum outputs to `num_classes` (default: `10`)

---

## Dataset layout (expected)

The notebooks expect a folder layout like:

```text
train/
  class0/
  class1/
  ...
test/
  class0/
  ...
val/
  class0/
  ...
```

Each class folder should contain the corresponding images.

If you are using VirusMNIST from MedMNIST (or another source), you may need a preprocessing/export step to convert it into the `ImageFolder` directory format above.

---

## Requirements

Typical environment requirements (based on notebook imports):

- Python 3.9+ (3.10+ recommended)
- PyTorch
- torchvision
- PennyLane
- numpy
- scikit-learn
- tqdm
- matplotlib, seaborn (for plots)

Example install:

```bash
pip install torch torchvision pennylane numpy scikit-learn tqdm matplotlib seaborn
```

If you want to use the wheel committed in the repo:

```bash
pip install Kathan2/timm-1.0.25-py3-none-any.whl
```

---

## How to run

### Option A — Run notebooks (recommended)

1. Create/activate a Python environment.
2. Install dependencies (see above).
3. Place your dataset folders in the repo root so paths match the notebooks:
   - `./train`, `./test`, `./val`
4. Open and run:
   - `Kathan2/exp2.ipynb` (hybrid QNN)
   - `Kathan2/exp2 - classical.ipynb` (baseline)

### Option B — Reuse saved checkpoints

The hybrid notebook saves best weights during training:

- `exp2.ipynb` saves to: `exp2.pth` (in the working directory)
- The repo also contains other `.pth` files under `Kathan2/` you can load similarly.

---

## Configuration knobs (in notebooks)

Common parameters set near the top of the notebooks:

- `n_qubits = 6`
- `batch_size = 64`
- `num_classes = 10`
- `num_epochs = 80`
- `lr = 0.0003`

The quantum circuit uses a trainable weight tensor shaped like:

```python
weight_shapes = {"weights": (3, n_qubits)}
```

---

## Outputs / evaluation

The hybrid notebook includes:
- Validation accuracy tracking + checkpointing
- Classification report (`sklearn.metrics.classification_report`)
- Confusion matrix plotting (seaborn heatmap)
- Additional plots saved in `Kathan2/*.png`

---

## Notes / caveats

- The GitHub repo contains many large binaries (`.pth`, `.png`, `.whl`). If you plan to collaborate, consider using Git LFS.
- Some directories/files may be experimental (“other testing”, “using RESNET”), so treat them as work-in-progress.
- Dataset preparation is not fully scripted in the repo yet (the notebooks assume `ImageFolder` directories exist).

---

## Citation / attribution

If you use MedMNIST / VirusMNIST, please cite the corresponding dataset/publication as required by the dataset authors.

---

## Author

- GitHub: **KATHAN-VYAS**

---

## License

Add a license file if you intend others to reuse this work (MIT/Apache-2.0/CC-BY, etc.).
