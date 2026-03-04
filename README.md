# CCoMAML — Cooperative MAML for Cattle Identification

Few-shot cattle muzzle identification using Cooperative Model-Agnostic Meta-Learning (CCoMAML).

---

## Method

CCoMAML extends MAML with a co-learner network $f_\psi$ that estimates a gradient correction $\Delta g_i$ from the previous episode's adapted parameters. The augmented gradient steers the meta-learner toward better generalisation across episodes.

| Symbol | Variable | Description |
|--------|----------|-------------|
| $\alpha$ | `lr_inner` | Inner loop learning rate |
| $\beta$ | `meta_lr` | Meta-learner outer loop lr |
| $\beta_{co}$ | `co_lr` | Co-learner learning rate |
| $\gamma$ | `loss_scaling` | Co-learner loss weight |
| $\mu$ | `mu` | Gradient correction magnitude |
| $R_{L2}$ | `weight_decay` | L2 regularisation |

**Key equations:**

```
Inner loop:          φ_i' = θ - α · ∇_θ L(θ, D_i^Su)
Meta gradient:       g_i  = ∇_θ L_meta(φ_i'(θ), D_i^Qu)
Co-learner:          Δg_i = f_ψ(φ_{i-1}', D_i^Qu)
Augmented gradient:  g̃_i  = g_i + μ · Δg_i
Meta update:         θ   ← θ - β · mean(g̃_i)
Co-learner update:   ψ   ← ψ - β_co · ∇_ψ (γ · L_co)
```

---

## Repository Structure

```
CCoMAML/
├── configs/
│   └── default.yaml        # All hyperparameters
├── src/
│   ├── __init__.py
│   ├── meta_modules.py     # MetaModule, MetaLinear, get_subdict
│   ├── models.py           # BaseNet (MHAFF), CoLearner
│   ├── dataset.py          # FSLCDataset — N-way K-shot episode sampler
│   └── utils.py            # Shared utility functions
├── train.py                # Training loop + CLI entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/CCoMAML.git
cd CCoMAML
pip install -r requirements.txt
```

---

## Data Preparation

Organise your dataset so each class has its own sub-folder:

```
data/
├── train/
│   ├── cow_001/
│   │   ├── img1.jpg
│   │   └── ...
│   └── cow_002/
│       └── ...
└── val/
    ├── cow_101/
    └── ...
```

Each class must have at least `num_shots + num_queries` images.

---

## Training

**Using default config:**
```bash
python train.py
```

**Override specific values:**
```bash
python train.py --train_folder data/train --val_folder data/val --num_epochs 100
```

**Custom config file:**
```bash
python train.py --config configs/my_config.yaml
```

Edit `configs/default.yaml` to change any hyperparameter persistently.

---

## Checkpoints

Saved to `save_dir` (default: `weights/CCoMAML/`):

| File | Description |
|------|-------------|
| `best_base_model.pt` | BaseNet weights at best validation accuracy |
| `best_colearner.pt` | CoLearner weights at best validation accuracy |
| `epoch_XXX_base_model.pt` | Per-epoch BaseNet snapshot |
| `epoch_XXX_colearner.pt` | Per-epoch CoLearner snapshot |


---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- See `requirements.txt` for full list

---

## Contact
If you have any questions, suggestions or need assistance, please don't hesitate to contact me at dulalatom@gmail.com
