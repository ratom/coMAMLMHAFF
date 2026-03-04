"""
train.py
--------
CCoMAML training loop — Cooperative MAML for Cattle Identification.

Algorithm overview (follows the paper exactly)
-----------------------------------------------
INNER LOOP          phi_i' = theta - alpha * grad_theta L(theta, D_i^Su)
META GRADIENT       g_i    = grad_theta L_meta(phi_i'(theta), D_i^Qu)
CO-LEARNER          Delta_g_i = f_psi(phi_{i-1}', D_i^Qu)
AUGMENTED GRADIENT  g_tilde_i = g_i + mu * Delta_g_i
AUGMENTED UPDATE    theta <- theta - beta * mean(g_tilde over batch)
CO-LEARNER UPDATE   psi   <- psi - beta_co * grad_psi(gamma * L_co)

Usage
-----
    # Using default config:
    python train.py

    # Override specific values:
    python train.py --train_folder /path/to/train --num_epochs 100

    # Point to a custom config file:
    python train.py --config configs/my_config.yaml
"""

import os
import math
import argparse
from collections import OrderedDict

import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score

from src import (
    BaseNet, CoLearner, FSLCDataset,
    get_accuracy, compute_grad_correction_dim,
    get_phi_flat, unflatten_delta_g,
)


# ===========================================================================
# Training function
# ===========================================================================

def train_fsl_model(
    train_folder:        str,
    val_folder:          str,
    num_epochs:          int   = 200,
    device:              str   = 'cuda:0',
    num_ways:            int   = 5,
    num_shots:           int   = 5,
    num_queries:         int   = 5,
    inner_updates:       int   = 1,
    lr_inner:            float = 0.01,
    meta_lr:             float = 1e-4,
    co_lr:               float = 1e-4,
    weight_decay:        float = 1e-5,
    early_stop_patience: int   = 10,
    loss_scaling:        float = 0.2,
    mu:                  float = 1.0,
    episodes_per_epoch:  int   = 1000,
    batch_size:          int   = 16,
    save_dir:            str   = 'weights/CCoMAML',
) -> dict:
    """Run CCoMAML meta-training.

    Args:
        train_folder        : Path to training class folders.
        val_folder          : Path to validation class folders.
        num_epochs          : Maximum training epochs.
        device              : CUDA device string or 'cpu'.
        num_ways            : N — classes per episode.
        num_shots           : K — support images per class.
        num_queries         : Query images per class.
        inner_updates       : Number of inner-loop gradient steps.
        lr_inner            : alpha — inner loop learning rate.
        meta_lr             : beta  — meta-learner outer loop lr.
        co_lr               : beta_co — co-learner learning rate.
        weight_decay        : R_L2 — L2 regularisation (Adam weight_decay).
        early_stop_patience : Epochs without improvement before stopping.
        loss_scaling        : gamma — co-learner loss weight.
        mu                  : Gradient correction magnitude.
        episodes_per_epoch  : Number of episodes to process per epoch.
        batch_size          : Episodes per DataLoader batch.
        save_dir            : Directory to save model checkpoints.

    Returns:
        history (dict): Per-epoch metrics with keys:
            epoch, loss_meta, loss_co, loss_total,
            val_accuracy, val_f1_macro, val_f1_weighted.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Checkpoint directory
    # Saved files:
    #   best_base_model.pt            — best validation checkpoint
    #   best_colearner.pt             — best validation checkpoint
    #   epoch_XXX_base_model.pt       — per-epoch checkpoint
    #   epoch_XXX_colearner.pt        — per-epoch checkpoint
    # ------------------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoints -> {save_dir}")

    best_base_path = os.path.join(save_dir, 'best_base_model.pt')
    best_co_path   = os.path.join(save_dir, 'best_colearner.pt')

    # ------------------------------------------------------------------
    # Model initialisation
    # ------------------------------------------------------------------
    base_model = BaseNet(num_classes=num_ways).to(device)
    grad_correction_dim = compute_grad_correction_dim(base_model)

    colearner = CoLearner(
        in_channels=3,
        num_classes=num_ways,
        hidden_size=64,
        grad_correction_dim=grad_correction_dim,
    ).to(device)

    # ------------------------------------------------------------------
    # Optimisers
    #   optimizer_base : updates theta via g_tilde (beta)
    #   optimizer_co   : updates psi via gamma * L_co (beta_co)
    # ------------------------------------------------------------------
    optimizer_base = torch.optim.Adam(
        base_model.parameters(), lr=meta_lr, weight_decay=weight_decay
    )
    optimizer_co = torch.optim.Adam(
        colearner.parameters(), lr=co_lr, weight_decay=weight_decay
    )

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------
    train_loader = DataLoader(
        FSLCDataset(train_folder, num_ways, num_shots, num_queries,
                    episodes=episodes_per_epoch),
        batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        FSLCDataset(val_folder, num_ways, num_shots, num_queries,
                    episodes=episodes_per_epoch),
        batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    best_val_acc     = 0.0
    patience_counter = 0

    history = {
        'epoch'          : [],
        'loss_meta'      : [],
        'loss_co'        : [],
        'loss_total'     : [],
        'val_accuracy'   : [],
        'val_f1_macro'   : [],
        'val_f1_weighted': [],
    }

    # phi_{i-1}' initialised from current theta at training start
    phi_prev_flat = get_phi_flat(base_model).to(device)

    # ==================================================================
    # EPOCH LOOP
    # ==================================================================
    for epoch in range(num_epochs):
        base_model.train()
        colearner.train()

        ep_loss_meta  = 0.0
        ep_loss_co    = 0.0
        ep_loss_total = 0.0
        ep_count      = 0

        pbar = tqdm(
            train_loader,
            total=min(
                math.ceil(episodes_per_epoch / batch_size),
                len(train_loader),
            ),
            desc=f"Epoch {epoch+1}/{num_epochs}",
            leave=False,
        )

        # --------------------------------------------------------------
        # BATCH LOOP
        # --------------------------------------------------------------
        for batch in pbar:
            if ep_count * batch_size >= episodes_per_epoch:
                break
            ep_count += 1

            (support_trans, support_res, support_labels,
             query_trans,   query_res,   query_labels) = batch

            batch_g_tilde   = None
            batch_loss_meta = 0.0
            batch_loss_co   = 0.0
            n_episodes      = support_trans.shape[0]

            # ----------------------------------------------------------
            # EPISODE LOOP — inner loop + gradient accumulation
            # ----------------------------------------------------------
            for ep_idx in range(n_episodes):
                s_trans  = support_trans[ep_idx].to(device)
                s_res    = support_res[ep_idx].to(device)
                s_labels = support_labels[ep_idx].to(device)
                q_trans  = query_trans[ep_idx].to(device)
                q_res    = query_res[ep_idx].to(device)
                q_labels = query_labels[ep_idx].to(device)

                # --- INNER LOOP (Eq. inner_loop) ---
                # phi_i' = theta - alpha * grad_theta L(theta, D^Su)
                # create_graph=True preserves graph for outer-loop grad.
                phi = OrderedDict(base_model.meta_named_parameters())

                for _ in range(inner_updates):
                    support_out, _ = base_model(s_trans, s_res, params=phi)
                    loss_inner = F.cross_entropy(support_out, s_labels)

                    grads = torch.autograd.grad(
                        loss_inner, list(phi.values()),
                        create_graph=True, allow_unused=True,
                    )
                    phi = OrderedDict(
                        (name,
                         param - lr_inner * (
                             grad if grad is not None
                             else torch.zeros_like(param)))
                        for (name, param), grad in zip(phi.items(), grads)
                    )

                # --- OUTER LOOP Step 1: Primary meta-gradient (Eq. meta_gradient) ---
                # g_i = grad_theta L_meta(phi_i'(theta), D^Qu)
                query_out, _ = base_model(q_trans, q_res, params=phi)
                loss_meta_ep = F.cross_entropy(query_out, q_labels)

                meta_params = list(base_model.meta_parameters())
                g_i_list = torch.autograd.grad(
                    loss_meta_ep, meta_params,
                    create_graph=False, allow_unused=True,
                )
                g_i_list = [
                    g if g is not None else torch.zeros_like(p)
                    for g, p in zip(g_i_list, meta_params)
                ]
                g_i_named = OrderedDict(
                    (name, g)
                    for (name, _), g in zip(
                        base_model.meta_named_parameters(), g_i_list)
                )

                # --- OUTER LOOP Step 2: Co-learner correction (Eq. co_learner_gradient) ---
                # Delta_g_i = f_psi(phi_{i-1}', D^Qu)
                delta_g_flat, co_logits = colearner(q_trans, phi_prev_flat)
                delta_g_named = unflatten_delta_g(delta_g_flat, base_model)

                # --- OUTER LOOP Step 3: Augmented gradient (Eq. augmented_gradient) ---
                # g_tilde_i = g_i + mu * Delta_g_i
                g_tilde_ep = OrderedDict(
                    (name, g_i_named[name] + mu * delta_g_named[name])
                    for name in g_i_named
                )

                # Accumulate g_tilde across episodes
                if batch_g_tilde is None:
                    batch_g_tilde = {k: v.detach().clone()
                                     for k, v in g_tilde_ep.items()}
                else:
                    for k in batch_g_tilde:
                        batch_g_tilde[k] += g_tilde_ep[k].detach().clone()

                # Co-learner loss and update (Eq. co_learner_update)
                loss_co_ep       = F.cross_entropy(co_logits, q_labels)
                loss_co_weighted = loss_scaling * loss_co_ep
                batch_loss_meta += loss_meta_ep.item()
                batch_loss_co   += loss_co_weighted.item()

                optimizer_co.zero_grad()
                loss_co_weighted.backward()
                optimizer_co.step()

                # Update phi_{i-1}' <- phi_i'
                phi_prev_flat = torch.cat(
                    [v.detach().view(-1) for v in phi.values()]
                ).to(device)

            # --- OUTER LOOP Step 4: Meta-parameter update (Eq. augmented_update) ---
            # theta <- theta - beta * mean(g_tilde over batch)
            if batch_g_tilde is not None:
                for k in batch_g_tilde:
                    batch_g_tilde[k] /= n_episodes

            optimizer_base.zero_grad()
            for name, param in base_model.meta_named_parameters():
                if name in batch_g_tilde:
                    param.grad = batch_g_tilde[name]
            optimizer_base.step()

            avg_loss_meta_batch  = batch_loss_meta / n_episodes
            avg_loss_co_batch    = batch_loss_co   / n_episodes
            avg_loss_total_batch = avg_loss_meta_batch + avg_loss_co_batch

            ep_loss_meta  += avg_loss_meta_batch
            ep_loss_co    += avg_loss_co_batch
            ep_loss_total += avg_loss_total_batch

            pbar.set_postfix({
                'L_meta' : f"{avg_loss_meta_batch:.4f}",
                'γ·L_co' : f"{avg_loss_co_batch:.4f}",
                'L_total': f"{avg_loss_total_batch:.4f}",
            })

        # Reset phi_{i-1}' to theta at epoch boundary
        phi_prev_flat = get_phi_flat(base_model).to(device)

        avg_loss_meta  = ep_loss_meta  / max(1, ep_count)
        avg_loss_co    = ep_loss_co    / max(1, ep_count)
        avg_loss_total = ep_loss_total / max(1, ep_count)

        # ==============================================================
        # VALIDATION
        # ==============================================================
        base_model.eval()

        all_preds, all_labels_list = [], []
        val_acc = 0.0
        vcount  = 0

        for batch in val_loader:
            (support_trans, support_res, support_labels,
             query_trans,   query_res,   query_labels) = batch

            for ep_idx in range(support_trans.shape[0]):
                s_trans  = support_trans[ep_idx].to(device)
                s_res    = support_res[ep_idx].to(device)
                s_labels = support_labels[ep_idx].to(device)
                q_trans  = query_trans[ep_idx].to(device)
                q_res    = query_res[ep_idx].to(device)
                q_labels = query_labels[ep_idx].to(device)

                phi_val = OrderedDict(
                    (name, param.clone().requires_grad_(True))
                    for name, param in base_model.meta_named_parameters()
                )

                with torch.enable_grad():
                    for _ in range(inner_updates):
                        support_out, _ = base_model(
                            s_trans, s_res, params=phi_val)
                        loss_val_inner = F.cross_entropy(support_out, s_labels)
                        grads = torch.autograd.grad(
                            loss_val_inner, list(phi_val.values()),
                            create_graph=False, allow_unused=True,
                        )
                        phi_val = OrderedDict(
                            (name,
                             (param - lr_inner * (
                                 grad if grad is not None
                                 else torch.zeros_like(param))
                              ).detach().requires_grad_(True))
                            for (name, param), grad in zip(
                                phi_val.items(), grads)
                        )

                with torch.no_grad():
                    query_out, _ = base_model(q_trans, q_res, params=phi_val)
                    val_acc += get_accuracy(query_out, q_labels).item()
                    vcount  += 1
                    _, preds = torch.max(query_out, dim=1)
                    all_preds.extend(preds.cpu().numpy().tolist())
                    all_labels_list.extend(q_labels.cpu().numpy().tolist())

        val_acc     /= max(1, vcount)
        all_preds_np = np.array(all_preds)
        all_lbls_np  = np.array(all_labels_list)

        f1_macro    = f1_score(all_lbls_np, all_preds_np,
                               average='macro',    zero_division=0)
        f1_weighted = f1_score(all_lbls_np, all_preds_np,
                               average='weighted', zero_division=0)

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        print(f"\n{'='*68}")
        print(f"  Epoch {epoch+1}/{num_epochs}  (batch_size={batch_size})")
        print(f"  Train | L_meta={avg_loss_meta:.4f}  "
              f"γ·L_co={avg_loss_co:.4f}  L_total={avg_loss_total:.4f}")
        print(f"  Val   | Accuracy={val_acc*100:.2f}%  "
              f"F1-Macro={f1_macro:.4f}  F1-Weighted={f1_weighted:.4f}")
        print(f"{'='*68}")

        history['epoch'].append(epoch + 1)
        history['loss_meta'].append(avg_loss_meta)
        history['loss_co'].append(avg_loss_co)
        history['loss_total'].append(avg_loss_total)
        history['val_accuracy'].append(val_acc)
        history['val_f1_macro'].append(f1_macro)
        history['val_f1_weighted'].append(f1_weighted)

        # ------------------------------------------------------------------
        # Checkpointing
        # ------------------------------------------------------------------
        epoch_base_path = os.path.join(
            save_dir, f'epoch_{epoch+1:03d}_base_model.pt')
        epoch_co_path = os.path.join(
            save_dir, f'epoch_{epoch+1:03d}_colearner.pt')

        torch.save(base_model.state_dict(), epoch_base_path)
        torch.save(colearner.state_dict(),  epoch_co_path)
        print(f"  Saved -> {epoch_base_path}")

        if val_acc > best_val_acc + 1e-6:
            best_val_acc     = val_acc
            patience_counter = 0
            torch.save(base_model.state_dict(), best_base_path)
            torch.save(colearner.state_dict(),  best_co_path)
            print(f"  ** New best val acc: {best_val_acc*100:.2f}% "
                  f"-> {best_base_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break

    # ------------------------------------------------------------------
    # Final summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*68}")
    print(f"  Training complete.  Best Val Acc: {best_val_acc*100:.2f}%")
    print(f"  Best weights: {best_base_path}")
    print(f"                {best_co_path}")
    print(f"{'='*68}")

    print(f"\n{'Epoch':>6} | {'L_meta':>8} | {'γ·L_co':>8} | "
          f"{'L_total':>8} | {'Acc(%)':>8} | {'F1-Mac':>8} | {'F1-Wgt':>8}")
    print('-' * 72)
    for i in range(len(history['epoch'])):
        print(
            f"{history['epoch'][i]:>6} | "
            f"{history['loss_meta'][i]:>8.4f} | "
            f"{history['loss_co'][i]:>8.4f} | "
            f"{history['loss_total'][i]:>8.4f} | "
            f"{history['val_accuracy'][i]*100:>7.2f}% | "
            f"{history['val_f1_macro'][i]:>8.4f} | "
            f"{history['val_f1_weighted'][i]:>8.4f}"
        )

    return history


# ===========================================================================
# CLI entry point
# ===========================================================================

def _parse_args():
    parser = argparse.ArgumentParser(
        description="CCoMAML — Cooperative MAML for Cattle Identification"
    )
    parser.add_argument(
        '--config', type=str, default='configs/default.yaml',
        help='Path to YAML config file (default: configs/default.yaml)',
    )
    # Allow any config key to be overridden on the command line
    parser.add_argument('--train_folder',        type=str)
    parser.add_argument('--val_folder',          type=str)
    parser.add_argument('--num_epochs',          type=int)
    parser.add_argument('--device',              type=str)
    parser.add_argument('--num_ways',            type=int)
    parser.add_argument('--num_shots',           type=int)
    parser.add_argument('--num_queries',         type=int)
    parser.add_argument('--inner_updates',       type=int)
    parser.add_argument('--lr_inner',            type=float)
    parser.add_argument('--meta_lr',             type=float)
    parser.add_argument('--co_lr',               type=float)
    parser.add_argument('--weight_decay',        type=float)
    parser.add_argument('--early_stop_patience', type=int)
    parser.add_argument('--loss_scaling',        type=float)
    parser.add_argument('--mu',                  type=float)
    parser.add_argument('--episodes_per_epoch',  type=int)
    parser.add_argument('--batch_size',          type=int)
    parser.add_argument('--save_dir',            type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    # Load config file
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Command-line arguments override config file values
    for key, val in vars(args).items():
        if key != 'config' and val is not None:
            cfg[key] = val

    train_fsl_model(**cfg)
