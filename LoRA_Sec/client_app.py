"""LoRA-Sec: A Flower / Hugging Face app."""

# import warnings

import torch
import random
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context
# from transformers import logging
from LoRA_Sec.task import (
    train,
    test,
    load_data,
    set_params,
    get_params,
    get_model,
)


import math
from typing import List
import copy

# --- Mixture attack ----

import hashlib

# ------- FedIMP -------- #
import os, time
import pickle
from pathlib import Path
import numpy as np
from torch.optim import AdamW
# ----------------------- #

global_benign_grads = []


# ====================================================================================================================================================== #

def flatten_lora(model, include_classifier: bool = False, device: str = "cpu") -> torch.Tensor:
    """Concatenate only LoRA (and optionally classifier) parameters."""
    parts = []
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ("lora_A" in name) or ("lora_B" in name) or (include_classifier and name.startswith("classifier")):
                parts.append(p.detach().to(device).reshape(-1))
    if not parts:
        # Safe empty tensor; avoids crashes if LoRA absent
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.cat(parts)

# --------------------------------------------------------------------------------- #

@torch.no_grad()
def unflatten_lora(model, flat_tensor: torch.Tensor, include_classifier: bool = False) -> None:
    """Write a flat LoRA-only vector back into the model (in named_parameters() order)."""
    idx = 0
    for name, p in model.named_parameters():
        use = ("lora_A" in name) or ("lora_B" in name) or (include_classifier and name.startswith("classifier"))
        if not use:
            continue
        n = p.numel()
        if idx + n > flat_tensor.numel():
            raise ValueError("unflatten_lora: vector too short for model parameters")
        chunk = flat_tensor[idx:idx + n].view_as(p).to(p.dtype).to(p.device)
        p.data.copy_(chunk)   # in-place update
        idx += n
    if idx != flat_tensor.numel():
        raise ValueError("unflatten_lora: vector has leftover elements (shape mismatch)")

@torch.no_grad()
def flatten_state_dict_lora(state: dict, include_classifier: bool = False, device: str = "cpu") -> torch.Tensor:
    """Concatenate only LoRA (and optionally classifier) entries from a state_dict-like dict."""
    parts = []
    for name, v in state.items():
        use = ("lora_A" in name) or ("lora_B" in name) or (include_classifier and str(name).startswith("classifier"))
        if use:
            t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
            parts.append(t.to(device).reshape(-1))
    return torch.cat(parts) if parts else torch.empty(0, dtype=torch.float32, device=device)

# --------------------------------------------------------------------------------- #

def compute_perturbation_vector(grads: List[torch.Tensor], mode: str = "std") -> torch.Tensor:
    if not grads:
        raise ValueError("compute_perturbation_vector: received empty gradient list.")

    grads = torch.stack(grads)
    if mode == "std":
        perturbation = torch.std(grads, dim=0)
    elif mode == "mean":
        perturbation = torch.mean(grads, dim=0)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return perturbation

# --------------------------------------------------------------------------------- #
# -------------------------- Helper functions for FedIMP -------------------------- #


def _fedimp_target_indices(model, include_classifier: bool, lora_only: bool) -> list[int]:
    """Return tensor indices in the *named_parameters()* order."""
    names = [n for n, _ in model.named_parameters()]
    idx: list[int] = []
    for i, name in enumerate(names):
        use = True
        if lora_only:
            use = ("lora_A" in name) or ("lora_B" in name)
        if not include_classifier and name.startswith("classifier."):
            use = False
        if use:
            idx.append(i)
    return idx


@torch.no_grad()
def _fedimp_flat_from_indices(arrs, idx):
    """Flatten selected ndarrays (or tensors) into a single 1-D torch tensor."""
    parts = []
    for i in idx:
        a = arrs[i]
        t = torch.from_numpy(a) if isinstance(a, np.ndarray) else a
        parts.append(t.reshape(-1))
    return torch.cat(parts)

def _fedimp_add_into_indices(params, idx, flat_delta):
    """Add a flat vector into a list-of-ndarrays at given indices; returns a NEW params list."""
    out = [np.array(p, copy=True) for p in params]
    cursor = 0
    for i in idx:
        shape = out[i].shape
        size = out[i].size
        chunk = flat_delta[cursor:cursor+size].view(shape).cpu().numpy()
        out[i] = out[i] + chunk
        cursor += size
    return out


# ------- FedIMP acceptance auditor (defense instrumentation) --------

def _flatten_update(params_dict_or_list):
    """Flatten a client's model update (delta) into 1-D np.array."""
    if isinstance(params_dict_or_list, dict):
        parts = []
        for k in sorted(params_dict_or_list.keys()):
            v = params_dict_or_list[k]
            a = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            parts.append(a.reshape(-1))
        return np.concatenate(parts)
    else:
        # list/tuple of ndarrays/tensors
        parts = []
        for v in params_dict_or_list:
            a = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            parts.append(a.reshape(-1))
        return np.concatenate(parts)

def fedimp_acceptance_metrics(flat_updates, n_total, m_compromised_guess=0):
    """
    Compute FedIMP-style k-NN sum^2 metrics for defense auditing.
    flat_updates: list[np.ndarray] of shape [num_clients_this_round, D]
    n_total: total selected clients this round
    m_compromised_guess: if unknown, pass 0; we’ll default k = ceil(n/2)
    Returns dict with per-client LHS sums and global RHS sum.
    """
    U = np.stack(flat_updates, axis=0)  # [N, D]
    N = U.shape[0]
    if N < 2:
        return {"k": 1, "lhs": np.zeros(N), "rhs_sum": 0.0}

    k = max(1, math.ceil(max(1, n_total - m_compromised_guess) / 2))

    # pairwise squared distances
    # (use float64 for numerical stability)
    Uh = U.astype(np.float64)
    # (u - v)^2 = u^2 + v^2 - 2uv
    sq = (Uh ** 2).sum(axis=1, keepdims=True)  # [N,1]
    D2 = sq + sq.T - 2 * (Uh @ Uh.T)           # [N,N]
    np.fill_diagonal(D2, np.inf)

    # LHS: for each client, sum k smallest distances to others
    lhs = np.partition(D2, kth=k-1, axis=1)[:, :k].sum(axis=1)  # [N]

    # RHS: sum over benign-to-benign k-NN squared distances (defensive proxy: use all)
    rhs_per = np.partition(D2, kth=k-1, axis=1)[:, :k].sum(axis=1)  # [N]
    rhs_sum = float(rhs_per.sum())

    return {"k": k, "lhs": lhs, "rhs_sum": rhs_sum}

# ====================================================================================================================================================== #

class ClientClass(NumPyClient):
    def __init__(self, model_name, trainloader, testloader, cid, context , use_ortho_loss, is_malicious=False) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainloader = trainloader
        self.testloader = testloader
        self.net = get_model(model_name, num_labels = int(context.run_config["num_labels"]))
        self.net.to(self.device)
        self.is_malicious = is_malicious
        self.poison_intensity: float = 0.35
        self.cid = cid
        self.CurrentRound = 1
        self.attack_type = context.run_config["attack_type"]  
        self.trim_ratio = context.run_config["attack_parameter"]      
        self.use_ortho_loss = use_ortho_loss
        self.lambda_ortho = context.run_config["lambda_ortho"]

        # --- Mixture config (used when attack_type == "Mixture") ---
        _mix_default = ["RandScaling", "Trim", "MinMax", "MinSum", "PoisonedFL", "FedIMP"]
        mcfg = context.run_config.get("mixture_choices", _mix_default)
        if isinstance(mcfg, str):
            if mcfg.strip() in {"*", "ALL", "all", "Auto", "auto"}:
                mcfg = _mix_default
            else:
                mcfg = [s.strip() for s in mcfg.split(",") if s.strip()]
        self.mixture_choices = mcfg
        self.mixture_seed = int(context.run_config.get("mixture_seed", 0))

        # ---------------------------------------------------------- #
        # ---------------------- FedIMP config --------------------- #
        self.fedimp_importance = str(context.run_config.get("fedimp_importance", "fisher")).lower()
        self.fedimp_topk_ratio = float(context.run_config.get("fedimp_topk_ratio", 0.10))
        self.fedimp_scale = float(context.run_config.get("fedimp_scale", 5.0))
        self.fedimp_flip_sign = bool(context.run_config.get("fedimp_flip_sign", False))
        self.fedimp_noise_std = float(context.run_config.get("fedimp_noise_std", 0.0))
        self.fedimp_match_norm = bool(context.run_config.get("fedimp_match_norm", False))
        self.fedimp_use_lora_only = bool(context.run_config.get("fedimp_use_lora_only", True))
        self.fedimp_include_classifier = bool(context.run_config.get("fedimp_include_classifier", False))
        self.fedimp_fisher_batches = int(context.run_config.get("fedimp_fisher_batches", 2))
        self.fedimp_fisher_batchsize = int(context.run_config.get("fedimp_fisher_batchsize", 16))

        # Build once so we can expand LoRA-only masks to full length quickly
        self._fedimp_elem_idx = self._fedimp_build_elem_index()

        self._fedimp_idx = _fedimp_target_indices(
            self.net,
            include_classifier=self.fedimp_include_classifier,
            lora_only=self.fedimp_use_lora_only,
        )


        self.fedimp_num_sims       = int(context.run_config.get("fedimp_num_sims", 6))
        self.fedimp_sim_batches    = int(context.run_config.get("fedimp_sim_batches", 2))
        self.fedimp_delta_max      = float(context.run_config.get("fedimp_delta_max", 10.0))
        self.fedimp_tau            = float(context.run_config.get("fedimp_tau", 1e-3))
        self.fedimp_eps            = float(context.run_config.get("fedimp_eps", 0.0))
        self.fedimp_prepoison_jump = float(context.run_config.get("fedimp_prepoison_jump", 0.02))
        self.fedimp_share_across_malicious = bool(context.run_config.get("fedimp_share_across_malicious", True))
        self.fedimp_shared_dir     = str(Path(context.run_config.get("fedimp_shared_dir", "./.fedimp_shared")).resolve())

        self.fedimp_leader_cid     = str(context.run_config.get("fedimp_leader_cid", "0"))

        self._fedimp_l0 = None  # stores the pre-poison baseline loss from round 1
        self._fedimp_last_round_checked = 0

        # Counts needed to compute k = ceil((n_t - m_t)/2) for acceptance test
        self._total_clients = int(context.run_config.get("number_of_clients", 0) or 0)
        self._fraction_fit = float(context.run_config.get("fraction-fit", 1.0))
        self._malicious_pct = float(context.run_config.get("Malicious-Client-Percentage", 0.0))

        # ---------------------------------------------------------- #

    # --------------------------------------------------------------------------------- #

    def fit(self, parameters, config) -> tuple[list, int, dict]:

        self.CurrentRound = int(config["round_number"])
        old_params = parameters


        # Only for MinMax and MinSum attacks ------------------------
        set_params(self.net, parameters)  # Load server model

        broadcast_params = get_params(self.net)    # snapshot of the broadcast/global (wRe)

        global global_benign_grads                 # needed to mutate it from inside the function


        flat_model = flatten_lora(self.net, include_classifier=False, device="cpu").to(torch.float16)

        global_benign_grads.append(flat_model)
        if len(global_benign_grads) > 2:
            global_benign_grads.pop(0)

        # ------------------------------------------------------------

        benign_grads = global_benign_grads

        # Set both model and LoRA parameters
        set_params(self.net, parameters)

        train(self.net, self.trainloader, epochs=1, device=self.device, use_ortho_loss=self.use_ortho_loss, lambda_ortho=self.lambda_ortho)

        before_attack_local = get_params(self.net)  # snapshot after local training, pre-attack


        # -------------------- For Mixture attack -------------------- #

        selected_attack = self.attack_type
        self._mix_mode = None
        if self.is_malicious and self.attack_type == "Mixture":
            selected_attack = self._mixture_pick(self.CurrentRound)
            self._mix_mode = selected_attack

        # ------------------------------------------------------------ #

        # If the client is malicious, simulate model poisoning by modifying the model weights
        if self.is_malicious:
            if selected_attack == "RandScaling":
                self.poison_model()


            elif selected_attack == "Trim":

                # compute mu, sigma (either via coordination A.2 or fallback B)
                mu, sigma, s_hat = self._compute_trim_stats_partial(
                    broadcast_params, before_attack_local, config
                )

                # craft LoRA-only vector x_star using μ±(3–4)σ bands and s_hat
                x_star = self._craft_trimmed_mean_vector(mu, sigma, s_hat)

                # write x_star back into LoRA tensors, keep non-LoRA untouched
                new_params = self._apply_crafted_lora(before_attack_local, x_star)
                set_params(self.net, new_params)

            elif selected_attack == "MinMax":
                # SAME-ROUND forging happens on the server; do nothing here
                pass

            elif selected_attack == "MinSum":
                # SAME-ROUND forging happens on the server; do nothing here
                pass

            elif selected_attack == "PoisonedFL":

                # Server-side forging; nothing to do on client
                pass

            elif selected_attack == "FedIMP" and self.is_malicious:
                # --- 0) Inputs/state ---
                # broadcast_params is the server model snapshot taken earlier
                # after local benign training you already have before_attack_local = get_params(self.net)

                # --- FedIMP safety: clean per-round shared artifacts to avoid shape drift ---
                if getattr(self, "fedimp_share_across_malicious", False):
                    from pathlib import Path
                    shared = Path(getattr(self, "fedimp_shared_dir", "./.fedimp_shared"))
                    tag = f"fedimp_round{self.CurrentRound}_"
                    try:
                        if shared.exists():
                            for f in shared.glob(tag + "*"):
                                f.unlink()  # delete stale round files
                    except Exception as e:
                        print(f"[FedIMP] cleanup warning: {e}")
                # -----------------------------------------------------------------------------

                # 1) Importance (Fisher proxy) -> mask M^t
                imp = self._fedimp_estimate_importance()
                if imp is None:
                    imp = torch.ones_like(_fedimp_flat_from_indices(get_params(self.net), self._fedimp_idx))

                # OPTIONAL BUT PAPER-FAITHFUL: aggregate Fisher importances across compromised clients
                if self.fedimp_share_across_malicious:
                    shared = Path(self.fedimp_shared_dir); shared.mkdir(parents=True, exist_ok=True)

                    # Save my Fisher diag and my weight = |D_aux_i| atomically
                    imp_path = shared / f"fedimp_round{self.CurrentRound}_imp_{self.cid}.pt"
                    w_path  = shared / f"fedimp_round{self.CurrentRound}_w_{self.cid}.txt"

                    tmp_imp = imp_path.with_suffix(imp_path.suffix + ".tmp")
                    torch.save(imp.detach().to("cpu"), tmp_imp)
                    os.replace(tmp_imp, imp_path)

                    aux_size = float(self._fedimp_aux_size())
                    tmp_w = w_path.with_suffix(w_path.suffix + ".tmp")
                    tmp_w.write_text(str(aux_size))
                    os.replace(tmp_w, w_path)


                    fused = shared / f"fedimp_round{self.CurrentRound}_imp_fused.pt"
                    done  = shared / f"fedimp_round{self.CurrentRound}_imp_fused.done"
                    if str(self.cid) == self.fedimp_leader_cid:
                        time.sleep(0.2)  # tiny barrier so peers can start writing

                        # Wait until at least one candidate .pt appears, up to ~10s
                        for _ in range(200):
                            if any(shared.glob(f"fedimp_round{self.CurrentRound}_imp_*.pt")):
                                break
                            time.sleep(0.05)

                        # Load whatever is readable; skip half-written files just in case
                        imps, weights = [], []
                        for p in shared.glob(f"fedimp_round{self.CurrentRound}_imp_*.pt"):
                            try:
                                imps.append(torch.load(p, map_location="cpu"))
                            except (EOFError, pickle.UnpicklingError):
                                continue  # ignore partial/corrupted file in this pass
                            cid_str = p.stem.split("_")[-1]
                            wfile = shared / f"fedimp_round{self.CurrentRound}_w_{cid_str}.txt"
                            try:
                                w = float(wfile.read_text()) if wfile.exists() else 1.0
                            except Exception:
                                w = 1.0
                            weights.append(w)

                        if len(imps) == 0:
                            # Fallback: use this leader's own importance so the round won’t crash
                            I = imp.detach().to("cpu").unsqueeze(0)     # [1, D]
                            W = torch.tensor([1.0]).view(-1, 1)         # [1, 1]
                        else:
                            I = torch.stack(imps, dim=0)                # [m, D]
                            W = torch.tensor(weights).view(-1, 1)       # [m, 1]

                        imp = (I * W).sum(dim=0) / W.sum()

                        # Atomic write + sentinel
                        tmp = fused.with_suffix(fused.suffix + ".tmp")
                        torch.save(imp.cpu(), tmp)
                        os.replace(tmp, fused)
                        done.write_text("ok")
                    else:
                        # wait for fused result (sentinel + guarded load)
                        for _ in range(200):
                            if fused.exists() and done.exists():
                                try:
                                    loaded = torch.load(fused, map_location="cpu")
                                    imp = loaded.to(imp.device)
                                    break
                                except (EOFError, pickle.UnpicklingError):
                                    time.sleep(0.05)
                            time.sleep(0.05)


                # ----------------------------------------------------------------------


                # Top-k binary mask
                new_params = get_params(self.net)
                flat_base  = _fedimp_flat_from_indices(broadcast_params, self._fedimp_idx)
                D = imp.numel()
                k = max(1, int(self.fedimp_topk_ratio * D))
                _, idxs = torch.topk(imp, k=k, largest=True, sorted=False)

                # Force to CPU so it matches μ/σ (which are CPU tensors)
                imp  = imp.detach().to("cpu")
                idxs = idxs.detach().to("cpu")
                mask = torch.zeros_like(imp, dtype=torch.float32)
                mask[idxs] = 1.0         

                # 2) (μ, σ) via K simulated benign updates (tiny steps from broadcast)
                mu, sigma = self._fedimp_simulate_benign_deltas(
                    starting_params=broadcast_params,
                    K=self.fedimp_num_sims,
                    batches_per_sim=self.fedimp_sim_batches,
                )
                if mu is None:
                    # fallback: use current benign local delta stats
                    flat_local = _fedimp_flat_from_indices(new_params, self._fedimp_idx)
                    benign_delta = flat_local - flat_base
                    mu = benign_delta.clone()
                    sigma = torch.ones_like(mu) * (benign_delta.abs().mean() + 1e-6)

                # 3) Optionally share (μ, σ, mask) via a leader so all compromised clients align
                if self.fedimp_share_across_malicious:
                    shared = Path(self.fedimp_shared_dir); shared.mkdir(parents=True, exist_ok=True)
                    mfile = shared / f"fedimp_round{self.CurrentRound}_mu.pt"
                    sfile = shared / f"fedimp_round{self.CurrentRound}_sigma.pt"
                    kfile = shared / f"fedimp_round{self.CurrentRound}_mask.pt"
                    stats_done = shared / f"fedimp_round{self.CurrentRound}_stats.done"
                    if str(self.cid) == self.fedimp_leader_cid:
                        # atomic saves
                        for file, tensor in [(mfile, mu), (sfile, sigma), (kfile, mask)]:
                            tmp = file.with_suffix(file.suffix + ".tmp")
                            torch.save(tensor.detach().to("cpu"), tmp)
                            os.replace(tmp, file)
                        stats_done.write_text("ok")
                    else:
                        # wait for leader to fully publish (incl. sentinel), then guarded loads
                        for _ in range(200):
                            if mfile.exists() and sfile.exists() and kfile.exists() and stats_done.exists():
                                try:
                                    target_dev = mu.device
                                    mu    = torch.load(mfile, map_location="cpu").to(target_dev)
                                    sigma = torch.load(sfile, map_location="cpu").to(target_dev)
                                    mask  = torch.load(kfile, map_location="cpu").to(target_dev)
                                    break
                                except (EOFError, pickle.UnpicklingError):
                                    time.sleep(0.05)
                            time.sleep(0.05)

                # ... existing share-across-malicious block ends with possibly loading mu/sigma/mask ...
                mu, sigma, mask = self._fedimp_align_vectors(mu, sigma, mask)


                # 4) Build benign delta set again (for acceptance check)
                #    (We reuse μ/σ sims by regenerating the same K; cheap but robust)
                benign_deltas = []
                for _ in range(max(1, self.fedimp_num_sims)):
                    batches = self._fedimp_take_batches(self.fedimp_sim_batches)
                    d = self._fedimp_train_k_batches(broadcast_params, batches)
                    benign_deltas.append(d.unsqueeze(0))
                benign_deltas = torch.cat(benign_deltas, dim=0)

                # 5) Pre-poison timing per paper (Algorithm 2)
                delta = None
                if self.CurrentRound == 1:
                    # record l0 on broadcast; send a big-δ this round
                    aux_batches = self._fedimp_take_batches(max(1, self.fedimp_sim_batches))
                    self._fedimp_l0 = self._fedimp_aux_loss(broadcast_params, aux_batches)
                    self._fedimp_last_round_checked = 1
                    delta = self.fedimp_delta_max

                elif (self.CurrentRound == 2 
                    and self._fedimp_last_round_checked == 1 
                    and self._fedimp_l0 is not None):
                    # check l1 on *new broadcast* (post-aggregation)
                    aux_batches = self._fedimp_take_batches(max(1, self.fedimp_sim_batches))
                    l1 = self._fedimp_aux_loss(broadcast_params, aux_batches)
                    if (l1 - self._fedimp_l0) > float(self.fedimp_prepoison_jump):
                        delta = self.fedimp_delta_max
                    # clear flags
                    self._fedimp_l0 = None
                    self._fedimp_last_round_checked = 2


                # 6) if not decided by pre-poison, do binary search:
                if delta is None:
                    delta = self._fedimp_binary_search_delta(mu, sigma, mask, benign_deltas)

                # 7) Final malicious delta

                # --- FedIMP safety: shape + device guard ---
                def _shape(t):
                    return None if t is None else tuple(t.shape)
                def _dev(t):
                    try:
                        return str(t.device)
                    except Exception:
                        return None

                shapes_ok = (
                    (mu is not None) and (sigma is not None) and (mask is not None)
                    and (mu.dim() == 1) and (sigma.dim() == 1) and (mask.dim() == 1)
                    and (mu.numel() == sigma.numel() == mask.numel())
                )
                devices_ok = _dev(mu) == _dev(sigma) == _dev(mask)

                if not shapes_ok:
                    print(f"[FedIMP] forge: shape mismatch, disabling this round. "
                        f"mu={_shape(mu)} sigma={_shape(sigma)} mask={_shape(mask)}")
                    basis = mu if (mu is not None and mu.dim() == 1) else \
                            sigma if (sigma is not None and sigma.dim() == 1) else \
                            mask if (mask is not None and mask.dim() == 1) else None
                    mal = torch.zeros_like(basis) if basis is not None else None
                elif not devices_ok:
                    print(f"[FedIMP] forge: device mismatch, disabling this round. "
                        f"mu={_dev(mu)} sigma={_dev(sigma)} mask={_dev(mask)}")
                    mal = torch.zeros_like(mu)
                else:
                    mal = mu - delta * (mask * sigma)

                # 8) Optional norm-match to benign local delta for stealth
                if self.fedimp_match_norm:
                    flat_local = _fedimp_flat_from_indices(get_params(self.net), self._fedimp_idx)
                    benign_delta = flat_local - flat_base
                    bnorm = benign_delta.norm(p=2) + 1e-12
                    fnorm = mal.norm(p=2) + 1e-12
                    mal = mal * (bnorm / fnorm)

                # 9) (Optional) share identical Δw across all malicious clients
                if self.fedimp_share_across_malicious:
                    shared = Path(self.fedimp_shared_dir); shared.mkdir(parents=True, exist_ok=True)
                    vfile = shared / f"fedimp_round{self.CurrentRound}_vector.pt"
                    vdone = shared / f"fedimp_round{self.CurrentRound}_vector.done"
                    if vfile.exists() and vdone.exists():
                        mal = torch.load(vfile, map_location="cpu").to(mal.device)
                    else:
                        # One writer wins; others will load after sentinel appears
                        try:
                            tmp = vfile.with_suffix(vfile.suffix + ".tmp")
                            torch.save(mal.detach().to("cpu"), tmp)
                            os.replace(tmp, vfile)
                            vdone.write_text("ok")
                        except Exception:
                            pass
                        for _ in range(200):
                            if vfile.exists() and vdone.exists():
                                try:
                                    mal = torch.load(vfile, map_location="cpu").to(mal.device)
                                    break
                                except (EOFError, pickle.UnpicklingError):
                                    time.sleep(0.05)
                            time.sleep(0.05)


                # 10) Commit forged update on targeted indices
                forged_params = _fedimp_add_into_indices(get_params(self.net), self._fedimp_idx, mal)
                set_params(self.net, forged_params)


        updated = get_params(self.net)
        metrics = {
            "I_am_malicious": int(self.is_malicious),  # 1 for malicious, 0 for benign
            "client_cid": str(self.cid),               # optional: handy for debug
        }

        if self.is_malicious and self.attack_type == "Mixture":
            metrics["mix_mode"] = self._mix_mode or ""

        return updated, len(self.trainloader), metrics


    # --------------------------------------------------------------------------------- #
    # Helper fuctions for Trim attack

    def _flat_lora(self, params_list):
        # params_list can be list[np.ndarray] or list[torch.Tensor]
        flat = []
        for (name, _), arr in zip(self.net.state_dict().items(), params_list):
            if ("lora_A" in name or "lora_B" in name):
                t = torch.as_tensor(arr, device=self.device)
                flat.append(t.view(-1))
        return torch.cat(flat)

    def _apply_crafted_lora(self, base_params, flat):
        # base_params: list[np.ndarray] or list[torch.Tensor]
        out, off = [], 0
        for (name, _), arr in zip(self.net.state_dict().items(), base_params):
            if ("lora_A" in name or "lora_B" in name):
                base_t   = torch.as_tensor(arr)
                cnt      = base_t.numel()
                patched  = flat[off:off+cnt].view(base_t.shape).to(base_t.dtype)
                out.append(patched)
                off += cnt
            else:
                out.append(torch.as_tensor(arr))
        return out

    def _compute_trim_stats_partial(self, wRe_params, before_params, config):
        # Single-client fallback (paper-style partial knowledge):
        x   = self._flat_lora(before_params)        # after-training local (pre-attack)
        mu  = x                                     # use this as the per-coord mean
        sigma = 0.1 * (x.abs() + 1e-12)             # non-zero proxy std (avoid collapse)
        wRe = self._flat_lora(wRe_params)           # broadcast/global (previous round)
        s_hat = torch.where(mu > wRe, torch.ones_like(mu), -torch.ones_like(mu))
        return mu, sigma, s_hat

    def _craft_trimmed_mean_vector(self, mu, sigma, s_hat):
        rand = torch.rand_like(mu)
        upper = (mu + 3*sigma) + rand * (1*sigma)   # μ+3σ..μ+4σ  (push down)
        lower = (mu - 4*sigma) + rand * (1*sigma)   # μ-4σ..μ-3σ  (push up)
        return torch.where(s_hat < 0, upper, lower)

    # --------------------------------------------------------------------------------- #

    def evaluate(self, parameters, config) -> tuple[float, int, dict]:
        """Evaluate model parameters."""
        set_params(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader, device=self.device)
        # return float(loss), len(self.testloader), {"accuracy": float(accuracy)}
        return float(loss), len(self.testloader), {"loss": float(loss), "accuracy": float(accuracy)}

    # --------------------------------------------------------------------------------- #
    # -------------------------- Helper functions for FedIMP -------------------------- #

    def _fedimp_estimate_importance(self):
        """Estimate diagonal Fisher on targeted coords using a small probe of local data."""
        self.net.train()
        device = next(self.net.parameters()).device

        # Build a tiny probe loader from self.trainloader
        it = iter(self.trainloader)
        batches = []
        for _ in range(self.fedimp_fisher_batches):
            try:
                batches.append(next(it))
            except StopIteration:
                break

        # --- 3A: record true |D_aux| used for Fisher this round (for Eq. 9 weighting) ---
        total_aux = 0
        for batch in batches:
            # batch may be a dict (common), tuple, or tensor; infer batch size robustly
            if isinstance(batch, dict):
                # try first tensor-like value
                for v in batch.values():
                    if hasattr(v, "size"):
                        total_aux += int(v.size(0))
                        break
            elif isinstance(batch, (list, tuple)):
                v = batch[0]
                total_aux += int(v.size(0)) if hasattr(v, "size") else 0
            else:
                total_aux += int(batch.size(0)) if hasattr(batch, "size") else 0
        self.fedimp_aux_len = total_aux  
        # -------------------------------------------------------------------------------

        if not batches:
            return None  # no data

        # Zero accumulators and grads
        for p in self.net.parameters():
            if p.grad is not None:
                p.grad.zero_()

        # Accumulate gradient^2 over probe
        fisher_acc = None
        for batch in batches:
            batch = {k: v.to(device) for k, v in batch.items()}
            self.net.zero_grad(set_to_none=True)
            out = self.net(**batch)
            loss = out.loss
            loss.backward()

            # Collect grads on targeted params
            grads = []
            for i, (_n, p) in enumerate(self.net.named_parameters()):
                if i in self._fedimp_idx and p.grad is not None:
                    grads.append(p.grad.detach().reshape(-1).float())
            if not grads:
                continue
            g_flat = torch.cat(grads)
            g2 = g_flat * g_flat
            fisher_acc = g2 if fisher_acc is None else fisher_acc + g2

        if fisher_acc is None:
            return None

        fisher_acc = fisher_acc / max(1, len(batches))
        # Normalize to [0,1] for stability
        fisher_acc = (fisher_acc - fisher_acc.min()) / (fisher_acc.max() - fisher_acc.min() + 1e-12)
        return fisher_acc  # 1-D tensor aligned with _fedimp_idx flattening


    def _fedimp_take_batches(self, n: int):
        """Take first n batches from trainloader as the small auxiliary/probe set."""
        it = iter(self.trainloader)
        batches = []
        for _ in range(n):
            try:
                batches.append(next(it))
            except StopIteration:
                break
        return batches

    # --- FedIMP index helpers (ADD) ---------------------------------------------
    def _fedimp_build_elem_index(self) -> torch.Tensor:
        """
        Return flat element indices (w.r.t. parameters_to_vector(self.net.parameters()))
        for the parameters we want FedIMP to act on:
        - If self.fedimp_use_lora_only: only LoRA params
        - Respect self.fedimp_include_classifier (usually False per your setup)
        """
        idx = []
        offset = 0
        for name, p in self.net.named_parameters():
            n = p.numel()

            # decide if this tensor is selected
            use = True
            if self.fedimp_use_lora_only:
                # keep only LoRA tensors
                use = ("lora_" in name) or ("loraA" in name) or ("loraB" in name)
            if not self.fedimp_include_classifier and name.startswith("classifier."):
                use = False

            if use:
                # add the element-wise flat indices for this parameter tensor
                idx.extend(range(offset, offset + n))

            offset += n

        if len(idx) == 0:
            # fallback to avoid empty masks
            return torch.zeros(0, dtype=torch.long)
        return torch.tensor(idx, dtype=torch.long)


    def _fedimp_expand_mask_to_full(self, mask_small: torch.Tensor, *, full_len: int, device) -> torch.Tensor:
        """
        Expand a LoRA-only (or otherwise sliced) mask to a full-length mask using the
        precomputed element index map.
        """
        if not hasattr(self, "_fedimp_elem_idx") or self._fedimp_elem_idx is None:
            self._fedimp_elem_idx = self._fedimp_build_elem_index()

        elem_idx = self._fedimp_elem_idx.to(device)
        out = torch.zeros(full_len, device=device, dtype=mask_small.dtype)

        # if lengths mismatch (e.g., due to tiny naming diffs), fill as much as possible safely
        k = min(mask_small.numel(), elem_idx.numel())
        if k > 0:
            out[elem_idx[:k]] = mask_small[:k]
        return out
    # -------------------------------------------------------------------

    def _fedimp_align_vectors(self, mu: torch.Tensor, sigma: torch.Tensor, mask: torch.Tensor):
        """Align (μ,σ,mask) to the same dimensionality, preferring LoRA-only when configured."""
        if (mu is None) or (sigma is None) or (mask is None):
            return mu, sigma, mask
        D_mu, D_ms = int(mu.numel()), int(mask.numel())

        if D_mu == D_ms:
            return mu, sigma, mask

        dev = mu.device
        # If μ/σ are full-model while mask is LoRA-only -> slice μ/σ down to LoRA elements
        if D_mu > D_ms and getattr(self, "fedimp_use_lora_only", False):
            if not hasattr(self, "_fedimp_elem_idx") or self._fedimp_elem_idx is None:
                self._fedimp_elem_idx = self._fedimp_build_elem_index()
            idx = self._fedimp_elem_idx.to(dev)
            # protect against length drift
            k = min(idx.numel(), D_mu)
            idx = idx[:k]
            mu    = mu[:D_mu][idx]
            sigma = sigma[:D_mu][idx]
            return mu, sigma, mask

        # If mask is full-model while μ/σ are LoRA-only -> expand mask to full
        if D_ms > D_mu:
            mask = self._fedimp_expand_mask_to_full(mask, full_len=D_mu, device=dev)
            return mu, sigma, mask

        return mu, sigma, mask


    def _fedimp_train_k_batches(self, starting_params, batches: list):
        """Do a tiny benign step from broadcast to mimic a benign client."""
        device = next(self.net.parameters()).device
        # reset to broadcast
        set_params(self.net, starting_params)
        self.net.train()
        opt = AdamW((p for p in self.net.parameters() if p.requires_grad), lr=5e-5)
        for batch in batches:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = self.net(**batch)
            loss = out.loss
            # optional orthogonality (same as task.train)
            if self.use_ortho_loss:
                orthogonality_loss = 0.0
                for name, param in self.net.named_parameters():
                    if ("lora_A" in name) or ("lora_B" in name):
                        A = param
                        if A.shape[0] < A.shape[1]:
                            prod = A @ A.T
                            identity = torch.eye(prod.shape[0], device=A.device)
                        else:
                            prod = A.T @ A
                            identity = torch.eye(prod.shape[0], device=A.device)
                        orthogonality_loss += torch.norm(prod - identity, p="fro")
                loss = loss + self.lambda_ortho * orthogonality_loss
            loss.backward()
            opt.step()
            opt.zero_grad()
        # delta on targeted coords
        new_params = get_params(self.net)
        flat_local = _fedimp_flat_from_indices(new_params, self._fedimp_idx)
        flat_base  = _fedimp_flat_from_indices(starting_params, self._fedimp_idx)
        return flat_local - flat_base

    def _fedimp_simulate_benign_deltas(self, starting_params, K: int, batches_per_sim: int):
        """Simulate K benign updates (tiny steps) to estimate μ and σ."""
        deltas = []
        for s in range(K):
            batches = self._fedimp_take_batches(batches_per_sim)
            if not batches:
                break
            d = self._fedimp_train_k_batches(starting_params, batches)
            deltas.append(d.unsqueeze(0))
        if not deltas:
            return None, None
        D = torch.cat(deltas, dim=0)  # [K, D]
        mu = D.mean(dim=0)
        sigma = D.std(dim=0, unbiased=False) + 1e-12
        return mu, sigma

    def _fedimp_aux_loss(self, params, batches: list):
        """Average loss on a small auxiliary set."""
        device = next(self.net.parameters()).device
        set_params(self.net, params)
        self.net.eval()
        losses = []
        with torch.no_grad():
            for batch in batches:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = self.net(**batch)
                losses.append(float(out.loss.item()))
        return float(np.mean(losses)) if losses else 0.0

    def _fedimp_accept(self, mal_vec: torch.Tensor, benign: torch.Tensor) -> bool:
        """
        Paper Eq.(11): compare SUM of squared distances to the k nearest benign
        vs the SUM of benign-to-benign k-NN squared distances (k = ceil((n_t - m_t)/2)).
        """
        if benign is None or benign.ndim != 2 or benign.size(0) < 2:
            return True  # nothing to compare against

        K_ben = benign.size(0)
        k = self._fedimp_knn_k(K_ben)

        # LHS: sum of k-NN squared distances from mal_vec to benign set
        dists_mal = ((benign - mal_vec) ** 2).sum(dim=1)
        lhs = dists_mal.topk(k=k, largest=False).values.sum().item()

        # RHS: sum over benign points of their own k-NN squared distances
        knn_sums = []
        for i in range(K_ben):
            b = benign[i]
            d = ((benign - b) ** 2).sum(dim=1)
            d_sorted, _ = torch.sort(d)
            # if self-distance appears as zero, skip it
            if d_sorted[0].item() == 0.0 and K_ben > k:
                knn = d_sorted[1:1+k]
            else:
                knn = d_sorted[0:k]
            knn_sums.append(knn.sum())
        rhs = torch.stack(knn_sums).sum().item()

        eps = float(getattr(self, "fedimp_eps", 0.0))
        return lhs <= (rhs + eps)



    def _fedimp_binary_search_delta(self, mu: torch.Tensor, sigma: torch.Tensor,
                                mask: torch.Tensor, benign: torch.Tensor) -> float:
        # --- FedIMP safety: device guard (skip search if tensors are on different devices)
        def _dev(t):
            try:
                return str(t.device)
            except Exception:
                return None

        same_device = (
            (mu is not None) and (sigma is not None) and (mask is not None)
            and (_dev(mu) == _dev(sigma) == _dev(mask))
        )

        if not same_device:
            print(f"[FedIMP] binary_search: device mismatch, skipping search. "
                f"mu={_dev(mu)} sigma={_dev(sigma)} mask={_dev(mask)}")
            return 0.0  # benign fallback for this round
        # --------------------------------------
        
        # Try to auto-align once
        if hasattr(self, "_fedimp_align_vectors"):
            mu, sigma, mask = self._fedimp_align_vectors(mu, sigma, mask)

        low, high = 0.0, float(self.fedimp_delta_max)
        mid = (low + high) / 2.0
        tau = float(self.fedimp_tau)

        # --- FedIMP safety: shapes must match, otherwise skip search this round ---
        if (mu is None) or (sigma is None) or (mask is None) \
        or (mu.dim() != 1) or (sigma.dim() != 1) or (mask.dim() != 1) \
        or (mu.numel() != sigma.numel()) or (mu.numel() != mask.numel()):
            print(f"[FedIMP] binary_search: shape mismatch, skipping search. "
                f"mu={None if mu is None else tuple(mu.shape)} "
                f"sigma={None if sigma is None else tuple(sigma.shape)} "
                f"mask={None if mask is None else tuple(mask.shape)}")
            return 0.0  # behave benignly (no boost) rather than crash

        while (high - low) > tau:
            mal = mu - mid * (mask * sigma)
            if self._fedimp_accept(mal, benign):
                low = mid
            else:
                high = mid
            mid = (low + high) / 2.0

        return float(mid)


    def _fedimp_knn_k(self, K_ben: int) -> int:
        """k = ceil((n_t - m_t)/2), capped by available benign sims."""
        n_round = max(1, int(math.ceil(self._fraction_fit * self._total_clients)))
        m_round = max(0, int(math.ceil(self._malicious_pct * n_round)))
        k = int(math.ceil(max(1, n_round - m_round) / 2))
        return max(1, min(k, max(1, K_ben - 1)))


    def _fedimp_aux_size(self) -> int:
        """True |D_aux| used for Fisher this round."""
        # return the number of aux samples used in Fisher estimation
        # If you already cache aux dataset/loader, return its true length.
        return int(getattr(self, "fedimp_aux_len", 0) or 0)

    # --- Helper functions for mixture attack ----------------------------------------- #

    def _mixture_pick(self, round_no: int) -> str:
        s = f"{self.cid}|{round_no}|{self.mixture_seed}".encode("utf-8")
        h = int.from_bytes(hashlib.blake2b(s, digest_size=8).digest(), "little", signed=False)
        return self.mixture_choices[h % len(self.mixture_choices)] if self.mixture_choices else ""

    # --------------------------------------------------------------------------------- #

    def poison_model(self) -> None:
        """Apply various model poisoning attacks."""
        with torch.no_grad():
            for name, param in self.net.named_parameters():
                # Only apply poisoning to certain layers for more targeted attack
                # if any(layer_type in name for layer_type in ['query', 'value']):    # 'query', 'value', 'dense'
                if any(key in name for key in ["lora_A", "lora_B"]):

                    seed = 42 + int(self.cid) * 1000 + self.CurrentRound  # Unique seed per client per round
                    torch.manual_seed(seed)
                    
                    # Strategy 2: Scale the weights
                    param.mul_(1 + self.poison_intensity)

                    eps = 0.5
                    param.mul_(1 + torch.empty_like(param).uniform_(-eps, eps))

    # --------------------------------------------------------------------------------- #

    def trim_attack(self, old_params, new_params):
        """Trim top-k largest weight changes in LoRA layers only."""
        trimmed = []
        param_names = list(self.net.state_dict().keys())

        for name, old, new in zip(param_names, old_params, new_params):
            # if any(lora_key in name for lora_key in ["lora", "query", "value"]):
            if any(key in name for key in ["lora_A", "lora_B"]):
                delta = new - old
                flat_delta = delta.flatten()
                abs_sorted_indices = torch.tensor(abs(flat_delta)).argsort()

                k = int(len(flat_delta) * self.trim_ratio)
                if k > 0:
                    flat_delta[abs_sorted_indices[-k:]] = 0  # Trim top-k by magnitude

                trimmed_delta = flat_delta.reshape(delta.shape)
                trimmed.append(old + trimmed_delta)
            else:
                # Leave non-LoRA layers untouched
                trimmed.append(new)

        return trimmed

    # --------------------------------------------------------------------------------- #

    def min_max_attack(self, benign_grads, base_update, gamma_init=100.0, tau=1e-5):

        # Ensure same device
        # benign_grads = [g.to(self.device) for g in benign_grads]
        benign_grads = [g.to(self.device, dtype=torch.float32) for g in benign_grads]

        perturb = compute_perturbation_vector(benign_grads, mode="std")
        # perturb = compute_perturbation_vector(benign_grads, mode="mean")    # Option 2

        gamma = gamma_init
        step = gamma

        base_net = copy.deepcopy(self.net)
        perturb_net = copy.deepcopy(self.net)
        unflatten_lora(base_net, base_update)
        unflatten_lora(perturb_net, perturb)

        base_dict = base_net.state_dict()
        perturb_dict = perturb_net.state_dict()

        benign_max = max(torch.norm(g1 - g2) for g1 in benign_grads for g2 in benign_grads)

        while step > tau:
            poisoned_state = {}
            for name in base_dict:
                if any(k in name for k in ["lora_A", "lora_B"]):
                    poisoned_state[name] = base_dict[name] + gamma * perturb_dict[name]
                else:
                    poisoned_state[name] = base_dict[name]  # Leave untouched

            poisoned_tensor = flatten_state_dict_lora(poisoned_state, include_classifier=False, device=self.device)

            max_dist = max(torch.norm(poisoned_tensor - g) for g in benign_grads)
            
            gamma = gamma + step if max_dist <= benign_max * 1 else gamma - step
            step /= 2


        return poisoned_state  # a dictionary of {parameter_name: tensor}   

    # --------------------------------------------------------------------------------- #

    def min_sum_attack(self, benign_grads, base_update, gamma_init=100.0, tau=1e-5):

        # Ensure benign gradients are on the same device
        benign_grads = [g.to(self.device, dtype=torch.float32) for g in benign_grads]

        # Compute perturbation direction
        perturb = compute_perturbation_vector(benign_grads, mode="std")

        # Initialize step sizes
        gamma = gamma_init
        step = gamma 

        # Unflatten flat vectors into model-shaped parameter sets
        import copy
        base_net = copy.deepcopy(self.net)
        perturb_net = copy.deepcopy(self.net)
        # unflatten_model(base_net, base_update)
        # unflatten_model(perturb_net, perturb)
        unflatten_lora(base_net, base_update)
        unflatten_lora(perturb_net, perturb)

        base_dict = base_net.state_dict()
        perturb_dict = perturb_net.state_dict()

        def sum_dist(x):
            return sum(torch.norm(x - g) ** 2 for g in benign_grads)

        # Flatten benign grads to match poisoned structure
        benign_sum = max(sum_dist(g) for g in benign_grads)

        while step > tau:
            # Apply perturbation to only LoRA parameters
            poisoned_state = {}
            for name in base_dict:
                if any(k in name for k in ["lora_A", "lora_B"]):
                    poisoned_state[name] = base_dict[name] + gamma * perturb_dict[name]
                else:
                    poisoned_state[name] = base_dict[name]  # keep original

            # poisoned_tensor = torch.cat([v.view(-1).to(self.device) for v in poisoned_state.values()])
            poisoned_tensor = flatten_state_dict_lora(poisoned_state, include_classifier=False, device=self.device)

            dist_sum = sum_dist(poisoned_tensor)

            # Adjust gamma based on distance
            gamma = gamma + step if dist_sum <= benign_sum * 1 else gamma - step
            step /= 2


        return poisoned_tensor

    # --------------------------------------------------------------------------------- #


# ====================================================================================================================================================== #
# ================================================================ Client Function ===================================================================== #

def client_fn(context: Context) -> Client:
    """Construct a Client that will be run in a ClientApp."""
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    partitioner_type =  context.run_config["partitioner_type"]
    partitioner_parameter = context.run_config["partitioner_parameter"]
    dataset_name = context.run_config["dataset_name"]
    use_ortho_loss = context.run_config["use_ortho_loss"]

    # Read the run config to get settings to configure the Client
    model_name = context.run_config["model-name"]

    number_of_samples = int(context.run_config.get("number_of_samples", 0))
    trainloader, testloader = load_data(partition_id, num_partitions, model_name, partitioner_type, dataset_name, partitioner_parameter, number_of_samples)

    # ---------------------------------------------------------------------
    # ---------------------- Select Malacious Clients ---------------------

    random.seed(42)      # Set seed for reproducibility



    # number_of_active_clients = math.ceil(context.run_config["number_of_clients"] * fraction_fit)
    threshold = context.run_config["Malicious-Client-Percentage"]
    num_malicious = math.ceil(threshold * context.run_config["number_of_clients"] )

    # Create and update dictionary
    client_dict = {}

    # Randomly choose `num_malicious` clients to be malicious
    client_ids = [str(i) for i in range(context.run_config["number_of_clients"])]
    malicious_ids = set(random.sample(client_ids, num_malicious))

    print(f"[ClientApp] Malicious client IDs: {sorted(malicious_ids)}")

    # Create the dictionary with exactly `num_malicious` True values
    client_dict = {cid: (cid in malicious_ids) for cid in client_ids}

    # context.run_config["client_dict"] = client_dict

    # print("Malacious clients: ", client_dict)

    # ---------------------------------------------------------------------
    # ---------------- Check if the client is malicious -------------------

    cid = str(context.node_config["partition-id"]) 

    is_malicious = client_dict[cid]   
    # is_malicious = False

    # ----------------------------------------------------------------------

    return ClientClass(model_name, trainloader, testloader, is_malicious=is_malicious, cid=cid, use_ortho_loss=use_ortho_loss, context=context).to_client()

# ====================================================================================================================================================== #

app = ClientApp(client_fn=client_fn)