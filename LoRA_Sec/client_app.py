"""LoRA-Sec: A Flower / Hugging Face app."""
# ====================================================================================================================================================== #
# import warnings

import os
# Set process-level reproducibility controls before importing PyTorch/Flower.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")

import torch
import random
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context
# from transformers import logging
from LoRA_Sec.task import (
    train,
    test,
    test_qa,
    load_data,
    set_params,
    get_params,
    get_model,
    set_global_determinism,
    deterministic_client_seed,
)

import math
from typing import List, Dict
import copy

# --- Mixture attack ----

import hashlib

# ------- FedIMP -------- #
import time
import pickle
from pathlib import Path
import numpy as np
from torch.optim import AdamW
# ----------------------- #

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
def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def lora_ortho_score_from_dict(named_tensors: Dict[str, torch.Tensor]) -> float:
    """Same metric as the server's ORLA `_compute_ortho_score`, computed locally."""
    score = 0.0
    for _name, A in named_tensors.items():
        A32 = A.detach().to(torch.float32)
        if A32.shape[0] < A32.shape[1]:
            prod = A32 @ A32.T
            ident = torch.eye(prod.shape[0], device=A32.device)
        else:
            prod = A32.T @ A32
            ident = torch.eye(prod.shape[0], device=A32.device)
        score += torch.norm(prod - ident, p="fro").item()
    return float(score)


def soft_orthogonal_projection(M: torch.Tensor, alpha: float) -> torch.Tensor:
    """Per-singular-value soft orthogonalization of M.

    For each singular value s_i of M, solves:
        sigma_i* = argmin  alpha*(sigma_i^2 - 1)^2 + (1-alpha)*(sigma_i - s_i)^2

    The first-order condition is:
        g(sigma) = 4*alpha*sigma*(sigma^2-1) + 2*(1-alpha)*(sigma-s) = 0

    The unique positive root lies in [min(s,1), max(s,1)] because:
        g(s) and g(1) have opposite signs (g(s)<0 iff s<1, g(1)>0 iff s<1).
    Bisection within this bracket is used instead of Newton's method because
    Newton can jump to a negative value when s is small (e.g. s=0.1, alpha=0.7),
    causing sig to collapse to the 1e-7 floor and remain there -- the opposite
    of the intended orthogonalization.  Bisection is unconditionally safe and
    converges to full float32 precision in 60 iterations.

    Returns U diag(sigma*) V^T, which:
      - has a strictly lower ORLA score than M (all sigma_i moved toward 1)
      - is closer to M in Frobenius norm than the blend gives at the same ORLA
        level (blend applies a single beta uniformly; SOP applies the optimal
        per-singular-value correction)
      - stays in M's own singular-vector frame (no rotation introduced)

    alpha=0 -> no change (M returned as-is).
    alpha=1 -> full Procrustes projection (singular values all set to 1).

    Edge case s<=0: treated as s=0; bisection solves on [0,1], converging to 1
    at alpha=1, consistent with the alpha=1 guarantee.
    """
    alpha = float(min(1.0, max(0.0, alpha)))
    if alpha == 0.0:
        return M

    orig_dtype = M.dtype
    M32 = M.detach().to(torch.float32)
    try:
        U, S, Vh = torch.linalg.svd(M32, full_matrices=False)
    except RuntimeError:
        return M  # degenerate; leave unchanged

    def _bisect(s: float) -> float:
        """Bisect g(sigma)=0 on the bracket [min(s,1), max(s,1)]."""
        lo, hi = min(s, 1.0), max(s, 1.0)
        if lo == hi:          # s == 1 exactly: already optimal
            return s
        for _ in range(60):  # 60 halvings -> |error| < 2^{-60} * |s-1| ~ machine eps
            mid = (lo + hi) * 0.5
            g   = 4.0 * alpha * mid * (mid * mid - 1.0) + 2.0 * (1.0 - alpha) * (mid - s)
            if g < 0.0:
                lo = mid
            else:
                hi = mid
        return (lo + hi) * 0.5

    sigma_star = [_bisect(float(s_val)) for s_val in S.tolist()]

    S_new = torch.tensor(sigma_star, dtype=torch.float32, device=M.device)
    M_new = U @ torch.diag(S_new) @ Vh
    return M_new.to(orig_dtype)


def calibrate_alpha_for_target_score(
    named_mal_lora: Dict[str, torch.Tensor],
    target_score: float,
    coarse_steps: int = 20,
    refine_iters: int = 15,
) -> float:
    """Binary search on alpha in [0,1] to find the minimal SOP strength whose
    resulting ORLA score is closest to `target_score`.

    The ORLA score is monotone non-increasing in alpha (higher alpha -> more
    orthogonal -> lower score), so a bisection after a coarse grid pass is exact.
    """
    def score_at(alpha: float) -> float:
        projected = {
            name: soft_orthogonal_projection(M, alpha)
            for name, M in named_mal_lora.items()
        }
        return lora_ortho_score_from_dict(projected)

    # Coarse grid to bracket the target
    alphas = np.linspace(0.0, 1.0, coarse_steps)
    scores = [score_at(float(a)) for a in alphas]
    diffs  = [abs(s - target_score) for s in scores]
    best_idx   = int(np.argmin(diffs))
    best_alpha = float(alphas[best_idx])
    best_diff  = diffs[best_idx]

    # Bisection refinement in the bracket around the best coarse point
    lo = float(alphas[max(0, best_idx - 1)])
    hi = float(alphas[min(len(alphas) - 1, best_idx + 1)])
    for _ in range(refine_iters):
        mid = (lo + hi) / 2.0
        d_mid = abs(score_at(mid) - target_score)
        if d_mid < best_diff:
            best_alpha, best_diff = mid, d_mid
        # Score is monotone decreasing in alpha: higher alpha -> lower score.
        # If score(mid) > target we need more alpha (go right), else go left.
        if score_at(mid) > target_score:
            lo = mid
        else:
            hi = mid

    return float(min(1.0, max(0.0, best_alpha)))


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
    def __init__(self, model_name, trainloader, testloader, cid, context , use_ortho_loss, lr, is_malicious=False) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainloader = trainloader
        self.testloader = testloader
        self.base_seed = int(context.run_config.get("base_seed", 42))
        set_global_determinism(self.base_seed + int(cid))
        self.problem_type = str(context.run_config.get("problem_type", "seq_cls")).lower()
        self.net = get_model(model_name, num_labels = int(context.run_config.get("num_labels")), problem_type=self.problem_type)
        self.net.to(self.device)
        self.is_malicious = is_malicious
        self.poison_intensity: float = 0.35
        self.cid = cid
        self.CurrentRound = 1
        self.attack_type = context.run_config["attack_type"]   
        self.use_ortho_loss = use_ortho_loss
        self.lambda_ortho = context.run_config["lambda_ortho"]
        self.lr = lr

        # --- Mixture config (used when attack_type == "Mixture") ---
        _mix_default = ["RandScaling", "Trim", "MinMax", "MinSum", "PoisonedFL", "FedIMP"]
        mcfg = context.run_config.get("mixture_choices", _mix_default)
        if isinstance(mcfg, str):
            if mcfg.strip() in {"*", "ALL", "all", "Auto", "auto"}:
                mcfg = _mix_default
            else:
                mcfg = [s.strip() for s in mcfg.split(",") if s.strip()]
        self.mixture_choices = mcfg
        self.mixture_seed = int(context.run_config.get("mixture_seed", self.base_seed))

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

        # ---------------------------------------------------------- #
        # The remaining SOP attacks still require these two local-score settings.
        self.adaptive_ortho_target_offset = float(context.run_config.get("adaptive_ortho_target_offset", 0.0))
        self.adaptive_ortho_search_iters = int(context.run_config.get("adaptive_ortho_search_iters", 10))

        # Counts needed to compute k = ceil((n_t - m_t)/2) for acceptance test
        self._total_clients = int(context.run_config.get("number_of_clients", 0) or 0)
        self._fraction_fit = float(context.run_config.get("fraction-fit", 1.0))
        self._malicious_pct = float(context.run_config.get("Malicious-Client-Percentage", 0.0))

        # ---------------------------------------------------------- #

    # --------------------------------------------------------------------------------- #

    def get_properties(self, config) -> dict:
        """Expose the stable logical/dataset partition id to the server.

        Flower 1.12's server-side ClientProxy.cid is a random per-run node id.
        The stable identity of this simulated client is the partition id from
        ``context.node_config["partition-id"]``, stored in ``self.cid``.
        """
        return {
            "logical_client_id": str(self.cid),
            "partition_id": int(self.cid),
        }

    # --------------------------------------------------------------------------------- #

    def fit(self, parameters, config) -> tuple[list, int, dict]:

        self.CurrentRound = int(config["round_number"])

        expected_logical_cid = config.get("logical_client_id", None)
        if expected_logical_cid is not None and str(expected_logical_cid) != str(self.cid):
            raise RuntimeError(
                f"Deterministic client mapping mismatch: server expected logical client "
                f"{expected_logical_cid}, but this ClientApp owns partition {self.cid}."
            )

        # The server assigns malicious status independently every round.
        # First, the malicious population is selected from ALL logical clients
        # using Random(round_number). Participating clients are selected
        # independently using Random(base_seed + round_number). Therefore,
        # only their intersection is malicious in the current round.
        # Do not keep a static malicious flag.
        try:
            self.is_malicious = bool(int(config.get("is_malicious", int(self.is_malicious))))
        except Exception:
            self.is_malicious = bool(config.get("is_malicious", self.is_malicious))
        self._round_malicious_count = int(config.get("round_malicious_count", 0) or 0)
        self._round_malicious_rank = int(config.get("malicious_rank", -1))
        self._round_attack_group_count = int(config.get("attack_group_count", self._round_malicious_count) or 0)
        self._round_attack_group_rank = int(config.get("attack_group_rank", self._round_malicious_rank))
        self._round_is_coord_leader = bool(self.is_malicious and self._round_attack_group_rank == 0)

        # Re-seed every client fit from the round and client id. This makes
        # dropout, attack randomness, and DataLoader shuffling independent of
        # actor reuse/scheduling in previous rounds.
        client_seed = deterministic_client_seed(self.base_seed, self.CurrentRound, int(self.cid))
        set_global_determinism(client_seed)
        loader_gen = getattr(self.trainloader, "generator", None)
        if loader_gen is not None:
            loader_gen.manual_seed(client_seed)

        old_params = parameters


        # Only for MinMax and MinSum attacks ------------------------
        set_params(self.net, parameters)  # Load server model

        # broadcast_params = get_params(self.net)    # snapshot of the broadcast/global (wRe)
        broadcast_params = old_params  # or broadcast_params = parameters

        # Do not maintain process-global per-client gradient history here. Flower's
        # Ray workers are reused for different logical clients, so such state depends
        # on worker scheduling. MinMax/MinSum forging is server-side in this code and
        # this history was dead state for the active attack paths.

        # Set both model and LoRA parameters
        # set_params(self.net, parameters)

        train(self.net, self.trainloader, self.lr , epochs=1, device=self.device, use_ortho_loss=self.use_ortho_loss, lambda_ortho=self.lambda_ortho)
        # train(self.net, self.trainloader, 1, self.device, self.use_ortho_loss, self.lambda_ortho, self.lr)

        # before_attack_local = get_params(self.net)  # snapshot after local training, pre-attack
        before_attack_local = None  # only compute if an attack needs it

        # -------------------- For Mixture attack -------------------- #

        selected_attack = self.attack_type
        self._mix_mode = None
        if self.is_malicious and self.attack_type == "Mixture":
            selected_attack = str(config.get("mixture_mode", "") or self._mixture_pick(self.CurrentRound))
            self._mix_mode = selected_attack

        self._round_attack_mode = str(selected_attack)
        self._round_coord_tag = "".join(ch if ch.isalnum() else "_" for ch in self._round_attack_mode) or "attack"

        # ------------------------------------------------------------ #

        # If the client is malicious, simulate model poisoning by modifying the model weights
        if self.is_malicious:
            if selected_attack == "RandScaling":
                self.poison_model()


            elif selected_attack == "Trim":

                if before_attack_local is None:
                    before_attack_local = get_params(self.net)  # snapshot after local training, pre-attack

                # compute mu, sigma (either via coordination A.2 or fallback B)
                mu, sigma, s_hat = self._compute_trim_stats_partial(
                    broadcast_params, before_attack_local, config
                )

                # craft LoRA-only vector x_star using μ±(3–4)σ bands and s_hat
                x_star = self._craft_trimmed_mean_vector(mu, sigma, s_hat)

                # write x_star back into LoRA tensors
                new_params = self._apply_crafted_lora(before_attack_local, x_star)
                set_params(self.net, new_params)

            elif selected_attack == "SOPTrim":
                # Soft Orthogonal Projection on top of the Trim base attack.
                #
                # SOP orthogonalises each matrix
                # independently by rescaling its singular values toward 1 while
                # keeping the singular *vectors* unchanged.  This minimises the
                # structural deviation from the original malicious matrix for a
                # given ORLA score reduction -- the matrices look like rescaled
                # versions of the original attack matrices, not rotated ones.
                #
                # Pipeline:
                #   1. Craft x_star via the existing Trim attack
                #   2. Compute honest-anchor target score from this client's own
                #      pre-attack honest training only -- no inter-client coordination
                #      (shared filesystem pooling violates the threat model: clients
                #      are independent and cannot communicate outside the FL protocol)
                #   3. Add deterministic per-client-per-round noise to the target
                #      offset so each malicious client lands at a different ORLA score,
                #      breaking the homogeneity fingerprint that clustering-aware
                #      defenses would otherwise detect
                #   4. Binary-search alpha in [0,1] so that SOP(A_mal, alpha) and
                #      SOP(B_mal, alpha) jointly land at the target ORLA score
                #   5. Apply soft_orthogonal_projection(M, alpha) to every A and B
                #      independently (no cross-factor coupling, no gauge rotation)

                if before_attack_local is None:
                    before_attack_local = get_params(self.net)

                mu, sigma, s_hat = self._compute_trim_stats_partial(
                    broadcast_params, before_attack_local, config
                )
                x_star = self._craft_trimmed_mean_vector(mu, sigma, s_hat)
                malicious_params = self._apply_crafted_lora(before_attack_local, x_star)

                param_names = list(self.net.state_dict().keys())
                named_mal_lora = {
                    name: torch.as_tensor(_to_numpy(arr))
                    for name, arr in zip(param_names, malicious_params)
                    if ("lora_A" in name) or ("lora_B" in name)
                }
                score_raw = lora_ortho_score_from_dict(named_mal_lora)

                # Threat-model-valid target: use only this client's own honest
                # pre-attack score. No shared filesystem, no inter-client pooling.
                honest_score = self._adaptive_honest_target_score(before_attack_local)
                std_proxy = 0.1 * abs(honest_score) + 1e-6

                # Deterministic per-client-per-round noise on the offset so each
                # malicious client targets a different ORLA score, mimicking the
                # natural variance of the honest population and breaking the
                # homogeneity fingerprint that clustering-aware defenses detect.
                noise_seed = deterministic_client_seed(
                    self.base_seed, self.CurrentRound, int(self.cid)
                ) + 77_777
                rng = np.random.default_rng(noise_seed)
                client_offset = self.adaptive_ortho_target_offset + rng.normal(0, 1.0)

                target_score = max(0.0, honest_score + client_offset * std_proxy)

                if score_raw > target_score:
                    try:
                        alpha = calibrate_alpha_for_target_score(
                            named_mal_lora, target_score,
                            refine_iters=self.adaptive_ortho_search_iters,
                        )
                    except Exception as e:
                        print(f"[SOPTrim] alpha calibration failed ({e}); alpha=0 (no change)")
                        alpha = 0.0
                    sop_lora = {
                        name: soft_orthogonal_projection(M, alpha)
                        for name, M in named_mal_lora.items()
                    }
                else:
                    alpha = 0.0
                    sop_lora = named_mal_lora

                final_params = []
                for name, arr in zip(param_names, malicious_params):
                    if name in sop_lora:
                        final_params.append(_to_numpy(sop_lora[name]))
                    else:
                        final_params.append(_to_numpy(arr))
                set_params(self.net, final_params)

                score_final = lora_ortho_score_from_dict(sop_lora)
                print(
                    f"[SOPTrim] round={self.CurrentRound} cid={self.cid} "
                    f"score_raw={score_raw:.4f} honest_score={honest_score:.4f} "
                    f"std_proxy={std_proxy:.4f} client_offset={client_offset:.4f} "
                    f"target_score={target_score:.4f} alpha={alpha:.4f} "
                    f"score_final={score_final:.4f}"
                )

            elif selected_attack == "SOPRandScaling":
                # Soft Orthogonal Projection on top of the RandScaling base attack.
                # Identical SOP pipeline to SOPTrim; only the base attack differs
                # (poison_model() instead of the Trim crafting step).
                #
                # The honest snapshot MUST be captured BEFORE poison_model()
                # mutates self.net; capture the honest parameters first.
                #
                # No inter-client coordination: each client uses only its own
                # honest pre-attack score (threat-model-valid). Per-client
                # deterministic noise on the offset diversifies the target scores
                # across malicious clients, breaking the homogeneity fingerprint.

                if before_attack_local is None:
                    before_attack_local = get_params(self.net)

                self.poison_model()
                malicious_params = get_params(self.net)
                param_names = list(self.net.state_dict().keys())
                named_mal_lora = {
                    name: torch.as_tensor(_to_numpy(arr))
                    for name, arr in zip(param_names, malicious_params)
                    if ("lora_A" in name) or ("lora_B" in name)
                }
                score_raw = lora_ortho_score_from_dict(named_mal_lora)

                # Threat-model-valid target: use only this client's own honest
                # pre-attack score. No shared filesystem, no inter-client pooling.
                honest_score = self._adaptive_honest_target_score(before_attack_local)
                std_proxy = 0.1 * abs(honest_score) + 1e-6

                # Deterministic per-client-per-round noise on the offset so each
                # malicious client targets a different ORLA score, mimicking the
                # natural variance of the honest population and breaking the
                # homogeneity fingerprint that clustering-aware defenses detect.
                noise_seed = deterministic_client_seed(
                    self.base_seed, self.CurrentRound, int(self.cid)
                ) + 77_777
                rng = np.random.default_rng(noise_seed)
                client_offset = self.adaptive_ortho_target_offset + rng.normal(0, 1.0)

                target_score = max(0.0, honest_score + client_offset * std_proxy)

                if score_raw > target_score:
                    try:
                        alpha = calibrate_alpha_for_target_score(
                            named_mal_lora, target_score,
                            refine_iters=self.adaptive_ortho_search_iters,
                        )
                    except Exception as e:
                        print(f"[SOPRandScaling] alpha calibration failed ({e}); alpha=0 (no change)")
                        alpha = 0.0
                    sop_lora = {
                        name: soft_orthogonal_projection(M, alpha)
                        for name, M in named_mal_lora.items()
                    }
                else:
                    alpha = 0.0
                    sop_lora = named_mal_lora

                final_params = []
                for name, arr in zip(param_names, malicious_params):
                    if name in sop_lora:
                        final_params.append(_to_numpy(sop_lora[name]))
                    else:
                        final_params.append(_to_numpy(arr))
                set_params(self.net, final_params)

                score_final = lora_ortho_score_from_dict(sop_lora)
                print(
                    f"[SOPRandScaling] round={self.CurrentRound} cid={self.cid} "
                    f"score_raw={score_raw:.4f} honest_score={honest_score:.4f} "
                    f"std_proxy={std_proxy:.4f} client_offset={client_offset:.4f} "
                    f"target_score={target_score:.4f} alpha={alpha:.4f} "
                    f"score_final={score_final:.4f}"
                )

            elif selected_attack == "SOPMinMax":
                # Soft Orthogonal Projection on top of the MinMax base attack.
                #
                # Unlike plain MinMax (which is fully server-side forged), SOPMinMax
                # crafts the MinMax malicious update entirely on the client so that
                # the SOP stealth layer can be applied before submission -- exactly
                # mirroring the SOPTrim / SOPRandScaling design.
                #
                # Pipeline:
                #   1. Simulate K benign deltas from the broadcast model using the
                #      same tiny-step helper used by FedIMP, to serve as G (the
                #      "observed benign gradient matrix") for the MinMax bisection.
                #   2. Compute the MinMax forged flat vector via self.min_max_attack()
                #      using the honest pre-attack flat update as base_update and the
                #      simulated benign deltas as benign_grads.
                #   3. Reconstruct named LoRA tensors from the poisoned state dict
                #      returned by min_max_attack().
                #   4. Apply the SOP stealth layer with per-client-per-round noise on
                #      the target offset (same as SOPTrim / SOPRandScaling) so each
                #      malicious client lands at a different ORLA score.
                #   5. Write the final (SOP-camouflaged) params back into self.net.
                #
                # The server's _maybe_forge_same_round skips SOPMinMax (it only acts
                # on {"MinMax", "MinSum", "Mixture"}), so these client-crafted updates
                # are submitted as-is and never overwritten.

                if before_attack_local is None:
                    before_attack_local = get_params(self.net)

                # Step 1: Simulate benign deltas to build G for MinMax bisection.
                # Reuse FedIMP's tiny-step helper; fedimp_num_sims / fedimp_sim_batches
                # control the number and size of simulations.
                # _fedimp_train_k_batches returns LoRA-flat DELTAS (new_params - broadcast).
                # min_max_attack expects absolute LoRA parameter values, so we add the
                # broadcast LoRA-flat base to convert each delta into an absolute.
                # Compute the broadcast LoRA-flat base in named_parameters() order
                # (same ordering used by flatten_lora / unflatten_lora and by
                # _fedimp_train_k_batches via _fedimp_flat_from_indices).
                set_params(self.net, broadcast_params)
                broadcast_lora_flat = flatten_lora(self.net, include_classifier=False, device=self.device)
                benign_sims = []
                for _ in range(max(2, self.fedimp_num_sims)):
                    batches = self._fedimp_take_batches(self.fedimp_sim_batches)
                    if not batches:
                        break
                    delta = self._fedimp_train_k_batches(broadcast_params, batches, self.lr)
                    # Convert delta -> absolute LoRA flat vector.
                    # delta is flat over _fedimp_idx (LoRA params in named_parameters
                    # order); broadcast_lora_flat is in the same named_parameters order.

                    delta = delta.to(
                        device=broadcast_lora_flat.device,
                        dtype=broadcast_lora_flat.dtype,
                    )
                    benign_sims.append(broadcast_lora_flat + delta)

                # Restore the model to the after-local-training state so that
                # base_update reflects the honest local delta.
                set_params(self.net, before_attack_local)

                if len(benign_sims) < 2:
                    # Insufficient simulations: fall back to the honest local update
                    # (no MinMax forging this round; SOP still runs below).
                    param_names = list(self.net.state_dict().keys())
                    named_mal_lora = {
                        name: torch.as_tensor(_to_numpy(arr))
                        for name, arr in zip(param_names, before_attack_local)
                        if ("lora_A" in name) or ("lora_B" in name)
                    }
                    malicious_params = before_attack_local
                    print(f"[SOPMinMax] round={self.CurrentRound} cid={self.cid} "
                          f"WARNING: <2 benign sims; skipping MinMax forging this round.")
                else:
                    # Step 2: Run MinMax bisection client-side.
                    # Use flatten_lora (named_parameters order) for consistency with
                    # unflatten_lora used inside min_max_attack.  self.net already
                    # holds before_attack_local (restored above).
                    base_update_flat = flatten_lora(self.net, include_classifier=False, device=self.device)
                    poisoned_state = self.min_max_attack(benign_sims, base_update_flat)

                    # Step 3: Reconstruct full param list and named LoRA tensors.
                    # poisoned_state is a dict {param_name: tensor} covering ALL
                    # parameters (LoRA entries crafted, others copied from broadcast).
                    param_names = list(self.net.state_dict().keys())
                    malicious_params = [
                        _to_numpy(poisoned_state[n]) if n in poisoned_state
                        else _to_numpy(torch.as_tensor(before_attack_local[i]))
                        for i, n in enumerate(param_names)
                    ]
                    named_mal_lora = {
                        name: poisoned_state[name].detach().to(torch.float32)
                        for name in poisoned_state
                        if ("lora_A" in name) or ("lora_B" in name)
                    }

                score_raw = lora_ortho_score_from_dict(named_mal_lora)

                # Step 4: SOP stealth layer -- identical calibration to SOPTrim.
                honest_score = self._adaptive_honest_target_score(before_attack_local)
                std_proxy = 0.1 * abs(honest_score) + 1e-6

                noise_seed = deterministic_client_seed(
                    self.base_seed, self.CurrentRound, int(self.cid)
                ) + 77_777
                rng = np.random.default_rng(noise_seed)
                client_offset = self.adaptive_ortho_target_offset + rng.normal(0, 1.0)

                target_score = max(0.0, honest_score + client_offset * std_proxy)

                if score_raw > target_score:
                    try:
                        alpha = calibrate_alpha_for_target_score(
                            named_mal_lora, target_score,
                            refine_iters=self.adaptive_ortho_search_iters,
                        )
                    except Exception as e:
                        print(f"[SOPMinMax] alpha calibration failed ({e}); alpha=0 (no change)")
                        alpha = 0.0
                    sop_lora = {
                        name: soft_orthogonal_projection(M, alpha)
                        for name, M in named_mal_lora.items()
                    }
                else:
                    alpha = 0.0
                    sop_lora = named_mal_lora

                # Step 5: Write final params back.
                final_params = []
                for name, arr in zip(param_names, malicious_params):
                    if name in sop_lora:
                        final_params.append(_to_numpy(sop_lora[name]))
                    else:
                        final_params.append(_to_numpy(arr))
                set_params(self.net, final_params)

                score_final = lora_ortho_score_from_dict(sop_lora)
                print(
                    f"[SOPMinMax] round={self.CurrentRound} cid={self.cid} "
                    f"n_benign_sims={len(benign_sims)} score_raw={score_raw:.4f} "
                    f"honest_score={honest_score:.4f} std_proxy={std_proxy:.4f} "
                    f"client_offset={client_offset:.4f} target_score={target_score:.4f} "
                    f"alpha={alpha:.4f} score_final={score_final:.4f}"
                )

            elif selected_attack == "SOPPoisonedFL":
                # Soft Orthogonal Projection on top of the PoisonedFL base attack.
                #
                # PoisonedFL forging is computed server-side (same as plain PoisonedFL)
                # because it depends on the server's global PFL_STATE (the shared sign
                # vector s, the momentum k^{t-1}, and the adaptive scaling c^t) which
                # are unavailable to individual clients.  The SOP stealth layer is
                # therefore also applied server-side in _maybe_forge_pfl(), using the
                # per-client target score that this client pre-computes and embeds in
                # the FitRes metrics dict.
                #
                # Client responsibilities:
                #   1. Capture the honest pre-attack state (before_attack_local).
                #   2. Compute honest_score, std_proxy, and a per-client-per-round
                #      noisy target_score (identical noise recipe to SOPTrim /
                #      SOPRandScaling so each malicious client targets a different
                #      ORLA score, breaking the homogeneity fingerprint).
                #   3. Embed "sop_target_score" in the returned metrics so the server
                #      can calibrate alpha per slot in _maybe_forge_pfl().
                #   4. Pass through without modifying self.net -- the server overwrites
                #      this client's update with the SOP-camouflaged PoisonedFL vector.

                if before_attack_local is None:
                    before_attack_local = get_params(self.net)

                honest_score = self._adaptive_honest_target_score(before_attack_local)
                std_proxy = 0.1 * abs(honest_score) + 1e-6

                noise_seed = deterministic_client_seed(
                    self.base_seed, self.CurrentRound, int(self.cid)
                ) + 77_777
                rng = np.random.default_rng(noise_seed)
                client_offset = self.adaptive_ortho_target_offset + rng.normal(0, 1.0)

                target_score = max(0.0, honest_score + client_offset * std_proxy)

                # Stash the target in a temporary attribute so the metrics block
                # below (after the if/elif chain) can pick it up.
                self._sop_pfl_target_score = target_score

                print(
                    f"[SOPPoisonedFL] round={self.CurrentRound} cid={self.cid} "
                    f"honest_score={honest_score:.4f} std_proxy={std_proxy:.4f} "
                    f"client_offset={client_offset:.4f} target_score={target_score:.4f} "
                    f"(server will apply SOP after PoisonedFL forging)"
                )
                # Server-side forging + SOP; nothing else to do on client.

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
                    if self._round_is_coord_leader:
                        expected = max(1, int(self._round_attack_group_count))
                        # Wait for the complete deterministic FedIMP subgroup, not
                        # merely whichever clients happen to write first.
                        for _ in range(1200):
                            candidates = sorted(
                                shared.glob(f"fedimp_round{self.CurrentRound}_imp_*.pt"),
                                key=lambda x: x.name,
                            )
                            if len(candidates) >= expected:
                                break
                            time.sleep(0.05)

                        imps, weights = [], []
                        for p in sorted(shared.glob(f"fedimp_round{self.CurrentRound}_imp_*.pt"), key=lambda x: x.name):
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
                        for _ in range(1200):
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
                    if self._round_is_coord_leader:
                        # atomic saves
                        for file, tensor in [(mfile, mu), (sfile, sigma), (kfile, mask)]:
                            tmp = file.with_suffix(file.suffix + ".tmp")
                            torch.save(tensor.detach().to("cpu"), tmp)
                            os.replace(tmp, file)
                        stats_done.write_text("ok")
                    else:
                        # wait for leader to fully publish (incl. sentinel), then guarded loads
                        for _ in range(1200):
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
                    d = self._fedimp_train_k_batches(broadcast_params, batches, self.lr)
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

                # 9) (Optional) share identical Δw across all malicious clients.
                # A deterministic subgroup leader is the ONLY writer; the old
                # "first writer wins" behavior depended on process scheduling.
                if self.fedimp_share_across_malicious:
                    shared = Path(self.fedimp_shared_dir); shared.mkdir(parents=True, exist_ok=True)
                    vfile = shared / f"fedimp_round{self.CurrentRound}_vector.pt"
                    vdone = shared / f"fedimp_round{self.CurrentRound}_vector.done"
                    if self._round_is_coord_leader:
                        tmp = vfile.with_suffix(vfile.suffix + ".tmp")
                        torch.save(mal.detach().to("cpu"), tmp)
                        os.replace(tmp, vfile)
                        vdone.write_text("ok")
                    else:
                        for _ in range(1200):
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

        # SOPPoisonedFL: pass the per-client target ORLA score to the server so
        # it can calibrate alpha after applying the PoisonedFL forging step.
        if self.is_malicious and selected_attack == "SOPPoisonedFL":
            metrics["sop_target_score"] = float(getattr(self, "_sop_pfl_target_score", -1.0))
            self._sop_pfl_target_score = None  # reset

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
        # s_hat=+1 means coordinate increased during honest training (mu > wRe).
        # To degrade the model we oppose the drift: push upward coords DOWN (lower)
        # and downward coords UP (upper).
        return torch.where(s_hat < 0, upper, lower)

    # --------------------------------------------------------------------------------- #

    def evaluate(self, parameters, config) -> tuple[float, int, dict]:
        """Evaluate model parameters deterministically."""
        expected_logical_cid = config.get("logical_client_id", None)
        if expected_logical_cid is not None and str(expected_logical_cid) != str(self.cid):
            raise RuntimeError(
                f"Deterministic evaluation mapping mismatch: server expected logical client "
                f"{expected_logical_cid}, but this ClientApp owns partition {self.cid}."
            )
        eval_round = int(config.get("round_number", self.CurrentRound))
        eval_seed = deterministic_client_seed(self.base_seed, eval_round, int(self.cid)) + 900_000_000
        set_global_determinism(eval_seed)
        set_params(self.net, parameters)
        if self.problem_type == "qa":
            loss, exact_match, f1 = test_qa(self.net, self.testloader, device=self.device)
            return float(loss), len(self.testloader.dataset), {"exact_match": float(exact_match), "f1": float(f1), "client_cid": str(self.cid)}
        if self.problem_type == "mc_qa":
            loss, accuracy = test(self.net, self.testloader, device=self.device)
            return float(loss), len(self.testloader.dataset), {"loss": float(loss), "accuracy": float(accuracy), "client_cid": str(self.cid)}
        loss, accuracy = test(self.net, self.testloader, device=self.device)
        # return float(loss), len(self.testloader), {"accuracy": float(accuracy)}
        return float(loss), len(self.testloader), {"loss": float(loss), "accuracy": float(accuracy), "client_cid": str(self.cid)}

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


    # def _fedimp_train_k_batches(self, starting_params, batches: list):
    def _fedimp_train_k_batches(self, starting_params, batches: list, lr):
        """Do a tiny benign step from broadcast to mimic a benign client."""
        device = next(self.net.parameters()).device
        # reset to broadcast
        set_params(self.net, starting_params)
        self.net.train()
        opt = AdamW((p for p in self.net.parameters() if p.requires_grad), lr=lr, foreach=False, fused=False)
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
            d = self._fedimp_train_k_batches(starting_params, batches, self.lr)
            deltas.append(d.unsqueeze(0))
        if not deltas:
            return None, None
        D = torch.cat(deltas, dim=0)  # [K, D]
        mu = D.mean(dim=0)
        sigma = D.std(dim=0, unbiased=False) + 1e-12
        return mu, sigma

    def _adaptive_honest_target_score(self, honest_params) -> float:
        """Exact, zero-noise ORLA-style ortho score of this client's own
        honest (pre-attack) trained LoRA tensors. No simulation is needed:
        the client already ran normal local training -- with the same
        `use_ortho_loss` regularizer every benign client trains under --
        before any attack is crafted, so `honest_params` (e.g.
        `before_attack_local`) is a real, directly-observed sample of what
        this client's score would have been had it stayed benign.
        """
        param_names = list(self.net.state_dict().keys())
        honest_lora = {
            name: torch.as_tensor(_to_numpy(arr))
            for name, arr in zip(param_names, honest_params)
            if ("lora_A" in name) or ("lora_B" in name)
        }
        return lora_ortho_score_from_dict(honest_lora)

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
            for param_index, (name, param) in enumerate(self.net.named_parameters()):
                # Only apply poisoning to certain layers for more targeted attack
                if any(key in name for key in ["lora_A", "lora_B"]):
                    seed = deterministic_client_seed(
                        self.base_seed, self.CurrentRound, int(self.cid)
                    ) + 50_000 + param_index
                    gen = torch.Generator(device=param.device)
                    gen.manual_seed(seed)

                    # Strategy 2: Scale the weights
                    param.mul_(1 + self.poison_intensity)

                    eps = 0.5
                    noise = torch.empty_like(param).uniform_(-eps, eps, generator=gen)
                    param.mul_(1 + noise)

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
    lr = context.run_config["learning_rate"]

    # Read the run config to get settings to configure the Client
    model_name = context.run_config["model-name"]

    number_of_samples = int(context.run_config.get("number_of_samples", 0))
    problem_type = str(context.run_config.get("problem_type", "seq_cls")).lower()
    base_seed = int(context.run_config.get("base_seed", 42))
    set_global_determinism(base_seed + int(partition_id))
    trainloader, testloader = load_data(
        partition_id, num_partitions, model_name, partitioner_type, dataset_name,
        partitioner_parameter, number_of_samples, problem_type=problem_type,
        base_seed=base_seed,
    )

    # Malicious status is intentionally NOT chosen here. The server chooses a
    # deterministic malicious population for every round and sends each selected
    # client an ``is_malicious`` flag in FitIns.config.
    cid = str(context.node_config["partition-id"])
    is_malicious = False

    return ClientClass(model_name, trainloader, testloader, is_malicious=is_malicious, cid=cid, use_ortho_loss=use_ortho_loss, lr=lr, context=context).to_client()

# ====================================================================================================================================================== #

app = ClientApp(client_fn=client_fn)