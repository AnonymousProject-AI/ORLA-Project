"""LoRA-Sec: A Flower / Hugging Face app using Krum with logging."""
# ====================================================================================================================================================== #
import os
# Establish strict process-level reproducibility controls before importing
# Flower/NumPy/PyTorch (Flower itself may import numerical libraries).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")

from flwr.common import Context, ndarrays_to_parameters, EvaluateIns, parameters_to_ndarrays, GetPropertiesIns
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import Krum
from flwr.common.typing import FitRes, Parameters, Scalar
from typing import List, Tuple, Optional, Dict
import random
import hashlib
import math
import numpy as np
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitIns

from LoRA_Sec.task import (
    get_params, get_model, set_params, load_data,
    _get_seqcls_train_dataset,
    _get_seqcls_fltrust_root_dataset, FLTRUST_SEQCLS_ROOT_SAMPLES,
    load_dataset_cache_first, tokenizer_from_pretrained_cache_first,
    set_global_determinism,
)

# -------------------- Problem switch (seq_cls vs extractive QA) --------------------
PROBLEM_TYPE = "seq_cls"

def get_model_pt(model_name, num_labels):
    """Route model construction based on run_config['problem_type']."""
    return get_model(model_name, num_labels, problem_type=PROBLEM_TYPE)
# -----------------------------------------------------------------------------------

from flwr.server.strategy import FedAvg

import torch

import atexit

# ----- FedDMC deps -----
try:
    from sklearn.decomposition import PCA
except Exception:
    PCA = None  # we'll fall back to NumPy SVD if sklearn isn't available
# -----------------------

torch.manual_seed(42); np.random.seed(42); random.seed(42)


def _cid_sort_key(client_or_cid):
    """Stable numeric-then-lexicographic ordering for Flower client IDs."""
    cid = getattr(client_or_cid, "cid", client_or_cid)
    cid = str(cid)
    try:
        return (0, int(cid))
    except Exception:
        return (1, cid)


def _result_logical_cid(pair):
    # Return the stable client partition ID from the result metrics.

    try:
        metrics = getattr(pair[1], "metrics", {}) or {}
        if "client_cid" in metrics:
            return str(metrics["client_cid"])
    except Exception:
        pass
    return str(getattr(pair[0], "cid", pair[0]))


def _sort_results_inplace(results):
    """Normalize asynchronous result arrival order by stable logical client id."""
    if isinstance(results, list):
        results.sort(key=lambda pair: _cid_sort_key(_result_logical_cid(pair)))
    return results


# === SAME-ROUND FORGING CONTEXT ===
LAST_BROADCAST = None  # parameters just sent to clients this round

# -------------------------------------------------------------------------------------- #
# === PoisonedFL global state (server-side coordinator) ===================
PREV_BROADCAST = None  # broadcast of previous round (as ndarrays list)
PFL_STATE = {
    "enabled": False,        # auto-enabled when attack_type == "PoisonedFL"
    "s": None,               # fixed sign vector over attacked coords (torch +/-1)
    "c": None,               # scaling factor c_t (init from config pfl_c0)
    "beta": 0.7,             # multiplicative decay when hypothesis test fails
    "e": 50,                 # window length (rounds) for the binomial test
    "min_c": 0.5,            # lower bound on c_t
    "k_prev": None,          # previous malicious k^{t-1} (pre-sign)
    "dim": None,             # attacked dimension
    "history_W": [],         # flattened attacked weights per aggregated round
    "w_prev": None,          # broadcast w^{t-1} (as ndarrays)
    "w_prev2": None,         # broadcast w^{t-2} (as ndarrays)
}
# -------------------------------------------------------------------------------------- #

recent_accuracies = []

# ====================================================================================================================================================== #

import flwr.common.logger as fl_logger
from logging import INFO

def custom_log_history(history) -> None:
    """Custom log_history that hides per-round loss but can print other info."""
    pass


# Override Flower's default
fl_logger.log_history = custom_log_history

# ====================================================================================================================================================== #

def _fedimp_acceptance_metrics_defense(flat_updates, n_total, m_compromised_guess=0):
    # Compute squared-distance k-NN metrics for FedIMP acceptance checks.

    U = np.stack(flat_updates, axis=0).astype(np.float64)  # [N, D]
    N = U.shape[0]
    if N < 2:
        return {"k": 1, "lhs": np.zeros(N, dtype=np.float64), "rhs_sum": 0.0}

    k = max(1, math.ceil(max(1, n_total - m_compromised_guess) / 2))

    # pairwise squared distances via (u - v)^2 = u^2 + v^2 - 2uv
    sq = (U ** 2).sum(axis=1, keepdims=True)   # [N,1]
    D2 = sq + sq.T - 2.0 * (U @ U.T)           # [N,N]
    np.fill_diagonal(D2, np.inf)

    # LHS: for each client, sum of k smallest squared distances to others
    lhs = np.partition(D2, k - 1, axis=1)[:, :k].sum(axis=1)  # [N]

    # RHS: sum over benign-to-benign k-NN squared distances (defensive proxy: all clients)
    rhs_per = np.partition(D2, k - 1, axis=1)[:, :k].sum(axis=1)  # [N]
    rhs_sum = float(rhs_per.sum())

    return {"k": k, "lhs": lhs, "rhs_sum": rhs_sum}

# ====================================================================================================================================================== #
#  PoisonedFL helpers 

def _pfl_target_indices(net, include_classifier: bool):
    names = list(net.state_dict().keys())
    idx = [i for i, n in enumerate(names) if ("lora_A" in n or "lora_B" in n)]
    if include_classifier:
        idx += [i for i, n in enumerate(names) if "classifier" in n]
    return idx

def _pfl_flat_from_indices(param_list, indices) -> torch.Tensor:
    # param_list is a list of numpy arrays (like from get_params)
    return torch.from_numpy(
        np.concatenate([param_list[i].ravel() for i in indices]).astype(np.float32)
    )

def _pfl_add_into_indices(param_list, indices, flat_vec: torch.Tensor, per_layer_scales=None):
    # returns a NEW list with added deltas on targeted indices
    out = [np.array(p, copy=True) for p in param_list]
    x = flat_vec.detach().cpu().numpy()
    off = 0
    for j, k in enumerate(indices):
        cnt = out[k].size
        seg = x[off:off + cnt].reshape(out[k].shape)
        if per_layer_scales is not None:
            seg = seg * per_layer_scales[j]
        out[k] = out[k] + seg
        off += cnt
    return out

# ====================================================================================================================================================== #
# ------------- Soft Orthogonal Projection (SOP) helpers for server-side use ----------

# Keep server-side SOP helpers separate to avoid circular imports.

def _sop_lora_ortho_score(named_tensors: dict) -> float:
    """ORLA-style Frobenius deviation from identity, summed over all LoRA matrices."""
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


def _sop_soft_project(M: torch.Tensor, alpha: float) -> torch.Tensor:
    """
       Move each singular value toward 1 with strength alpha.

       Solve alpha*(sigma^2-1)^2 + (1-alpha)*(sigma-s)^2 by bisection.
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
        lo, hi = min(s, 1.0), max(s, 1.0)
        if lo == hi:
            return s
        for _ in range(60):
            mid = (lo + hi) * 0.5
            g = 4.0 * alpha * mid * (mid * mid - 1.0) + 2.0 * (1.0 - alpha) * (mid - s)
            if g < 0.0:
                lo = mid
            else:
                hi = mid
        return (lo + hi) * 0.5

    sigma_star = [_bisect(float(sv)) for sv in S.tolist()]
    S_new = torch.tensor(sigma_star, dtype=torch.float32, device=M.device)
    return (U @ torch.diag(S_new) @ Vh).to(orig_dtype)


def _sop_calibrate_alpha(named_mal_lora: dict, target_score: float,
                         coarse_steps: int = 20, refine_iters: int = 15) -> float:
    """Binary search on alpha in [0,1] to match `target_score` (monotone in alpha)."""
    def score_at(a: float) -> float:
        return _sop_lora_ortho_score(
            {name: _sop_soft_project(M, a) for name, M in named_mal_lora.items()}
        )

    alphas = np.linspace(0.0, 1.0, coarse_steps)
    scores = [score_at(float(a)) for a in alphas]
    diffs = [abs(s - target_score) for s in scores]
    best_idx = int(np.argmin(diffs))
    best_alpha = float(alphas[best_idx])
    best_diff = diffs[best_idx]

    lo = float(alphas[max(0, best_idx - 1)])
    hi = float(alphas[min(len(alphas) - 1, best_idx + 1)])
    for _ in range(refine_iters):
        mid = (lo + hi) / 2.0
        d_mid = abs(score_at(mid) - target_score)
        if d_mid < best_diff:
            best_alpha, best_diff = mid, d_mid
        if score_at(mid) > target_score:
            lo = mid
        else:
            hi = mid
    return float(min(1.0, max(0.0, best_alpha)))


def _sop_apply_to_ndarrays(base_nds: list, lora_idx: list, flat_mal: torch.Tensor,
                            param_names: list, target_score: float) -> list:
    
    # Apply calibrated SOP to the forged LoRA update and return full weights.

    # Build named LoRA dict from the forged flat vector
    named_mal_lora: dict = {}
    off = 0
    for i in lora_idx:
        name = param_names[i]
        shp = base_nds[i].shape
        cnt = base_nds[i].size
        chunk = flat_mal[off:off + cnt].view(shp).to(torch.float32)
        # flat_mal encodes the delta; the full weight is base + delta
        full_w = torch.from_numpy(base_nds[i].astype(np.float32)) + chunk
        named_mal_lora[name] = full_w
        off += cnt

    score_raw = _sop_lora_ortho_score(named_mal_lora)

    if score_raw > target_score:
        try:
            alpha = _sop_calibrate_alpha(named_mal_lora, target_score)
        except Exception as e:
            print(f"[SOP-server] alpha calibration failed ({e}); alpha=0 (no change)")
            alpha = 0.0
        sop_lora = {name: _sop_soft_project(M, alpha) for name, M in named_mal_lora.items()}
    else:
        alpha = 0.0
        sop_lora = named_mal_lora

    score_final = _sop_lora_ortho_score(sop_lora)

    # Reconstruct full ndarrays list: replace LoRA entries with SOP result (as delta)
    out = [arr.copy() for arr in base_nds]
    for i in lora_idx:
        name = param_names[i]
        if name in sop_lora:
            # sop_lora[name] is the full weight; convert back to delta
            sop_delta = (sop_lora[name] - torch.from_numpy(base_nds[i].astype(np.float32))).detach().cpu().numpy()
            out[i] = base_nds[i] + sop_delta.reshape(base_nds[i].shape).astype(base_nds[i].dtype, copy=False)

    return out, alpha, score_raw, score_final

# ====================================================================================================================================================== #

class DeterministicClientSelectionMixin:

    # ---------------------------------------------------------------------------------------------------------------------------- #

    def _get_logical_client_map(self, client_manager, group_id: int = 0):
        """Map Flower's random proxy/node IDs to stable dataset partition IDs.

        Flower 1.12 simulation assigns a random node_id to every SuperNode on each
        run. 'ClientProxy.cid' is that node_id, while the client loads data using
        'context.node_config["partition-id"]'. Sampling by proxy.cid therefore
        selects different real data partitions across repeated runs even when the
        RNG seed is identical.

        We query each client once via 'get_properties' for its stable logical
        partition id and cache the mapping for the run. Any missing/duplicate id is
        treated as a hard error instead of silently falling back to nondeterminism.
        """
        target_total = int(getattr(self, "total_clients", 0) or 0)
        if target_total > 0 and hasattr(client_manager, "wait_for"):
            client_manager.wait_for(target_total)

        proxies = list(client_manager.all().values())
        proxy_ids = frozenset(str(p.cid) for p in proxies)
        cached_proxy_ids = getattr(self, "_logical_map_proxy_ids", None)
        cached = getattr(self, "_logical_to_proxy", None)
        if cached is not None and cached_proxy_ids == proxy_ids:
            return cached

        logical_to_proxy = {}
        proxy_to_logical = {}
        request = GetPropertiesIns(config={"request": "logical_client_id"})

        # Query in a deterministic proxy-id order. The query order does not define
        # identity; it only keeps logs/reductions stable. Identity comes from the
        # partition id returned by the ClientApp itself.
        for proxy in sorted(proxies, key=_cid_sort_key):
            try:
                res = proxy.get_properties(request, timeout=None, group_id=int(group_id))
                props = getattr(res, "properties", {}) or {}
                logical = props.get("logical_client_id", props.get("partition_id", None))
            except Exception as exc:
                raise RuntimeError(
                    f"Could not resolve logical partition id for Flower proxy {proxy.cid}: {exc}"
                ) from exc

            if logical is None:
                raise RuntimeError(
                    f"Flower proxy {proxy.cid} did not return 'logical_client_id'. "
                    "The ClientClass.get_properties implementation is required for "
                    "deterministic client selection."
                )

            logical = str(int(logical)) if str(logical).lstrip("-").isdigit() else str(logical)
            if logical in logical_to_proxy:
                raise RuntimeError(
                    f"Duplicate logical client/partition id {logical} reported by "
                    f"proxies {logical_to_proxy[logical].cid} and {proxy.cid}."
                )
            logical_to_proxy[logical] = proxy
            proxy_to_logical[str(proxy.cid)] = logical

        if target_total > 0:
            expected = {str(i) for i in range(target_total)}
            actual = set(logical_to_proxy)
            if actual != expected:
                missing = sorted(expected - actual, key=_cid_sort_key)
                extra = sorted(actual - expected, key=_cid_sort_key)
                raise RuntimeError(
                    "Logical client mapping is incomplete/inconsistent: "
                    f"missing={missing}, extra={extra}."
                )

        self._logical_to_proxy = logical_to_proxy
        self._proxy_to_logical = proxy_to_logical
        self._logical_map_proxy_ids = proxy_ids

        # Print once per run so reproducibility can be verified directly.
        mapping_for_log = [
            f"partition {lid} -> proxy {logical_to_proxy[lid].cid}"
            for lid in sorted(logical_to_proxy, key=_cid_sort_key)
        ]
        print("[Determinism] resolved logical client mapping: " + "; ".join(mapping_for_log))
        return logical_to_proxy

    # ---------------------------------------------------------------------------------------------------------------------------- #

    def fit_config(self, server_round: int) -> Dict[str, Scalar]:
        # Return any config needed by clients
        return {
            "epochs": 1,
            "batch_size": 32,
            "learning_rate": 0.001,
        }


    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> List[Tuple[ClientProxy, FitIns]]:

        from flwr.common import parameters_to_ndarrays
        global LAST_BROADCAST
        # LAST_BROADCAST = parameters_to_ndarrays(parameters)

        # ---------------------------------------------------------- #
        global LAST_BROADCAST, PREV_BROADCAST, PFL_STATE
        # Rotate the two most recent *previous* broadcasts so g^{t-1}=w^{t-1}-w^{t-2}
        PFL_STATE["w_prev2"] = PFL_STATE.get("w_prev", None)
        PFL_STATE["w_prev"]  = LAST_BROADCAST  # this was w^{t-1} at end of last round

        # Now set the broadcast for THIS round
        PREV_BROADCAST = LAST_BROADCAST
        LAST_BROADCAST = parameters_to_ndarrays(parameters)
        # ---------------------------------------------------------- #

        # Resolve Flower's random per-run proxy/node ids to stable logical
        # partition ids before any sampling. This is essential for true
        # repeatability under the Flower 1.12 Simulation Engine.
        base_seed = int(getattr(self, "base_seed", 42))

        round_seed = 42 + int(server_round)

        logical_to_proxy = self._get_logical_client_map(
            client_manager, group_id=server_round
        )
        logical_ids = sorted(logical_to_proxy.keys(), key=_cid_sort_key)

        malicious_percentage = float(getattr(self, "malicious_percentage", 0.0))
        if malicious_percentage > 0.0 and logical_ids:
            num_malicious_population = min(
                len(logical_ids),
                max(1, int(malicious_percentage * len(logical_ids))),
            )

            mal_rng = random.Random(42)

            malicious_population = set(
                mal_rng.sample(logical_ids, num_malicious_population)
            )
        else:
            num_malicious_population = 0
            malicious_population = set()

        # Choose round participants independently using seed 42 + round.
        num_clients = int(self.fraction_fit * len(logical_ids))
        num_clients = min(len(logical_ids), max(0, num_clients))
        participation_rng = random.Random(round_seed)
        sampled_logical_ids = participation_rng.sample(logical_ids, num_clients)
        sampled_clients = [logical_to_proxy[cid] for cid in sampled_logical_ids]

        selected_malicious = sorted(
            [cid for cid in sampled_logical_ids if cid in malicious_population],
            key=_cid_sort_key,
        )
        malicious_rank = {cid: rank for rank, cid in enumerate(selected_malicious)}

        # For Mixture, choose the attack mode on the server as well, so every
        # malicious subgroup has a deterministic active leader/count for file
        # coordination (FedIMP/adaptive attacks cannot depend on arrival order).
        attack_mode_by_cid = {}
        attack_group_rank = {}
        attack_group_count = {}
        attack_type = str(getattr(self, "attack_type", ""))
        if attack_type == "Mixture":
            choices = list(getattr(self, "mixture_choices", []) or [])
            mix_seed = int(getattr(self, "mixture_seed", base_seed))
            groups = {}
            for cid in selected_malicious:
                if choices:
                    token = f"{cid}|{server_round}|{mix_seed}".encode("utf-8")
                    h = int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), "little", signed=False)
                    mode = str(choices[h % len(choices)])
                else:
                    mode = ""
                attack_mode_by_cid[cid] = mode
                groups.setdefault(mode, []).append(cid)
            for mode, members in groups.items():
                members = sorted(members, key=_cid_sort_key)
                for rank, cid in enumerate(members):
                    attack_group_rank[cid] = rank
                    attack_group_count[cid] = len(members)
        else:
            for cid in selected_malicious:
                attack_mode_by_cid[cid] = attack_type
                attack_group_rank[cid] = malicious_rank[cid]
                attack_group_count[cid] = len(selected_malicious)

        # Robust aggregators such as Krum/MultiKrum need f for THIS round.
        if hasattr(self, "num_malicious_clients"):
            try:
                self.num_malicious_clients = int(len(selected_malicious))
            except Exception:
                pass

        print(f"[Round {server_round}] seed={round_seed} selected logical clients: {sampled_logical_ids}")
        print(f"[Round {server_round}] malicious logical population: {sorted(map(str, malicious_population), key=_cid_sort_key)}")
        print(f"[Round {server_round}] selected malicious logical clients: {selected_malicious}")

        instructions = []
        for cid, client in zip(sampled_logical_ids, sampled_clients):
            is_mal = int(cid in malicious_population)
            config = {
                "round_number": int(server_round),
                "round_seed": int(round_seed),
                "logical_client_id": str(cid),
                "is_malicious": is_mal,
                "round_malicious_count": int(len(selected_malicious)),
                "malicious_rank": int(malicious_rank.get(cid, -1)),
                "mixture_mode": str(attack_mode_by_cid.get(cid, "")),
                "attack_group_count": int(attack_group_count.get(cid, 0)),
                "attack_group_rank": int(attack_group_rank.get(cid, -1)),
            }
            instructions.append((client, FitIns(parameters, config=config)))

        return instructions

    # ---------------------------------------------------------------------------------------------------------------------------- #

    def configure_evaluate(self, server_round, parameters, client_manager):
        # --- NEW: evaluate only in the last K rounds of every W-round window ---
        cfg = globals().get("Config_Store", {}) or {}

        W = int(cfg.get("eval_window_rounds", 100))         # every 100 rounds
        K = int(cfg.get("eval_last_k_in_window", 10))       # last 10 rounds

        # Handle partial final evaluation windows.
        total_rounds = int(cfg.get("num-server-rounds", 0))

        if W > 0 and K > 0:
            if total_rounds > 0:
                # end of the current window, capped by total_rounds
                window_end = min(((server_round - 1) // W + 1) * W, total_rounds)
                window_start = max(1, window_end - K + 1)
                if server_round < window_start or server_round > window_end:
                    return []
            else:
                # fallback: pure periodic windows of size W
                pos = ((server_round - 1) % W) + 1  # 1..W
                if pos <= (W - K):
                    return []

        # Sample evaluation clients with a local RNG.
        round_seed = int(getattr(self, "base_seed", 42)) + int(server_round)
        rng = random.Random(round_seed)

        fraction_evaluate = self.fraction_evaluate

        logical_to_proxy = self._get_logical_client_map(
            client_manager, group_id=server_round
        )
        logical_ids = sorted(logical_to_proxy.keys(), key=_cid_sort_key)
        num_clients = min(
            len(logical_ids), int(self.fraction_evaluate * len(logical_ids))
        )
        selected_logical_ids = rng.sample(logical_ids, num_clients)

        global CURRENT_EVAL_ROUND
        CURRENT_EVAL_ROUND = server_round

        out = []
        for cid in selected_logical_ids:
            evaluate_ins = EvaluateIns(
                parameters,
                config={
                    "round_number": int(server_round),
                    "logical_client_id": str(cid),
                },
            )
            out.append((logical_to_proxy[cid], evaluate_ins))
        return out


# ====================================================================================================================================================== #

def aggregate_evaluate(metrics):
    """
       Aggregate evaluation results.

         - seq_cls: averages accuracy (and loss if provided), keeps rolling last-10 accuracy.
         - qa: averages F1 / Exact Match (and loss), keeps rolling last-10 F1.
    """
    # Evaluation responses can also arrive asynchronously. Sort by client id
    # before floating-point summation so the reduction order is reproducible.
    metrics = sorted(
        metrics,
        key=lambda item: _cid_sort_key((item[1] or {}).get("client_cid", "")),
    )
    global recent_accuracies
    global avg10_blocks
    global window_summaries

    global eval_history
    if "eval_history" not in globals():
        eval_history = []

    if "avg10_blocks" not in globals():
        avg10_blocks = []
    if "window_summaries" not in globals():
        window_summaries = []
    if "recent_accuracies" not in globals():
        recent_accuracies = []

    # Read evaluation window config (defaults match the gating in configure_evaluate)
    cfg = globals().get("Config_Store", {}) or {}
    W = int(cfg.get("eval_window_rounds", 100))
    K = int(cfg.get("eval_last_k_in_window", 10))
    total_rounds = int(cfg.get("num-server-rounds", 0))
    # Rolling average length (defaults to K; can be overridden by rolling_avg_k)
    rolling_k = int(cfg.get("rolling_avg_k", K if K > 0 else 10))

    # Detect QA vs classification from reported metrics
    is_qa = any(("f1" in m or "exact_match" in m) for _, m in metrics)

    loss_scores = [m["loss"] for _, m in metrics if "loss" in m]
    avg_loss = sum(loss_scores) / len(loss_scores) if loss_scores else 0.0

    if is_qa:
        f1_scores = [m["f1"] for _, m in metrics if "f1" in m]
        em_scores = [m["exact_match"] for _, m in metrics if "exact_match" in m]
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0

        recent_accuracies.append(avg_f1)
        if len(recent_accuracies) > rolling_k:
            recent_accuracies.pop(0)
        recent_avg = sum(recent_accuracies) / len(recent_accuracies)

        round_num = globals().get("CURRENT_EVAL_ROUND", None)
        eval_history.append({
            "round": int(round_num) if round_num is not None else None,
            "avg_loss": float(avg_loss),
            "avg_f1": float(avg_f1),
            "avg_em": float(avg_em),
            "rolling_avg_f1": float(recent_avg),
        })

        summary = f"[SUMMARY] Average F1: {avg_f1:.4f} | Average EM: {avg_em:.4f}"
        print(summary)
        print(f"[INFO]    Rolling Avg F1 (Last {len(recent_accuracies)} Rounds): {recent_avg:.4f}")

        with open("results.txt", "a") as file:
            file.write(summary + "\n")
            if avg_loss:
                file.write(f"[INFO]    Average loss: {avg_loss:.6f}\n")
            file.write(f"[INFO]    Rolling Avg F1 (Last {len(recent_accuracies)} Rounds): {recent_avg:.4f}\n")

        # Record one summary per window (e.g., at rounds 100/200/300/...)
        round_num = globals().get("CURRENT_EVAL_ROUND", None)
        if round_num is not None and W > 0:
            r = int(round_num)
            window_end = min(((r - 1) // W + 1) * W, total_rounds) if total_rounds > 0 else (((r - 1) // W + 1) * W)
            if r == int(window_end):
                # True window start (handles partial last window when total_rounds is not a multiple of W)
                window_start = ((int(window_end) - 1) // W) * W + 1
                window_summaries.append({
                    "window_start": window_start,
                    "window_end": int(window_end),
                    "metric": "f1",
                    "rolling_k": int(len(recent_accuracies)),
                    "value": float(recent_avg),
                })
                avg10_blocks.append(float(recent_avg))

        out = {"f1": avg_f1, "exact_match": avg_em}
        if avg_loss:
            out["loss"] = avg_loss
        return out

    # ---------------- seq_cls ----------------
    accuracy_scores = [m["accuracy"] for _, m in metrics if "accuracy" in m]
    avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0

    recent_accuracies.append(avg_accuracy)
    if len(recent_accuracies) > rolling_k:
        recent_accuracies.pop(0)
    recent_avg = sum(recent_accuracies) / len(recent_accuracies)

    round_num = globals().get("CURRENT_EVAL_ROUND", None)
    eval_history.append({
        "round": int(round_num) if round_num is not None else None,
        "avg_loss": float(avg_loss),
        "avg_accuracy": float(avg_accuracy),
        "rolling_avg_accuracy": float(recent_avg),
    })

    summary = f"[SUMMARY] Average accuracy: {avg_accuracy:.4f}"
    print(summary)
    print(f"[INFO]    Rolling Avg Accuracy (Last {len(recent_accuracies)} Rounds): {recent_avg:.4f}")

    with open("results.txt", "a") as file:
        file.write(summary + "\n")
        if avg_loss:
            file.write(f"[INFO]    Average loss: {avg_loss:.6f}\n")
        file.write(f"[INFO]    Rolling Avg Accuracy (Last {len(recent_accuracies)} Rounds): {recent_avg:.4f}\n")

    # Record one summary per window (e.g., at rounds 100/200/300/...)
    round_num = globals().get("CURRENT_EVAL_ROUND", None)
    if round_num is not None and W > 0:
        r = int(round_num)
        window_end = min(((r - 1) // W + 1) * W, total_rounds) if total_rounds > 0 else (((r - 1) // W + 1) * W)
        if r == int(window_end):
            # True window start (handles partial last window when total_rounds is not a multiple of W)
            window_start = ((int(window_end) - 1) // W) * W + 1
            window_summaries.append({
                "window_start": window_start,
                "window_end": int(window_end),
                "metric": "accuracy",
                "rolling_k": int(len(recent_accuracies)),
                "value": float(recent_avg),
            })
            avg10_blocks.append(float(recent_avg))

    out = {"accuracy": avg_accuracy}
    if avg_loss:
        out["loss"] = avg_loss
    return out

# ====================================================================================================================================================== #

def print_final_results():

    """
        At exit, print and also append final evaluation results to a per-experiment text file.

            File name: dataset-attack-strategy-use_ortho_loss.txt
            Example: IMDB-RandScaling-MultiKrum-True.txt
    """

    # Grab run config saved at server startup
    cfg = globals().get("Config_Store", {}) or {}


    # Build output filename
    ds = cfg.get('dataset_name','unknown')
    atk = cfg.get('attack_type','unknown')
    strat = cfg.get('strategy','unknown')
    ortho = cfg.get('use_ortho_loss','unknown')

    fname = f"{ds}-{atk}-{strat}-{ortho}.txt"

    # Compact per-run file: one final rolling-average accuracy value per line.
    # Example: IMDB-SOPTrim-ORLA-True-final-accuracy.txt
    final_accuracy_fname = f"{ds}-{atk}-{strat}-{ortho}-final-accuracy.txt"
    final_accuracy_value = None


    # --- Run config header ---
    lines = []
    lines.append("")  # leading newline to separate runs
    lines.append("[RUN CONFIG]")
    lines.append(f"attack_type: {cfg.get('attack_type', 'N/A')}")
    lines.append(f"strategy: {cfg.get('strategy', 'N/A')}")
    lines.append(f"dataset_name: {cfg.get('dataset_name', 'N/A')}")
    lines.append(f"partitioner_parameter: {cfg.get('partitioner_parameter', 'N/A')}")
    lines.append(f"base_seed: {cfg.get('base_seed', 42)}")
    lines.append(f"Malicious_Client_Percentage: {cfg.get('Malicious-Client-Percentage', 'N/A')}")
    lines.append(f"Orthogonality regularization: {cfg.get('use_ortho_loss', 'N/A')}")
    lines.append("")

    # --- Per-round aggregated metrics ---
    if "eval_history" in globals() and eval_history:
        lines.append("[PER-ROUND AGGREGATED METRICS]")
        for row in eval_history:
            lines.append(str(row))
        lines.append("")

    # --- Window-level summaries (one line per window) ---
    W = int(cfg.get("eval_window_rounds", 100))
    K = int(cfg.get("eval_last_k_in_window", 10))
    rolling_k = int(cfg.get("rolling_avg_k", K if K > 0 else 10))

    if "window_summaries" in globals() and window_summaries:
        lines.append("[FINAL] Window rolling averages")
        for ws in sorted(window_summaries, key=lambda x: int(x.get("window_end", 0) or 0)):
            w_start = int(ws.get("window_start", 1) or 1)
            w_end = int(ws.get("window_end", 0) or 0)
            k_used = int(ws.get("rolling_k", rolling_k) or rolling_k)
            val = float(ws.get("value", 0.0) or 0.0)
            metric = (ws.get("metric") or "").lower()

            if metric == "f1":
                lines.append(f"[FINAL][Rounds {w_start}-{w_end}] Rolling Avg F1 (Last {k_used} Rounds): {val:.4f}")
            else:
                lines.append(f"[FINAL][Rounds {w_start}-{w_end}] Rolling Avg Accuracy (Last {k_used} Rounds): {val:.4f}")
                # Keep the final evaluation window's accuracy.
                final_accuracy_value = val
        lines.append("")
    else:
        # Fallback: print only the most recent rolling average
        is_qa_final = False
        if "eval_history" in globals() and eval_history:
            is_qa_final = any(("avg_f1" in r) for r in eval_history)

        if len(recent_accuracies) >= 1:
            last_k = recent_accuracies[-min(len(recent_accuracies), rolling_k):]
            recent_avg = sum(last_k) / len(last_k)
            if is_qa_final:
                lines.append(f"[FINAL] Rolling Avg F1 (Last {len(last_k)} Rounds): {recent_avg:.4f}")
            else:
                lines.append(f"[FINAL] Rolling Avg Accuracy (Last {len(last_k)} Rounds): {recent_avg:.4f}")
                final_accuracy_value = float(recent_avg)
            lines.append("")
        else:
            lines.append("[FINAL] No evaluation rounds were recorded.")
            lines.append("")

    # Print to console
    print("\n".join(lines))

    # Append to file
    try:
        with open(fname, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        print(f"[WARN] Could not write results to {fname}: {e}")

    # Append ONLY the final rolling-average accuracy for this run.
    # No header/config text is written, so repeated runs produce one value per line.
    if final_accuracy_value is not None:
        try:
            with open(final_accuracy_fname, "a", encoding="utf-8") as f:
                f.write(f"{final_accuracy_value:.4f}\n")
        except Exception as e:
            print(
                f"[WARN] Could not write final accuracy to "
                f"{final_accuracy_fname}: {e}"
            )

# ====================================================================================================================================================== #

# SAME-ROUND MinMax/MinSum forging (LoRA-only) shared mixin

class _SameRoundForgeMixin:
    """Inject NDSS'21 MinMax/MinSum forging into aggregate_fit for same-round use."""

    def _flat_lora_delta(self, nds, base_nds, lora_idx):
        import numpy as np, torch
        deltas = [nds[i] - base_nds[i] for i in range(len(nds))]
        parts = [deltas[i].ravel() for i in lora_idx]
        return torch.tensor(np.concatenate(parts), dtype=torch.float32)

    def _unflatten_into_lora(self, base_nds, lora_idx, flat_delta):
        import numpy as np
        forged = [arr.copy() for arr in base_nds]
        x = flat_delta.detach().cpu().numpy()
        off = 0
        for i in lora_idx:
            shp = base_nds[i].shape
            cnt = base_nds[i].size
            forged[i] = base_nds[i] + x[off:off+cnt].reshape(shp).astype(base_nds[i].dtype, copy=False)
            off += cnt
        return forged

    def _pairwise_max(self, G):
        import torch
        n = G.shape[0]
        m = 0.0
        for i in range(n):
            gi = G[i]
            for j in range(i + 1, n):
                d = torch.norm(gi - G[j]).item()
                if d > m: m = d
        return m

    # --- paper's direction choices (inverse-std / inverse-uv / inverse-sign)
    def _pick_rp(self, G, mode: str = "std"):
        import torch
        mu = G.mean(dim=0)
        if mode == "uv":
            v = mu
            if torch.norm(v) == 0: v = torch.randn_like(v)
            p = -v / (torch.norm(v) + 1e-12)
        elif mode == "sgn":
            p = -torch.sign(mu)
            if torch.count_nonzero(p) == 0:
                p = -torch.randn_like(mu)
                p = p / (torch.norm(p) + 1e-12)
        else:  # "std" (paper default)
            p = -G.std(dim=0)
            if torch.norm(p) == 0: p = -torch.randn_like(p)
            p = p / (torch.norm(p) + 1e-12)
        return p

    def _forge_minmax(self, G):
        import torch
        mu_b = G.mean(dim=0)
        p = self._pick_rp(G, getattr(self, "rp_mode", "std"))
        tgt = self._pairwise_max(G)
        def max_to_point(x):
            return torch.norm(x.unsqueeze(0) - G, dim=1).max().item()
        gamma = 100.0
        step = gamma / 2.0
        last_ok = 0.0
        while step > 1e-5:
            cand = mu_b + gamma * p
            if max_to_point(cand) <= tgt:
                last_ok = gamma; gamma += step
            else:
                gamma -= step
            step *= 0.5
        return (mu_b + last_ok * p).detach()

    def _forge_minsum(self, G):
        import torch, math
        n = float(G.shape[0])
        mu_b = G.mean(dim=0)
        p = self._pick_rp(G, getattr(self, "rp_mode", "std"))
        max_sum = 0.0
        for i in range(int(n)):
            s = torch.norm(G[i].unsqueeze(0) - G, dim=1).pow(2).sum().item()
            if s > max_sum: max_sum = s
        sumsq = (G.pow(2).sum(dim=1)).sum().item()
        S_mu = sumsq - n * (mu_b @ mu_b).item()
        numer = max(0.0, max_sum - S_mu)
        denom = n * (p @ p).item()
        gamma = math.sqrt(numer / denom) if denom > 0 else 0.0
        return (mu_b + gamma * p).detach()

    def _maybe_forge_same_round(self, server_round, results):
        # Normalize asynchronous result order for every strategy/attack.
        _sort_results_inplace(results)

        # Seed any rare random fallback used by the forging direction.
        set_global_determinism(int(getattr(self, "base_seed", 42)) + int(server_round))

        # Only act for MinMax/MinSum
        if getattr(self, "attack_type", None) not in {"MinMax", "MinSum", "Mixture"}:
            return
        from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, FitRes
        import torch
        global LAST_BROADCAST
        if LAST_BROADCAST is None or not results:
            return

        if getattr(self, "mixture_debug", False) and getattr(self, "_last_mix_log_round", None) != server_round:
            picks = []
            for _, fit_res in results:
                m = getattr(fit_res, "metrics", {}) or {}
                # robust malicious parse
                flag = m.get("I_am_malicious", m.get("I_am_malacious", 0))
                try:
                    is_mal = bool(int(flag))
                except Exception:
                    is_mal = bool(flag)
                if is_mal:
                    picks.append(f"{m.get('client_cid', '?')}:{m.get('mix_mode', '')}")
            if picks:
                print(f"[Mixture][Server] round={server_round} picks=" + ", ".join(picks))
            self._last_mix_log_round = server_round
    
        try:
            base = LAST_BROADCAST
            # figure LoRA indices once
            lora_idx = getattr(self, "lora_indices", None)
            if not lora_idx:
                from LoRA_Sec.task import get_model
                param_names = list(get_model_pt(self.model_name, self.num_labels).state_dict().keys())
                lora_idx = [i for i, n in enumerate(param_names) if ("lora_A" in n or "lora_B" in n)]
                self.lora_indices = lora_idx

            benign_vecs, mal_slots = [], []
            for idx, (client, fit_res) in enumerate(results):
                nds = parameters_to_ndarrays(fit_res.parameters)
                delta_flat = self._flat_lora_delta(nds, base, lora_idx)

                m = getattr(fit_res, "metrics", {}) or {}
                flag = m.get("I_am_malicious", m.get("I_am_malacious", 0))  # accept either spelling
                try:
                    is_mal = bool(int(flag))
                except Exception:
                    is_mal = bool(flag)


                mixture = (getattr(self, "attack_type", None) == "Mixture")
                if is_mal:
                    if mixture:
                        mode = (m.get("mix_mode") or "")
                        if mode in {"MinMax", "MinSum"}:
                            mal_slots.append((idx, base, mode))
                    else:
                        mal_slots.append((idx, base))
                else:
                    benign_vecs.append(delta_flat)


            if len(benign_vecs) < 2 or not mal_slots:
                return

            G = torch.stack(benign_vecs)
            if getattr(self, "attack_type", None) == "Mixture":
                x_minmax = self._forge_minmax(G)
                x_minsum = self._forge_minsum(G)
            else:
                x = self._forge_minmax(G) if self.attack_type == "MinMax" else self._forge_minsum(G)


            # Overwrite each malicious client's LoRA delta with the correct forged vector
            if getattr(self, "attack_type", None) == "Mixture":
                for slot_idx, base_nds, mode in mal_slots:
                    x_use = x_minmax if mode == "MinMax" else (x_minsum if mode == "MinSum" else None)
                    if x_use is None:
                        continue
                    forged_nds = self._unflatten_into_lora(base_nds, lora_idx, x_use)
                    client, fit_res = results[slot_idx]
                    results[slot_idx] = (
                        client,
                        FitRes(
                            status=fit_res.status,
                            parameters=ndarrays_to_parameters(forged_nds),
                            num_examples=fit_res.num_examples,
                            metrics=fit_res.metrics,
                        ),
                    )
            else:
                for slot_idx, base_nds in mal_slots:
                    forged_nds = self._unflatten_into_lora(base_nds, lora_idx, x)
                    client, fit_res = results[slot_idx]
                    results[slot_idx] = (
                        client,
                        FitRes(
                            status=fit_res.status,
                            parameters=ndarrays_to_parameters(forged_nds),
                            num_examples=fit_res.num_examples,
                            metrics=fit_res.metrics,
                        ),
                    )

            print(f"[Forge/{self.attack_type}] Round {server_round}: forged {len(mal_slots)} malicious updates (same-round).")
        except Exception as e:
            print(f"[Forge] Exception: {e}")

# ====================================================================================================================================================== #

class _PoisonedFLForgeMixin:
    """Server-side PoisonedFL forging so all fake clients submit the same malicious update."""

    def _flat_lora_params(self, nds, lora_idx):
        import numpy as np, torch
        parts = [nds[i].ravel() for i in lora_idx]
        return torch.tensor(np.concatenate(parts), dtype=torch.float32)

    def _maybe_forge_pfl(self, server_round, results):
        if getattr(self, "attack_type", None) not in {"PoisonedFL", "SOPPoisonedFL", "Mixture"}:
            return
        from flwr.common import ndarrays_to_parameters, FitRes
        import torch
        global LAST_BROADCAST, PFL_STATE
        if LAST_BROADCAST is None or PFL_STATE.get("w_prev") is None or PFL_STATE.get("w_prev2") is None:
            return
        if not results:
            return

        base = LAST_BROADCAST
        lora_idx = getattr(self, "lora_indices", None)
        if not lora_idx:
            from LoRA_Sec.task import get_model
            net = get_model_pt(self.model_name, self.num_labels)
            include_cls = bool(getattr(self, "pfl_include_classifier", False))
            lora_idx = _pfl_target_indices(net, include_cls)
            self.lora_indices = lora_idx

        # g^{t-1} over attacked coords (LoRA only)
        g_prev = self._flat_lora_params(PFL_STATE["w_prev"], lora_idx) - \
                 self._flat_lora_params(PFL_STATE["w_prev2"], lora_idx)

        dim = int(g_prev.numel())
        # Init state once
        if PFL_STATE["s"] is None or PFL_STATE.get("dim") != dim:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(getattr(self, "base_seed", 42)) + 1337)
            PFL_STATE["s"] = torch.sign(torch.randn(dim, generator=gen))
            PFL_STATE["dim"] = dim
        if PFL_STATE["c"] is None:
            PFL_STATE["c"]     = float(getattr(self, "pfl_c0", 8.0))
            PFL_STATE["beta"]  = float(getattr(self, "pfl_beta", 0.7))
            PFL_STATE["e"]     = int(getattr(self, "pfl_e", 50))
            PFL_STATE["min_c"] = float(getattr(self, "pfl_min_c", 0.5))
        if PFL_STATE["k_prev"] is None:
            PFL_STATE["k_prev"] = torch.zeros_like(g_prev)

        s = PFL_STATE["s"]

        # v^t = | g^{t-1} - (k^{t-1}/||k^{t-1}||) * ||g^{t-1}|| |, then L2-normalize
        g_norm = torch.norm(g_prev) + 1e-12
        k_prev = PFL_STATE["k_prev"]
        k_prev_norm = torch.norm(k_prev) + 1e-12

        # v_raw = torch.abs(g_prev - (k_prev / k_prev_norm) * g_norm)
        signed_kprev = k_prev * s                      # ⊙ s
        v_raw = torch.abs(g_prev - (signed_kprev / (torch.norm(signed_kprev)+1e-12)) * g_norm)
        v_t = v_raw / (torch.norm(v_raw) + 1e-12)
        
        v = v_raw / (torch.norm(v_raw) + 1e-12)

        # λ^t = c^t * ||g^{t-1}||,  k^t = λ^t * v^t,  forged = k^t ⊙ s
        lam = float(PFL_STATE["c"]) * float(g_norm)
        k_t = lam * v

        forged_flat = k_t * s

        # Resolve parameter names (needed for SOP's named-tensor dict)
        param_names = getattr(self, "_pfl_param_names", None)
        if param_names is None:
            from LoRA_Sec.task import get_model
            net = get_model_pt(self.model_name, self.num_labels)
            param_names = list(net.state_dict().keys())
            self._pfl_param_names = param_names

        # Overwrite every matching malicious client's LoRA delta with the forged vector.
        # For SOPPoisonedFL, additionally apply SOP to the forged LoRA matrices using
        # the per-client target score that the client embedded in its metrics dict.
        mixture = (getattr(self, "attack_type", None) == "Mixture")
        attack_type = getattr(self, "attack_type", None)
        mal_slots = []
        for idx, (client, fit_res) in enumerate(results):
            m = (getattr(fit_res, "metrics", {}) or {})
            flag = m.get("I_am_malicious", m.get("I_am_malacious", 0))
            try:
                is_mal = bool(int(flag))
            except Exception:
                is_mal = bool(flag)
            if not is_mal:
                continue
            if mixture:
                mix_mode = m.get("mix_mode", "")
                if mix_mode not in {"PoisonedFL", "SOPPoisonedFL"}:
                    continue
                mal_slots.append((idx, mix_mode))
            else:
                mal_slots.append((idx, attack_type))

        if not mal_slots:
            return

        for slot_idx, mode in mal_slots:
            client, fit_res = results[slot_idx]
            m = (getattr(fit_res, "metrics", {}) or {})

            if mode == "SOPPoisonedFL":
                # Apply SOP to the forged PoisonedFL vector using the per-client
                # target score reported by the client in its metrics.
                target_score = float(m.get("sop_target_score", -1.0))
                if target_score < 0.0:
                    # Client did not report a target (e.g., first round before SOP
                    # initialization); fall back to plain PoisonedFL forging.
                    forged_nds = self._unflatten_into_lora(base, lora_idx, forged_flat)
                    print(f"[SOPPoisonedFL] round={server_round} slot={slot_idx}: "
                          f"no sop_target_score in metrics; using plain PoisonedFL vector.")
                else:
                    try:
                        forged_nds, alpha, score_raw, score_final = _sop_apply_to_ndarrays(
                            base_nds=base,
                            lora_idx=lora_idx,
                            flat_mal=forged_flat,
                            param_names=param_names,
                            target_score=target_score,
                        )
                        print(
                            f"[SOPPoisonedFL] round={server_round} slot={slot_idx} "
                            f"target_score={target_score:.4f} alpha={alpha:.4f} "
                            f"score_raw={score_raw:.4f} score_final={score_final:.4f}"
                        )
                    except Exception as e:
                        print(f"[SOPPoisonedFL] SOP step failed ({e}); using plain PoisonedFL vector.")
                        forged_nds = self._unflatten_into_lora(base, lora_idx, forged_flat)
            else:
                # Plain PoisonedFL (or Mixture with mix_mode=="PoisonedFL")
                forged_nds = self._unflatten_into_lora(base, lora_idx, forged_flat)

            results[slot_idx] = (
                client,
                FitRes(
                    status=fit_res.status,
                    parameters=ndarrays_to_parameters(forged_nds),
                    num_examples=fit_res.num_examples,
                    metrics=fit_res.metrics,
                ),
            )

        PFL_STATE["k_prev"] = k_t.detach().clone()  # remember k^t for next round

    def _pfl_after_aggregate_update(self, server_round, aggregated_result):
        """Update c via the binomial test on aggregated W_t; keep a length-(e+1) history."""
        if getattr(self, "attack_type", None) not in {"PoisonedFL", "SOPPoisonedFL", "Mixture"}:
            return
        from flwr.common import parameters_to_ndarrays
        import torch, math
        global PFL_STATE
        if not aggregated_result or aggregated_result[0] is None:
            return

        params = parameters_to_ndarrays(aggregated_result[0])
        lora_idx = getattr(self, "lora_indices", None)
        if not lora_idx:
            return
        W_flat = self._flat_lora_params(params, lora_idx)
        PFL_STATE["history_W"].append(W_flat)
        e_win = int(PFL_STATE.get("e", 50))
        if len(PFL_STATE["history_W"]) > e_win + 1:
            PFL_STATE["history_W"] = PFL_STATE["history_W"][-(e_win+1):]

        # If enough history, H0: sign agreement with s is Bin(d, 0.5)
        if len(PFL_STATE["history_W"]) >= e_win + 1 and PFL_STATE["s"] is not None:
            W_t  = PFL_STATE["history_W"][-1]
            W_te = PFL_STATE["history_W"][-(e_win+1)]
            delta = W_t - W_te
            s = PFL_STATE["s"]
            X = int((torch.sign(delta) == s).sum().item())
            d = int(s.numel())

            # Normal approximation for tail prob P(Bin(d,0.5) >= X)
            mu = 0.5 * d
            sigma = (d * 0.25) ** 0.5 + 1e-12
            z = (X - 0.5 - mu) / sigma                 # continuity correction
            Phi = 0.5 * (1.0 + math.erf(z / (2**0.5))) # normal CDF
            p_val = 1.0 - Phi
            p_thresh = float(getattr(self, "pfl_pval", 0.01))
            beta   = float(PFL_STATE.get("beta", 0.7))
            min_c  = float(PFL_STATE.get("min_c", 0.5))
            c_max  = float(getattr(self, "pfl_cmax", 50.0))  # optional cap


            if p_val > p_thresh:
                PFL_STATE["c"] = max(min_c, beta * float(PFL_STATE["c"]))  # decrease on failure
            else:
                PFL_STATE["c"] = float(PFL_STATE["c"])                     # unchanged on success (paper)


# ====================================================================================================================================================== #

class DeterministicFedAvg(_SameRoundForgeMixin, _PoisonedFLForgeMixin, DeterministicClientSelectionMixin, FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        self._maybe_forge_same_round(server_round, results)  # MinMax/MinSum
        self._maybe_forge_pfl(server_round, results)         # PoisonedFL
        aggregated_result = super().aggregate_fit(server_round, results, failures)
        self._pfl_after_aggregate_update(server_round, aggregated_result)
        return aggregated_result

# ====================================================================================================================================================== #

class _ORLABase(_SameRoundForgeMixin, _PoisonedFLForgeMixin, DeterministicClientSelectionMixin, FedAvg):
    """Shared setup/scoring for ORLA and ORLA-Flex; no legacy aggregation."""
    def __init__(self, *, model_name: str, num_labels: int, m: int, selection_mode: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.num_labels = num_labels
        self.m = m                                   # number of clients to eliminate
        self.selection_mode = int(selection_mode)    # 1=fixed-m trim, 2=auto-trim
        # Reuse static PEFT/LoRA metadata across rounds for this strategy only.
        self._v9_model_metadata = None
        self._v9_lora_pair_indices = None


    def _compute_ortho_score(self, weights: List[np.ndarray], param_names: List[str]) -> float:
        score = 0.0
        for name, param in zip(param_names, weights):
            if ("lora_A" in name) or ("lora_B" in name):
                A = torch.tensor(param)
                try:
                    if A.shape[0] < A.shape[1]:
                        prod = A @ A.T
                        identity = torch.eye(prod.shape[0])
                    else:
                        prod = A.T @ A
                        identity = torch.eye(prod.shape[0])

                    # Orthogonality score calculation
                    score += torch.norm(prod - identity, p="fro").item()

                except Exception as e:
                    print(f"[OrthoScore] Skipping {name}: {e}")
        return score




# ====================================================================================================================================================== #

class ORLA(_ORLABase):
    """
      Three-stage robust aggregation for federated LoRA.

            Stage 1 ranks clients by LoRA orthogonality deviation.
            Stage 2 refines selection using unusually low singular-value spread.
            Stage 3 evaluates effective BA distances, with safeguards and influence caps.
    
            ORLA-Flex uses an adaptive cutoff in Stage 1.
    """

    def __init__(
        self,
        *,
        model_name: str,
        num_labels: int,
        m: int,
        removal_ratio: Optional[float] = None,
        selection_mode: int = 1,
        spread_z_threshold: float = 1.0,
        spread_ratio_threshold: float = 0.60,
        factor_protect_z: float = 3.0,
        max_helper_swaps: int = 3,
        enable_spread_helper: bool = True,
        enable_factor_stage: bool = True,
        matched_random_stage1: bool = False,
        safety_gate_mode: str = "none",
        safety_z_threshold: float = 4.0,
        safety_ratio_threshold: float = 3.0,
        safety_max_extra_rejections: int = 3,
        safety_core_fraction: float = 0.50,
        safety_attack_scale_quantile: float = 0.75,
        safety_attack_scale_ratio: float = 0.80,
        safety_separation_gap_threshold: float = 2.50,
        safety_rescue_ratio_threshold: float = 2.00,
        safety_max_rescues: int = 0,
        safety_bootstrap_rounds: int = 10,
        safety_bootstrap_required_trusted_rounds: int = 2,
        safety_bootstrap_z_threshold: float = 4.0,
        safety_bootstrap_ratio_threshold: float = 2.0,
        influence_cap_ratio: float = 3.0,
        influence_extreme_residual_ratio: float = 10.0,
        # ORLA ablation switches. Defaults preserve the published/full A6 path.
        safety_use_core_reference: bool = True,
        safety_require_attack_scale: bool = True,
        safety_require_separation: bool = True,
        safety_enable_rescue: bool = True,
        safety_enable_bootstrap: bool = True,
        enable_influence_cap: bool = True,
        enable_extreme_residual_override: bool = True,
        ablation_label: str = "none",
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            num_labels=num_labels,
            m=m,
            selection_mode=selection_mode,
            **kwargs,
        )
        # For fixed ORLA, removal_ratio sets the per-round rejection fraction.
        # For ORLA-Flex, m is the minimum number of retained clients.
        if removal_ratio is None:
            self.v9_removal_ratio = None
        else:
            self.v9_removal_ratio = float(removal_ratio)

        self.v9_spread_z_threshold = float(spread_z_threshold)
        self.v9_spread_ratio_threshold = float(spread_ratio_threshold)
        self.v9_factor_protect_z = float(factor_protect_z)
        self.v9_max_helper_swaps = int(max_helper_swaps)
        # Explicit Stage-2 switch used for the leave-one-stage-out ablation.
        # When False, singular-spread features are not computed for Stage 2 and
        # no promote/rescue swap is allowed. Stage 1 and Stage 3 remain unchanged.
        self.v9_enable_spread_helper = bool(enable_spread_helper)
        # Stage-1 switch used only for the leave-one-stage-out ablation.
        # When False, the orthogonality score is not computed or used.
        self.v9_enable_factor_stage = bool(enable_factor_stage)

        # Stage-1 ablation: randomize ranking but preserve its rejection quota.
        self.v9_matched_random_stage1 = bool(matched_random_stage1)
        if self.v9_enable_factor_stage and self.v9_matched_random_stage1:
            raise ValueError(
                "ORLA-v9 cannot enable both orthogonality Stage 1 and the matched-random Stage-1 control."
            )

        # Stage 3 supports exact BA distance (trace) and cheaper norm proxies.
        self.v9_safety_gate_mode = str(safety_gate_mode).strip().lower()
        self.v9_safety_z_threshold = float(safety_z_threshold)
        self.v9_safety_ratio_threshold = float(safety_ratio_threshold)
        self.v9_safety_max_extra_rejections = int(safety_max_extra_rejections)

        # Use the lowest-distance survivors as the Stage-3 benign reference.
        self.v9_safety_core_fraction = float(safety_core_fraction)

        # Confirm BA outliers against the attack-scale of Stage-1/2 rejections.
        self.v9_safety_attack_scale_quantile = float(safety_attack_scale_quantile)
        self.v9_safety_attack_scale_ratio = float(safety_attack_scale_ratio)

        # Require clear BA separation before rejection or rescue decisions.
        self.v9_safety_separation_gap_threshold = float(safety_separation_gap_threshold)
        self.v9_safety_rescue_ratio_threshold = float(safety_rescue_ratio_threshold)
        # 0 means no arbitrary rescue cap; separation + benign-like BA evidence
        # controls the rescue decision.
        self.v9_safety_max_rescues = int(safety_max_rescues)

        # Bootstrap can reject extreme BA outliers before reference estimates stabilize.
        # End bootstrap after enough consecutive trusted rounds.
        self.v9_safety_bootstrap_rounds = int(safety_bootstrap_rounds)
        self.v9_safety_bootstrap_required_trusted_rounds = int(
            safety_bootstrap_required_trusted_rounds
        )
        self.v9_safety_bootstrap_z_threshold = float(safety_bootstrap_z_threshold)
        self.v9_safety_bootstrap_ratio_threshold = float(
            safety_bootstrap_ratio_threshold
        )

        # Cap unusually large survivor influence while preserving equal base weights.
        self.v9_influence_cap_ratio = float(influence_cap_ratio)
        # ORLA extends A5's cap eligibility only. A selected survivor whose
        # effective-BA distance is extreme relative to the same low-D core,
        # D_i >= influence_extreme_residual_ratio * M_core, becomes cap-eligible
        # even when it did not enter A5's A3-qualified uncertain set. The actual
        # soft cap remains the A5 value influence_cap_ratio * M_core / D_i.
        self.v9_influence_extreme_residual_ratio = float(
            influence_extreme_residual_ratio
        )
        # Fine-grained A6 ablation controls. These gates are intentionally
        # orthogonal: each numeric ablation disables exactly one mechanism while
        # leaving all other A6 settings and thresholds unchanged.
        self.v9_safety_use_core_reference = bool(safety_use_core_reference)
        self.v9_safety_require_attack_scale = bool(safety_require_attack_scale)
        self.v9_safety_require_separation = bool(safety_require_separation)
        self.v9_safety_enable_rescue = bool(safety_enable_rescue)
        self.v9_safety_enable_bootstrap = bool(safety_enable_bootstrap)
        self.v9_enable_influence_cap = bool(enable_influence_cap)
        self.v9_enable_extreme_residual_override = bool(enable_extreme_residual_override)
        self.v9_ablation_label = str(ablation_label or "none")
        self._v9a5_trusted_separation_streak = 0
        self._v9a5_bootstrap_completed = False


        if self.v9_removal_ratio is not None and (
            (not np.isfinite(self.v9_removal_ratio))
            or self.v9_removal_ratio < 0.0
            or self.v9_removal_ratio >= 1.0
        ):
            raise ValueError("ORLA-v9 removal_ratio must be finite and in [0, 1).")
        if (not np.isfinite(self.v9_spread_z_threshold)) or self.v9_spread_z_threshold < 0.0:
            raise ValueError("ORLA-v9 spread_z_threshold must be finite and >= 0.")
        if (
            (not np.isfinite(self.v9_spread_ratio_threshold))
            or self.v9_spread_ratio_threshold <= 0.0
            or self.v9_spread_ratio_threshold > 1.0
        ):
            raise ValueError("ORLA-v9 spread_ratio_threshold must be in (0, 1].")
        if (not np.isfinite(self.v9_factor_protect_z)) or self.v9_factor_protect_z < 0.0:
            raise ValueError("ORLA-v9 factor_protect_z must be finite and >= 0.")
        if self.v9_max_helper_swaps < 0:
            raise ValueError("ORLA-v9 max_helper_swaps must be >= 0.")
        if self.v9_safety_gate_mode not in {
            "none", "trace", "trace_core", "trace_core_confirm",
            "trace_core_confirm_bidir", "trace_core_confirm_bidir_weighted",
            "trace_core_confirm_bidir_weighted_extreme", "sigma", "fro"
        }:
            raise ValueError(
                "ORLA-v9 safety_gate_mode must be one of: none, trace, trace_core, "
                "trace_core_confirm, trace_core_confirm_bidir, "
                "trace_core_confirm_bidir_weighted, "
                "trace_core_confirm_bidir_weighted_extreme, sigma, fro."
            )
        if (not np.isfinite(self.v9_safety_z_threshold)) or self.v9_safety_z_threshold < 0.0:
            raise ValueError("ORLA-v9 safety_z_threshold must be finite and >= 0.")
        if (
            (not np.isfinite(self.v9_safety_ratio_threshold))
            or self.v9_safety_ratio_threshold <= 1.0
        ):
            raise ValueError("ORLA-v9 safety_ratio_threshold must be finite and > 1.")
        if self.v9_safety_max_extra_rejections < 0:
            raise ValueError("ORLA-v9 safety_max_extra_rejections must be >= 0.")
        if (
            (not np.isfinite(self.v9_safety_core_fraction))
            or self.v9_safety_core_fraction <= 0.0
            or self.v9_safety_core_fraction > 1.0
        ):
            raise ValueError("ORLA-v9 safety_core_fraction must be in (0, 1].")
        if (
            (not np.isfinite(self.v9_safety_attack_scale_quantile))
            or self.v9_safety_attack_scale_quantile < 0.0
            or self.v9_safety_attack_scale_quantile > 1.0
        ):
            raise ValueError("ORLA-v9 safety_attack_scale_quantile must be in [0, 1].")
        if (
            (not np.isfinite(self.v9_safety_attack_scale_ratio))
            or self.v9_safety_attack_scale_ratio <= 0.0
            or self.v9_safety_attack_scale_ratio > 1.0
        ):
            raise ValueError("ORLA-v9 safety_attack_scale_ratio must be in (0, 1].")
        if (
            (not np.isfinite(self.v9_safety_separation_gap_threshold))
            or self.v9_safety_separation_gap_threshold <= 1.0
        ):
            raise ValueError(
                "ORLA-v9 safety_separation_gap_threshold must be finite and > 1."
            )
        if (
            (not np.isfinite(self.v9_safety_rescue_ratio_threshold))
            or self.v9_safety_rescue_ratio_threshold <= 0.0
        ):
            raise ValueError(
                "ORLA-v9 safety_rescue_ratio_threshold must be finite and > 0."
            )
        if self.v9_safety_max_rescues < 0:
            raise ValueError("ORLA-v9 safety_max_rescues must be >= 0.")
        if self.v9_safety_bootstrap_rounds < 0:
            raise ValueError("ORLA-v9 safety_bootstrap_rounds must be >= 0.")
        if self.v9_safety_bootstrap_required_trusted_rounds < 1:
            raise ValueError(
                "ORLA-v9 safety_bootstrap_required_trusted_rounds must be >= 1."
            )
        if (
            (not np.isfinite(self.v9_safety_bootstrap_z_threshold))
            or self.v9_safety_bootstrap_z_threshold < 0.0
        ):
            raise ValueError(
                "ORLA-v9 safety_bootstrap_z_threshold must be finite and >= 0."
            )
        if (
            (not np.isfinite(self.v9_safety_bootstrap_ratio_threshold))
            or self.v9_safety_bootstrap_ratio_threshold <= 1.0
        ):
            raise ValueError(
                "ORLA-v9 safety_bootstrap_ratio_threshold must be finite and > 1."
            )
        if (
            (not np.isfinite(self.v9_influence_cap_ratio))
            or self.v9_influence_cap_ratio <= 0.0
        ):
            raise ValueError("ORLA-v9 influence_cap_ratio must be finite and > 0.")
        if (
            (not np.isfinite(self.v9_influence_extreme_residual_ratio))
            or self.v9_influence_extreme_residual_ratio <= 0.0
        ):
            raise ValueError(
                "ORLA-v9 influence_extreme_residual_ratio must be finite and > 0."
            )

    def _v9_get_model_metadata(self):
        """Cache immutable state_dict names and LoRA scale for this strategy."""
        if self._v9_model_metadata is None:
            model = get_model_pt(self.model_name, self.num_labels)
            names = list(model.state_dict().keys())
            scale, alpha, rank = self._v9_resolve_lora_scale(model)
            name_to_index = {name: i for i, name in enumerate(names)}
            self._v9_lora_pair_indices = tuple(
                (i, name_to_index[name.replace("lora_A", "lora_B")])
                for i, name in enumerate(names)
                if "lora_A" in name
                and name.replace("lora_A", "lora_B") in name_to_index
            )
            self._v9_model_metadata = (names, scale, alpha, rank)
            del model
        return self._v9_model_metadata

    @staticmethod
    def _v9_stable_client_id(client, fit_res):
        metrics = getattr(fit_res, "metrics", {}) or {}
        return str(metrics.get("client_cid", getattr(client, "cid", "")))

    def _v9_matched_random_order(self, records, server_round: int):

        # Return deterministic random Stage-1 ranking for the ablation.
        
        cfg = globals().get("Config_Store", {}) or {}
        base_seed = int(cfg.get("base_seed", 42) or 42)
        keyed = []
        for i, rec in enumerate(records):
            cid = self._v9_stable_client_id(rec["client"], rec["fit_res"])
            token = (
                f"ORLA-v9A-NoStage1-Matched|{base_seed}|"
                f"{int(server_round)}|{cid}"
            ).encode("utf-8")
            digest = hashlib.sha256(token).digest()
            keyed.append((digest, _cid_sort_key(cid), i))
        keyed.sort(key=lambda x: (x[0], x[1]))
        return [i for _digest, _cid, i in keyed]

    @staticmethod
    def _v9_robust_positive_z(values: np.ndarray) -> np.ndarray:
        """One-sided robust z for unusually HIGH values, with a numerical floor."""
        x = np.asarray(values, dtype=np.float64)
        if x.size == 0:
            return np.zeros(0, dtype=np.float64)
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        # The 5% center floor prevents tiny round-to-round numerical differences

        scale = max(1.4826 * mad, 0.05 * max(abs(med), 1e-12), 1e-12)
        return np.maximum(0.0, (x - med) / scale)

    @staticmethod
    def _v9_robust_low_z(values: np.ndarray) -> np.ndarray:
        """One-sided robust z for unusually LOW values, with a numerical floor."""
        x = np.asarray(values, dtype=np.float64)
        if x.size == 0:
            return np.zeros(0, dtype=np.float64)
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        scale = max(1.4826 * mad, 0.05 * max(abs(med), 1e-12), 1e-12)
        return np.maximum(0.0, (med - x) / scale)

    def _compute_singular_spread_score(
        self,
        weights: List[np.ndarray],
        param_names: List[str],
    ) -> float:
        """
            Measure LoRA singular-value spread.
                Lower values indicate more uniform singular values (SOP-like updates).
        """
        per_factor = []
        for name, param in zip(param_names, weights):
            if ("lora_A" not in name) and ("lora_B" not in name):
                continue
            try:
                M = torch.as_tensor(param, dtype=torch.float64, device="cpu")
                if M.ndim != 2:
                    continue
                if M.shape[0] < M.shape[1]:
                    G = M @ M.T
                else:
                    G = M.T @ M
                G = 0.5 * (G + G.T)
                eigvals = torch.linalg.eigvalsh(G).clamp_min(0.0)
                sigma = torch.sqrt(eigvals)
                if sigma.numel() < 2:
                    continue
                mean_sigma = float(torch.mean(sigma).item())
                std_sigma = float(torch.std(sigma, unbiased=False).item())
                cv = std_sigma / max(abs(mean_sigma), 1e-12)
                if np.isfinite(cv):
                    per_factor.append(float(cv))
            except Exception as e:
                print(f"[ORLA/Spread] Skipping {name}: {e}")

        if not per_factor:
            # Fail safe: no usable helper evidence.  Returning +inf ensures this
            # client cannot look like an anomalously LOW-spread SOP candidate.
            return float("inf")
        return float(np.median(np.asarray(per_factor, dtype=np.float64)))

    @staticmethod
    def _v9_singular_features(
        weights: List[np.ndarray],
        param_names: List[str],
        lora_scale: float,
    ) -> Tuple[float, float, int]:
        
        # Compute singular-spectrum features from LoRA Gram matrices.
        per_factor_cv = []
        sigma_max = {}
        by_name = {name: arr for name, arr in zip(param_names, weights)}

        for name in param_names:
            if ("lora_A" not in name) and ("lora_B" not in name):
                continue
            if name not in by_name:
                continue
            try:
                M = torch.as_tensor(by_name[name], dtype=torch.float64, device="cpu")
                if M.ndim != 2:
                    continue
                if M.shape[0] < M.shape[1]:
                    G = M @ M.T
                else:
                    G = M.T @ M
                G = 0.5 * (G + G.T)
                eigvals = torch.linalg.eigvalsh(G).clamp_min(0.0)
                sigma = torch.sqrt(eigvals)
                if sigma.numel() == 0:
                    continue
                sigma_max[name] = float(torch.max(sigma).item())
                if sigma.numel() >= 2:
                    mean_sigma = float(torch.mean(sigma).item())
                    std_sigma = float(torch.std(sigma, unbiased=False).item())
                    cv = std_sigma / max(abs(mean_sigma), 1e-12)
                    if np.isfinite(cv):
                        per_factor_cv.append(float(cv))
            except Exception as exc:
                print(f"[ORLA/Singular] Skipping {name}: {exc}")

        spread_score = (
            float(np.median(np.asarray(per_factor_cv, dtype=np.float64)))
            if per_factor_cv else float("inf")
        )

        total_sq = 0.0
        pair_count = 0
        for a_name in param_names:
            if "lora_A" not in a_name:
                continue
            b_name = a_name.replace("lora_A", "lora_B")
            if a_name not in sigma_max or b_name not in sigma_max:
                continue
            pair_score = (
                float(lora_scale)
                * float(sigma_max[a_name])
                * float(sigma_max[b_name])
            )
            if np.isfinite(pair_score):
                total_sq += pair_score * pair_score
                pair_count += 1

        sigma_score = (
            float(math.sqrt(max(total_sq, 0.0)))
            if pair_count > 0 else float("nan")
        )
        return spread_score, sigma_score, int(pair_count)

    @staticmethod
    def _v9_resolve_lora_scale(model) -> Tuple[float, float, float]:
        # Return effective LoRA scale, alpha, and rank.
        try:
            peft_cfg = getattr(model, "peft_config", None)
            if isinstance(peft_cfg, dict):
                cfg = peft_cfg.get("default", None)
                if cfg is None and peft_cfg:
                    cfg = next(iter(peft_cfg.values()))
            else:
                cfg = peft_cfg
            if cfg is None:
                raise ValueError("model has no PEFT config")

            r = float(getattr(cfg, "r"))
            alpha = float(getattr(cfg, "lora_alpha"))
            if (not np.isfinite(r)) or r <= 0.0:
                raise ValueError(f"invalid LoRA rank r={r}")
            if (not np.isfinite(alpha)):
                raise ValueError(f"invalid LoRA alpha={alpha}")

            if bool(getattr(cfg, "use_rslora", False)):
                scale = alpha / math.sqrt(r)
            else:
                scale = alpha / r
            return float(scale), float(alpha), float(r)
        except Exception as exc:
            # Diagnostic fallback only.  A scale of 1 preserves the relative
            # comparison even if an unusual PEFT wrapper hides its config.
            print(
                f"[ORLA/BA-Diag] Could not resolve LoRA alpha/r from model: {exc}. "
                "Using diagnostic scale=1.0."
            )
            return 1.0, float("nan"), float("nan")

    @staticmethod
    def _v9_effective_ba_update_norm(
        client_weights: List[np.ndarray],
        global_weights: List[np.ndarray],
        param_names: List[str],
        lora_scale: float,
        lora_pair_indices=None,
    ) -> Tuple[float, int]:
        """
           Compute the exact scaled BA-update distance without materializing BA.
                Sum ||scale*(B_i A_i - B_g A_g)||_F^2 over LoRA layers, then take the root.
        """
        if global_weights is None:
            return float("nan"), 0
        if len(client_weights) != len(global_weights) or len(client_weights) != len(param_names):
            return float("nan"), 0

        if lora_pair_indices is None:
            # Preserve the original name-based fallback for any external caller.
            c_by_name = {name: arr for name, arr in zip(param_names, client_weights)}
            g_by_name = {name: arr for name, arr in zip(param_names, global_weights)}
            pairs = (
                (a_name, b_name, c_by_name[a_name], c_by_name[b_name],
                 g_by_name[a_name], g_by_name[b_name])
                for a_name in param_names if "lora_A" in a_name
                for b_name in [a_name.replace("lora_A", "lora_B")]
                if b_name in c_by_name and b_name in g_by_name
            )
        else:
            # Identical LoRA pair ordering; avoid building maps for each client.
            pairs = (
                (param_names[ia], param_names[ib], client_weights[ia],
                 client_weights[ib], global_weights[ia], global_weights[ib])
                for ia, ib in lora_pair_indices
            )

        total_sq = 0.0
        pair_count = 0
        for a_name, b_name, ai_arr, bi_arr, ag_arr, bg_arr in pairs:
            try:
                Ai = torch.as_tensor(ai_arr, dtype=torch.float64, device="cpu")
                Bi = torch.as_tensor(bi_arr, dtype=torch.float64, device="cpu")
                Ag = torch.as_tensor(ag_arr, dtype=torch.float64, device="cpu")
                Bg = torch.as_tensor(bg_arr, dtype=torch.float64, device="cpu")

                if (
                    Ai.ndim != 2 or Bi.ndim != 2 or Ag.ndim != 2 or Bg.ndim != 2
                    or Ai.shape != Ag.shape or Bi.shape != Bg.shape
                    or Ai.shape[0] != Bi.shape[1]
                    or Ag.shape[0] != Bg.shape[1]
                ):
                    continue

                bi_t_bi = Bi.T @ Bi
                ai_ai_t = Ai @ Ai.T
                bg_t_bg = Bg.T @ Bg
                ag_ag_t = Ag @ Ag.T
                bi_t_bg = Bi.T @ Bg
                ag_ai_t = Ag @ Ai.T

                ni_sq = torch.trace(bi_t_bi @ ai_ai_t)
                ng_sq = torch.trace(bg_t_bg @ ag_ag_t)
                cross = torch.trace(bi_t_bg @ ag_ai_t)
                diff_sq = float((ni_sq + ng_sq - 2.0 * cross).item())
                # Numerical roundoff can make an exact zero slightly negative.
                diff_sq = max(diff_sq, 0.0)
                total_sq += (float(lora_scale) ** 2) * diff_sq
                pair_count += 1
            except Exception as exc:
                print(f"[ORLA/BA-Diag] Skipping pair {a_name}/{b_name}: {exc}")

        if pair_count == 0:
            return float("nan"), 0
        return float(math.sqrt(max(total_sq, 0.0))), int(pair_count)

    @staticmethod
    def _v9_fro_product_score(
        weights: List[np.ndarray],
        param_names: List[str],
        lora_scale: float,
    ) -> Tuple[float, int]:
        """
            Very cheap Stage-3 proxy.
                Estimate BA-update size using LoRA factor Frobenius norms.
        """
        by_name = {name: arr for name, arr in zip(param_names, weights)}
        total_sq = 0.0
        pair_count = 0
        for a_name in param_names:
            if "lora_A" not in a_name:
                continue
            b_name = a_name.replace("lora_A", "lora_B")
            if b_name not in by_name:
                continue
            try:
                A = np.asarray(by_name[a_name], dtype=np.float64)
                B = np.asarray(by_name[b_name], dtype=np.float64)
                if A.ndim != 2 or B.ndim != 2 or A.shape[0] != B.shape[1]:
                    continue
                pair_score = (
                    float(lora_scale)
                    * float(np.linalg.norm(A, ord="fro"))
                    * float(np.linalg.norm(B, ord="fro"))
                )
                if np.isfinite(pair_score):
                    total_sq += pair_score * pair_score
                    pair_count += 1
            except Exception as exc:
                print(f"[ORLAC/Safety] Skipping pair {a_name}/{b_name}: {exc}")

        if pair_count == 0:
            return float("nan"), 0
        return float(math.sqrt(max(total_sq, 0.0))), int(pair_count)

# ------------------------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Tuple[Parameters, Dict[str, Scalar]]]:
        self._maybe_forge_same_round(server_round, results)
        self._maybe_forge_pfl(server_round, results)

        if not results:
            return None, {}
        if self.selection_mode not in {1, 2}:
            raise ValueError(
                f"Unknown ORLA-v9 selection_mode={self.selection_mode}; expected 1 "
                "(fixed removal-ratio Stage 1) or 2 (ORLA-Flex gap Stage 1)."
            )
        if self.selection_mode == 2 and not self.v9_enable_factor_stage:
            raise ValueError(
                "ORLA-Flex with the ORLA pipeline requires the orthogonality "
                "Stage 1, because its adaptive cutoff is defined on orthogonality scores."
            )

        param_names, v9_lora_scale, v9_lora_alpha, v9_lora_r = self._v9_get_model_metadata()
        records = []
        need_singular_features = (
            self.v9_enable_spread_helper or self.v9_safety_gate_mode == "sigma"
        )
        for client, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            if need_singular_features:
                spread_score, sigma_product_score, sigma_pair_count = self._v9_singular_features(
                    weights, param_names, v9_lora_scale
                )
            else:
                # Stage-2 ablation: do not even compute the singular-spread signal.
                spread_score = float("inf")
                sigma_product_score = float("nan")
                sigma_pair_count = 0
            factor_score = (
                float(self._compute_ortho_score(weights, param_names))
                if self.v9_enable_factor_stage else float("nan")
            )
            records.append({
                "client": client,
                "fit_res": fit_res,
                "weights": weights,
                "factor_score": factor_score,
                "spread_score": float(spread_score),
                "sigma_product_score": float(sigma_product_score),
                "sigma_product_pair_count": int(sigma_pair_count),
            })

        n = len(records)


        flex_cutoff_k = None
        flex_max_gap = float("nan")

        if self.selection_mode == 1:
            if self.v9_removal_ratio is None:
                # Backward-compatible direct-construction behavior.
                remove_count = int(self.m)
            else:
                # IMPORTANT: derive the quota from the clients that actually returned
                # updates in THIS round, not from the total population size.
                remove_count = int(self.v9_removal_ratio * n)

            if remove_count < 0:
                raise ValueError(
                    f"ORLA-v9 remove count must be >= 0, got {remove_count}."
                )
            if remove_count >= n:
                raise ValueError(
                    f"ORLA-v9 would remove {remove_count} clients, but only {n} participated. "
                    "Reduce orla_removal_ratio."
                )

            # Preserve the pre-existing fixed-v9 Stage-1 implementations exactly.
            if self.v9_enable_factor_stage:
                factor_order = sorted(
                    range(n),
                    key=lambda i: (
                        records[i]["factor_score"],
                        _cid_sort_key(self._v9_stable_client_id(
                            records[i]["client"], records[i]["fit_res"]
                        )),
                    ),
                )
                keep_count = n - remove_count
                baseline_kept = set(factor_order[:keep_count])
                baseline_rejected = set(factor_order[keep_count:])
                final_rejected = set(baseline_rejected)

                factor_values = np.asarray(
                    [rec["factor_score"] for rec in records], dtype=np.float64
                )
                factor_z = self._v9_robust_positive_z(factor_values)
                for rec in records:
                    rec["stage1_control_rank"] = None
                for i, rec in enumerate(records):
                    rec["factor_z"] = float(factor_z[i])
            elif self.v9_matched_random_stage1:
                factor_order = self._v9_matched_random_order(records, server_round)
                keep_count = n - remove_count
                baseline_kept = set(factor_order[:keep_count])
                baseline_rejected = set(factor_order[keep_count:])
                final_rejected = set(baseline_rejected)
                control_rank = {
                    idx: rank + 1 for rank, idx in enumerate(factor_order)
                }
                for i, rec in enumerate(records):
                    rec["factor_z"] = 0.0
                    rec["stage1_control_rank"] = int(control_rank[i])
            else:
                factor_order = sorted(
                    range(n),
                    key=lambda i: _cid_sort_key(self._v9_stable_client_id(
                        records[i]["client"], records[i]["fit_res"]
                    )),
                )
                baseline_kept = set(range(n))
                baseline_rejected = set()
                final_rejected = set()
                for rec in records:
                    rec["factor_z"] = 0.0
                    rec["stage1_control_rank"] = None

        else:
            # New ORLA-Flex Stage 1. This is the same gap rule as the original
            # ORLA(selection_mode=2); only the downstream pipeline changes.
            min_keep = int(self.m)
            if min_keep < 1:
                raise ValueError(
                    f"ORLA-Flex minimum retention must be >= 1, got {min_keep}."
                )

            factor_order = sorted(
                range(n),
                key=lambda i: (
                    records[i]["factor_score"],
                    _cid_sort_key(self._v9_stable_client_id(
                        records[i]["client"], records[i]["fit_res"]
                    )),
                ),
            )
            factor_values = np.asarray(
                [rec["factor_score"] for rec in records], dtype=np.float64
            )
            factor_z = self._v9_robust_positive_z(factor_values)
            for rec in records:
                rec["stage1_control_rank"] = None
            for i, rec in enumerate(records):
                rec["factor_z"] = float(factor_z[i])

            if n <= min_keep:
                flex_cutoff_k = n
                baseline_kept = set(factor_order)
                baseline_rejected = set()
                print(
                    f"[ORLA-Flex/A6][Stage1] n={n} <= min_keep={min_keep}; "
                    "kept all clients at the orthogonality-gap stage."
                )
            else:
                sorted_factor_scores = [
                    float(records[i]["factor_score"]) for i in factor_order
                ]
                start_i = min_keep - 1
                tail_diffs = [
                    sorted_factor_scores[i + 1] - sorted_factor_scores[i]
                    for i in range(start_i, n - 1)
                ]
                if tail_diffs:
                    local_argmax = int(
                        np.argmax(np.asarray(tail_diffs, dtype=np.float64))
                    )
                    flex_cutoff_k = min_keep + local_argmax
                    flex_cutoff_k = max(min_keep, min(flex_cutoff_k, n))
                    flex_max_gap = float(tail_diffs[local_argmax])
                    baseline_kept = set(factor_order[:flex_cutoff_k])
                    baseline_rejected = set(factor_order[flex_cutoff_k:])
                    print(
                        f"[ORLA-Flex/A6][Stage1] min_keep={min_keep}, "
                        f"cutoff_k={flex_cutoff_k}, max_gap={flex_max_gap:.6g}, "
                        f"rejected={len(baseline_rejected)}/{n}."
                    )
                else:
                    flex_cutoff_k = n
                    baseline_kept = set(factor_order)
                    baseline_rejected = set()
                    print(
                        f"[ORLA-Flex/A6][Stage1] no legal tail gap; kept all {n} clients."
                    )

            final_rejected = set(baseline_rejected)
            remove_count = int(len(baseline_rejected))

        # Stage 2: refine Stage 1 using unusually low singular-value spread.
        # Preserve the rejection quota through promotion/rescue swaps.
        finite_spread_idx = [
            i for i, rec in enumerate(records) if np.isfinite(rec["spread_score"])
        ]
        spread_median = float("nan")
        promoted = []
        rescued = []

        if (
            self.v9_enable_spread_helper
            and self.v9_max_helper_swaps > 0
            and len(finite_spread_idx) >= 3
        ):
            finite_values = np.asarray(
                [records[i]["spread_score"] for i in finite_spread_idx],
                dtype=np.float64,
            )
            spread_median = float(np.median(finite_values))
            low_z_finite = self._v9_robust_low_z(finite_values)
            for local_i, global_i in enumerate(finite_spread_idx):
                records[global_i]["spread_low_z"] = float(low_z_finite[local_i])
            for i in range(n):
                records[i].setdefault("spread_low_z", 0.0)

            if self.v9_enable_factor_stage or self.v9_matched_random_stage1:
                # Stage 2 remains a boundary-correction SWAP. In the matched
                # random control it uses spread information only.
                helper_candidates = [
                    i for i in baseline_kept
                    if np.isfinite(records[i]["spread_score"])
                    and records[i]["spread_low_z"] >= self.v9_spread_z_threshold
                    and spread_median > 0.0
                    and records[i]["spread_score"]
                        <= self.v9_spread_ratio_threshold * spread_median
                ]

                if self.v9_enable_factor_stage:
                    helper_candidates.sort(
                        key=lambda i: (
                            -records[i]["spread_low_z"],
                            records[i]["spread_score"],
                            -records[i]["factor_score"],
                            _cid_sort_key(self._v9_stable_client_id(
                                records[i]["client"], records[i]["fit_res"]
                            )),
                        )
                    )
                    rescue_pool = [
                        i for i in baseline_rejected
                        if records[i]["factor_z"] < self.v9_factor_protect_z
                        and not (
                            records[i].get("spread_low_z", 0.0) >= self.v9_spread_z_threshold
                            and np.isfinite(records[i]["spread_score"])
                            and spread_median > 0.0
                            and records[i]["spread_score"]
                                <= self.v9_spread_ratio_threshold * spread_median
                        )
                    ]
                    rescue_pool.sort(
                        key=lambda i: (
                            records[i]["factor_score"],
                            records[i].get("spread_low_z", 0.0),
                            _cid_sort_key(self._v9_stable_client_id(
                                records[i]["client"], records[i]["fit_res"]
                            )),
                        )
                    )
                else:
                    # Matched NoStage1: absolutely no orthogonality-derived term.
                    helper_candidates.sort(
                        key=lambda i: (
                            -records[i]["spread_low_z"],
                            records[i]["spread_score"],
                            _cid_sort_key(self._v9_stable_client_id(
                                records[i]["client"], records[i]["fit_res"]
                            )),
                        )
                    )
                    rescue_pool = [
                        i for i in baseline_rejected
                        if not (
                            records[i].get("spread_low_z", 0.0) >= self.v9_spread_z_threshold
                            and np.isfinite(records[i]["spread_score"])
                            and spread_median > 0.0
                            and records[i]["spread_score"]
                                <= self.v9_spread_ratio_threshold * spread_median
                        )
                    ]
                    rescue_pool.sort(
                        key=lambda i: (
                            records[i].get("spread_low_z", 0.0),
                            abs(records[i]["spread_score"] - spread_median)
                                if np.isfinite(records[i]["spread_score"])
                                else float("inf"),
                            _cid_sort_key(self._v9_stable_client_id(
                                records[i]["client"], records[i]["fit_res"]
                            )),
                        )
                    )

                swap_budget = min(
                    self.v9_max_helper_swaps,
                    len(helper_candidates),
                    len(rescue_pool),
                )
                for promote_i, rescue_i in zip(
                    helper_candidates[:swap_budget], rescue_pool[:swap_budget]
                ):
                    final_rejected.discard(rescue_i)
                    final_rejected.add(promote_i)
                    promoted.append(promote_i)
                    rescued.append(rescue_i)
            else:
                # Stage-1 ablation: Stage 2 stands alone.  Every client begins
                # as kept, so low-spread anomalies are directly promoted into
                # the rejection set.  Capacity and thresholds are unchanged.
                helper_candidates = [
                    i for i in range(n)
                    if np.isfinite(records[i]["spread_score"])
                    and records[i]["spread_low_z"] >= self.v9_spread_z_threshold
                    and spread_median > 0.0
                    and records[i]["spread_score"]
                        <= self.v9_spread_ratio_threshold * spread_median
                ]
                helper_candidates.sort(
                    key=lambda i: (
                        -records[i]["spread_low_z"],
                        records[i]["spread_score"],
                        _cid_sort_key(self._v9_stable_client_id(
                            records[i]["client"], records[i]["fit_res"]
                        )),
                    )
                )
                direct_budget = min(
                    self.v9_max_helper_swaps, len(helper_candidates), max(0, n - 1)
                )
                promoted = helper_candidates[:direct_budget]
                final_rejected.update(promoted)
        else:
            for rec in records:
                rec["spread_low_z"] = 0.0

        # Freeze the Stage-2 decision for diagnostics before Stage 3 can add
        # any extra rejections.
        post_helper_rejected = set(final_rejected)

        # Stage 3: compare effective BA distances with a low-distance reference.
        # Use outlier and separation evidence to reject or rescue clients.
        # Apply early-round safeguards and cap suspicious survivors' influence.
        global LAST_BROADCAST
        v9_global_weights = LAST_BROADCAST
        pre_safety_selected = [i for i in factor_order if i not in final_rejected]
        if not pre_safety_selected:
            raise RuntimeError("ORLA-v9 selected zero clients before Stage 3.")

        for rec in records:
            rec["effective_ba_update_norm"] = float("nan")
            rec["effective_ba_pair_count"] = 0
            rec["safety_gate_score"] = float("nan")
            rec["safety_gate_z"] = 0.0
            rec["safety_gate_ratio_to_median"] = float("nan")
            rec["safety_attack_scale_reference"] = float("nan")
            rec["safety_attack_scale_threshold"] = float("nan")
            rec["safety_gate_ratio_to_attack_scale"] = float("nan")
            rec["safety_separation_gap"] = float("nan")
            rec["safety_separation_trusted"] = 0
            rec["safety_rescue_threshold"] = float("nan")
            rec["safety_bootstrap_active"] = 0
            rec["safety_bootstrap_candidate"] = 0
            rec["influence_cap_threshold"] = float("nan")
            rec["influence_cap_eligible"] = 0
            rec["influence_extreme_residual_threshold"] = float("nan")
            rec["influence_extreme_residual_eligible"] = 0
            rec["influence_weight"] = 1.0

        # Compute the exact BA-distance when Stage 3 is active.
        if self.v9_safety_gate_mode in {"none", "trace", "trace_core", "trace_core_confirm", "trace_core_confirm_bidir", "trace_core_confirm_bidir_weighted", "trace_core_confirm_bidir_weighted_extreme"}:
            if v9_global_weights is None or len(v9_global_weights) != len(param_names):
                print(
                    f"[ORLA/BA-Diag][Round {server_round}] current broadcast unavailable "
                    "or misaligned; exact trace score unavailable this round."
                )
            else:
                for rec in records:
                    d_i, pair_count = self._v9_effective_ba_update_norm(
                        rec["weights"],
                        v9_global_weights,
                        param_names,
                        v9_lora_scale,
                        lora_pair_indices=self._v9_lora_pair_indices,
                    )
                    rec["effective_ba_update_norm"] = float(d_i)
                    rec["effective_ba_pair_count"] = int(pair_count)
                    if self.v9_safety_gate_mode in {"trace", "trace_core", "trace_core_confirm", "trace_core_confirm_bidir", "trace_core_confirm_bidir_weighted", "trace_core_confirm_bidir_weighted_extreme"}:
                        rec["safety_gate_score"] = float(d_i)

        if self.v9_safety_gate_mode == "sigma":
            for rec in records:
                # Reuse the singular values already computed for Stage 2.
                rec["safety_gate_score"] = float(rec.get("sigma_product_score", float("nan")))
                rec["safety_gate_pair_count"] = int(
                    rec.get("sigma_product_pair_count", 0) or 0
                )
        elif self.v9_safety_gate_mode == "fro":
            for rec in records:
                score, pair_count = self._v9_fro_product_score(
                    rec["weights"], param_names, v9_lora_scale
                )
                rec["safety_gate_score"] = float(score)
                rec["safety_gate_pair_count"] = int(pair_count)

        safety_rejected = []
        safety_rescued = []
        safety_median = float("nan")
        self._v9_last_attack_scale_reference = float("nan")
        self._v9_last_attack_scale_threshold = float("nan")
        self._v9_last_attack_scale_reference_count = 0
        self._v9_last_separation_gap = float("nan")
        self._v9_last_separation_trusted = False
        self._v9_last_rescue_threshold = float("nan")
        self._v9_last_bootstrap_active = False
        self._v9_last_influence_cap_threshold = float("nan")
        self._v9_last_influence_extreme_residual_threshold = float("nan")
        if self.v9_safety_gate_mode != "none":
            finite_survivors = [
                i for i in pre_safety_selected
                if np.isfinite(records[i].get("safety_gate_score", float("nan")))
            ]
            if len(finite_survivors) >= 3:
                vals = np.asarray(
                    [records[i]["safety_gate_score"] for i in finite_survivors],
                    dtype=np.float64,
                )

                if (
                    self.v9_safety_use_core_reference
                    and self.v9_safety_gate_mode in {"trace_core", "trace_core_confirm", "trace_core_confirm_bidir", "trace_core_confirm_bidir_weighted", "trace_core_confirm_bidir_weighted_extreme"}
                ):
                    # Use the lowest-distance survivor core as the BA reference.
                    # Keep at least three reference clients for stable robust statistics.
                    core_count = max(
                        3,
                        int(math.ceil(self.v9_safety_core_fraction * len(finite_survivors))),
                    )
                    core_count = min(core_count, len(finite_survivors))
                    core_vals = np.sort(vals)[:core_count]
                    safety_median = float(np.median(core_vals))
                    core_mad = float(np.median(np.abs(core_vals - safety_median)))
                    core_scale = max(
                        1.4826 * core_mad,
                        0.05 * max(abs(safety_median), 1e-12),
                        1e-12,
                    )
                    safety_z = np.maximum(0.0, (vals - safety_median) / core_scale)
                else:
                    core_count = len(finite_survivors)
                    safety_median = float(np.median(vals))
                    safety_z = self._v9_robust_positive_z(vals)

                for local_i, global_i in enumerate(finite_survivors):
                    records[global_i]["safety_gate_z"] = float(safety_z[local_i])
                    records[global_i]["safety_reference_count"] = int(core_count)
                    if safety_median > 0.0:
                        records[global_i]["safety_gate_ratio_to_median"] = float(
                            records[global_i]["safety_gate_score"] / safety_median
                        )

                # Require a suspicious survivor to match the BA scale of rejected updates.
                attack_scale_reference = float("nan")
                attack_scale_threshold = float("nan")
                attack_scale_reference_count = 0
                if self.v9_safety_gate_mode in {"trace_core_confirm", "trace_core_confirm_bidir", "trace_core_confirm_bidir_weighted", "trace_core_confirm_bidir_weighted_extreme"}:
                    rejected_attack_scale_vals = np.asarray(
                        [
                            records[i]["safety_gate_score"]
                            for i in sorted(post_helper_rejected)
                            if np.isfinite(records[i].get("safety_gate_score", float("nan")))
                        ],
                        dtype=np.float64,
                    )
                    attack_scale_reference_count = int(rejected_attack_scale_vals.size)
                    if attack_scale_reference_count > 0:
                        attack_scale_reference = float(
                            np.quantile(
                                rejected_attack_scale_vals,
                                self.v9_safety_attack_scale_quantile,
                            )
                        )
                        if np.isfinite(attack_scale_reference) and attack_scale_reference > 0.0:
                            attack_scale_threshold = float(
                                self.v9_safety_attack_scale_ratio * attack_scale_reference
                            )
                            for i in finite_survivors:
                                records[i]["safety_attack_scale_reference"] = attack_scale_reference
                                records[i]["safety_attack_scale_threshold"] = attack_scale_threshold
                                records[i]["safety_gate_ratio_to_attack_scale"] = float(
                                    records[i]["safety_gate_score"] / attack_scale_reference
                                )

                self._v9_last_attack_scale_reference = attack_scale_reference
                self._v9_last_attack_scale_threshold = attack_scale_threshold
                self._v9_last_attack_scale_reference_count = attack_scale_reference_count

                candidates = [
                    i for i in finite_survivors
                    if records[i]["safety_gate_z"] >= self.v9_safety_z_threshold
                    and safety_median > 0.0
                    and records[i]["safety_gate_score"]
                        >= self.v9_safety_ratio_threshold * safety_median
                    and (
                        (not self.v9_safety_require_attack_scale)
                        or self.v9_safety_gate_mode
                            not in {"trace_core_confirm", "trace_core_confirm_bidir", "trace_core_confirm_bidir_weighted", "trace_core_confirm_bidir_weighted_extreme"}
                        or (
                            np.isfinite(attack_scale_threshold)
                            and records[i]["safety_gate_score"] >= attack_scale_threshold
                        )
                    )
                ]
                candidates.sort(
                    key=lambda i: (
                        -records[i]["safety_gate_z"],
                        -records[i]["safety_gate_score"],
                        _cid_sort_key(self._v9_stable_client_id(
                            records[i]["client"], records[i]["fit_res"]
                        )),
                    )
                )

                # Trust bidirectional BA evidence only when high-distance candidates are separated.
                # Bootstrap uses strong core outliers and does not rescue earlier rejections.
                if self.v9_safety_gate_mode in {
                    "trace_core_confirm_bidir",
                    "trace_core_confirm_bidir_weighted",
                    "trace_core_confirm_bidir_weighted_extreme",
                }:
                    # Preserve the fully A3-qualified candidates for measuring
                    # A4-quality separation even during A5 bootstrap.
                    a3_candidates = list(candidates)
                    candidate_set = set(a3_candidates)
                    noncandidate_survivors = [
                        i for i in finite_survivors if i not in candidate_set
                    ]
                    separation_gap = float("nan")
                    separation_trusted = False
                    if a3_candidates and noncandidate_survivors:
                        min_candidate = min(
                            float(records[i]["safety_gate_score"]) for i in a3_candidates
                        )
                        max_other = max(
                            float(records[i]["safety_gate_score"])
                            for i in noncandidate_survivors
                        )
                        if max_other > 0.0:
                            separation_gap = float(min_candidate / max_other)
                        elif min_candidate > 0.0:
                            separation_gap = float("inf")
                        else:
                            separation_gap = 1.0
                        separation_trusted = bool(
                            separation_gap >= self.v9_safety_separation_gap_threshold
                        )

                    # Ablation can remove the separation requirement while still
                    # computing/logging the observed gap. Decisions then treat the
                    # separation check as satisfied, with all other A6 evidence intact.
                    decision_separation_trusted = bool(
                        separation_trusted or not self.v9_safety_require_separation
                    )

                    rescue_threshold = (
                        float(self.v9_safety_rescue_ratio_threshold * safety_median)
                        if np.isfinite(safety_median) and safety_median > 0.0
                        else float("nan")
                    )
                    self._v9_last_separation_gap = separation_gap
                    self._v9_last_separation_trusted = separation_trusted
                    self._v9_last_rescue_threshold = rescue_threshold

                    for i in range(n):
                        records[i]["safety_separation_gap"] = separation_gap
                        records[i]["safety_separation_trusted"] = int(
                            separation_trusted
                        )
                        records[i]["safety_rescue_threshold"] = rescue_threshold

                    if self.v9_safety_gate_mode in {"trace_core_confirm_bidir_weighted", "trace_core_confirm_bidir_weighted_extreme"}:
                        # Decide bootstrap status from *prior* rounds so the current
                        # round cannot use its own separation evidence to bypass the
                        # intended early protection. Once completed, bootstrap never
                        # reactivates.
                        bootstrap_active = bool(
                            self.v9_safety_enable_bootstrap
                            and server_round <= self.v9_safety_bootstrap_rounds
                            and not self._v9a5_bootstrap_completed
                        )
                        self._v9_last_bootstrap_active = bootstrap_active

                        # Update the A4-quality separation streak for future rounds.
                        if separation_trusted:
                            self._v9a5_trusted_separation_streak += 1
                        else:
                            self._v9a5_trusted_separation_streak = 0
                        if (
                            self._v9a5_trusted_separation_streak
                            >= self.v9_safety_bootstrap_required_trusted_rounds
                        ):
                            self._v9a5_bootstrap_completed = True

                        if bootstrap_active:
                            # Early safeguard: ignore attack-scale and cluster-gap
                            # requirements for rejection, but retain the same robust
                            # low-D core reference. No bidirectional rescue is allowed
                            # during bootstrap because the benign/attack geometry is
                            # precisely what is still being established.
                            bootstrap_candidates = [
                                i for i in finite_survivors
                                if records[i]["safety_gate_z"]
                                    >= self.v9_safety_bootstrap_z_threshold
                                and safety_median > 0.0
                                and records[i]["safety_gate_score"]
                                    >= self.v9_safety_bootstrap_ratio_threshold
                                        * safety_median
                            ]
                            bootstrap_candidates.sort(
                                key=lambda i: (
                                    -records[i]["safety_gate_z"],
                                    -records[i]["safety_gate_score"],
                                    _cid_sort_key(self._v9_stable_client_id(
                                        records[i]["client"], records[i]["fit_res"]
                                    )),
                                )
                            )
                            candidates = bootstrap_candidates
                            safety_rescued = []
                            for i in range(n):
                                records[i]["safety_bootstrap_active"] = 1
                            for i in bootstrap_candidates:
                                records[i]["safety_bootstrap_candidate"] = 1
                                # Normally rejected; if the final keep-one safety
                                # net leaves one such outlier selected, cap it.
                                records[i]["influence_cap_eligible"] = 1
                        elif not decision_separation_trusted:
                            # Post-bootstrap A5 inherits A4's conservative
                            # decision (do not reject), but these A3-qualified
                            # clients are exactly the "suspicious but uncertain"
                            # survivors for which soft influence capping was
                            # designed. Ordinary survivors are left untouched.
                            candidates = []
                            for i in a3_candidates:
                                records[i]["influence_cap_eligible"] = 1
                        else:
                            candidates = a3_candidates
                    elif not decision_separation_trusted:
                        # A4 behavior: ambiguous BA geometry preserves Stage 1+2.
                        candidates = []
                    else:
                        candidates = a3_candidates

                    # Bidirectional rescue is allowed only outside A5 bootstrap and
                    # only when A4-quality separation is trusted.
                    allow_rescue = bool(
                        self.v9_safety_enable_rescue
                        and decision_separation_trusted
                        and not (
                            self.v9_safety_gate_mode
                                in {"trace_core_confirm_bidir_weighted", "trace_core_confirm_bidir_weighted_extreme"}
                            and self._v9_last_bootstrap_active
                        )
                    )
                    if allow_rescue:
                        rescue_candidates = [
                            i for i in sorted(post_helper_rejected)
                            if np.isfinite(
                                records[i].get("safety_gate_score", float("nan"))
                            )
                            and np.isfinite(rescue_threshold)
                            and records[i]["safety_gate_score"] <= rescue_threshold
                        ]
                        rescue_candidates.sort(
                            key=lambda i: (
                                records[i]["safety_gate_score"],
                                _cid_sort_key(self._v9_stable_client_id(
                                    records[i]["client"], records[i]["fit_res"]
                                )),
                            )
                        )
                        if self.v9_safety_max_rescues > 0:
                            rescue_candidates = rescue_candidates[
                                : self.v9_safety_max_rescues
                            ]
                        safety_rescued = rescue_candidates
                        for i in safety_rescued:
                            final_rejected.discard(i)

                if self.v9_safety_max_extra_rejections > 0:
                    candidates = candidates[: self.v9_safety_max_extra_rejections]

                # Never let the safety gate eliminate every Stage-1+2 survivor.
                # For A4, a trusted separation necessarily has at least one
                # non-candidate survivor, so this is also a final safety net.
                max_legal = max(0, len(pre_safety_selected) - 1)
                safety_rejected = candidates[:max_legal]
                final_rejected.update(safety_rejected)

        selected_indices = [i for i in factor_order if i not in final_rejected]
        if not selected_indices:
            raise RuntimeError("ORLA-v9 selected zero clients after Stage 3.")

        # Cap suspicious survivors by min(1, cap_ratio * core_median / distance).
        # Equal base client weights are otherwise unchanged.
        influence_cap_threshold = float("nan")
        if (
            self.v9_enable_influence_cap
            and self.v9_safety_gate_mode in {"trace_core_confirm_bidir_weighted", "trace_core_confirm_bidir_weighted_extreme"}
            and np.isfinite(safety_median)
            and safety_median > 0.0
        ):
            influence_cap_threshold = float(
                self.v9_influence_cap_ratio * safety_median
            )
        self._v9_last_influence_cap_threshold = influence_cap_threshold
        selected_index_set = set(selected_indices)

        # Extreme BA residuals change cap eligibility, not client selection.
        # client whose D_i is at least 10x M_core by default is cap-eligible even
        influence_extreme_residual_threshold = float("nan")
        if (
            self.v9_enable_influence_cap
            and self.v9_enable_extreme_residual_override
            and self.v9_safety_gate_mode
                == "trace_core_confirm_bidir_weighted_extreme"
            and np.isfinite(safety_median)
            and safety_median > 0.0
        ):
            influence_extreme_residual_threshold = float(
                self.v9_influence_extreme_residual_ratio * safety_median
            )
            for i in selected_indices:
                d_i = float(
                    records[i].get("safety_gate_score", float("nan"))
                )
                if (
                    np.isfinite(d_i)
                    and d_i >= influence_extreme_residual_threshold
                ):
                    records[i]["influence_cap_eligible"] = 1
                    records[i]["influence_extreme_residual_eligible"] = 1
        self._v9_last_influence_extreme_residual_threshold = (
            influence_extreme_residual_threshold
        )

        for i, rec in enumerate(records):
            rec["influence_cap_threshold"] = influence_cap_threshold
            rec["influence_extreme_residual_threshold"] = (
                influence_extreme_residual_threshold
            )
            if i not in selected_index_set:
                rec["influence_weight"] = 0.0
                continue
            weight = 1.0
            if (
                self.v9_enable_influence_cap
                and self.v9_safety_gate_mode in {"trace_core_confirm_bidir_weighted", "trace_core_confirm_bidir_weighted_extreme"}
                and np.isfinite(influence_cap_threshold)
                and influence_cap_threshold > 0.0
            ):
                d_i = float(rec.get("safety_gate_score", float("nan")))
                if (
                    int(rec.get("influence_cap_eligible", 0) or 0) == 1
                    and np.isfinite(d_i)
                    and d_i > influence_cap_threshold
                ):
                    weight = float(influence_cap_threshold / max(d_i, 1e-12))
            rec["influence_weight"] = float(min(1.0, max(0.0, weight)))

        # Diagnostics intentionally expose both signals and the exact role of
        # the helper so experiments can verify that non-adaptive rounds usually
        # remain identical to base ORLA.
        stage1_policy = (
            "fixed_ratio" if self.selection_mode == 1 and self.v9_removal_ratio is not None
            else "fixed_m" if self.selection_mode == 1
            else "flex_gap"
        )
        print(
            f"[ORLA] round={server_round} n={n} base_remove={remove_count} "
            f"stage1_policy={stage1_policy} "
            f"min_keep={int(self.m) if self.selection_mode == 2 else 'N/A'} "
            f"flex_cutoff={flex_cutoff_k if self.selection_mode == 2 else 'N/A'} "
            f"flex_gap={flex_max_gap if self.selection_mode == 2 else float('nan'):.6g} "
            f"ablation={self.v9_ablation_label} "
            f"ratio={self.v9_removal_ratio if self.v9_removal_ratio is not None else ('flex' if self.selection_mode == 2 else 'fixed-m')} "
            f"stage1_mode={'orthogonality' if self.v9_enable_factor_stage else ('matched_random_control' if self.v9_matched_random_stage1 else 'disabled')} "
            f"factor_enabled={self.v9_enable_factor_stage} "
            f"spread_enabled={self.v9_enable_spread_helper} "
            f"spread_median={spread_median:.6g} swaps={len(promoted)} "
            f"safety_mode={self.v9_safety_gate_mode} "
            f"safety_extra={len(safety_rejected)} safety_rescues={len(safety_rescued)} "
            f"bootstrap={int(getattr(self, '_v9_last_bootstrap_active', False))} "
            f"cap={getattr(self, '_v9_last_influence_cap_threshold', float('nan')):.6g} "
            f"extreme_cap={getattr(self, '_v9_last_influence_extreme_residual_threshold', float('nan')):.6g} "
            f"final_remove={len(final_rejected)}"
        )
        if promoted:
            if self.v9_enable_factor_stage:
                for p, r in zip(promoted, rescued):
                    print(
                        "[ORLA] helper swap: "
                        f"reject cid={self._v9_stable_client_id(records[p]['client'], records[p]['fit_res'])} "
                        f"factor={records[p]['factor_score']:.6f} "
                        f"spread={records[p]['spread_score']:.6f} "
                        f"low_z={records[p]['spread_low_z']:.3f}; "
                        f"rescue cid={self._v9_stable_client_id(records[r]['client'], records[r]['fit_res'])} "
                        f"factor={records[r]['factor_score']:.6f} "
                        f"factor_z={records[r]['factor_z']:.3f}"
                    )
            elif self.v9_matched_random_stage1:
                for p, r in zip(promoted, rescued):
                    print(
                        "[ORLA/NoStage1-Matched] Stage-2 spread-only swap: "
                        f"reject cid={self._v9_stable_client_id(records[p]['client'], records[p]['fit_res'])} "
                        f"spread={records[p]['spread_score']:.6f} "
                        f"low_z={records[p]['spread_low_z']:.3f}; "
                        f"rescue cid={self._v9_stable_client_id(records[r]['client'], records[r]['fit_res'])} "
                        f"spread={records[r]['spread_score']:.6f} "
                        f"low_z={records[r].get('spread_low_z', 0.0):.3f}"
                    )
            else:
                for p in promoted:
                    print(
                        "[ORLA/NoStage1] Stage-2 direct reject: "
                        f"cid={self._v9_stable_client_id(records[p]['client'], records[p]['fit_res'])} "
                        f"spread={records[p]['spread_score']:.6f} "
                        f"low_z={records[p]['spread_low_z']:.3f}"
                    )

        if safety_rejected:
            for i in safety_rejected:
                print(
                    f"[ORLA/Safety] extra reject "
                    f"cid={self._v9_stable_client_id(records[i]['client'], records[i]['fit_res'])} "
                    f"mode={self.v9_safety_gate_mode} "
                    f"score={records[i]['safety_gate_score']:.6g} "
                    f"z={records[i]['safety_gate_z']:.3f} "
                    f"ratio={records[i]['safety_gate_ratio_to_median']:.3f} "
                    f"attack_scale_ratio={records[i].get('safety_gate_ratio_to_attack_scale', float('nan')):.3f}"
                )

        if safety_rescued:
            for i in safety_rescued:
                print(
                    f"[ORLA/Safety] rescue to keep "
                    f"cid={self._v9_stable_client_id(records[i]['client'], records[i]['fit_res'])} "
                    f"mode={self.v9_safety_gate_mode} "
                    f"score={records[i]['safety_gate_score']:.6g} "
                    f"rescue_threshold={records[i].get('safety_rescue_threshold', float('nan')):.6g} "
                    f"gap={records[i].get('safety_separation_gap', float('nan')):.3f}"
                )

        score_log = []
        for i in factor_order:
            rec = records[i]
            cid = self._v9_stable_client_id(rec["client"], rec["fit_res"])
            state = "REJECT" if i in final_rejected else "KEEP"
            base_state = (
                ("baseR" if i in baseline_rejected else "baseK")
                if self.v9_enable_factor_stage
                else (
                    ("randR" if i in baseline_rejected else "randK")
                    if self.v9_matched_random_stage1 else "stage1OFF"
                )
            )
            score_log.append(
                f"{cid}: factor={rec['factor_score']:.4f}, "
                f"spread={rec['spread_score']:.6f}, low_z={rec.get('spread_low_z', 0.0):.2f}, "
                f"safety={rec.get('safety_gate_score', float('nan')):.4g}, "
                f"sz={rec.get('safety_gate_z', 0.0):.2f}, "
                f"{base_state}->{state}"
            )
        print("[ORLA] " + " | ".join(score_log))


        # Aggregate selected client models. All pre-A5 variants retain the exact
        # original ORLA equal-client mean. A5/A6 multiply those equal base weights
        # by the soft influence cap for cap-eligible survivors; A6 only broadens
        # eligibility with the extreme-residual rule. Neither variant introduces
        # num_examples/sample-count weighting.
        selected = [records[i] for i in selected_indices]
        num_layers = len(selected[0]["fit_res"].parameters.tensors)
        aggregated = []
        if (
            self.v9_enable_influence_cap
            and self.v9_safety_gate_mode in {"trace_core_confirm_bidir_weighted", "trace_core_confirm_bidir_weighted_extreme"}
        ):
            agg_weights = np.asarray(
                [float(rec.get("influence_weight", 1.0)) for rec in selected],
                dtype=np.float64,
            )
            if (
                agg_weights.size != len(selected)
                or (not np.all(np.isfinite(agg_weights)))
                or float(np.sum(agg_weights)) <= 1e-12
            ):
                # Defensive fallback: never let a diagnostic/numerical issue
                # change training semantics unpredictably.
                agg_weights = np.ones(len(selected), dtype=np.float64)
            agg_weights = agg_weights / float(np.sum(agg_weights))

            for layer_idx in range(num_layers):
                layer_updates = [rec["weights"][layer_idx] for rec in selected]
                stacked = np.stack(layer_updates, axis=0)
                weighted = np.tensordot(agg_weights, stacked, axes=(0, 0))
                aggregated.append(weighted.astype(stacked.dtype, copy=False))
        else:
            for layer_idx in range(num_layers):
                layer_updates = [rec["weights"][layer_idx] for rec in selected]
                aggregated.append(np.mean(layer_updates, axis=0))

        result = (ndarrays_to_parameters(aggregated), {})
        self._pfl_after_aggregate_update(server_round, result)
        return result


# ====================================================================================================================================================== #

class ORLATraceOnly(ORLA):
    """
        Stage-3-only ablation: rank clients by effective BA distance.

            Fixed mode rejects floor(removal_ratio*N) largest distances.
            Adaptive mode rejects robust upper-tail outliers, subject to a quota.
            Stages 1 and 2 do not participate in selection.
    """

    def __init__(self, *, trace_only_mode: str, **kwargs):
        kwargs["safety_gate_mode"] = "trace"
        # Stage-1/2 knobs are retained only because ORLA validates/stores them;
        # this subclass never uses them in its decision path.
        kwargs["max_helper_swaps"] = 0
        super().__init__(**kwargs)
        self.v9_enable_factor_stage = False
        self.v9_enable_spread_helper = False
        mode = str(trace_only_mode).strip().lower()
        if mode not in {"fixed", "adaptive"}:
            raise ValueError(
                "ORLA trace-only ablation mode must be 'fixed' or 'adaptive'."
            )
        self.v9_trace_only_mode = mode

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Tuple[Parameters, Dict[str, Scalar]]]:
        self._maybe_forge_same_round(server_round, results)
        self._maybe_forge_pfl(server_round, results)

        if not results:
            return None, {}
        if self.selection_mode != 1:
            raise ValueError(
                "ORLA trace-only ablations require selection_mode=1."
            )

        # Only model metadata needed for Stage 3 is loaded.  Orthogonality and
        # singular-spread features are deliberately NOT computed here.
        param_names, lora_scale, lora_alpha, lora_r = self._v9_get_model_metadata()

        global LAST_BROADCAST
        global_weights = LAST_BROADCAST
        if global_weights is None or len(global_weights) != len(param_names):
            raise RuntimeError(
                "ORLA trace-only ablation requires the current LAST_BROADCAST "
                "to compute D_i, but it is unavailable or misaligned."
            )

        records = []
        for client, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            d_i, pair_count = self._v9_effective_ba_update_norm(
                weights, global_weights, param_names, lora_scale,
                lora_pair_indices=self._v9_lora_pair_indices,
            )
            records.append({
                "client": client,
                "fit_res": fit_res,
                "weights": weights,
                # Dummy Stage-1/2 fields exist only so the common diagnostic
                # writer can serialize a uniform schema. They are NOT computed
                # and do NOT influence selection.
                "factor_score": float("nan"),
                "factor_z": 0.0,
                "spread_score": float("nan"),
                "spread_low_z": 0.0,
                "effective_ba_update_norm": float(d_i),
                "effective_ba_pair_count": int(pair_count),
                "safety_gate_score": float(d_i),
                "safety_gate_z": 0.0,
                "safety_gate_ratio_to_median": float("nan"),
            })

        n = len(records)
        if self.v9_removal_ratio is None:
            max_remove = int(self.m)
        else:
            max_remove = int(self.v9_removal_ratio * n)
        if max_remove < 0:
            raise ValueError(f"Trace-only removal count must be >= 0, got {max_remove}.")
        if max_remove >= n:
            raise ValueError(
                f"Trace-only ablation would remove up to {max_remove} clients, "
                f"but only {n} participated. Reduce orla_removal_ratio."
            )

        finite_idx = [
            i for i, rec in enumerate(records)
            if np.isfinite(rec["safety_gate_score"])
        ]
        if len(finite_idx) != n:
            raise RuntimeError(
                "ORLA trace-only ablation could not compute a finite D_i for "
                "every participating client."
            )

        # Ascending D_i: low scores look more benign; high scores more suspicious.
        score_order = sorted(
            range(n),
            key=lambda i: (
                records[i]["safety_gate_score"],
                _cid_sort_key(self._v9_stable_client_id(
                    records[i]["client"], records[i]["fit_res"]
                )),
            ),
        )

        vals = np.asarray(
            [records[i]["safety_gate_score"] for i in finite_idx],
            dtype=np.float64,
        )
        safety_median = float(np.median(vals)) if vals.size else float("nan")
        zvals = self._v9_robust_positive_z(vals) if vals.size else np.asarray([])
        for local_i, global_i in enumerate(finite_idx):
            records[global_i]["safety_gate_z"] = float(zvals[local_i])
            if safety_median > 0.0:
                records[global_i]["safety_gate_ratio_to_median"] = float(
                    records[global_i]["safety_gate_score"] / safety_median
                )

        if self.v9_trace_only_mode == "fixed":
            # Use the fixed rejection quota, ranking only by BA distance.

            final_rejected = set(score_order[n - max_remove:]) if max_remove > 0 else set()
        else:
            # Variable-size one-sided outlier rejection.  The rejection threshold is
            # a MAXIMUM rejection ratio, not a quota in this mode.
            candidates = [
                i for i in finite_idx
                if records[i]["safety_gate_z"] >= self.v9_safety_z_threshold
                and safety_median > 0.0
                and records[i]["safety_gate_score"]
                    >= self.v9_safety_ratio_threshold * safety_median
            ]
            candidates.sort(
                key=lambda i: (
                    -records[i]["safety_gate_z"],
                    -records[i]["safety_gate_score"],
                    _cid_sort_key(self._v9_stable_client_id(
                        records[i]["client"], records[i]["fit_res"]
                    )),
                )
            )
            final_rejected = set(candidates[:max_remove]) if max_remove > 0 else set()

        if len(final_rejected) >= n:
            # Defensive guard; normal parameter validation already prevents this.
            keep_best = score_order[0]
            final_rejected.discard(keep_best)

        selected_indices = [i for i in score_order if i not in final_rejected]
        if not selected_indices:
            raise RuntimeError("ORLA-v9A trace-only ablation selected zero clients.")

        # No Stage 1 or Stage 2 decisions exist in this ablation.
        baseline_rejected = set()
        post_helper_rejected = set()
        promoted = []
        rescued = []
        spread_median = float("nan")
        safety_rejected = sorted(final_rejected)

        print(
            f"[ORLAA-Ablation-{self.v9_trace_only_mode}] round={server_round} "
            f"n={n} max_remove={max_remove} actual_remove={len(final_rejected)} "
            f"D_median={safety_median:.6g} z_thr={self.v9_safety_z_threshold:.3g} "
            f"ratio_thr={self.v9_safety_ratio_threshold:.3g}"
        )
        print(
            "[ORLAA-Ablation] " + " | ".join(
                f"{self._v9_stable_client_id(records[i]['client'], records[i]['fit_res'])}:"
                f"D={records[i]['safety_gate_score']:.6g},"
                f"z={records[i]['safety_gate_z']:.2f},"
                f"ratio={records[i]['safety_gate_ratio_to_median']:.2f},"
                f"{'REJECT' if i in final_rejected else 'KEEP'}"
                for i in score_order
            )
        )


        # Average the clients retained by BA-distance ranking.
        selected = [records[i] for i in selected_indices]
        num_layers = len(selected[0]["fit_res"].parameters.tensors)
        aggregated = []
        for layer_idx in range(num_layers):
            aggregated.append(np.mean(
                [rec["weights"][layer_idx] for rec in selected], axis=0
            ))

        result = (ndarrays_to_parameters(aggregated), {})
        self._pfl_after_aggregate_update(server_round, result)
        return result


# ====================================================================================================================================================== #

class KrumWithLogging(_SameRoundForgeMixin, _PoisonedFLForgeMixin, DeterministicClientSelectionMixin , Krum):
    """Krum strategy that logs untrusted (excluded) client updates per round."""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[str, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        self._maybe_forge_same_round(server_round, results)  # MinMax/MinSum
        self._maybe_forge_pfl(server_round, results)         # PoisonedFL

        # Use standard Krum to select best client
        aggregated_result = super().aggregate_fit(server_round, results, failures)

        # Krum selects one index based on scoring
        selected_cid = aggregated_result[1].get("krum_selected_cid", None)

        # If 'krum_selected_cid' is not provided, infer it
        if selected_cid is None:
            selected_cid = results[0][0]  # fallback to the first one if not available

        # Identify and log suspicious clients (not selected)
        suspicious_clients = [cid for cid, _ in results if cid != selected_cid]

        # return aggregated_result
        self._pfl_after_aggregate_update(server_round, aggregated_result)
        return aggregated_result

# ====================================================================================================================================================== #

class MultiKrum(_SameRoundForgeMixin, _PoisonedFLForgeMixin, DeterministicClientSelectionMixin, FedAvg):
    
    # Select and average the m clients with lowest Krum scores.

    def __init__(
        self,
        *,
        num_malicious_clients: int,
        m: int | None = None,  # if None, we pick a conservative default per round
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_malicious_clients = int(num_malicious_clients)
        self.m = None if m is None else int(m)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ):
        # Allow same-round forging hooks (MinMax/MinSum, PoisonedFL)
        self._maybe_forge_same_round(server_round, results)
        self._maybe_forge_pfl(server_round, results)

        if not results:
            return None, {}

        # Convert client parameters to flat vectors
        client_params = [parameters_to_ndarrays(fr.parameters) for _, fr in results]
        flat = [np.concatenate([w.ravel() for w in ws]).astype(np.float32) for ws in client_params]
        X = np.vstack(flat)  # shape: (n, d)
        n = X.shape[0]
        f = int(self.num_malicious_clients)

        # condition check
        if 2 * f + 2 >= n:
            print(f"[MultiKrum] WARNING: violates 2f+2<n (n={n}, f={f}). Falling back to FedAvg this round.")
            avg = [np.mean([ws[layer] for ws in client_params], axis=0) for layer in range(len(client_params[0]))]
            result = (ndarrays_to_parameters(avg), {})
            self._pfl_after_aggregate_update(server_round, result)
            return result

          # Use n - f - 2 neighbors in the Krum score.
        k_neighbors = n - f - 2
        if k_neighbors <= 0:
            print(f"[MultiKrum] k_neighbors={k_neighbors} <= 0. Falling back to FedAvg.")
            avg = [np.mean([ws[layer] for ws in client_params], axis=0) for layer in range(len(client_params[0]))]
            result = (ndarrays_to_parameters(avg), {})
            self._pfl_after_aggregate_update(server_round, result)
            return result

        # Compute pairwise squared distances: O(n^2 d)
        diffs = X[:, None, :] - X[None, :, :]
        d2 = np.sum(diffs * diffs, axis=2)  # (n, n)
        np.fill_diagonal(d2, np.inf)       # exclude self

        # Krum score for each i: sum of its k_neighbors smallest distances
        # Use partition to avoid full sort cost
        scores = np.sum(np.partition(d2, kth=k_neighbors-1, axis=1)[:, :k_neighbors], axis=1)

        # Decide m (how many winners to average)
        # Default: conservative m = n - f - 2 (common choice); you can pass absolute m or a ratio via config
        m = self.m
        if m is None:
            m = max(1, n - f - 2)
        elif 0 < m < 1:
            # If user supplied a ratio (0<m<1), convert to count
            m = max(1, int(round(m * n)))
        m = min(m, n)  # safety

        # Select indices with smallest scores with a deterministic index tie-break.
        winner_idx = np.lexsort((np.arange(n), scores))[:m]

        # Average selected updates layer-wise
        num_layers = len(client_params[0])
        aggregated = []
        for layer in range(num_layers):
            layer_stack = np.stack([client_params[i][layer] for i in winner_idx], axis=0)
            aggregated.append(layer_stack.mean(axis=0))

        print(f"[MultiKrum] Round {server_round}: selected m={m}/{n} with k_neighbors={k_neighbors}.")
        # Optional: log scores summary

        result = (ndarrays_to_parameters(aggregated), {})
        self._pfl_after_aggregate_update(server_round, result)
        return result

# ====================================================================================================================================================== #

class TrimmedMean(_SameRoundForgeMixin, _PoisonedFLForgeMixin, DeterministicClientSelectionMixin , FedAvg):
    def __init__(self, *, trim_ratio: float, **kwargs):
        super().__init__(**kwargs)
        self.trim_ratio = trim_ratio  # e.g., 0.1 means trim top/bottom 10%

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Parameters]:

        self._maybe_forge_same_round(server_round, results)  # MinMax/MinSum
        self._maybe_forge_pfl(server_round, results)         # PoisonedFL

        if not results:
            return None

        # Extract model updates
        weights: List[np.ndarray] = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]
        
        # Transpose to get per-layer updates
        num_layers = len(weights[0])
        num_clients = len(weights)
        trimmed_weights = []

        for layer_idx in range(num_layers):
            # Get all client updates for the current layer
            layer_updates = np.array([weights[i][layer_idx] for i in range(num_clients)])

            # Flatten to 2D (clients, values)
            shape = layer_updates[0].shape
            flat_layer = layer_updates.reshape(num_clients, -1)

            # Trim top/bottom based on norm across clients
            # Stable sorting plus float64 accumulation makes the reduction independent
            # of asynchronous arrival order and minimizes platform/thread reduction drift.
            # Cast back to the original parameter dtype before returning to Flower.
            original_dtype = flat_layer.dtype
            sorted_vals = np.sort(flat_layer, axis=0, kind="stable")
            k = int(self.trim_ratio * num_clients)
            if k * 2 >= num_clients:
                raise ValueError("Too many clients trimmed. Reduce `trim_ratio`.")

            trimmed = sorted_vals[k : num_clients - k]
            mean_trimmed = np.mean(trimmed.astype(np.float64, copy=False), axis=0)
            trimmed_weights.append(mean_trimmed.astype(original_dtype, copy=False).reshape(shape))

        result = (ndarrays_to_parameters(trimmed_weights), {})
        self._pfl_after_aggregate_update(server_round, result)
        return result

# ====================================================================================================================================================== #


class FLTrustStrategy(_SameRoundForgeMixin, _PoisonedFLForgeMixin, DeterministicClientSelectionMixin, FedAvg):
    def __init__(self, *, server_update: Optional[List[np.ndarray]] = None, context=None, **kwargs):
        super().__init__(**kwargs)

        if context is None:
            raise ValueError("Context is required.")

        self.model_name = context.run_config["model-name"]

        global PROBLEM_TYPE
        PROBLEM_TYPE = str(context.run_config.get("problem_type", "seq_cls")).lower()

        self.dataset_name = context.run_config["dataset_name"]
        self.num_labels = int(context.run_config["num_labels"])
        self.base_seed = int(context.run_config.get("base_seed", 42))
        self.client_number_of_samples = int(
            context.run_config.get("number_of_samples", 0)
        )

        self.base_params = None

        self.trusted_loader = self._load_fltrust_data(
            dataset_name=self.dataset_name,
            model_name=self.model_name,
            num_samples=FLTRUST_SEQCLS_ROOT_SAMPLES,
            seed=42
        )

    # ------------------------------------------------------------------------------------------------------------- #

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Tuple[Parameters, Dict[str, Scalar]]]:

        # Existing poisoning hooks (keep as-is)
        self._maybe_forge_same_round(server_round, results)  # MinMax/MinSum
        self._maybe_forge_pfl(server_round, results)         # PoisonedFL

        if not results:
            return None

        # Get current global parameters (Δ_t) and server trusted update g_s
        base_params: List[np.ndarray] = self.base_params  

        server_update = self.get_server_trusted_update(
            model_name=self.model_name,
            dataset_name=self.dataset_name,
            num_labels=self.num_labels,
            seed=self.base_seed + int(server_round),
        )  # g_s

        # Convert client parameters to ndarray (Δ_i)
        client_updates: List[Tuple[ClientProxy, List[np.ndarray]]] = [
            (client, parameters_to_ndarrays(fit_res.parameters))
            for client, fit_res in results
        ]

        # Compute client updates relative to global: g_i = Δ_i - Δ_t
        relative_updates: List[Tuple[ClientProxy, List[np.ndarray]]] = []
        for client, weights in client_updates:
            update = [cw - bw for cw, bw in zip(weights, base_params)]
            relative_updates.append((client, update))

        # FLTrust norm clipping:  ĝ_i = g_i * min(1, ||g_s|| / ||g_i||)
        def _clip_update(g_i: List[np.ndarray], g_s: List[np.ndarray]) -> List[np.ndarray]:
            flat_gi = self._flatten(g_i)
            flat_gs = self._flatten(g_s)
            norm_gi = np.linalg.norm(flat_gi)
            norm_gs = np.linalg.norm(flat_gs)
            if norm_gi == 0.0:
                return [np.zeros_like(x) for x in g_i]

            scale = min(1.0, norm_gs / (norm_gi + 1e-12))
            return [x * scale for x in g_i]

        clipped_updates: List[Tuple[ClientProxy, List[np.ndarray]]] = []
        for client, g_i in relative_updates:
            g_hat = _clip_update(g_i, server_update)
            clipped_updates.append((client, g_hat))

        # Trust scores: t_i = max(0, cos(ĝ_i, g_s))
        similarities = []
        for client, g_hat in clipped_updates:
            sim = self._cosine_similarity(
                self._flatten(g_hat),
                self._flatten(server_update),
            )
            similarities.append(max(0.0, float(sim)))  # ReLU

        similarities = np.array(similarities, dtype=float)
        if np.sum(similarities) == 0.0:
            similarities = np.ones_like(similarities)

        weights = similarities / (np.sum(similarities) + 1e-12)

        # Aggregate in UPDATE space and add to base: Δ_{t+1} = Δ_t + Σ α_i ĝ_i
        num_layers = len(base_params)
        aggregated: List[np.ndarray] = []
        for layer in range(num_layers):
            # Σ_i α_i ĝ_i[layer]
            delta_layer = sum(
                w * g_hat[layer]
                for w, (_, g_hat) in zip(weights, clipped_updates)
            )
            # Δ_t + aggregated update
            new_layer = base_params[layer] + delta_layer
            aggregated.append(new_layer)

        self.base_params = aggregated  # store new global parameters

        result = (ndarrays_to_parameters(aggregated), {})
        self._pfl_after_aggregate_update(server_round, result)
        return result


    # ------------------------------------------------------------------------------------------------------------- #

    def initialize_parameters(self, client_manager):
        params = super().initialize_parameters(client_manager)
        if params is not None:
            self.base_params = parameters_to_ndarrays(params)
        return params

    # ------------------------------------------------------------------------------------------------------------- #

    def get_server_trusted_update(self, model_name: str, dataset_name: str, num_samples: int = 100, seed: int = 42, num_labels: int = 2,
       ) -> List[np.ndarray]:
        """
        Compute the trusted server update g_s for FLTrust.

            1. Start from the CURRENT global model (self.base_params).
            2. Train on the trusted root dataset.
            3. Return g_s = after - before.
        """

        import torch
        import numpy as np

        set_global_determinism(int(seed))

        # ---------------------- Build model and load global params --------------------------

        model = get_model_pt(model_name, num_labels)

        # Load current global LoRA/full parameters if available
        if self.base_params is not None:
            set_params(model, self.base_params)

        model.train()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Only train parameters which actually require gradients (LoRA + classifier)
        optimizer = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad),
            lr=1e-3,
            foreach=False,
            fused=False,
        )

        # ------------------- Record parameters before trusted training ----------------------

        w_before = get_params(model)
        flat_before = np.concatenate([w.flatten() for w in w_before])
        
        # ----------------------------- Trusted training loop --------------------------------

        dataset = self.trusted_loader 

        total_loss = 0.0
        n_batches = 0

        for batch in dataset:
            n_batches += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
 
        # ---------------------- Record parameters after trusted training --------------------

        w_after = get_params(model)
        flat_after = np.concatenate([w.flatten() for w in w_after])

        # ----------------------------- Compute trusted update g_s ---------------------------

        server_update = [after - before for after, before in zip(w_after, w_before)]
        flat_update = np.concatenate([u.flatten() for u in server_update])

        return server_update


    # ------------------------------------------------------------------------------------------------------------- #

    @staticmethod
    def _flatten(ndarrays: List[np.ndarray]) -> np.ndarray:
        return np.concatenate([arr.ravel() for arr in ndarrays])

    # ------------------------------------------------------------------------------------------------------------- #

    @staticmethod
    def _cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
        dot = np.dot(x, y)
        norm = np.linalg.norm(x) * np.linalg.norm(y)
        return dot / norm if norm > 0 else 0.0

    # ------------------------------------------------------------------------------------------------------------- #

    def _load_fltrust_data(self, dataset_name: str, model_name: str, num_samples: int, seed: int):
        """
            Build the trusted/root DataLoader for FLTrust.

                For seq_cls tasks we sample from the training split using Dirichlet(label) sampling.
                For multiple-choice QA we support:
                - SWAG
                - PIQA

            Extractive QA uses the existing QA tokenization schema:
            {id, question, context, answers}.
        """

        from torch.utils.data import DataLoader
        from datasets import load_dataset
        from transformers import AutoTokenizer, DataCollatorWithPadding, DefaultDataCollator, DataCollatorForMultipleChoice

        # Decide task mode
        ds_name_l = str(dataset_name).lower()
        problem_type = str(globals().get("PROBLEM_TYPE", "seq_cls")).lower()
        is_qa = (problem_type == "qa")
        is_mcqa = (problem_type == "mc_qa") or (ds_name_l in {"swag", "piqa"})
        subset = None

        # ----------------------------- Load raw dataset ----------------------------- #
        if is_qa:
            raise ValueError(f"Unsupported extractive QA dataset: {dataset_name}")

        elif is_mcqa:
            def _swag_stem_type(sent2: str) -> int:
                import re

                wh_words = ("what", "why", "how", "when", "where", "who", "which")
                aux_words = {
                    "is", "are", "was", "were", "do", "does", "did", "can", "could",
                    "would", "should", "will", "may", "might", "must", "has", "have", "had",
                }

                text = (sent2 or "").strip().lower()
                m = re.match(r"[a-z]+", text)
                first = m.group(0) if m else ""
                if first in wh_words:
                    return wh_words.index(first)
                if first in aux_words:
                    return len(wh_words)
                return len(wh_words) + 1

            def _piqa_goal_intent(goal: str) -> int:
                text = (goal or "").strip().lower()

                keyword_groups = [
                    ("clean_wash", ("clean", "wash", "wipe", "dry", "rinse", "scrub", "polish")),
                    ("fix_repair", ("fix", "repair", "mend", "patch", "restore")),
                    ("open_close", ("open", "close", "shut", "lock", "unlock", "seal", "unseal")),
                    ("move_carry", ("carry", "move", "transport", "lift", "pull", "push", "drag", "load", "unload")),
                    ("cook_food", ("cook", "bake", "boil", "fry", "grill", "heat", "eat", "drink")),
                    ("cut_break", ("cut", "slice", "break", "tear", "crack", "crush", "chop")),
                    ("attach_connect", ("attach", "connect", "tie", "glue", "tape", "fasten", "hang", "mount", "install")),
                ]

                for idx, (_name, words) in enumerate(keyword_groups):
                    if any(word in text for word in words):
                        return idx
                return len(keyword_groups)

            if dataset_name in ["SWAG", "swag"]:
                raw_dataset = load_dataset_cache_first("swag", "regular", split="train")

                raw_dataset = raw_dataset.rename_column("label", "mc_label")
                pseudo_labels = [int(_swag_stem_type(x)) for x in raw_dataset["sent2"]]
                raw_dataset = raw_dataset.add_column("label", pseudo_labels)

                keep_cols = {"sent1", "sent2", "ending0", "ending1", "ending2", "ending3", "label", "mc_label"}
                drop_cols = [c for c in raw_dataset.column_names if c not in keep_cols]
                if drop_cols:
                    raw_dataset = raw_dataset.remove_columns(drop_cols)

            elif dataset_name in ["PIQA", "piqa"]:
                raw_dataset = load_dataset_cache_first("ybisk/piqa", split="train")

                raw_dataset = raw_dataset.rename_column("label", "mc_label")
                pseudo_labels = [int(_piqa_goal_intent(x)) for x in raw_dataset["goal"]]
                raw_dataset = raw_dataset.add_column("label", pseudo_labels)

                keep_cols = {"goal", "sol1", "sol2", "label", "mc_label"}
                drop_cols = [c for c in raw_dataset.column_names if c not in keep_cols]
                if drop_cols:
                    raw_dataset = raw_dataset.remove_columns(drop_cols)

            else:
                raise ValueError(f"Unsupported multiple-choice QA dataset: {dataset_name}")

        else:
            # Sequence classification uses the exact same deterministic root
            # selector that is also used to REMOVE these rows before client
            # subsampling/partitioning. This prevents FLTrust data leakage.
            subset, reserved_source_indices, root_digest = _get_seqcls_fltrust_root_dataset(
                dataset_name,
                seed=int(seed),
                client_number_of_samples=self.client_number_of_samples,
                num_samples=int(num_samples),
            )
            print(
                f"[FLTrust] trusted root selected from reserved rows: "
                f"n={len(subset)}, overlap_with_clients=0, root_digest={root_digest[:16]}"
            )

        # QA/MCQA retain the existing trusted-root sampler. Sequence
        # classification already has `subset` from the shared reservation helper.
        if subset is None:
            # -------------------- Dirichlet(label) sampling for trusted/root set (with min-per-class) --------------------
        
            dirichlet_alpha = 0.5
            min_per_class = 5

            raw_dataset = raw_dataset.shuffle(seed=int(seed))

            rng = np.random.default_rng(int(seed))

            # Collect label values
            labels = np.array(raw_dataset["label"], dtype=int)

            # Keep only valid labels (ignore -1 if any)
            valid_mask = labels >= 0
            valid_indices = np.where(valid_mask)[0]
            labels_valid = labels[valid_mask]

            classes = np.unique(labels_valid)
            num_classes = len(classes)
            if num_classes == 0:
                raise ValueError("Trusted/root set sampling failed: no valid labels found.")

            # Build index list per class
            indices_by_class = {
                c: valid_indices[np.where(labels_valid == c)[0]].tolist()
                for c in classes
            }

            # Dirichlet proportions over classes (alpha=5 => near-IID-ish)
            alpha_vec = np.ones(num_classes, dtype=float) * float(dirichlet_alpha)
            proportions = rng.dirichlet(alpha_vec)

            # --- Enforce a minimum per class ---
            min_per_class = int(min_per_class)
            required = min_per_class * num_classes
            if required > num_samples:
                raise ValueError(
                    f"min_per_class={min_per_class} too large for num_samples={num_samples} "
                    f"and num_classes={num_classes}"
                )

            # Allocate counts: start with minimum, then distribute remaining via multinomial
            counts = np.full(num_classes, min_per_class, dtype=int)
            remaining = num_samples - required
            if remaining > 0:
                counts += rng.multinomial(remaining, proportions)

            # Sample without replacement per class
            chosen = []
            for c, k in zip(classes, counts):
                pool = indices_by_class[c]
                if len(pool) == 0 or k <= 0:
                    continue
                k_eff = min(int(k), len(pool))
                chosen.extend(rng.choice(pool, size=k_eff, replace=False).tolist())

            # If we picked fewer than num_samples (due to small class pools), top up uniformly from remaining
            if len(chosen) < num_samples:
                remaining_pool = sorted(set(valid_indices.tolist()) - set(chosen))
                if len(remaining_pool) > 0:
                    extra = rng.choice(
                        remaining_pool,
                        size=min(num_samples - len(chosen), len(remaining_pool)),
                        replace=False,
                    ).tolist()
                    chosen.extend(extra)

            # Final subset: deterministic ordering, exact size
            chosen = sorted(chosen[:num_samples])
            subset = raw_dataset.select(chosen)
            # -------------------------------------------------------------------------------------------------------------


        # ---------------------------- Tokenize + DataLoader ---------------------------- #
        tokenizer = tokenizer_from_pretrained_cache_first(model_name, model_max_length=512)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

        if is_qa:
            # Build train features with start/end positions so loss is defined
            max_length = 384
            doc_stride = 128

            def _prepare_train_features_squad(examples):
                questions = [q.lstrip() for q in examples["question"]]
                tokenized = tokenizer(
                    questions,
                    examples["context"],
                    truncation="only_second",
                    max_length=max_length,
                    stride=doc_stride,
                    return_overflowing_tokens=True,
                    return_offsets_mapping=True,
                    padding="max_length",
                )

                sample_mapping = tokenized.pop("overflow_to_sample_mapping")
                offset_mapping = tokenized.pop("offset_mapping")

                start_positions = []
                end_positions = []

                for i, offsets in enumerate(offset_mapping):
                    input_ids = tokenized["input_ids"][i]
                    cls_index = input_ids.index(tokenizer.cls_token_id)

                    seq_ids = tokenized.sequence_ids(i)
                    sample_index = sample_mapping[i]
                    answers = examples["answers"][sample_index]

                    if len(answers.get("answer_start", [])) == 0:
                        start_positions.append(cls_index)
                        end_positions.append(cls_index)
                        continue

                    start_char = answers["answer_start"][0]
                    end_char = start_char + len(answers["text"][0])

                    token_start_index = 0
                    while seq_ids[token_start_index] != 1:
                        token_start_index += 1
                    token_end_index = len(input_ids) - 1
                    while seq_ids[token_end_index] != 1:
                        token_end_index -= 1

                    if not (
                        offsets[token_start_index][0] <= start_char
                        and offsets[token_end_index][1] >= end_char
                    ):
                        start_positions.append(cls_index)
                        end_positions.append(cls_index)
                        continue

                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    start_positions.append(token_start_index - 1)

                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    end_positions.append(token_end_index + 1)

                tokenized["start_positions"] = start_positions
                tokenized["end_positions"] = end_positions
                return tokenized

            subset = subset.map(
                _prepare_train_features_squad,
                batched=True,
                remove_columns=subset.column_names,
            )

            collator = DefaultDataCollator()
            loader = DataLoader(subset, batch_size=8, shuffle=False, collate_fn=collator)
            return loader

        if is_mcqa:
            def _prepare_swag_features(examples):
                ending_names = ["ending0", "ending1", "ending2", "ending3"]
                first_sentences = [[context] * 4 for context in examples["sent1"]]
                question_headers = examples["sent2"]
                second_sentences = [
                    [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
                ]
                first_sentences = sum(first_sentences, [])
                second_sentences = sum(second_sentences, [])
                tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
                out = {k: [v[i: i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
                if "mc_label" in examples:
                    out["label"] = examples["mc_label"]
                elif "label" in examples:
                    out["label"] = examples["label"]
                return out

            def _prepare_piqa_features(examples):
                first_sentences = [[goal, goal] for goal in examples["goal"]]
                second_sentences = [[examples["sol1"][i], examples["sol2"][i]] for i in range(len(examples["goal"]))]
                first_sentences = sum(first_sentences, [])
                second_sentences = sum(second_sentences, [])
                tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
                out = {k: [v[i: i + 2] for i in range(0, len(v), 2)] for k, v in tokenized_examples.items()}
                if "mc_label" in examples:
                    out["label"] = examples["mc_label"]
                elif "label" in examples:
                    out["label"] = examples["label"]
                return out

            if dataset_name in ["SWAG", "swag"]:
                preprocess_fn = _prepare_swag_features
            elif dataset_name in ["PIQA", "piqa"]:
                preprocess_fn = _prepare_piqa_features
            else:
                raise ValueError(f"Unsupported multiple-choice QA dataset: {dataset_name}")

            subset = subset.map(
                preprocess_fn,
                batched=True,
                remove_columns=subset.column_names,
            )

            collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
            loader = DataLoader(subset, batch_size=8, shuffle=False, collate_fn=collator)
            return loader

        # ---------------------------- seq_cls pipeline ---------------------------- #
        def tokenize_fn(batch):
            return tokenizer(batch["text"], truncation=True, padding=True)

        subset = subset.map(tokenize_fn, batched=True)
        subset = subset.remove_columns("text")
        subset = subset.rename_column("label", "labels")

        collator = DataCollatorWithPadding(tokenizer)
        loader = DataLoader(subset, batch_size=32, shuffle=False, collate_fn=collator)
        return loader

# ====================================================================================================================================================== #


class DnC(_SameRoundForgeMixin, _PoisonedFLForgeMixin, DeterministicClientSelectionMixin , FedAvg):
    def __init__(self, *, trim_ratio: float, projection_dim: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.trim_ratio = trim_ratio
        self.projection_dim = projection_dim

    def safe_svd(self, X: np.ndarray, *, max_retries: int = 5, eps: float = 1e-4, clip_threshold: float = 1e3, seed: int = 42):
    # def safe_svd(self, X, max_retries=5, eps=1e-4, clip_threshold=1e3):
        """
            Perform SVD safely by handling convergence failures using perturbation.

            Parameters:
                X (np.ndarray): Centered input matrix.
                max_retries (int): Number of retry attempts if SVD fails.
                eps (float): Noise scale for perturbation.
                clip_threshold (float or None): Optional clipping threshold.

            Returns:
                u, s, vh: Results from SVD.
        """
        X_mod = X.copy()

        if clip_threshold is not None:
            X_mod = np.clip(X_mod, -clip_threshold, clip_threshold)

        for i in range(max_retries):
            try:
                u, s, vh = np.linalg.svd(X_mod, full_matrices=False)
                return u, s, vh
            except np.linalg.LinAlgError:
                retry_rng = np.random.default_rng(int(seed) + i)
                noise = eps * retry_rng.standard_normal(X_mod.shape)
                X_mod += noise
                print(f"[safe_svd] Retry {i+1}/{max_retries} after SVD failure")

        raise np.linalg.LinAlgError("SVD did not converge after multiple retries")


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Tuple[Parameters, Dict[str, Scalar]]]:

        self._maybe_forge_same_round(server_round, results)  # MinMax/MinSum
        self._maybe_forge_pfl(server_round, results)         # PoisonedFL

        if not results:
            return None

        client_updates = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

        # Flatten updates
        flattened_updates = [np.concatenate([w.flatten() for w in update]) for update in client_updates]
        update_matrix = np.vstack(flattened_updates)

        # Step 1: Dimensionality reduction
        if update_matrix.shape[1] > self.projection_dim:
            rng = np.random.default_rng(int(getattr(self, "base_seed", 42)) + int(server_round))
            random_matrix = rng.standard_normal((update_matrix.shape[1], self.projection_dim))

            # random_matrix = np.random.randn(update_matrix.shape[1], self.projection_dim)
            update_matrix_reduced = update_matrix @ random_matrix
        else:
            update_matrix_reduced = update_matrix

        # Step 2: SVD to find principal component
        # u, s, vh = np.linalg.svd(update_matrix_reduced - update_matrix_reduced.mean(axis=0), full_matrices=False)
        X_centered = update_matrix_reduced - update_matrix_reduced.mean(axis=0)


        # u, s, vh = self.safe_svd(X_centered, max_retries=5, eps=1e-4, clip_threshold=1e3, seed=int(getattr(self, "base_seed", 42)) + int(server_round))
        try:
            u, s, vh = self.safe_svd(
                X_centered, max_retries=5, eps=1e-4, clip_threshold=1e3,
                seed=int(getattr(self, "base_seed", 42)) + int(server_round),
            )
        except np.linalg.LinAlgError:
            print(f"[DnC] Skipping round {server_round} due to SVD failure.")
            return None, {} # Skip this round


        principal_direction = vh[0]

        # Step 3: Project *centered* updates onto principal direction (paper: <x_i - mean, v>)
        projections = X_centered @ principal_direction
        scores = projections ** 2  # paper uses squared projection as outlier score

        # Step 4: Remove top trim_ratio fraction of scores (assumed malicious)
        num_trim = int(self.trim_ratio * len(scores))
        indices_sorted = np.argsort(scores)  # keep smallest scores
        reliable_indices = indices_sorted[:len(scores) - num_trim]

        # Step 5: Aggregate remaining updates
        selected_updates = [client_updates[i] for i in reliable_indices]

        # Average layer-wise
        num_layers = len(selected_updates[0])
        averaged = [
            np.mean([update[layer] for update in selected_updates], axis=0)
            for layer in range(num_layers)
        ]

        result = (ndarrays_to_parameters(averaged) , {})
        self._pfl_after_aggregate_update(server_round , result)
        return result

# ====================================================================================================================================================== #

class IdealDefenseStrategy(DeterministicClientSelectionMixin, FedAvg):
    def __init__(self, *, context: Context, **kwargs):
        super().__init__(**kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Tuple[Parameters, Dict[str, Scalar]]]:

        _sort_results_inplace(results)
        if not results:
            return None, {}

        benign_results = []
        removed = []

        for client, fit_res in results:
            metrics = getattr(fit_res, "metrics", {}) or {}
            is_malicious = bool(int(metrics.get("I_am_malicious", 0)))  # flag from client
            part_id = metrics.get("client_cid", "?")

            if is_malicious:
                removed.append(part_id)
            else:
                benign_results.append((client, fit_res))

        print(f"[IdealDefense] Round {server_round}: removed malicious clients {removed}")

        if not benign_results:
            print("[IdealDefense] WARNING: all clients were malicious, falling back to all")
            benign_results = results

        # FedAvg over benign
        weights = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in benign_results]
        num_layers = len(weights[0])
        averaged = [
            np.mean([w[layer] for w in weights], axis=0)
            for layer in range(num_layers)
        ]

        return ndarrays_to_parameters(averaged), {}

# ====================================================================================================================================================== #

class FedDMCStrategy(_SameRoundForgeMixin, _PoisonedFLForgeMixin, DeterministicClientSelectionMixin, FedAvg):
    
    # Filter client updates using PCA, BTBCN clustering, and SEDC trust.


    def __init__(
        self,
        *,
        context,
        k: int = 10,                         # PCA dimension.
        min_cluster_size: int = 3,           # Minimum BTBCN cluster size.
        alpha: float = 0.7,                  # SEDC EMA weight (0<=alpha<1)
        threshold: float = 0.5,              # SEDC decision threshold (paper recommends 0.5)
        use_lora_only: bool = True,          # default aligns with LoRA-only training
        include_classifier: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.context = context
        self.k = int(k)
        self.min_cluster_size = int(min_cluster_size)
        self.alpha = float(alpha)
        self.threshold = float(threshold)
        self.use_lora_only = bool(use_lora_only)
        self.include_classifier = bool(include_classifier)

        # Initialize EMA trust to 0.5.
        self._trust = {}  # cid -> float

        self.model_name = context.run_config["model-name"]
        self.num_labels = int(context.run_config["num_labels"])
        self.lora_indices = None

    # ---------- helpers ----------
    @staticmethod
    def _flatten_all(nds):
        return np.concatenate([w.ravel() for w in nds]).astype(np.float32, copy=False)

    def _flatten_selected(self, nds):
        if not self.use_lora_only:
            return self._flatten_all(nds)
        try:
            if self.lora_indices is None:
                from LoRA_Sec.task import get_model
                net = get_model_pt(self.model_name, self.num_labels)
                # use the module-level helpers (not self.*)
                idx = _pfl_target_indices(net, self.include_classifier)
                self.lora_indices = idx
            return _pfl_flat_from_indices(nds, self.lora_indices).numpy().astype(np.float32, copy=False)
        except Exception:
            # conservative fallback
            return self._flatten_all(nds)

    def _pca_reduce(self, X: np.ndarray, k: int) -> np.ndarray:
        """Row=client, col=feature. Center then reduce to k dims."""
        if X.shape[0] <= 1:
            return X
        Xc = X - X.mean(axis=0, keepdims=True)
        k_eff = max(1, min(k, Xc.shape[1] - 1)) if Xc.shape[1] > 1 else 1
        if PCA is not None:
            try:
                return PCA(n_components=k_eff, svd_solver="auto", random_state=int(getattr(self, "_current_round_seed", getattr(self, "base_seed", 42)))).fit_transform(Xc)
            except Exception:
                pass
        # fallback: NumPy SVD
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        Vt_k = Vt[:k_eff]
        return Xc @ Vt_k.T

    def _btbcn_split(self, Z: np.ndarray, min_cluster_size: int):
        # Split a feature cluster using the BTBCN criterion.
        n = Z.shape[0]
        all_idx = list(range(n))
        active = all_idx.copy()
        noise = set()
        benign = set()
        malicious = set()

        def farthest_pair(idxs):
            # O(m^2) search (m<=participating clients per round, small)
            P = Z[idxs]
            # pairwise distances
            D = np.sum(P**2, axis=1, keepdims=True) + np.sum(P**2, axis=1) - 2 * (P @ P.T)
            np.fill_diagonal(D, -np.inf)
            i_rel, j_rel = divmod(np.argmax(D), D.shape[1])
            return idxs[i_rel], idxs[j_rel]

        cur = active
        while len(cur) >= 2:
            a, b = farthest_pair(cur)
            A, B = [], []
            za, zb = Z[a], Z[b]
            for u in cur:
                if np.linalg.norm(Z[u] - za) <= np.linalg.norm(Z[u] - zb):
                    A.append(u)
                else:
                    B.append(u)

            if len(A) < min_cluster_size:
                noise.update(A)
                cur = [u for u in cur if u not in noise]
                continue
            if len(B) < min_cluster_size:
                noise.update(B)
                cur = [u for u in cur if u not in noise]
                continue

            # Both sides sufficiently large -> decide
            if len(A) >= len(B):
                benign.update(A); malicious.update(B)
            else:
                benign.update(B); malicious.update(A)
            break

        # Any leftovers that never formed a valid split are noise -> treat as malicious (conservative)
        leftovers = set(all_idx) - benign - malicious - noise
        malicious.update(leftovers)
        noise.update(leftovers)

        is_benign = np.zeros(n, dtype=bool)
        if benign:
            is_benign[list(benign)] = True
        return is_benign

    # ---------- core aggregation ----------
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ):
        # allow attacks (MinMax/MinSum, PoisonedFL) to modify results consistently with testbed
        self._maybe_forge_same_round(server_round, results)
        self._maybe_forge_pfl(server_round, results)

        if not results:
            return None

        # 1) Collect client vectors (LoRA-only by default)
        cids = []
        mats = []
        raw_updates = []
        for client, fit_res in results:
            arrs = parameters_to_ndarrays(fit_res.parameters)
            flat = self._flatten_selected(arrs)
            mats.append(flat)
            raw_updates.append(arrs)
            # Use the stable logical partition id reported by the client.
            # Flower's proxy cid is a random per-run node id in simulation.
            metrics = getattr(fit_res, "metrics", {}) or {}
            cid = metrics.get("client_cid", None)
            if cid is None:
                raise RuntimeError(
                    "FedDMC requires the stable 'client_cid' metric for deterministic history tracking."
                )
            cids.append(str(cid))
        W = np.vstack(mats)  # [n_clients, d]

        # 2) PCA (DR)
        self._current_round_seed = int(getattr(self, "base_seed", 42)) + int(server_round)
        Z = self._pca_reduce(W, self.k)  # highlights benign/malicious separability
        # 3) BTBCN (binary tree with noise pruning)
        is_benign_bt = self._btbcn_split(Z, self.min_cluster_size)

        # 4) SEDC (EMA trust across rounds), init 0.5
        #    S_hat[t] = alpha * S_hat[t-1] + (1-alpha) * S[t], then threshold
        selected_results = []
        removed = []
        for (client, fit_res), benign_now, cid in zip(results, is_benign_bt, cids):
            prev = self._trust.get(cid, 0.5)
            cur = 1.0 if benign_now else 0.0
            score = self.alpha * prev + (1.0 - self.alpha) * cur
            self._trust[cid] = float(score)

        keep_mask = [self._trust[cid] >= self.threshold for cid in cids]
        for keep, pair, cid in zip(keep_mask, results, cids):
            if keep:
                selected_results.append(pair)
            else:
                removed.append(cid)

        print(f"[FedDMC] Round {server_round}: kept={len(selected_results)}/{len(results)}; removed={removed}")

        # fallback if everything was filtered
        if not selected_results:
            print("[FedDMC] WARNING: no clients passed SEDC threshold; using BTBCN benign cluster directly.")
            selected_results = [pair for (pair, ok) in zip(results, is_benign_bt) if ok]
        if not selected_results:
            print("[FedDMC] WARNING: BTBCN benign set empty; falling back to FedAvg over all.")
            selected_results = results

        # 5) FedAvg over selected (paper aggregates benign via FedAvg)
        weights = [parameters_to_ndarrays(fr.parameters) for _, fr in selected_results]
        num_layers = len(weights[0])
        averaged = [np.mean([w[layer] for w in weights], axis=0) for layer in range(num_layers)]
        result = (ndarrays_to_parameters(averaged), {})

        # PoisonedFL post-aggregate hook (keeps the PFL bookkeeping intact)
        self._pfl_after_aggregate_update(server_round, result)
        return result

# ====================================================================================================================================================== #

# ================================================================================================= #
# ShieldFL: cosine-weighted aggregation.
# Compare updates with the least-similar client and weight by 1 - cosine.
# The first round uses FedAvg until a previous global update is available.

class ShieldFLStrategy(_SameRoundForgeMixin, _PoisonedFLForgeMixin, DeterministicClientSelectionMixin, FedAvg):
    def __init__(self, *, use_lora_only: bool = True, include_classifier: bool = False, **kwargs):
        super().__init__(**kwargs)
        # Compute cosine similarity on the selected parameters (LoRA by default).
        self.use_lora_only = use_lora_only
        self.include_classifier = include_classifier
        self.prev_global_vec: Optional[np.ndarray] = None

    # Cosine-similarity helpers.
    def _unit(self, v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v)) + 1e-12
        return v / n

    def _cos(self, a: np.ndarray, b: np.ndarray) -> float:
        # a,b are assumed already normalized; still guard for safety
        denom = (np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)
        return float(np.dot(a, b) / denom)

    def _flat_for_cos(self, nds, base=None):
        """Flatten selected coords for cosine. If base is given, use deltas (nds - base)."""
        arrs = [a - b for a, b in zip(nds, base)] if base is not None else nds
        if getattr(self, "use_lora_only", True):
            # Build and cache LoRA(+classifier) indices once
            if not hasattr(self, "_shield_idx") or self._shield_idx is None:
                net = get_model_pt(self.model_name, self.num_labels)
                self._shield_idx = _pfl_target_indices(net, getattr(self, "include_classifier", False))
            return _pfl_flat_from_indices(arrs, self._shield_idx).numpy().astype(np.float32, copy=False)
        # fallback: all params
        return np.concatenate([a.ravel() for a in arrs]).astype(np.float32, copy=False)

    def _vec_from_layers(self, arrays: List[np.ndarray]) -> np.ndarray:
        # Prefer the shared helper if available in the codebase:
        try:
            return self._flatten_selected(arrays)
        except Exception:
            # Fallback: flatten everything
            return np.concatenate([a.ravel() for a in arrays])

    # --- core aggregation ---
    def aggregate_fit(self, server_round, results, failures):
        self._maybe_forge_same_round(server_round, results)
        self._maybe_forge_pfl(server_round, results)

        if not results:
            return None

        from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
        global LAST_BROADCAST, PFL_STATE

        # Base/broadcast model for this round
        base = LAST_BROADCAST
        if base is None:
            # First ever call: plain FedAvg
            params0 = [parameters_to_ndarrays(fr.parameters) for _, fr in results]
            num_layers = len(params0[0])
            avg_layers = [np.mean([p[l] for p in params0], axis=0) for l in range(num_layers)]
            result = (ndarrays_to_parameters(avg_layers), {})
            self._pfl_after_aggregate_update(server_round, result)
            return result

        # Collect per-client weights and deltas
        weights_i = []
        deltas_i = []
        vecs_i = []   # unit deltas for cosine
        for _, fr in results:
            w_i = parameters_to_ndarrays(fr.parameters)
            d_i = [w_i[l] - base[l] for l in range(len(w_i))]
            weights_i.append(w_i)
            deltas_i.append(d_i)
            v_i = self._unit(self._flat_for_cos(w_i, base=base))
            vecs_i.append(v_i)
        n = len(vecs_i)

        # Previous aggregated update g^{t-1}.
        if PFL_STATE.get("w_prev") is not None and PFL_STATE.get("w_prev2") is not None:
            g_prev = self._flat_for_cos(PFL_STATE["w_prev"], base=PFL_STATE["w_prev2"])
            g_prev_u = self._unit(g_prev)
        else:
            # Fallback: mean client delta this round
            g_prev_u = self._unit(np.mean(np.stack(vecs_i), axis=0))

        # Baseline: client with lowest cosine to g^{t-1}
        cos_prev = np.array([self._cos(v, g_prev_u) for v in vecs_i], dtype=float)
        baseline_idx = int(np.argmin(cos_prev))
        g_star = vecs_i[baseline_idx]

        # Weights: η_i ∝ (1 - cos(Δ_i, Δ_*)), normalize
        cos_star = np.array([self._cos(v, g_star) for v in vecs_i], dtype=float)
        eta = np.clip(1.0 - cos_star, 0.0, None)
        s = float(eta.sum())
        weights = (eta / s) if s > 1e-12 else (np.ones(n, dtype=float) / n)

        # Aggregate WEIGHTED DELTAS and add to base
        num_layers = len(base)
        agg = []
        for l in range(num_layers):
            delta_l = sum(float(w) * deltas_i[i][l] for i, w in enumerate(weights))
            agg.append(base[l] + delta_l)

        result = (ndarrays_to_parameters(agg), {})

        # Keep a normalized version of the aggregated delta for the next round (optional)
        self.prev_global_vec = self._unit(self._flat_for_cos(agg, base=base))

        try:
            kept_like = int((weights > (1.0 / n)).sum())
            print(f"[ShieldFL] r{server_round}: baseline_idx={baseline_idx}, "
                f"min_cos_prev={cos_prev[baseline_idx]:.4f}, "
                f"mean_eta={float(eta.mean()):.4f}, kept_like≈{kept_like}/{n}")
        except Exception:
            pass

        self._pfl_after_aggregate_update(server_round, result)
        return result


# ====================================================================================================================================================== #
# ================================================================ Server Function ===================================================================== #

def server_fn(context: Context) -> ServerAppComponents:

    global Config_Store
    global LAST_BROADCAST, PREV_BROADCAST, PFL_STATE
    global recent_accuracies, eval_history, avg10_blocks, window_summaries
    global CURRENT_EVAL_ROUND

    Config_Store = dict(context.run_config)

    # Reset every piece of module-level state at the start of a run. This matters
    # when Flower executes more than one run in a long-lived Python process.
    LAST_BROADCAST = None
    PREV_BROADCAST = None
    PFL_STATE = {
        "enabled": False,
        "s": None,
        "c": None,
        "beta": 0.7,
        "e": 50,
        "min_c": 0.5,
        "k_prev": None,
        "dim": None,
        "history_W": [],
        "w_prev": None,
        "w_prev2": None,
    }
    recent_accuracies = []
    eval_history = []
    avg10_blocks = []
    window_summaries = []
    CURRENT_EVAL_ROUND = None

    base_seed = int(context.run_config.get("base_seed", 42))
    set_global_determinism(base_seed)

    # Remove stale cross-client coordination artifacts from previous runs.
    # Per-client round cleanup was race-prone because one client could delete
    # files another client had just written.
    try:
        import shutil
        from pathlib import Path
        shared_dir = Path(context.run_config.get("fedimp_shared_dir", "./.fedimp_shared")).resolve()
        shutil.rmtree(shared_dir, ignore_errors=True)
        shared_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[Determinism] shared-dir cleanup warning: {e}")

    # Ensure server uses the same problem_type as clients BEFORE building init params
    global PROBLEM_TYPE
    PROBLEM_TYPE = str(context.run_config.get("problem_type", "seq_cls")).lower()

    """Construct components for ServerApp using Krum strategy with logging."""
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    model_name = context.run_config["model-name"]

    # Pre-build the deterministic sequence-classification partition cache once
    # in the server process. Without this, each Ray ClientAppActor can create its
    # own FederatedDataset for a new base_seed and independently query Hugging
    # Face Hub metadata before one actor finishes building the shared partition
    # cache. One server-side prewarm eliminates that request fan-out.
    if PROBLEM_TYPE == "seq_cls":
        # Materialize the persistent seed-independent raw dataset cache first.
        # Root reservation and client partition generation then share the same
        # stable raw ordering without concurrent Hub lookups.
        _get_seqcls_train_dataset(
            context.run_config["dataset_name"],
            number_of_samples=None,
            seed=42,
        )
        _prewarm_num_clients = int(context.run_config["number_of_clients"])
        _prewarm_train, _prewarm_test = load_data(
            partition_id=0,
            num_partitions=_prewarm_num_clients,
            model_name=model_name,
            partitioner_type=context.run_config["partitioner_type"],
            dataset_name=context.run_config["dataset_name"],
            partitioner_parameter=float(context.run_config["partitioner_parameter"]),
            number_of_samples=int(context.run_config["number_of_samples"]),
            problem_type=PROBLEM_TYPE,
            base_seed=base_seed,
        )
        del _prewarm_train, _prewarm_test
        print("[HF cache] deterministic seq-cls partition cache is ready before client sampling")

    ndarrays = get_params(get_model_pt(model_name , num_labels = int(context.run_config["num_labels"])))
    global_model_init = ndarrays_to_parameters(ndarrays)


    fraction_fit = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction-evaluate"]

    # --------------------------------------------------------------------- #
    # ------------------ Per-round malicious-client setup ----------------- #

    malicious_percentage = float(context.run_config["Malicious-Client-Percentage"])
    num_clients = int(context.run_config["number_of_clients"])
    # This value is only an initialization/default for strategies that require f.
    # configure_fit updates f to the actually selected malicious count each round.
    if malicious_percentage > 0.0 and num_clients > 0:
        num_malicious = min(
            num_clients,
            max(1, int(malicious_percentage * num_clients)),
        )
    else:
        num_malicious = 0

    print(f"[Determinism] base_seed={base_seed}")
    print(
        "[Determinism] malicious population is re-sampled from ALL logical clients "
        "with seed=round_number; participating clients are sampled independently "
        "with seed=base_seed+round_number"
    )

    # --------------------------------------------------------------------- #
    # -------------------------- Select Strategy -------------------------- #

    strategy_name = str(context.run_config.get("strategy", "")).strip()
    expected_active = max(1, int(fraction_fit * num_clients))
    print(
        f"[Strategy] requested={strategy_name!r}, "
        f"number_of_clients={num_clients}, expected_active={expected_active}"
    )

    if strategy_name == "FedAvg":
        strategy = DeterministicFedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            initial_parameters=global_model_init,
            evaluate_metrics_aggregation_fn=aggregate_evaluate,
        )

    elif strategy_name in {
        "ORLA",
        "ORLA-ablation1",
        "ORLA-ablation2",
        "ORLA-ablation3",
        "ORLA-ablation4",
        "ORLA-ablation5",
        "ORLA-ablation6",
        "ORLA-ablation7",
    }:
        # Each retained ablation preserves full ORLA settings except for the
        # explicitly named component. Numbering follows the seven experiments
        # retained for the final code release.
        a6_ablation_specs = {
            "ORLA-ablation1": {
                "label": "NoStage1-Matched",
                "enable_factor_stage": False,
                "matched_random_stage1": True,
            },
            "ORLA-ablation2": {
                "label": "NoStage2",
                "enable_spread_helper": False,
            },
            "ORLA-ablation3": {
                "label": "NoStage3",
                "safety_gate_mode": "none",
            },
            "ORLA-ablation4": {
                "label": "NoBootstrap",
                "safety_enable_bootstrap": False,
            },
            "ORLA-ablation5": {
                "label": "NoInfluenceCap",
                "enable_influence_cap": False,
            },
            "ORLA-ablation6": {
                "label": "NoExtremeResidualOverride",
                "enable_extreme_residual_override": False,
            },
            "ORLA-ablation7": {
                "label": "OnlyStage3-BAFixed",
                # BA-only fixed-ratio ranking: bypass Stage 1/2 entirely.
                "trace_only_mode": "fixed",
            },
        }
        a6_ablation = a6_ablation_specs.get(strategy_name, {})

        orla_ratio = float(context.run_config.get("orla_removal_ratio", 0.4))
        if not (0.0 <= orla_ratio < 1.0):
            raise ValueError(
                f"{strategy_name} expects orla_removal_ratio to be in [0,1), "
                f"e.g. 0.4. Got {orla_ratio!r}."
            )
        # ORLA/ablations interpret orla_removal_ratio as a PER-ROUND removal ratio.
        nominal_remove_count = int(orla_ratio * expected_active)
        if nominal_remove_count >= expected_active:
            raise ValueError(
                f"{strategy_name} nominal removal would be {nominal_remove_count} clients while about "
                f"{expected_active} participate per round. Reduce orla_removal_ratio."
            )

        # All ORLA ablations share the same default parameters.
        # Specific configuration keys override the common ORLA settings.
        safety_mode = str(a6_ablation.get(
            "safety_gate_mode", "trace_core_confirm_bidir_weighted_extreme"
        ))
        variant_key = "orla_v9a6"

        def _v9_cfg(name, default):
            # Use version-specific settings first; fall back to common settings.
            return context.run_config.get(
                f"{variant_key}_{name}",
                context.run_config.get(f"orla_v9_{name}", default),
            )

        # Stage-3-only uses an independent BA-ranking selection path.
        # All other configurations use the full ORLA pipeline.
        v9_strategy_class = (
            ORLATraceOnly if "trace_only_mode" in a6_ablation else ORLA
        )
        trace_only_kwargs = (
            {"trace_only_mode": a6_ablation["trace_only_mode"]}
            if "trace_only_mode" in a6_ablation else {}
        )
        strategy = v9_strategy_class(
            **trace_only_kwargs,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            initial_parameters=global_model_init,
            evaluate_metrics_aggregation_fn=aggregate_evaluate,
            model_name=model_name,
            num_labels=int(context.run_config["num_labels"]),
            m=nominal_remove_count,
            removal_ratio=orla_ratio,
            selection_mode=1,
            spread_z_threshold=float(
                context.run_config.get("orla_v9_spread_z_threshold", 1.0)
            ),
            spread_ratio_threshold=float(
                context.run_config.get("orla_v9_spread_ratio_threshold", 0.60)
            ),
            factor_protect_z=float(
                context.run_config.get("orla_v9_factor_protect_z", 3.0)
            ),
            max_helper_swaps=int(
                context.run_config.get(
                    f"{variant_key}_max_helper_swaps",
                    5,
                )
            ),
            enable_spread_helper=bool(
                a6_ablation.get("enable_spread_helper", True)
            ),
            enable_factor_stage=bool(
                a6_ablation.get("enable_factor_stage", True)
            ),
            matched_random_stage1=bool(
                a6_ablation.get("matched_random_stage1", False)
            ),
            safety_gate_mode=safety_mode,
            safety_z_threshold=float(_v9_cfg("safety_z_threshold", 4.0)),
            safety_ratio_threshold=float(_v9_cfg("safety_ratio_threshold", 3.0)),
            safety_max_extra_rejections=int(
                _v9_cfg(
                    "safety_max_extra_rejections",
                    0,
                )
            ),
            safety_core_fraction=float(
                _v9_cfg("safety_core_fraction", 0.50)
            ),
            safety_attack_scale_quantile=float(
                _v9_cfg("safety_attack_scale_quantile", 0.75)
            ),
            safety_attack_scale_ratio=float(
                _v9_cfg("safety_attack_scale_ratio", 0.80)
            ),
            safety_separation_gap_threshold=float(
                _v9_cfg("safety_separation_gap_threshold", 2.50)
            ),
            safety_rescue_ratio_threshold=float(
                _v9_cfg("safety_rescue_ratio_threshold", 2.00)
            ),
            safety_max_rescues=int(
                _v9_cfg("safety_max_rescues", 0)
            ),
            safety_bootstrap_rounds=int(
                _v9_cfg("safety_bootstrap_rounds", 10)
            ),
            safety_bootstrap_required_trusted_rounds=int(
                _v9_cfg("safety_bootstrap_required_trusted_rounds", 2)
            ),
            safety_bootstrap_z_threshold=float(
                _v9_cfg("safety_bootstrap_z_threshold", 4.0)
            ),
            safety_bootstrap_ratio_threshold=float(
                _v9_cfg("safety_bootstrap_ratio_threshold", 2.0)
            ),
            influence_cap_ratio=float(
                _v9_cfg("influence_cap_ratio", 3.0)
            ),
            influence_extreme_residual_ratio=float(
                _v9_cfg("influence_extreme_residual_ratio", 10.0)
            ),
            safety_use_core_reference=bool(
                a6_ablation.get("safety_use_core_reference", True)
            ),
            safety_require_attack_scale=bool(
                a6_ablation.get("safety_require_attack_scale", True)
            ),
            safety_require_separation=bool(
                a6_ablation.get("safety_require_separation", True)
            ),
            safety_enable_rescue=bool(
                a6_ablation.get("safety_enable_rescue", True)
            ),
            safety_enable_bootstrap=bool(
                a6_ablation.get("safety_enable_bootstrap", True)
            ),
            enable_influence_cap=bool(
                a6_ablation.get("enable_influence_cap", True)
            ),
            enable_extreme_residual_override=bool(
                a6_ablation.get("enable_extreme_residual_override", True)
            ),
            ablation_label=str(a6_ablation.get("label", "none")),
        )

    elif strategy_name == "ORLA-Flex":
        # ORLA-Flex uses adaptive Stage 1 with shared Stages 2 and 3.

        try:
            min_keep_float = float(context.run_config.get("orla_flex_min_retained", 6))
            min_keep = int(min_keep_float)
        except (TypeError, ValueError):
            raise ValueError(
                f"ORLA-Flex expects an integer orla_flex_min_retained, got {context.run_config.get('orla_flex_min_retained', 6)!r}."
            )
        if float(min_keep) != min_keep_float:
            raise ValueError(
                f"ORLA-Flex expects an integer minimum-retention count, got {context.run_config.get('orla_flex_min_retained', 6)!r}."
            )
        if not (1 <= min_keep <= expected_active):
            raise ValueError(
                f"ORLA-Flex orla_flex_min_retained must be between 1 and expected active clients "
                f"({expected_active}); got {min_keep}."
            )

        def _orla_flex_a6_cfg(name, default):
            # Match ORLA exactly: A6-specific key wins, then the common v9 key.
            return context.run_config.get(
                f"orla_v9a6_{name}",
                context.run_config.get(f"orla_v9_{name}", default),
            )

        strategy = ORLA(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            initial_parameters=global_model_init,
            evaluate_metrics_aggregation_fn=aggregate_evaluate,
            model_name=model_name,
            num_labels=int(context.run_config["num_labels"]),
            m=min_keep,
            removal_ratio=None,
            selection_mode=2,
            # Stage 2: EXACT ORLA settings.
            spread_z_threshold=float(
                context.run_config.get("orla_v9_spread_z_threshold", 1.0)
            ),
            spread_ratio_threshold=float(
                context.run_config.get("orla_v9_spread_ratio_threshold", 0.60)
            ),
            factor_protect_z=float(
                context.run_config.get("orla_v9_factor_protect_z", 3.0)
            ),
            max_helper_swaps=int(
                context.run_config.get("orla_v9a6_max_helper_swaps", 5)
            ),
            enable_spread_helper=True,
            enable_factor_stage=True,
            matched_random_stage1=False,
            # Stage 3: EXACT ORLA mode and defaults.
            safety_gate_mode="trace_core_confirm_bidir_weighted_extreme",
            safety_z_threshold=float(
                _orla_flex_a6_cfg("safety_z_threshold", 4.0)
            ),
            safety_ratio_threshold=float(
                _orla_flex_a6_cfg("safety_ratio_threshold", 3.0)
            ),
            safety_max_extra_rejections=int(
                _orla_flex_a6_cfg("safety_max_extra_rejections", 0)
            ),
            safety_core_fraction=float(
                _orla_flex_a6_cfg("safety_core_fraction", 0.50)
            ),
            safety_attack_scale_quantile=float(
                _orla_flex_a6_cfg("safety_attack_scale_quantile", 0.75)
            ),
            safety_attack_scale_ratio=float(
                _orla_flex_a6_cfg("safety_attack_scale_ratio", 0.80)
            ),
            safety_separation_gap_threshold=float(
                _orla_flex_a6_cfg("safety_separation_gap_threshold", 2.50)
            ),
            safety_rescue_ratio_threshold=float(
                _orla_flex_a6_cfg("safety_rescue_ratio_threshold", 2.00)
            ),
            safety_max_rescues=int(
                _orla_flex_a6_cfg("safety_max_rescues", 0)
            ),
            safety_bootstrap_rounds=int(
                _orla_flex_a6_cfg("safety_bootstrap_rounds", 10)
            ),
            safety_bootstrap_required_trusted_rounds=int(
                _orla_flex_a6_cfg("safety_bootstrap_required_trusted_rounds", 2)
            ),
            safety_bootstrap_z_threshold=float(
                _orla_flex_a6_cfg("safety_bootstrap_z_threshold", 4.0)
            ),
            safety_bootstrap_ratio_threshold=float(
                _orla_flex_a6_cfg("safety_bootstrap_ratio_threshold", 2.0)
            ),
            influence_cap_ratio=float(
                _orla_flex_a6_cfg("influence_cap_ratio", 3.0)
            ),
            influence_extreme_residual_ratio=float(
                _orla_flex_a6_cfg("influence_extreme_residual_ratio", 10.0)
            ),
            safety_use_core_reference=True,
            safety_require_attack_scale=True,
            safety_require_separation=True,
            safety_enable_rescue=True,
            safety_enable_bootstrap=True,
            enable_influence_cap=True,
            enable_extreme_residual_override=True,
            ablation_label="none",
        )

    elif strategy_name == "Krum":
        strategy = KrumWithLogging(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            initial_parameters=global_model_init,
            evaluate_metrics_aggregation_fn=aggregate_evaluate,
            num_malicious_clients=num_malicious,
        )

    elif strategy_name == "MultiKrum":
        m = context.run_config.get("multikrum_m", 6)
        f_override = context.run_config.get("multikrum_f", 6)
        strategy = MultiKrum(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            initial_parameters=global_model_init,
            evaluate_metrics_aggregation_fn=aggregate_evaluate,
            num_malicious_clients=int(f_override),
            m=m,
        )

    elif strategy_name == "TrimmedMean":
        strategy = TrimmedMean(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            initial_parameters=global_model_init,
            evaluate_metrics_aggregation_fn=aggregate_evaluate,
            trim_ratio=float(context.run_config.get("trimmed_mean_trim_ratio", 0.2)),
        )

    elif strategy_name == "FLTrust":
        strategy = FLTrustStrategy(
            context=context,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            initial_parameters=global_model_init,
            evaluate_metrics_aggregation_fn=aggregate_evaluate,
        )

    elif strategy_name == "DnC":
        strategy = DnC(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            initial_parameters=global_model_init,
            evaluate_metrics_aggregation_fn=aggregate_evaluate,
            trim_ratio=float(context.run_config.get("dnc_trim_ratio", 0.4)),
            projection_dim=100,
        )

    elif strategy_name == "IdealDefense":
        strategy = IdealDefenseStrategy(
            context=context,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            initial_parameters=global_model_init,
            evaluate_metrics_aggregation_fn=aggregate_evaluate,
        )


    elif strategy_name == "FedDMC":
        strategy = FedDMCStrategy(
            context=context,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            initial_parameters=global_model_init,
            evaluate_metrics_aggregation_fn=aggregate_evaluate,
            k=int(context.run_config.get("feddmc_k", 10)),
            min_cluster_size=int(context.run_config.get("feddmc_min_cluster_size", 3)),
            alpha=float(context.run_config.get("feddmc_alpha", 0.7)),
            threshold=float(context.run_config.get("feddmc_threshold", 0.5)),
            use_lora_only=bool(context.run_config.get("feddmc_use_lora_only", True)),
            include_classifier=bool(context.run_config.get("feddmc_include_classifier", False)),
        )

    elif strategy_name == "ShieldFL":
        strategy = ShieldFLStrategy(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            initial_parameters=global_model_init,
            evaluate_metrics_aggregation_fn=aggregate_evaluate,
            use_lora_only=bool(context.run_config.get("shieldfl_use_lora_only", True)),
            include_classifier=bool(context.run_config.get("shieldfl_include_classifier", False)),
        )

    else:
        raise ValueError(
            f"Unknown strategy {strategy_name!r}. Check [tool.flwr.app.config].strategy in pyproject.toml."
        )

    # Attach forging attributes (no-ops if attack not MinMax/MinSum)
    try:
        strategy.attack_type = context.run_config.get("attack_type", None)    # "MinMax"/"MinSum"/...
        strategy.client_dict = {}  # malicious status is assigned per-round via FitIns.config
        strategy.base_seed = base_seed
        strategy.malicious_percentage = malicious_percentage
        strategy.total_clients = num_clients
        strategy.model_name  = context.run_config["model-name"]
        strategy.num_labels  = int(context.run_config["num_labels"])
        # Pre-compute LoRA indices once
        param_names = list(get_model_pt(strategy.model_name, strategy.num_labels).state_dict().keys())
        strategy.lora_indices = [i for i, n in enumerate(param_names) if ("lora_A" in n or "lora_B" in n)]
        # Forging direction: std, uv or sign.
        strategy.rp_mode = context.run_config.get("rp_mode", "std")
        strategy.mixture_debug = bool(context.run_config.get("mixture_debug", True))
        _mix_default = ["RandScaling", "Trim", "MinMax", "MinSum", "PoisonedFL", "FedIMP"]
        _mcfg = context.run_config.get("mixture_choices", _mix_default)
        if isinstance(_mcfg, str):
            if _mcfg.strip() in {"*", "ALL", "all", "Auto", "auto"}:
                _mcfg = _mix_default
            else:
                _mcfg = [x.strip() for x in _mcfg.split(",") if x.strip()]
        strategy.mixture_choices = list(_mcfg or [])
        strategy.mixture_seed = int(context.run_config.get("mixture_seed", base_seed))
    except Exception as e:
        print(f"[server_fn] Could not attach forging attributes: {e}")

    # --------------------------------------------------------------------- #

    # Enable PoisonedFL defaults on the strategy
    strategy.pfl_c0    = context.run_config.get("pfl_c0", 8.0)
    strategy.pfl_beta  = context.run_config.get("pfl_beta", 0.7)
    strategy.pfl_e     = int(context.run_config.get("pfl_e", 50))
    strategy.pfl_min_c = context.run_config.get("pfl_min_c", 0.5)
    strategy.pfl_pval  = context.run_config.get("pfl_pval", 0.01)

    strategy.pfl_include_classifier = context.run_config.get("pfl_include_classifier", False)

    # Turn on the coordinator state if attack is PoisonedFL or SOPPoisonedFL
    try:
        mcfg = context.run_config.get("mixture_choices", [])
        if isinstance(mcfg, str):
            mcfg = [s.strip() for s in mcfg.split(",")] if mcfg.strip() else []
        PFL_STATE["enabled"] = (
            strategy.attack_type in {"PoisonedFL", "SOPPoisonedFL"}
            or (strategy.attack_type == "Mixture" and (
                "PoisonedFL" in mcfg or "SOPPoisonedFL" in mcfg
            ))
        )
    except Exception:
        pass

    # Pre-cache parameter names for _sop_apply_to_ndarrays (SOPPoisonedFL)
    try:
        strategy._pfl_param_names = list(
            get_model_pt(strategy.model_name, strategy.num_labels).state_dict().keys()
        )
    except Exception:
        pass


    # --------------------------------------------------------------------- #

    return ServerAppComponents(config=config, strategy=strategy)

# ====================================================================================================================================================== #

app = ServerApp(server_fn=server_fn)

atexit.register(print_final_results)
