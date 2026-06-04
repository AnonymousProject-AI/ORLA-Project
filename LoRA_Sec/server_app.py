"""LoRA-Sec: A Flower / Hugging Face app using Krum with logging."""

from flwr.common import Context, ndarrays_to_parameters, EvaluateIns, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import Krum
from flwr.common.typing import FitRes, Parameters, Scalar
from typing import List, Tuple, Optional, Dict
import random
import math
import numpy as np
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitIns

from LoRA_Sec.task import get_params, get_model, set_params, _get_seqcls_train_dataset, DataCollatorForMultipleChoice

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
# from collections.abc import Mapping

torch.manual_seed(42); np.random.seed(42); random.seed(42)

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
    """
    Compute FedIMP-style k-NN sum of squared distances for defense auditing.
    flat_updates: list[np.ndarray] each shape [D]
    n_total: number of selected clients this round
    m_compromised_guess: if unknown, pass 0; k will default to ceil(n/2)
    Returns dict with per-client LHS sums and global RHS sum.
    """
    U = np.stack(flat_updates, axis=0).astype(np.float64)  # [N, D]
    N = U.shape[0]
    if N < 2:
        return {"k": 1, "lhs": np.zeros(N, dtype=np.float64), "rhs_sum": 0.0}

    # k = ceil((n_t - m_t)/2); with unknown m, use ceil(N/2) as a defensive proxy
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

class DeterministicClientSelectionMixin:

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

        # Fix seed per round for deterministic sampling
        random.seed(42 + server_round)  # Seed can vary by round to keep it deterministic but not identical

        # Sample clients manually
        available_clients = list(client_manager.all().values()) 
        num_clients = int(self.fraction_fit * len(available_clients))

        # sampled_clients = random.sample(available_clients, num_clients)
        indices = random.sample(range(len(available_clients)), num_clients)
        sampled_clients = [available_clients[i] for i in indices]

        # Print selected client IDs
        print(f"[Round {server_round}] Selected client IDs:", indices)

        config = {"round_number": server_round}

        fit_ins = FitIns(parameters, config={"round_number": server_round})
  
        return [(client, fit_ins) for client in sampled_clients]

    # ---------------------------------------------------------------------------------------------------------------------------- #


    def configure_evaluate(self, server_round, parameters, client_manager):
        # --- NEW: evaluate only in the last K rounds of every W-round window ---
        cfg = globals().get("Config_Store", {}) or {}

        W = int(cfg.get("eval_window_rounds", 100))         # every 100 rounds
        K = int(cfg.get("eval_last_k_in_window", 10))       # last 10 rounds

        # Optional: if you have total rounds, handle the last (possibly partial) window nicely
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

        # Set a deterministic seed based on round number
        random.seed(42 + server_round)

        fraction_evaluate = self.fraction_evaluate

        available_clients = list(client_manager.all().values())
        num_clients = min(len(available_clients), int(self.fraction_evaluate * len(available_clients)))

        selected_clients = random.sample(available_clients, num_clients)

        # Print selected client IDs
        indexes = [available_clients.index(c) for c in selected_clients]

        global CURRENT_EVAL_ROUND
        CURRENT_EVAL_ROUND = server_round

        evaluate_ins = EvaluateIns(parameters, config={})
        return [(client, evaluate_ins) for client in selected_clients]


# ====================================================================================================================================================== #

def aggregate_evaluate(metrics):
    """Aggregate evaluation results.

    - seq_cls: averages accuracy (and loss if provided), keeps rolling last-10 accuracy.
    - qa: averages F1 / Exact Match (and loss), keeps rolling last-10 F1.
    """
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

    # Read evaluation window config (defaults match your gating in configure_evaluate)
    cfg = globals().get("Config_Store", {}) or {}
    W = int(cfg.get("eval_window_rounds", 100))
    K = int(cfg.get("eval_last_k_in_window", 10))
    total_rounds = int(cfg.get("num-server-rounds", 0))
    # Rolling average length (defaults to K; can be overridden by rolling_avg_k)
    rolling_k = int(cfg.get("rolling_avg_k", K if K > 0 else 10))

    loss_scores = [m["loss"] for _, m in metrics if "loss" in m]
    avg_loss = sum(loss_scores) / len(loss_scores) if loss_scores else 0.0


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
    """At exit, print and also APPEND results to a per-config text file.

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


    # --- Run config header ---
    lines = []
    lines.append("")  # leading newline to separate runs
    lines.append("[RUN CONFIG]")
    lines.append(f"attack_type: {cfg.get('attack_type', 'N/A')}")
    lines.append(f"strategy: {cfg.get('strategy', 'N/A')}")
    lines.append(f"dataset_name: {cfg.get('dataset_name', 'N/A')}")
    lines.append(f"partitioner_parameter: {cfg.get('partitioner_parameter', 'N/A')}")
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
            lines.append("")
        else:
            lines.append("[FINAL] No evaluation rounds were recorded.")
            lines.append("")

    # Print to console (keeps your current behavior)
    print("\n".join(lines))

    # Append to file
    try:
        with open(fname, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        print(f"[WARN] Could not write results to {fname}: {e}")

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
                # cid = getattr(client, "cid", None)
                # is_mal = bool(getattr(self, "client_dict", {}).get(str(cid), False)) if cid is not None else False

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
        if getattr(self, "attack_type", None) not in {"PoisonedFL", "Mixture"}:
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
            torch.manual_seed(1337)
            PFL_STATE["s"] = torch.sign(torch.randn(dim))
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

        # Overwrite every malicious client's LoRA delta with the same forged vector
        mixture = (getattr(self, "attack_type", None) == "Mixture")
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
            if mixture and m.get("mix_mode") != "PoisonedFL":
                continue
            mal_slots.append(idx)


        if not mal_slots:
            return

        for slot_idx in mal_slots:
            forged_nds = self._unflatten_into_lora(base, lora_idx, forged_flat)
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

        PFL_STATE["k_prev"] = k_t.detach().clone()  # remember k^t for next round

    def _pfl_after_aggregate_update(self, server_round, aggregated_result):
        """Update c via the binomial test on aggregated W_t; keep a length-(e+1) history."""
        if getattr(self, "attack_type", None) not in {"PoisonedFL", "Mixture"}:
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
# ====================================================================================================================================================== #


class DeterministicFedAvg(_SameRoundForgeMixin, _PoisonedFLForgeMixin, DeterministicClientSelectionMixin, FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        self._maybe_forge_same_round(server_round, results)  # MinMax/MinSum
        self._maybe_forge_pfl(server_round, results)         # PoisonedFL
        aggregated_result = super().aggregate_fit(server_round, results, failures)
        self._pfl_after_aggregate_update(server_round, aggregated_result)
        return aggregated_result

# ====================================================================================================================================================== #

class ORLA(_SameRoundForgeMixin, _PoisonedFLForgeMixin, DeterministicClientSelectionMixin , FedAvg): 
    def __init__(self, *, model_name: str, num_labels: int, m: int, selection_mode: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.num_labels = num_labels
        self.m = m                                   # number of clients to eliminate
        self.selection_mode = int(selection_mode)    # 1=fixed-m trim, 2=auto-trim


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Tuple[Parameters, Dict[str, Scalar]]]:

        self._maybe_forge_same_round(server_round, results)  # MinMax/MinSum
        self._maybe_forge_pfl(server_round, results)         # PoisonedFL

        if not results:
            return None, {}

        param_names = list(get_model_pt(self.model_name, self.num_labels).state_dict().keys())

        # Compute orthogonality score for each client
        client_scores = []
        for client, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            score = self._compute_ortho_score(weights, param_names)
            client_scores.append((client, fit_res, score))

        # Sort by orthogonality score (lower is better)
        client_scores.sort(key=lambda x: x[2])

        # ----------------------------------------------------------------------------------------------------- #
        # ---------------------------------------------- ORLA ------------------------------------------------- #

        if self.selection_mode == 1:
            selected = client_scores[:-self.m] if self.m > 0 else client_scores
            print(f"[FedAvg-OrthoTrim] Removed top {self.m} clients with worst orthogonality.")


        # ------------------------ ORLA-Flex : Min Threshold + Max change in [min-N] -------------------------- #

        elif self.selection_mode == 2:
            # Earliest legal cutoff (i.e., minimum number of clients you may keep)
            MIN_CUTOFF = self.m

            n = len(client_scores)               # client_scores sorted best→worst already
            scores = [s for _, _, s in client_scores]

            guaranteed = min(MIN_CUTOFF - 1, n)  # always keep [0 .. guaranteed-1]

            if n <= guaranteed:
                # Fewer/equal clients than the guaranteed set: keep all of them
                selected = client_scores[:n]
                print(f"[FedAvg-OrthoAutoTrim] Mode=2 auto, n={n} <= guaranteed={guaranteed}; kept all.")
            else:
                # Consider cutoffs k in [MIN_CUTOFF .. n]
                # These correspond to gap indices i in [guaranteed .. n-2]
                start_i = guaranteed
                tail_diffs = [scores[i+1] - scores[i] for i in range(start_i, n - 1)]

                if tail_diffs:
                    local_argmax = int(np.argmax(tail_diffs))
                    cutoff_k = (guaranteed + 1) + local_argmax     # number of clients to keep
                    selected = client_scores[:cutoff_k]
                    print(
                        f"[FedAvg-OrthoAutoTrim] Mode=2 auto, guaranteed={guaranteed}, "
                        f"cutoff_k={cutoff_k}, max jump={tail_diffs[local_argmax]:.4f}"
                    )
                else:
                    # No tail gaps distinguishable; fall back to keeping only the guaranteed block
                    selected = client_scores[:guaranteed]
                    print(f"[FedAvg-OrthoAutoTrim] Mode=2 auto, no tail diffs; kept guaranteed={guaranteed}.")

        # ------------------------------------------------------------------------------------------------ #

        else:
            # Fallback: keep all
            selected = client_scores
            print(f"[FedAvg-OrthoTrim] Unknown selection_mode={self.selection_mode}; kept all.")

        # -------------------------------------------------------------------------------------------------- #

        id_score_list = [f"{client.cid}: {score:.4f}" for client, _, score in client_scores]
        print(id_score_list)

        # Average updates
        num_layers = len(selected[0][1].parameters.tensors)
        aggregated = []
        for layer_idx in range(num_layers):
            layer_updates = [
                parameters_to_ndarrays(fit_res.parameters)[layer_idx]
                for _, fit_res, _ in selected
            ]
            aggregated.append(np.mean(layer_updates, axis=0))

        # return ndarrays_to_parameters(aggregated), {}
        result = (ndarrays_to_parameters(aggregated), {})
        self._pfl_after_aggregate_update(server_round, result)
        return result

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
    """Multi-Krum: pick m updates with the best Krum scores and average them.

    Krum score s(i) = sum of squared distances to the (n - f - 2) closest vectors.
    Requires 2f + 2 < n (paper condition). If the neighbor count becomes <= 0, falls back to FedAvg.
    """

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

        # Number of neighbors used in the score (n - f - 2), per paper
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

        # Select indices with smallest scores
        winner_idx = np.argpartition(scores, kth=m-1)[:m]
        winner_idx = winner_idx[np.argsort(scores[winner_idx])]  # sort winners by score (nice for logging)

        # Average selected updates layer-wise
        num_layers = len(client_params[0])
        aggregated = []
        for layer in range(num_layers):
            layer_stack = np.stack([client_params[i][layer] for i in winner_idx], axis=0)
            aggregated.append(layer_stack.mean(axis=0))

        print(f"[MultiKrum] Round {server_round}: selected m={m}/{n} with k_neighbors={k_neighbors}.")
        # Optional: log scores summary
        # print(f"[MultiKrum] winners (idx:score): {[(int(i), float(scores[i])) for i in winner_idx]}")

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
            sorted_vals = np.sort(flat_layer, axis=0)
            k = int(self.trim_ratio * num_clients)
            if k * 2 >= num_clients:
                raise ValueError("Too many clients trimmed. Reduce `trim_ratio`.")

            trimmed = sorted_vals[k : num_clients - k]
            mean_trimmed = np.mean(trimmed, axis=0)
            trimmed_weights.append(mean_trimmed.reshape(shape))

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

        self.base_params = None

        self.trusted_loader = self._load_fltrust_data(
            dataset_name=self.dataset_name,
            model_name=self.model_name,
            num_samples=100,
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
            seed=42,
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
        """Build the trusted/root DataLoader for FLTrust.

        For seq_cls tasks we sample from the training split using Dirichlet(label) sampling.
        For multiple-choice QA we support:
          - SWAG
          - PIQA
        """

        from torch.utils.data import DataLoader
        from datasets import load_dataset
        from transformers import AutoTokenizer, DataCollatorWithPadding

        # Decide task mode
        ds_name_l = str(dataset_name).lower()
        problem_type = str(globals().get("PROBLEM_TYPE", "seq_cls")).lower()
        is_mcqa = (problem_type == "mc_qa") or (ds_name_l in {"swag", "piqa"})

        # ----------------------------- Load raw dataset ----------------------------- #

        if is_mcqa:
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
                raw_dataset = load_dataset("swag", "regular", split="train")

                raw_dataset = raw_dataset.rename_column("label", "mc_label")
                pseudo_labels = [int(_swag_stem_type(x)) for x in raw_dataset["sent2"]]
                raw_dataset = raw_dataset.add_column("label", pseudo_labels)

                keep_cols = {"sent1", "sent2", "ending0", "ending1", "ending2", "ending3", "label", "mc_label"}
                drop_cols = [c for c in raw_dataset.column_names if c not in keep_cols]
                if drop_cols:
                    raw_dataset = raw_dataset.remove_columns(drop_cols)

            elif dataset_name in ["PIQA", "piqa"]:
                raw_dataset = load_dataset("ybisk/piqa", split="train")

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
            raw_dataset = _get_seqcls_train_dataset(dataset_name, number_of_samples=None)

        
        # -------------------- Dirichlet(label) sampling for trusted/root set (with min-per-class) --------------------
        
        dirichlet_alpha = 0.5
        min_per_class = 5

        raw_dataset = raw_dataset.shuffle()

        rng = np.random.default_rng()    #  rng = np.random.default_rng(seed)

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
            remaining_pool = list(set(valid_indices.tolist()) - set(chosen))
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
        tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

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

    def safe_svd(self, X: np.ndarray, *, max_retries: int = 5, eps: float = 1e-4, clip_threshold: float = 1e3):
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
                noise = eps * np.random.randn(*X_mod.shape)
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
            rng = np.random.default_rng(42 + server_round)
            random_matrix = rng.standard_normal((update_matrix.shape[1], self.projection_dim))

            # random_matrix = np.random.randn(update_matrix.shape[1], self.projection_dim)
            update_matrix_reduced = update_matrix @ random_matrix
        else:
            update_matrix_reduced = update_matrix

        # Step 2: SVD to find principal component
        # u, s, vh = np.linalg.svd(update_matrix_reduced - update_matrix_reduced.mean(axis=0), full_matrices=False)
        X_centered = update_matrix_reduced - update_matrix_reduced.mean(axis=0)


        # u, s, vh = self.safe_svd(X_centered, max_retries=5, eps=1e-4, clip_threshold=1e3)
        try:
            u, s, vh = self.safe_svd(X_centered, max_retries=5, eps=1e-4, clip_threshold=1e3)
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

        # return ndarrays_to_parameters(averaged), {}

        result = (ndarrays_to_parameters(averaged), {})
        self._pfl_after_aggregate_update(server_round, result)
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
    """
    FedDMC (Mu et al., TDSC 2024): PCA -> BTBCN (binary tree w/ noise pruning) -> SEDC (EMA trust).
    - BTBCN: repeatedly bipartitions by farthest-pair and prunes clusters smaller than min_cluster_size as noise;
             once both sides >= min_cluster_size, the larger cluster is benign.
    - SEDC: per-client EMA trust with thresholding.
    """

    def __init__(
        self,
        *,
        context,
        k: int = 10,                         # PCA dimension (paper uses adaptable k)
        min_cluster_size: int = 3,           # BTBCN pruning param; 3 worked well in paper figs
        alpha: float = 0.7,                  # SEDC EMA weight (0<=alpha<1)
        threshold: float = 0.5,              # SEDC decision threshold (paper recommends 0.5)
        use_lora_only: bool = True,          # default aligns with your LoRA-only training
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

        # SEDC state (EMA trust), init at 0.5 per paper
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
                # use your module-level helpers (not self.*)
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
                return PCA(n_components=k_eff, svd_solver="auto", random_state=42).fit_transform(Xc)
            except Exception:
                pass
        # fallback: NumPy SVD
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        Vt_k = Vt[:k_eff]
        return Xc @ Vt_k.T

    def _btbcn_split(self, Z: np.ndarray, min_cluster_size: int):
        """
        Binary tree-based clustering with noise:
        - While a bipartition creates a side with size < min_cluster_size, treat that side as noise and remove it,
          then bipartition the remaining set again.
        - Once both sides >= min_cluster_size, larger side = benign.
        Returns: boolean mask is_benign for all original indices (noise+malicious=False, benign=True).
        """
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
        # allow attacks (MinMax/MinSum, PoisonedFL) to modify results consistently with your testbed
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
            # robust way to get a string cid
            cid = getattr(client, "cid", None) or fit_res.metrics.get("client_cid", "?")
            cids.append(str(cid))
        W = np.vstack(mats)  # [n_clients, d]

        # 2) PCA (DR)
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

        # PoisonedFL post-aggregate hook (keeps your PFL bookkeeping intact)
        self._pfl_after_aggregate_update(server_round, result)
        return result

# ====================================================================================================================================================== #

# ================================================================================================= #
# ShieldFL (plaintext adaptation): cosine-based weighting against model poisoning
# - Paper key ideas we keep (no-HE): 
#   • Pick baseline g* as the client update with the lowest cosine similarity to previous global g(t-1)
#   • Weight clients by η_i ∝ (1 - cos(g_i, g*)), then aggregate with those weights
#   • Normalize vectors only for cosine measurement; aggregate original updates
#   • First round falls back to FedAvg (no previous global)
# ================================================================================================= #

class ShieldFLStrategy(_SameRoundForgeMixin, _PoisonedFLForgeMixin, DeterministicClientSelectionMixin, FedAvg):
    def __init__(self, *, use_lora_only: bool = True, include_classifier: bool = False, **kwargs):
        super().__init__(**kwargs)
        # Similar to your other strategies: we compute cos on selected params (LoRA-only by default)
        self.use_lora_only = use_lora_only
        self.include_classifier = include_classifier
        self.prev_global_vec: Optional[np.ndarray] = None

    # --- small helpers (use existing helpers if present in your base/mixins) ---
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
        # Prefer the shared helper if available in your codebase:
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

        # Previous aggregated gradient/delta g^{t-1}
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
    Config_Store = dict(context.run_config)

    # Ensure server uses the same problem_type as clients BEFORE building init params
    global PROBLEM_TYPE
    PROBLEM_TYPE = str(context.run_config.get("problem_type", "seq_cls")).lower()

    """Construct components for ServerApp using Krum strategy with logging."""
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    model_name = context.run_config["model-name"]
    ndarrays = get_params(get_model_pt(model_name , num_labels = int(context.run_config["num_labels"])))
    global_model_init = ndarrays_to_parameters(ndarrays)


    fraction_fit = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction-evaluate"]

    # --------------------------------------------------------------------- #
    # ---------------------- Select Malacious Clients --------------------- #

    random.seed(42)     # Set seed for reproducibility

    # number_of_active_clients = math.ceil(context.run_config["number_of_clients"] * fraction_fit)
    malicious_percentage = context.run_config["Malicious-Client-Percentage"]
    # num_malicious = int(threshold * number_of_active_clients)
    num_malicious = math.ceil(malicious_percentage * context.run_config["number_of_clients"] )
    num_clients = context.run_config["number_of_clients"]
    
    # Create and update dictionary
    client_dict = {}

    # Randomly choose `num_malicious` clients to be malicious
    client_ids = [str(i) for i in range(context.run_config["number_of_clients"])]
    malicious_ids = set(random.sample(client_ids, num_malicious))


    # Create the dictionary with exactly `num_malicious` True values
    client_dict = {cid: (cid in malicious_ids) for cid in client_ids}
    
    context.run_config["client_dict"] = client_dict
    
    print("Malacious clients: ", client_dict)

    # --------------------------------------------------------------------- #
    # -------------------------- Select Strategy -------------------------- #

    if context.run_config["strategy"] == "FedAvg":

        strategy = DeterministicFedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            initial_parameters=global_model_init,
            evaluate_metrics_aggregation_fn=aggregate_evaluate,
        )

    if context.run_config["strategy"] == "ORLA":

        strategy = ORLA(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            initial_parameters=global_model_init,
            evaluate_metrics_aggregation_fn=aggregate_evaluate,
            model_name=model_name,
            num_labels=int(context.run_config["num_labels"]),
            m=int(context.run_config["strategy_parameter"] * num_clients),   # Use strategy_parameter as number to eliminate
            selection_mode=1,   
        )

    elif context.run_config["strategy"] == "ORLA-Flex":

        strategy = ORLA(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            initial_parameters=global_model_init,
            evaluate_metrics_aggregation_fn=aggregate_evaluate,
            model_name=model_name,
            num_labels=int(context.run_config["num_labels"]),
            m=int(context.run_config["strategy_parameter"]),                 # Use strategy_parameter as number to eliminate
            selection_mode=2,   
        )


    elif context.run_config["strategy"] == "Krum":

        strategy = KrumWithLogging(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            initial_parameters=global_model_init,
            evaluate_metrics_aggregation_fn=aggregate_evaluate,
            num_malicious_clients = num_malicious  # adjust based on threat model
        )


    elif context.run_config["strategy"] == "MultiKrum":

        m = context.run_config.get("strategy_parameter", None)
        f_override = context.run_config.get("strategy_parameter2", None)

        # Build strategy (use f from config unless overridden by strategy_parameter)
        strategy = MultiKrum(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            initial_parameters=global_model_init,
            evaluate_metrics_aggregation_fn=aggregate_evaluate,
            num_malicious_clients=int(f_override),
            m=m,
        )


    elif context.run_config["strategy"] == "TrimmedMean":
        strategy = TrimmedMean(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            initial_parameters=global_model_init,
            evaluate_metrics_aggregation_fn=aggregate_evaluate,
            trim_ratio=context.run_config["strategy_parameter"],
        )


    elif context.run_config["strategy"] == "FLTrust":
        strategy = FLTrustStrategy(
            context=context,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            initial_parameters=global_model_init,
            evaluate_metrics_aggregation_fn=aggregate_evaluate,
        )

    elif context.run_config["strategy"] == "DnC":
        strategy = DnC(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            initial_parameters=global_model_init,
            evaluate_metrics_aggregation_fn=aggregate_evaluate,
            trim_ratio=context.run_config["strategy_parameter"],
            projection_dim=100,          # Simple models (e.g., MNIST, shallow MLP): 30–50 , Complex models (e.g., BERT, ResNet): 50–100 , Ensure projection_dim < num_model_params
        )

    elif context.run_config["strategy"] == "IdealDefense":
        strategy = IdealDefenseStrategy(
            context=context,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            initial_parameters=global_model_init,
            evaluate_metrics_aggregation_fn=aggregate_evaluate,
        )


    elif context.run_config["strategy"] == "FedDMC":
        strategy = FedDMCStrategy(
            context=context,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            initial_parameters=global_model_init,
            evaluate_metrics_aggregation_fn=aggregate_evaluate,

            # --- FedDMC params (read from run_config if present) ---
            k=int(context.run_config.get("feddmc_k", 10)),
            min_cluster_size=int(context.run_config.get("feddmc_min_cluster_size", 3)),
            alpha=float(context.run_config.get("feddmc_alpha", 0.7)),
            threshold=float(context.run_config.get("feddmc_threshold", 0.5)),
            use_lora_only=bool(context.run_config.get("feddmc_use_lora_only", True)),
            include_classifier=bool(context.run_config.get("feddmc_include_classifier", False)),
        )

    elif context.run_config["strategy"] == "ShieldFL":
        strategy = ShieldFLStrategy(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            initial_parameters=global_model_init,
            evaluate_metrics_aggregation_fn=aggregate_evaluate,
            use_lora_only=bool(context.run_config.get("shieldfl_use_lora_only", True)),
            include_classifier=bool(context.run_config.get("shieldfl_include_classifier", False)),
        )



    # --------------------------------------------------------------------- #

    # Attach forging attributes (no-ops if attack not MinMax/MinSum)
    try:
        strategy.attack_type = context.run_config.get("attack_type", None)    # "MinMax"/"MinSum"/...
        strategy.client_dict = context.run_config.get("client_dict", {})
        strategy.model_name  = context.run_config["model-name"]
        strategy.num_labels  = int(context.run_config["num_labels"])
        # Pre-compute LoRA indices once
        param_names = list(get_model_pt(strategy.model_name, strategy.num_labels).state_dict().keys())
        strategy.lora_indices = [i for i, n in enumerate(param_names) if ("lora_A" in n or "lora_B" in n)]
        # Optional: paper direction choice ("std" default, also accepts "uv" or "sgn")
        strategy.rp_mode = context.run_config.get("rp_mode", "std")
        strategy.mixture_debug = bool(context.run_config.get("mixture_debug", True))
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

    # Turn on the coordinator state if attack is PoisonedFL
    try:
        global PFL_STATE
        mcfg = context.run_config.get("mixture_choices", [])
        if isinstance(mcfg, str):
            mcfg = [s.strip() for s in mcfg.split(",")] if mcfg.strip() else []
        PFL_STATE["enabled"] = (
            strategy.attack_type == "PoisonedFL"
            or (strategy.attack_type == "Mixture" and "PoisonedFL" in mcfg)
        )
    except Exception:
        pass


    # --------------------------------------------------------------------- #

    return ServerAppComponents(config=config, strategy=strategy)

# ====================================================================================================================================================== #

app = ServerApp(server_fn=server_fn)

atexit.register(print_final_results)
