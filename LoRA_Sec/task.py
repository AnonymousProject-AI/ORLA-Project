"""LoRA-Sec: A Flower / Hugging Face app."""
# ====================================================================================================================================================== #
import hashlib
import os
import random
import time

# Strict reproducibility settings. These must be established before NumPy/PyTorch
# initialize their thread pools or CUDA libraries.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")
from pathlib import Path

from datasets import Value

from typing import Any, Dict, Iterable, List, Optional, Union
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
from evaluate import load as load_metric
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForMultipleChoice,
    DefaultDataCollator,
)
from transformers import BertForSequenceClassification

from datasets.utils.logging import disable_progress_bar
from flwr_datasets import FederatedDataset
from peft import get_peft_model, LoraConfig, TaskType

from datasets import load_dataset, load_from_disk

import inspect

# Cache Hugging Face resources to avoid repeated Hub requests.
HF_CACHE_LOCK_ROOT = Path(os.path.expanduser("~/.cache/lora_sec_hf_locks"))
HF_CACHE_LOCK_ROOT.mkdir(parents=True, exist_ok=True)
HF_CACHE_LOCK_STALE_SECONDS = 15 * 60

SEQCLS_RAW_CACHE_ROOT = Path(os.path.expanduser("~/.cache/lora_sec_seqcls_raw"))
SEQCLS_RAW_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
SEQCLS_RAW_CACHE_VERSION = "v1"

# ====================================================================================================================================================== #

def _hf_safe_key(kind: str, name: str) -> str:
    token = hashlib.sha256(f"{kind}|{name}".encode("utf-8")).hexdigest()[:20]
    return f"{kind}_{token}"

def _hf_rate_limited(exc: BaseException) -> bool:
    text = str(exc).lower()
    return ("429" in text) or ("too many requests" in text) or ("rate limit" in text)

def _acquire_hf_lock(lock_dir: Path) -> None:
    while True:
        try:
            lock_dir.mkdir(parents=False, exist_ok=False)
            return
        except FileExistsError:
            try:
                if time.time() - lock_dir.stat().st_mtime > HF_CACHE_LOCK_STALE_SECONDS:
                    import shutil
                    shutil.rmtree(lock_dir, ignore_errors=True)
                    continue
            except OSError:
                pass
            time.sleep(0.25)

def _release_hf_lock(lock_dir: Path) -> None:
    try:
        lock_dir.rmdir()
    except OSError:
        pass

def load_dataset_cache_first(path, *args, **kwargs):

    # Load a dataset locally, downloading it under a shared lock if needed.
    try:
        try:
            from datasets import DownloadConfig
        except ImportError:
            from datasets.utils import DownloadConfig

        local_kwargs = dict(kwargs)
        local_kwargs["download_config"] = DownloadConfig(local_files_only=True)
        local_kwargs.setdefault("download_mode", "reuse_dataset_if_exists")
        return load_dataset(path, *args, **local_kwargs)
    except Exception as local_exc:
        lock_dir = HF_CACHE_LOCK_ROOT / _hf_safe_key("dataset", f"{path}|{args}")
        _acquire_hf_lock(lock_dir)
        try:
            # Another process may have populated the cache while this one waited.
            try:
                try:
                    from datasets import DownloadConfig
                except ImportError:
                    from datasets.utils import DownloadConfig
                local_kwargs = dict(kwargs)
                local_kwargs["download_config"] = DownloadConfig(local_files_only=True)
                local_kwargs.setdefault("download_mode", "reuse_dataset_if_exists")
                return load_dataset(path, *args, **local_kwargs)
            except Exception:
                pass

            try:
                return load_dataset(path, *args, **kwargs)
            except Exception as online_exc:
                if _hf_rate_limited(online_exc):
                    raise RuntimeError(
                        f"Hugging Face rate limit while loading dataset {path!r}. "
                        "The artifact was not available through the local cache-first path. "
                        "Run once after the quota resets (or authenticate with HF_TOKEN); "
                        "subsequent runs will use the local cache."
                    ) from online_exc
                raise
        finally:
            _release_hf_lock(lock_dir)

def tokenizer_from_pretrained_cache_first(model_name: str, **kwargs):
    """AutoTokenizer.from_pretrained with local-only first and a shared download lock."""
    try:
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True, **kwargs)
    except Exception:
        lock_dir = HF_CACHE_LOCK_ROOT / _hf_safe_key("tokenizer", model_name)
        _acquire_hf_lock(lock_dir)
        try:
            try:
                return AutoTokenizer.from_pretrained(model_name, local_files_only=True, **kwargs)
            except Exception:
                return AutoTokenizer.from_pretrained(model_name, **kwargs)
        finally:
            _release_hf_lock(lock_dir)

def model_from_pretrained_cache_first(model_cls, model_name: str, **kwargs):
    """Transformers model .from_pretrained with local-only first and shared lock."""
    try:
        return model_cls.from_pretrained(model_name, local_files_only=True, **kwargs)
    except Exception:
        lock_dir = HF_CACHE_LOCK_ROOT / _hf_safe_key("model", f"{model_cls.__name__}|{model_name}")
        _acquire_hf_lock(lock_dir)
        try:
            try:
                return model_cls.from_pretrained(model_name, local_files_only=True, **kwargs)
            except Exception:
                return model_cls.from_pretrained(model_name, **kwargs)
        finally:
            _release_hf_lock(lock_dir)

def _seqcls_raw_cache_dir(dataset_name: str) -> Path:
    digest = hashlib.sha256(
        f"{SEQCLS_RAW_CACHE_VERSION}|{str(dataset_name).strip().lower()}".encode("utf-8")
    ).hexdigest()[:20]
    return SEQCLS_RAW_CACHE_ROOT / digest

def _seqcls_raw_cache_complete(cache_dir: Path) -> bool:
    return cache_dir.is_dir() and (cache_dir / "_SUCCESS").is_file()

def _save_seqcls_raw_cache(ds, cache_dir: Path) -> None:
    """Persist the processed, seed-independent raw seq-cls training split atomically."""
    if _seqcls_raw_cache_complete(cache_dir):
        return
    lock_dir = HF_CACHE_LOCK_ROOT / _hf_safe_key("seqcls_raw", cache_dir.name)
    _acquire_hf_lock(lock_dir)
    tmp_dir = cache_dir.with_name(f"{cache_dir.name}.tmp-{os.getpid()}")
    try:
        if _seqcls_raw_cache_complete(cache_dir):
            return
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        ds.save_to_disk(str(tmp_dir))
        (tmp_dir / "_SUCCESS").write_text("ok\n", encoding="utf-8")
        shutil.rmtree(cache_dir, ignore_errors=True)
        tmp_dir.replace(cache_dir)
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        _release_hf_lock(lock_dir)



# FLTrust draws trusted samples only after assigning client data.
FLTRUST_SEQCLS_ROOT_SAMPLES = 100
FLTRUST_SEQCLS_DIRICHLET_ALPHA = 0.5
FLTRUST_SEQCLS_MIN_PER_CLASS = 5
FLTRUST_SOURCE_INDEX_COL = "__lora_sec_fltrust_source_idx__"


def set_global_determinism(seed: int) -> int:

    # Seed randomness and request deterministic CPU/GPU execution.
    seed = int(seed)
    # Set PYTHONHASHSEED for child processes; sort order-sensitive collections.
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    os.environ.setdefault("MKL_CBWR", "COMPATIBLE")

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Apply deterministic CPU settings; inter-op thread count can only be set once.
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda"):
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        # Use deterministic math attention kernels.
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass
    return seed


def deterministic_client_seed(base_seed: int, round_number: int, partition_id: int) -> int:
    # Compute the client-local runtime seed from the base seed, round, and partition.

    return int(base_seed) + int(round_number) * 1_000_003 + int(partition_id) * 10_007


# ====================================================================================================================================================== #

from flwr_datasets.partitioner import (
    IidPartitioner,
    # LabelDistributionPartitioner,
    # QuantitySkewPartitioner,
    # SplitByLabelPartitioner,
    DirichletPartitioner,
    # SortPartitioner
)


disable_progress_bar()
fds = None
fds_cache_key = None
mcqa_eval_cache = {}
seqcls_dataset_cache = {}
SEQCLS_PARTITION_CACHE_ROOT = Path(
    os.path.expanduser("~/.cache/lora_sec_seqcls_flower_partitions")
)
SEQCLS_PARTITION_LOCK_DIR = SEQCLS_PARTITION_CACHE_ROOT / "locks"
SEQCLS_PARTITION_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
SEQCLS_PARTITION_LOCK_DIR.mkdir(parents=True, exist_ok=True)

# Increase the cache version after changing preprocessing or partitioning.
SEQCLS_PARTITION_CACHE_VERSION = "old_client_assignment_fltrust_remaining_v1"
SEQCLS_LOCK_STALE_SECONDS = 60 * 60


def _normalize_seqcls_limit(limit: int | None) -> int:
    if limit is None:
        return -1
    try:
        return int(limit)
    except Exception:
        return -1



def _get_seqcls_train_dataset(
    dataset_name: str,
    number_of_samples: int | None = None,
    seed: int = 42,
):
    """
        Load sequence-classification training data as text and integer labels.
        Non-positive sample limits select the full training split.
    """
    global seqcls_dataset_cache

    name = str(dataset_name).strip()
    name_l = name.lower()
    limit = _normalize_seqcls_limit(number_of_samples)
    cache_key = (name_l, limit, int(seed))
    if cache_key in seqcls_dataset_cache:
        return seqcls_dataset_cache[cache_key]

    raw_cache_dir = _seqcls_raw_cache_dir(name_l)
    if _seqcls_raw_cache_complete(raw_cache_dir):
        ds = load_from_disk(str(raw_cache_dir))
    else:
        if name_l == "imdb":
            ds = load_dataset_cache_first("stanfordnlp/imdb", split="train")

        elif name_l == "goemotions":
            ds = load_dataset_cache_first("go_emotions", "simplified", split="train")

            def _to_single_label(ex):
                labels = ex.get("labels", []) or []
                return {"label": int(labels[0]) if labels else -1}

            ds = ds.map(_to_single_label)
            if "labels" in ds.column_names:
                ds = ds.remove_columns("labels")

        elif name_l == "dbpedia":
            ds = load_dataset_cache_first("dbpedia_14", split="train")

            def _fuse_dbpedia(ex):
                return {
                    "text": (
                        str(ex.get("title", "") or "")
                        + " "
                        + str(ex.get("content", "") or "")
                    ).strip()
                }

            ds = ds.map(_fuse_dbpedia)
            drop_cols = [c for c in ("title", "content") if c in ds.column_names]
            if drop_cols:
                ds = ds.remove_columns(drop_cols)

        else:
            raise ValueError(
                "Unsupported sequence-classification dataset: "
                f"{dataset_name}. Expected one of IMDB, GoEmotions, "
                "DBPedia."
            )


        # Save the seed-independent processed training split once. FLTrust can
        # then draw a different deterministic trusted subset for every base seed
        # without any additional Hugging Face Hub request.
        _save_seqcls_raw_cache(ds, raw_cache_dir)

    # FLTrust's seq_cls path expects exactly these semantic fields to exist.
    if "text" not in ds.column_names or "label" not in ds.column_names:
        raise ValueError(
            f"Sequence-classification preprocessing for {dataset_name} did not "
            "produce the required 'text' and 'label' columns. "
            f"Got columns: {ds.column_names}"
        )

    if limit > 0:
        ds = ds.shuffle(seed=int(seed))
        ds = ds.select(range(min(limit, len(ds))))

    seqcls_dataset_cache[cache_key] = ds
    return ds



def _sample_fltrust_root_subset(
    raw_dataset,
    *,
    seed: int,
    num_samples: int = FLTRUST_SEQCLS_ROOT_SAMPLES,
    dirichlet_alpha: float = FLTRUST_SEQCLS_DIRICHLET_ALPHA,
    min_per_class: int = FLTRUST_SEQCLS_MIN_PER_CLASS,
):
    """Sample a deterministic FLTrust root set from an already-disjoint pool."""
    seed = int(seed)
    num_samples = int(num_samples)
    if num_samples < 0:
        raise ValueError(f"FLTrust num_samples must be >= 0, got {num_samples}.")
    if "label" not in raw_dataset.column_names:
        raise ValueError(
            "FLTrust root sampling requires a 'label' column. "
            f"Got columns: {raw_dataset.column_names}"
        )
    if len(raw_dataset) < num_samples:
        raise ValueError(
            f"FLTrust requested {num_samples} trusted samples, but only "
            f"{len(raw_dataset)} samples remain after assigning client data."
        )

    shuffled = raw_dataset.shuffle(seed=seed)
    labels = np.asarray(shuffled["label"], dtype=int)
    valid_mask = labels >= 0
    valid_indices = np.where(valid_mask)[0]
    labels_valid = labels[valid_mask]

    classes = np.unique(labels_valid)
    num_classes = len(classes)
    if num_classes == 0:
        raise ValueError("Trusted/root set sampling failed: no valid labels found.")

    min_per_class = int(min_per_class)
    required = min_per_class * num_classes
    if required > num_samples:
        raise ValueError(
            f"min_per_class={min_per_class} too large for num_samples={num_samples} "
            f"and num_classes={num_classes}"
        )

    indices_by_class = {
        c: valid_indices[np.where(labels_valid == c)[0]].tolist()
        for c in classes
    }
    rng = np.random.default_rng(seed)
    proportions = rng.dirichlet(
        np.ones(num_classes, dtype=float) * float(dirichlet_alpha)
    )

    counts = np.full(num_classes, min_per_class, dtype=int)
    remaining_count = num_samples - required
    if remaining_count > 0:
        counts += rng.multinomial(remaining_count, proportions)

    chosen = []
    for c, k in zip(classes, counts):
        pool = indices_by_class[c]
        if len(pool) == 0 or int(k) <= 0:
            continue
        k_eff = min(int(k), len(pool))
        chosen.extend(rng.choice(pool, size=k_eff, replace=False).tolist())

    if len(chosen) < num_samples:
        remaining_pool = sorted(set(valid_indices.tolist()) - set(chosen))
        if remaining_pool:
            extra = rng.choice(
                remaining_pool,
                size=min(num_samples - len(chosen), len(remaining_pool)),
                replace=False,
            ).tolist()
            chosen.extend(extra)

    chosen = sorted(chosen[:num_samples])
    if len(chosen) != num_samples:
        raise ValueError(
            f"FLTrust could select only {len(chosen)} valid trusted samples "
            f"from the {len(raw_dataset)} samples remaining after client assignment."
        )

    subset = shuffled.select(chosen)
    digest = hashlib.sha256(
        ",".join(str(i) for i in chosen).encode("utf-8")
    ).hexdigest()
    return subset, chosen, digest


def _exclude_used_source_rows(indexed_dataset, used_source_ids):
    """Return rows whose internal source IDs were not consumed by clients."""
    used = set(int(x) for x in used_source_ids)
    source_ids = [int(x) for x in indexed_dataset[FLTRUST_SOURCE_INDEX_COL]]
    keep = [pos for pos, source_id in enumerate(source_ids) if source_id not in used]
    out = indexed_dataset.select(keep)
    if FLTRUST_SOURCE_INDEX_COL in out.column_names:
        out = out.remove_columns([FLTRUST_SOURCE_INDEX_COL])
    return out


def _get_fltrust_remaining_train_dataset(
    dataset_name: str,
    *,
    problem_type: str,
    client_number_of_samples: int | None,
    seed: int = 42,
):
    # Return training rows not assigned to federated clients.
    seed = int(seed)
    limit = _normalize_seqcls_limit(client_number_of_samples)
    problem_type = str(problem_type).lower()
    name = str(dataset_name).strip()
    name_l = name.lower()

    def _take_client_limit(ds):
        if limit > 0:
            return ds.select(range(min(limit, len(ds))))
        return ds

    def _with_source_ids(ds):
        if FLTRUST_SOURCE_INDEX_COL in ds.column_names:
            raise ValueError(
                f"Unexpected pre-existing internal column {FLTRUST_SOURCE_INDEX_COL!r}."
            )
        return ds.add_column(FLTRUST_SOURCE_INDEX_COL, list(range(len(ds))))

    # ------------------------ Sequence classification ------------------------ #
    if problem_type == "seq_cls":
        raw = _get_seqcls_train_dataset(name, number_of_samples=None, seed=seed)
        indexed = _with_source_ids(raw)

        if name_l == "goemotions":
            # GoEmotions uses the full dataset without a sample cap.
            client_selected = indexed
        else:
            # Client subsets follow two deterministic shuffles.
            client_selected = indexed.shuffle(seed=seed).shuffle(seed=seed)
            client_selected = _take_client_limit(client_selected)

        used_ids = [int(x) for x in client_selected[FLTRUST_SOURCE_INDEX_COL]]
        remaining = _exclude_used_source_rows(indexed, used_ids)
        return remaining, len(used_ids)

    # ------------------------ Multiple-choice QA ---------------------------- #
    if problem_type == "mc_qa":
        if name_l == "swag":
            raw = load_dataset_cache_first("swag", "regular", split="train")

            def _stem_type(sent2: str) -> int:
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

            def _prepare(ds):
                ds = ds.rename_column("label", "mc_label")
                ds = ds.add_column("label", [int(_stem_type(x)) for x in ds["sent2"]])
                keep_cols = {
                    "sent1", "sent2", "ending0", "ending1", "ending2", "ending3",
                    "label", "mc_label", FLTRUST_SOURCE_INDEX_COL,
                }
                drop_cols = [c for c in ds.column_names if c not in keep_cols]
                return ds.remove_columns(drop_cols) if drop_cols else ds

        elif name_l == "piqa":
            raw = load_dataset_cache_first("ybisk/piqa", split="train")

            def _goal_intent(goal: str) -> int:
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
                for idx, (_group, words) in enumerate(keyword_groups):
                    if any(word in text for word in words):
                        return idx
                return len(keyword_groups)

            def _prepare(ds):
                ds = ds.rename_column("label", "mc_label")
                ds = ds.add_column("label", [int(_goal_intent(x)) for x in ds["goal"]])
                keep_cols = {"goal", "sol1", "sol2", "label", "mc_label", FLTRUST_SOURCE_INDEX_COL}
                drop_cols = [c for c in ds.column_names if c not in keep_cols]
                return ds.remove_columns(drop_cols) if drop_cols else ds
        else:
            raise ValueError(f"Unsupported multiple-choice QA dataset: {dataset_name}")

        indexed = _with_source_ids(raw)
        remaining_base = _prepare(indexed)

        # Apply the same two shuffles used for client allocation.
        client_flow = _prepare(indexed.shuffle(seed=seed)).shuffle(seed=seed)
        client_selected = _take_client_limit(client_flow)
        used_ids = [int(x) for x in client_selected[FLTRUST_SOURCE_INDEX_COL]]
        remaining = _exclude_used_source_rows(remaining_base, used_ids)
        return remaining, len(used_ids)

    raise ValueError(f"Unsupported problem_type for FLTrust remaining-pool selection: {problem_type}")


def _get_seqcls_fltrust_root_dataset(
    dataset_name: str,
    *,
    seed: int,
    client_number_of_samples: int | None,
    num_samples: int = FLTRUST_SEQCLS_ROOT_SAMPLES,
):
    """Select FLTrust root rows only after reproducing old client sampling."""
    remaining, _used_count = _get_fltrust_remaining_train_dataset(
        dataset_name,
        problem_type="seq_cls",
        client_number_of_samples=client_number_of_samples,
        seed=int(seed),
    )
    return _sample_fltrust_root_subset(
        remaining,
        seed=int(seed),
        num_samples=int(num_samples),
    )


def _safe_cache_component(value: str) -> str:
    import re

    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return value or "dataset"


def _seqcls_partition_cache_dir(
    dataset_name: str,
    num_partitions: int,
    partitioner_type: str,
    partitioner_parameter: float,
    number_of_samples: int | None,
    base_seed: int = 42,
) -> Path:
    """Return the cache directory for one exact Flower partitioning setup."""
    cache_spec = (
        SEQCLS_PARTITION_CACHE_VERSION,
        str(dataset_name),
        int(num_partitions),
        str(partitioner_type),
        repr(float(partitioner_parameter)),
        _normalize_seqcls_limit(number_of_samples),
        int(base_seed),
    )
    digest = hashlib.sha256(repr(cache_spec).encode("utf-8")).hexdigest()[:20]
    dataset_component = _safe_cache_component(str(dataset_name).lower())
    return SEQCLS_PARTITION_CACHE_ROOT / f"{dataset_component}__{digest}"


def _cache_is_complete(cache_dir: Path, num_partitions: int) -> bool:
    if not (cache_dir / "_SUCCESS").is_file():
        return False
    return all((cache_dir / f"partition_{pid}").is_dir() for pid in range(int(num_partitions)))


def _remove_path(path: Path) -> None:
    import shutil

    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        try:
            path.unlink()
        except OSError:
            pass


def _acquire_partition_cache_lock(lock_dir: Path, cache_dir: Path, num_partitions: int) -> bool:
    """Acquire the build lock, or return False if another process finished it."""
    while True:
        if _cache_is_complete(cache_dir, num_partitions):
            return False
        try:
            lock_dir.mkdir()
            return True
        except FileExistsError:
            # Recover stale cache locks.
            try:
                lock_age = time.time() - lock_dir.stat().st_mtime
                if lock_age > SEQCLS_LOCK_STALE_SECONDS:
                    _remove_path(lock_dir)
                    continue
            except OSError:
                pass
            time.sleep(1.0)


def _get_flower_seqcls_partition(
    federated_dataset,
    partition_id: int,
    num_partitions: int,
    dataset_name: str,
    partitioner_type: str,
    partitioner_parameter: float,
    number_of_samples: int | None,
    base_seed: int = 42,
):
    
    # Load a cached client partition or construct and cache it atomically.
    import json

    partition_id = int(partition_id)
    num_partitions = int(num_partitions)
    if partition_id < 0 or partition_id >= num_partitions:
        raise ValueError(
            f"partition_id must be in [0, {num_partitions - 1}], got {partition_id}"
        )

    cache_dir = _seqcls_partition_cache_dir(
        dataset_name=dataset_name,
        num_partitions=num_partitions,
        partitioner_type=partitioner_type,
        partitioner_parameter=partitioner_parameter,
        number_of_samples=number_of_samples,
        base_seed=base_seed,
    )
    partition_dir = cache_dir / f"partition_{partition_id}"

    if _cache_is_complete(cache_dir, num_partitions):
        return load_from_disk(str(partition_dir))

    lock_dir = SEQCLS_PARTITION_LOCK_DIR / f"{cache_dir.name}.lock"
    acquired = _acquire_partition_cache_lock(lock_dir, cache_dir, num_partitions)
    if not acquired:
        return load_from_disk(str(partition_dir))

    tmp_dir = cache_dir.with_name(f"{cache_dir.name}.tmp-{os.getpid()}")
    try:
        if _cache_is_complete(cache_dir, num_partitions):
            return load_from_disk(str(partition_dir))

        _remove_path(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=False)

        # Assign client partitions first; FLTrust later uses only unassigned rows.
        for pid in range(num_partitions):
            client_partition = federated_dataset.load_partition(pid)
            client_partition.save_to_disk(str(tmp_dir / f"partition_{pid}"))

        metadata = {
            "cache_version": SEQCLS_PARTITION_CACHE_VERSION,
            "dataset_name": str(dataset_name),
            "num_partitions": num_partitions,
            "partitioner_type": str(partitioner_type),
            "partitioner_parameter": float(partitioner_parameter),
            "number_of_samples": _normalize_seqcls_limit(number_of_samples),
            "seed": int(base_seed),
            "source": "FederatedDataset.load_partition (old client-assignment compatible; FLTrust root chosen afterward)",
        }
        (tmp_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        (tmp_dir / "_SUCCESS").write_text("ok\n", encoding="utf-8")

        # The lock guarantees a single writer. Rename the complete temporary
        # tree atomically so readers never observe partially saved partitions.
        _remove_path(cache_dir)
        tmp_dir.replace(cache_dir)
    finally:
        _remove_path(tmp_dir)
        _remove_path(lock_dir)

    return load_from_disk(str(partition_dir))


# ====================================================================================================================================================== #

def _get_mcqa_eval_dataset(dataset_name: str, seed: int = 42):
    global mcqa_eval_cache

    key = (str(dataset_name).lower(), int(seed))

    if key not in mcqa_eval_cache:
        if key[0] == "swag":
            ds = load_dataset_cache_first("swag", "regular", split="validation")
        elif key[0] == "piqa":
            ds = load_dataset_cache_first("ybisk/piqa", split="validation")
        else:
            raise ValueError(f"Unsupported multiple-choice QA dataset: {dataset_name}")

        if len(ds) > 10000:
            ds = ds.shuffle(seed=int(seed)).select(range(10000))

        mcqa_eval_cache[key] = ds

    return mcqa_eval_cache[key]


def _select_deterministic_subset(ds, limit: int, seed: int = 42):
    ds = ds.shuffle(seed=seed)
    if int(limit) <= 0:
        return ds
    take = min(int(limit), len(ds))
    return ds.select(range(take))


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: Any
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0] else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])

        flattened_features = []
        for feature in features:
            for i in range(num_choices):
                flattened_features.append({k: v[i] for k, v in feature.items()})

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


# ====================================================================================================================================================== #



def load_data(
    partition_id: int, num_partitions: int, model_name: str, partitioner_type: str, dataset_name: str, partitioner_parameter: float, number_of_samples: int,
    problem_type: str = "seq_cls",
    base_seed: int = 42,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Load dataset partition and create train/eval dataloaders."""
    global fds, fds_cache_key

    # Keep runtime/training randomness tied to the experiment seed.
    runtime_seed = int(base_seed)
    set_global_determinism(runtime_seed + int(partition_id))

    # Use fixed seed 42 for dataset sampling and client partitions.
    # Keep training randomness tied to the experiment seed.
    data_seed = 42
    base_seed = data_seed

    # Check dataset configuration before reusing the cached partitioner.
    current_fds_key = (
        str(dataset_name), int(num_partitions), str(partitioner_type),
        float(partitioner_parameter), int(number_of_samples),
        str(problem_type), int(base_seed),
    )
    if fds_cache_key != current_fds_key:
        fds = None
        fds_cache_key = current_fds_key

    # Skip FederatedDataset construction when the partition cache is complete.
    seqcls_partition_cache_ready = False
    if problem_type == "seq_cls":
        _cached_partition_dir = _seqcls_partition_cache_dir(
            dataset_name=dataset_name,
            num_partitions=num_partitions,
            partitioner_type=partitioner_type,
            partitioner_parameter=partitioner_parameter,
            number_of_samples=number_of_samples,
            base_seed=base_seed,
        )
        seqcls_partition_cache_ready = _cache_is_complete(
            _cached_partition_dir, num_partitions
        )

    if fds is None and not seqcls_partition_cache_ready:

        # Extractive QA uses pseudo-labels for label-based partitioning.
        # Partition the selected dataset across clients.
        # Choose the configured partitioner.

        # ============================================================================ #
        
        if partitioner_type == "IID":
            # ------------------------ IID Partitioning ------------------------ #

            partitioner = IidPartitioner(num_partitions=num_partitions)

            # ------------------------------------------------------------------ #

        # elif partitioner_type == "LabelDis":
            # -------------- Label distribution skew Partitioning -------------- #

            # partitioner = LabelDistributionPartitioner(num_partitions = num_partitions , concentration=0.2)

            # ------------------------------------------------------------------ #

        elif partitioner_type == "QuantSkew":
            # ------------------- Quantity skew Partitioning ------------------- #

            partitioner = QuantitySkewPartitioner(num_partitions = num_partitions , concentration=0.2)

            # ------------------------------------------------------------------ #

        elif partitioner_type == "SplitByLabel":
            # ------------------- Split by class Partitioning ------------------ #

            partitioner = SplitByLabelPartitioner( num_partitions = num_partitions ) # Each partition gets 1/10 of classes
            
            # ------------------------------------------------------------------ #

        elif partitioner_type == "Dirichlet":
            # ---------------------- Dirichlet Partitioning -------------------- #

            partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by="label",
                                   alpha=partitioner_parameter, min_partition_size=10, seed=base_seed ,
                                   self_balancing=True)

            # ------------------------------------------------------------------ #

        elif partitioner_type == "SortBased":
            # ---------------------- Sort-based Partitioning ------------------- #

            sort_partitioner = SortPartitioner(
                num_partitions=20,
                sort_key=lambda x: x[1]  # Sort by label
            )

            # ------------------------------------------------------------------ #

        elif partitioner_type == "Pathological":
            # ------------------------------------------------------------------ #

            partitioner = PathologicalPartitioner(
                num_partitions=10,
                partition_by="label",
                num_classes_per_partition=2,  # Each client has 2 classes
                class_assignment_mode="random",  # Assign classes randomly
                seed=base_seed
            )
            # ------------------------------------------------------------------ #

        else:
            print("Invalid partitioner type. Please choose from: IID, LabelDis, QuantSkew, SplitByLabel, Dirichlet, SortBased")

        # ============================================================================ #

        # Assign client samples before selecting FLTrust trusted data.

        def preprocess_go_emotions(x):     # only for go_emotions
            raw_dataset = load_dataset_cache_first("go_emotions", "simplified")
            raw_dataset = raw_dataset.map(lambda x: {"label": x["labels"][0] if x["labels"] else -1})
            raw_dataset = raw_dataset.remove_columns("labels")
            return raw_dataset


        def preprocess_imdb(dataset_dict):
            for split in dataset_dict:
                ds = dataset_dict[split]
                ds = ds.shuffle(seed=base_seed)
                if int(number_of_samples) > 0:
                    ds = ds.select(range(min(int(number_of_samples), len(ds))))
                dataset_dict[split] = ds
            return dataset_dict


        def preprocess_dbpedia(dataset_dict):
            # 1) fuse title+content -> text (Create a single "text" field and drop string columns that would break the collator)
            def fuse(ex):
                t = ex.get("title", "")
                c = ex.get("content", "")
                return {"text": (t + " " + c).strip()}

            for split in dataset_dict:
                ds = dataset_dict[split]

                # fuse text
                if ("title" in ds.column_names) or ("content" in ds.column_names):
                    ds = ds.map(fuse)
                    # remove the raw string columns so they don't survive into DataLoader
                    to_drop = [c for c in ["title", "content"] if c in ds.column_names]
                    if to_drop:
                        ds = ds.remove_columns(to_drop)

                # 2) deterministic client subset
                ds = ds.shuffle(seed=base_seed)
                if int(number_of_samples) > 0:
                    take = min(int(number_of_samples), len(ds))
                    ds = ds.select(range(take))

                dataset_dict[split] = ds

            return dataset_dict

        # ------------------------------ New MCQA datasets ------------------------------ #
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

        def preprocess_swag(dataset_dict):
            for split in list(dataset_dict.keys()):
                ds = dataset_dict[split]

                ds = ds.rename_column("label", "mc_label")
                pseudo_labels = [int(_swag_stem_type(x)) for x in ds["sent2"]]
                ds = ds.add_column("label", pseudo_labels)

                keep_cols = {"sent1", "sent2", "ending0", "ending1", "ending2", "ending3", "label", "mc_label"}
                drop_cols = [c for c in ds.column_names if c not in keep_cols]
                if drop_cols:
                    ds = ds.remove_columns(drop_cols)

                ds = _select_deterministic_subset(ds, number_of_samples, seed=base_seed)
                dataset_dict[split] = ds
            return dataset_dict

        def preprocess_piqa(dataset_dict):
            for split in list(dataset_dict.keys()):
                ds = dataset_dict[split]

                ds = ds.rename_column("label", "mc_label")
                pseudo_labels = [int(_piqa_goal_intent(x)) for x in ds["goal"]]
                ds = ds.add_column("label", pseudo_labels)

                keep_cols = {"goal", "sol1", "sol2", "label", "mc_label"}
                drop_cols = [c for c in ds.column_names if c not in keep_cols]
                if drop_cols:
                    ds = ds.remove_columns(drop_cols)

                ds = _select_deterministic_subset(ds, number_of_samples, seed=base_seed)
                dataset_dict[split] = ds
            return dataset_dict

        # --------------------------------------------------------------------------------------------- #
        # -------------------------------------- Dataset Loading -------------------------------------- # 
        
        if dataset_name == "IMDB":
            fds = FederatedDataset(dataset="stanfordnlp/imdb", partitioners={"train": partitioner}, preprocessor=preprocess_imdb, seed=base_seed)
        elif dataset_name == "GoEmotions":
            fds = FederatedDataset(dataset="go_emotions", partitioners={"train": partitioner}, preprocessor=preprocess_go_emotions , seed=base_seed ) # preprocessor={"train": preprocess_go_emotions}
        elif dataset_name == "DBPedia":     # HF name: "dbpedia_14"  We keep the same partitioner; preprocessor will create a "text" field
            fds = FederatedDataset(dataset="dbpedia_14", partitioners={"train": partitioner}, preprocessor=preprocess_dbpedia, seed=base_seed)
        elif dataset_name in ["SWAG", "swag"]:
            fds = FederatedDataset(dataset="swag", subset="regular", partitioners={"train": partitioner}, preprocessor=preprocess_swag, seed=base_seed)
        elif dataset_name in ["PIQA", "piqa"]:
            fds = FederatedDataset(dataset="ybisk/piqa", partitioners={"train": partitioner}, preprocessor=preprocess_piqa, seed=base_seed)


        # --------------------------------------------------------------------------------------------- #

    if problem_type == "seq_cls":
        partition = _get_flower_seqcls_partition(
            federated_dataset=fds,
            partition_id=partition_id,
            num_partitions=num_partitions,
            dataset_name=dataset_name,
            partitioner_type=partitioner_type,
            partitioner_parameter=partitioner_parameter,
            number_of_samples=number_of_samples,
            base_seed=base_seed,
        )
    else:
        partition = fds.load_partition(partition_id)


    # ---------------------- Check if Partitions are Identical in different runs ----------------- #
    def partition_hash(partition_id):
        if problem_type == "seq_cls":
            partition_local = _get_flower_seqcls_partition(
                federated_dataset=fds,
                partition_id=partition_id,
                num_partitions=num_partitions,
                dataset_name=dataset_name,
                partitioner_type=partitioner_type,
                partitioner_parameter=partitioner_parameter,
                number_of_samples=number_of_samples,
                base_seed=base_seed,
            )
            return hashlib.sha256(str(partition_local).encode()).hexdigest()
        partition = fds.load_partition(partition_id)
        return hashlib.sha256(str(partition).encode()).hexdigest()

    tokenizer = tokenizer_from_pretrained_cache_first(model_name, model_max_length=512)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

    # ------------------------------ Multiple-choice QA ------------------------------ #
    if problem_type == "mc_qa":

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

        eval_ds_raw = _get_mcqa_eval_dataset(dataset_name, seed=base_seed)

        train_ds = partition.map(
            preprocess_fn,
            batched=True,
            remove_columns=partition.column_names,
        )
        eval_ds = eval_ds_raw.map(
            preprocess_fn,
            batched=True,
            remove_columns=eval_ds_raw.column_names,
        )

        generator = torch.Generator()
        generator.manual_seed(runtime_seed + int(partition_id))
        mc_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)

        trainloader = DataLoader(
            train_ds,
            shuffle=True,
            batch_size=8,
            generator=generator,
            collate_fn=mc_collator,
        )
        testloader = DataLoader(
            eval_ds,
            shuffle=False,
            batch_size=8,
            collate_fn=mc_collator,
        )
        return trainloader, testloader
    # ------------------------------------------------------------------------------- #

    partition_train_test = partition.train_test_split(test_size=0.4 if problem_type == "qa" else 0.2, seed=base_seed)

    # -------------------------------------------------------------------------------------------- #

    # Split each client's local dataset into train/test.
    test_size = 0.4 if problem_type == "qa" else 0.2
    partition_train_test = partition.train_test_split(test_size=test_size, seed=base_seed)
    train_partition = partition_train_test["train"]

    tokenizer = tokenizer_from_pretrained_cache_first(model_name, model_max_length=512)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"


    # ------------------------------ Extractive QA ------------------------------ #
    
    if problem_type == "qa":
        # We create start/end token positions from character-level answer spans.
        # This stays local to this function to avoid touching the classification pipeline.

        max_length = 384
        doc_stride = 128

        def _load_official_qa_eval_split(dataset_name: str):
            return partition_train_test["test"]

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

                if len(answers["answer_start"]) == 0:
                    start_positions.append(cls_index)
                    end_positions.append(cls_index)
                    continue

                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Find the start/end of the context in token space
                token_start_index = 0
                while seq_ids[token_start_index] != 1:
                    token_start_index += 1
                token_end_index = len(input_ids) - 1
                while seq_ids[token_end_index] != 1:
                    token_end_index -= 1

                # If answer not fully inside the context window, label as CLS
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    start_positions.append(cls_index)
                    end_positions.append(cls_index)
                    continue

                # Otherwise move token_start_index and token_end_index to the answer boundaries
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                start_positions.append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_positions.append(token_end_index + 1)

            tokenized["start_positions"] = start_positions
            tokenized["end_positions"] = end_positions
            return tokenized

        def _prepare_eval_features_squad(examples):
            """Tokenize eval set while keeping what we need for SQuAD-style post-processing."""

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

            # For each feature, keep an example_id and mask offsets for non-context tokens.
            example_ids = []
            contexts = []
            answers = []
            for i in range(len(tokenized["input_ids"])):
                sample_index = sample_mapping[i]
                example_ids.append(examples["id"][sample_index])
                contexts.append(examples["context"][sample_index])
                answers.append(examples["answers"][sample_index])

                seq_ids = tokenized.sequence_ids(i)
                offsets = tokenized["offset_mapping"][i]
                tokenized["offset_mapping"][i] = [
                    o if seq_ids[k] == 1 else None for k, o in enumerate(offsets)
                ]

            tokenized["example_id"] = example_ids
            tokenized["context"] = contexts
            tokenized["answers"] = answers
            return tokenized

        eval_source = _load_official_qa_eval_split(dataset_name)

        # Tokenize train split with labels (start/end positions)
        train_ds = train_partition.map(
            _prepare_train_features_squad,
            batched=True,
            remove_columns=train_partition.column_names,
        )

        # Tokenize eval split *without* labels and keep context/answers for post-processing
        eval_ds = eval_source.map(
            _prepare_eval_features_squad,
            batched=True,
            remove_columns=eval_source.column_names,
        )

        # Collate: tensorize model inputs, keep python objects for post-processing
        base_collator = DefaultDataCollator()

        def _qa_eval_collate(batch):
            keep_keys = {"example_id", "offset_mapping", "context", "answers"}
            meta = {k: [b[k] for b in batch] for k in keep_keys}
            batch_for_model = [{k: v for k, v in b.items() if k not in keep_keys} for b in batch]
            out = base_collator(batch_for_model)
            out.update(meta)
            return out

        generator = torch.Generator()
        generator.manual_seed(runtime_seed + int(partition_id))

        trainloader = DataLoader(
            train_ds,
            shuffle=True,
            batch_size=8,
            generator=generator,
            collate_fn=base_collator,
        )
        testloader = DataLoader(
            eval_ds,
            batch_size=8,
            collate_fn=_qa_eval_collate,
            shuffle=False,
        )

        return trainloader, testloader

    # ------------------------------------------------------------------------------------------------ #

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, add_special_tokens=True)

    partition_train_test = partition_train_test.map(tokenize_function, batched=True)
    partition_train_test = partition_train_test.remove_columns("text")
    partition_train_test = partition_train_test.rename_column("label", "labels")


    # Keep only model inputs and labels; drop anything else (e.g., stray strings like "title")
    keep_cols = {"input_ids", "attention_mask", "token_type_ids", "labels"}
    for split in ["train", "test"]:
        cols = set(partition_train_test[split].column_names)
        drop = list(cols - keep_cols)
        if drop:
            partition_train_test[split] = partition_train_test[split].remove_columns(drop)


    generator = torch.Generator()
    generator.manual_seed(runtime_seed + int(partition_id))  # Deterministic shuffle per client

    if "generator" not in locals():
        generator = torch.Generator()
        generator.manual_seed(runtime_seed)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainloader = DataLoader(
        partition_train_test["train"],
        shuffle=True,
        batch_size=32,
        generator=generator,
        collate_fn=data_collator,
    )

    testloader = DataLoader(
        partition_train_test["test"], 
        batch_size=32, 
        collate_fn=data_collator, 
        shuffle=False
    )

    return trainloader, testloader

# ====================================================================================================================================================== #

def get_model(model_name, num_labels, problem_type: str = "seq_cls"):

    if problem_type == "qa":
        
        # Extractive QA head: predicts start/end token positions
        model = model_from_pretrained_cache_first(AutoModelForQuestionAnswering, model_name)

        qa_task = getattr(TaskType, "QUESTION_ANSWERING", None)
        if qa_task is None:
            qa_task = getattr(TaskType, "QUESTION_ANS", None)
        if qa_task is None:
            # fallback for peft versions without a dedicated QA task type
            qa_task = getattr(TaskType, "TOKEN_CLS", TaskType.SEQ_CLS)

        peft_config = LoraConfig(
            task_type=qa_task,
            r=16,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=["query", "value"],
            modules_to_save=["qa_outputs"],
        )
        model = get_peft_model(model, peft_config)
        for n, p in model.named_parameters():
            if "qa_outputs" in n:
                p.requires_grad = True
        return model

    # Initialize the BERT model, wrap with LoRA, then UNFREEZE classifier head

    if problem_type == "mc_qa":
        model = model_from_pretrained_cache_first(AutoModelForMultipleChoice, model_name)
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=["query", "value"],
            modules_to_save=["classifier"],
        )
        model = get_peft_model(model, peft_config)
        for n, p in model.named_parameters():
            if ("classifier" in n) or ("sequence_summary" in n):
                p.requires_grad = True
        return model
    model = model_from_pretrained_cache_first(BertForSequenceClassification, model_name, num_labels=num_labels)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["query", "value"],  # keep LoRA on attention; classifier trains normally
        modules_to_save=["classifier"]
    )
    model = get_peft_model(model, peft_config)
    # make sure classifier participates in training
    for n, p in model.named_parameters():
        if "classifier" in n:
            p.requires_grad = True
    return model

# ====================================================================================================================================================== #

def get_params(model):
    # Must match set_params() ordering (state_dict keys)
    state_dict = model.state_dict()
    return [v.detach().cpu().numpy() for _, v in state_dict.items()]

# ====================================================================================================================================================== #

def set_params(model, parameters) -> None:
    # Set both model and LoRA parameters
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

# ====================================================================================================================================================== #

def train(net, trainloader, lr , epochs, device, use_ortho_loss, lambda_ortho):


    # optimizer = AdamW((p for p in net.parameters() if p.requires_grad), lr=5e-5)
    optimizer = AdamW((p for p in net.parameters() if p.requires_grad), lr=lr, foreach=False, fused=False)

    net.train()

    for _ in range(epochs):
        for batch in trainloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = net(**batch)

            loss = outputs.loss

            # Enforce orthogonality on LoRA A matrices
            if use_ortho_loss:

                orthogonality_loss = 0.0

                for name, param in net.named_parameters():
                    if ("lora_A" in name) or ("lora_B" in name):    # !!!!
                        A = param
                        if A.shape[0] < A.shape[1]:
                            # Orthogonal rows: A A^T ≈ I
                            prod = A @ A.T
                            identity = torch.eye(prod.shape[0], device=A.device)
                        else:
                            # Orthogonal columns: A^T A ≈ I
                            prod = A.T @ A
                            identity = torch.eye(prod.shape[0], device=A.device)
                        
                        orthogonality_loss += torch.norm(prod - identity, p="fro")

                # Add orthogonality regularization
                loss += lambda_ortho * orthogonality_loss

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

# ====================================================================================================================================================== #


def test_qa(net, testloader, device) -> tuple[float, float, float]:

    # Evaluate extractive question answering with Exact Match and F1.

    import re
    import string
    import numpy as np

    net.eval()
    total_loss = 0.0
    em_sum = 0.0
    f1_sum = 0.0
    n = 0

    def _normalize_answer(s: str) -> str:
        def remove_articles(text: str) -> str:
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text: str) -> str:
            return " ".join(text.split())

        def remove_punc(text: str) -> str:
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text: str) -> str:
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _f1_score(prediction: str, ground_truth: str) -> float:
        pred_tokens = _normalize_answer(prediction).split()
        gt_tokens = _normalize_answer(ground_truth).split()
        if len(pred_tokens) == 0 and len(gt_tokens) == 0:
            return 1.0
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            return 0.0

        common = {}
        for t in pred_tokens:
            common[t] = common.get(t, 0) + 1
        num_same = 0
        for t in gt_tokens:
            if common.get(t, 0) > 0:
                num_same += 1
                common[t] -= 1

        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        return 2 * precision * recall / (precision + recall)

    def _exact_match_score(prediction: str, ground_truth: str) -> float:
        return 1.0 if _normalize_answer(prediction) == _normalize_answer(ground_truth) else 0.0

    def _best_em_f1(prediction: str, gold_texts: list[str]) -> tuple[float, float]:
        # Use the highest score across reference answers.
        em = 0.0
        f1 = 0.0
        for gt in gold_texts:
            em = max(em, _exact_match_score(prediction, gt))
            f1 = max(f1, _f1_score(prediction, gt))
        return em, f1

    def _pick_best_span(start_logits, end_logits, offset_mapping, max_answer_length: int = 30) -> tuple[int, int, float]:
        # offset_mapping: list[tuple[int,int] | None], with None for question/special tokens
        s_log = start_logits.detach().cpu().numpy()
        e_log = end_logits.detach().cpu().numpy()

        valid = [j for j, o in enumerate(offset_mapping) if o is not None]
        if not valid:
            return 0, 0, float("-inf")

        v = np.array(valid, dtype=int)
        k = min(100, len(v))
        start_idx = v[np.argsort(s_log[v])[-k:]][::-1]
        end_idx = v[np.argsort(e_log[v])[-k:]][::-1]

        best_score = -1e30
        best_s, best_e = int(valid[0]), int(valid[0])

        for s in start_idx:
            for e in end_idx:
                if e < s:
                    continue
                if (e - s + 1) > max_answer_length:
                    continue
                score = float(s_log[s] + e_log[e])
                if score > best_score:
                    best_score = score
                    best_s, best_e = int(s), int(e)

        return best_s, best_e, best_score

    # Keep one best prediction per original example across all overflow windows.
    best_predictions = {}

    with torch.no_grad():
        for batch in testloader:
            # Non-tensor fields (python lists)
            example_ids = batch.pop("example_id")
            offset_mapping = batch.pop("offset_mapping")
            contexts = batch.pop("context")
            answers = batch.pop("answers")

            # Move tensors to device; keep python objects as-is
            batch_on_device = {}
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch_on_device[k] = v.to(device)
                else:
                    batch_on_device[k] = v
            batch = batch_on_device

            # Keep only args accepted by the model forward
            forward_params = set(inspect.signature(net.forward).parameters.keys())
            batch = {k: v for k, v in batch.items() if k in forward_params}

            outputs = net(**batch)

            if hasattr(outputs, "loss") and outputs.loss is not None:
                total_loss += float(outputs.loss.item())

            bs = outputs.start_logits.shape[0]
            for i in range(bs):
                s, e, span_score = _pick_best_span(
                    outputs.start_logits[i],
                    outputs.end_logits[i],
                    offset_mapping[i],
                )

                off_s = offset_mapping[i][s] if s < len(offset_mapping[i]) else None
                off_e = offset_mapping[i][e] if e < len(offset_mapping[i]) else None
                if off_s is None or off_e is None:
                    pred_text = ""
                else:
                    start_char, _ = off_s
                    _, end_char = off_e
                    pred_text = contexts[i][start_char:end_char]

                gold_texts = answers[i].get("text", []) if isinstance(answers[i], dict) else []
                if not gold_texts:
                    gold_texts = [""]

                example_id = str(example_ids[i])
                prev = best_predictions.get(example_id)
                if (prev is None) or (span_score > prev["score"]):
                    best_predictions[example_id] = {
                        "score": float(span_score),
                        "prediction": pred_text,
                        "gold_texts": list(gold_texts),
                    }

    em_sum = 0.0
    f1_sum = 0.0
    n = 0
    for item in best_predictions.values():
        em, f1 = _best_em_f1(item["prediction"], item["gold_texts"])
        em_sum += em
        f1_sum += f1
        n += 1

    avg_loss = total_loss / max(1, len(testloader))
    exact_match = em_sum / max(1, n)
    f1 = f1_sum / max(1, n)
    return avg_loss, exact_match, f1

# ====================================================================================================================================================== #

def test(net, testloader, device) -> tuple[Any | float, Any]:
    metric = load_metric("accuracy")
    loss = 0
    net.eval()
    for batch in testloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = net(**batch)
        logits = outputs.logits
        loss += outputs.loss.item()
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    loss /= len(testloader.dataset)
    accuracy = metric.compute()["accuracy"]
    return loss, accuracy

# ====================================================================================================================================================== #
