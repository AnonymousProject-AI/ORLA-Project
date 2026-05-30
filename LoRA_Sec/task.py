"""LoRA-Sec: A Flower / Hugging Face app."""

import hashlib
import os
import time
from pathlib import Path

from datasets import Value

from typing import Any, Dict, Iterable, List, Optional, Union
from collections import OrderedDict
from dataclasses import dataclass

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
mcqa_eval_cache = {}
seqcls_dataset_cache = {}
seqcls_partition_cache = {}

SEQCLS_SNAPSHOT_ROOT = Path(os.path.expanduser("~/.cache/lora_sec_seqcls_snapshots"))
SEQCLS_LOCK_DIR = SEQCLS_SNAPSHOT_ROOT / "locks"
SEQCLS_SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)
SEQCLS_LOCK_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_seqcls_limit(limit: int | None) -> int:
    if limit is None:
        return -1
    try:
        return int(limit)
    except Exception:
        return -1


def _resolve_seqcls_dataset_spec(dataset_name: str):
    name = str(dataset_name).strip()
    if name == "IMDB":
        return "stanfordnlp/imdb", None
    if name == "Yelp":
        return "yelp_review_full", None
    if name == "DBPedia":
        return "dbpedia_14", None
    raise ValueError(f"Unsupported classification dataset: {dataset_name}")


def _snapshot_name_for_seqcls(dataset_name: str, number_of_samples: int | None) -> str:
    limit = _normalize_seqcls_limit(number_of_samples)
    suffix = "full" if limit <= 0 else str(limit)
    return f"{str(dataset_name).lower()}__{suffix}"


def _prepare_seqcls_train_split(ds, dataset_name: str, number_of_samples: int | None):
    name = str(dataset_name).strip()

    if name == "DBPedia":
        ds = ds.map(lambda ex: {"text": (ex.get("title", "") + " " + ex.get("content", "")).strip()})
        drop_cols = [c for c in ["title", "content"] if c in ds.column_names]
        if drop_cols:
            ds = ds.remove_columns(drop_cols)

    ds = ds.shuffle(seed=42)
    limit = _normalize_seqcls_limit(number_of_samples)
    if limit > 0:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def _load_seqcls_from_hf_cache_only(dataset_name: str, number_of_samples: int | None):
    dataset_id, subset = _resolve_seqcls_dataset_spec(dataset_name)
    prev_datasets_offline = os.environ.get("HF_DATASETS_OFFLINE")
    prev_hub_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        if subset is None:
            ds = load_dataset(dataset_id, split="train")
        else:
            ds = load_dataset(dataset_id, subset, split="train")
        return _prepare_seqcls_train_split(ds, dataset_name, number_of_samples)
    finally:
        if prev_datasets_offline is None:
            os.environ.pop("HF_DATASETS_OFFLINE", None)
        else:
            os.environ["HF_DATASETS_OFFLINE"] = prev_datasets_offline
        if prev_hub_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev_hub_offline


def _ensure_seqcls_snapshot(dataset_name: str, number_of_samples: int | None):
    snapshot_dir = SEQCLS_SNAPSHOT_ROOT / _snapshot_name_for_seqcls(dataset_name, number_of_samples)
    if snapshot_dir.exists():
        return load_from_disk(str(snapshot_dir))

    lock_dir = SEQCLS_LOCK_DIR / (snapshot_dir.name + ".lock")
    while True:
        try:
            lock_dir.mkdir()
            break
        except FileExistsError:
            if snapshot_dir.exists():
                return load_from_disk(str(snapshot_dir))
            time.sleep(1.0)

    try:
        if snapshot_dir.exists():
            return load_from_disk(str(snapshot_dir))

        ds = None
        try:
            ds = _load_seqcls_from_hf_cache_only(dataset_name, number_of_samples)
        except Exception:
            ds = None

        if ds is None:
            dataset_id, subset = _resolve_seqcls_dataset_spec(dataset_name)
            if subset is None:
                ds = load_dataset(dataset_id, split="train")
            else:
                ds = load_dataset(dataset_id, subset, split="train")
            ds = _prepare_seqcls_train_split(ds, dataset_name, number_of_samples)

        tmp_dir = snapshot_dir.with_name(snapshot_dir.name + ".tmp")
        if tmp_dir.exists():
            import shutil
            shutil.rmtree(tmp_dir)
        ds.save_to_disk(str(tmp_dir))
        tmp_dir.replace(snapshot_dir)
        return load_from_disk(str(snapshot_dir))
    finally:
        try:
            lock_dir.rmdir()
        except Exception:
            pass


def _get_seqcls_train_dataset(dataset_name: str, number_of_samples: int | None = None):
    global seqcls_dataset_cache
    key = (str(dataset_name), _normalize_seqcls_limit(number_of_samples))
    if key not in seqcls_dataset_cache:
        seqcls_dataset_cache[key] = _ensure_seqcls_snapshot(dataset_name, number_of_samples)
    return seqcls_dataset_cache[key]


def _get_seqcls_partition(partition_id: int, num_partitions: int, dataset_name: str, partitioner_type: str, partitioner_parameter: float, number_of_samples: int):
    global seqcls_partition_cache

    cache_key = (str(dataset_name), int(num_partitions), str(partitioner_type), float(partitioner_parameter), _normalize_seqcls_limit(number_of_samples))
    if cache_key not in seqcls_partition_cache:
        ds = _get_seqcls_train_dataset(dataset_name, number_of_samples)
        if str(partitioner_type) == "Dirichlet":
            indices = _dirichlet_partition_indices(ds["label"], num_partitions=num_partitions, alpha=partitioner_parameter, min_partition_size=10, seed=42)
        else:
            indices = _iid_partition_indices(len(ds), num_partitions=num_partitions, seed=42)
        seqcls_partition_cache[cache_key] = [ds.select(part_idx) for part_idx in indices]

    return seqcls_partition_cache[cache_key][int(partition_id)]



def _iid_partition_indices(num_items: int, num_partitions: int, seed: int = 42):
    import numpy as np

    indices = np.arange(num_items)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    return [sorted(split.tolist()) for split in np.array_split(indices, num_partitions)]


def _dirichlet_partition_indices(labels, num_partitions: int, alpha: float, min_partition_size: int = 10, seed: int = 42):
    import numpy as np

    labels = np.asarray(labels, dtype=int)
    unique_labels = sorted(np.unique(labels).tolist())
    if len(unique_labels) == 0:
        return _iid_partition_indices(len(labels), num_partitions, seed=seed)

    for attempt in range(100):
        rng = np.random.default_rng(seed + attempt)
        parts = [[] for _ in range(num_partitions)]

        for label in unique_labels:
            idx = np.where(labels == label)[0]
            idx = idx.copy()
            rng.shuffle(idx)
            if len(idx) == 0:
                continue

            probs = rng.dirichlet(np.full(num_partitions, float(alpha), dtype=float))
            counts = rng.multinomial(len(idx), probs)

            start = 0
            for pid, count in enumerate(counts.tolist()):
                if count > 0:
                    parts[pid].extend(idx[start:start + count].tolist())
                start += count

        if len(labels) < num_partitions * min_partition_size:
            return [sorted(p) for p in parts]

        sizes = [len(p) for p in parts]
        if min(sizes) >= min_partition_size:
            return [sorted(p) for p in parts]

    parts = [sorted(p) for p in parts]
    sizes = [len(p) for p in parts]
    while parts and min(sizes) < min_partition_size and max(sizes) > min_partition_size:
        small = int(min(range(num_partitions), key=lambda i: sizes[i]))
        large = int(max(range(num_partitions), key=lambda i: sizes[i]))
        moved = parts[large].pop()
        parts[small].append(moved)
        sizes[large] -= 1
        sizes[small] += 1
    return [sorted(p) for p in parts]


# ====================================================================================================================================================== #

def _get_mcqa_eval_dataset(dataset_name: str):
    global mcqa_eval_cache

    key = str(dataset_name).lower()

    if key not in mcqa_eval_cache:
        if key == "swag":
            ds = load_dataset("swag", "regular", split="validation")
        elif key == "piqa":
            ds = load_dataset("ybisk/piqa", split="validation")
        else:
            raise ValueError(f"Unsupported multiple-choice QA dataset: {dataset_name}")

        if len(ds) > 10000:
            ds = ds.shuffle(seed=42).select(range(10000))

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
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Load dataset partition and create train/eval dataloaders."""
    global fds
    if fds is None:


        # NOTE: For extractive QA (SQuAD v1.1) we support non-IID by creating a
        # pseudo-label ("label") in a preprocessor, so label-based partitioners
        # such as DirichletPartitioner(partition_by="label") work for QA too.

        # Partition the IMDB dataset into N partitions
        # Types:   IID, LabelDis, QuantSkew, SplitByLabel, Dirichlet, SortBased

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

            # partitioner = DirichletPartitioner(
            #     num_partitions = num_partitions,
            #     concentration=0.2  # Concentration parameter
            #     # partition_by="label"
            # )

            partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by="label",
                                   alpha=partitioner_parameter, min_partition_size=10, seed=42 ,
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
                seed=42
            )
            # ------------------------------------------------------------------ #

        else:
            print("Invalid partitioner type. Please choose from: IID, LabelDis, QuantSkew, SplitByLabel, Dirichlet, SortBased")

        # ============================================================================ #

        def preprocess_go_emotions(x):     # only for go_emotions
            raw_dataset = load_dataset("go_emotions", "simplified")
            raw_dataset = raw_dataset.map(lambda x: {"label": x["labels"][0] if x["labels"] else -1})
            raw_dataset = raw_dataset.remove_columns("labels")
            return raw_dataset

        def random_subset_preprocessor(dataset_dict):
            for split in dataset_dict:
                # Shuffle with a fixed seed for reproducibility
                dataset_dict[split] = dataset_dict[split].shuffle(seed=42)
                # Select the first N samples (e.g., 1000)
                dataset_dict[split] = dataset_dict[split].select(range(number_of_samples))
            return dataset_dict

        def preprocess_imdb(dataset_dict):
            for split in dataset_dict:
                # Shuffle for determinism
                dataset_dict[split] = dataset_dict[split].shuffle(seed=42)
                # Select first 10k samples
                take = min(number_of_samples, len(dataset_dict[split]))
                dataset_dict[split] = dataset_dict[split].select(range(take))
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

                # 2) deterministic subset to 25k 
                ds = ds.shuffle(seed=42)
                # take = min(25000, len(ds))
                take = min(number_of_samples, len(ds))
                ds = ds.select(range(take))

                dataset_dict[split] = ds

            return dataset_dict


        def preprocess_amazon_polarity(dataset_dict):
            # Amazon Polarity fields: {"title", "content", "label"}
            def fuse(ex):
                # fuse title + content → text
                t = ex.get("title", "")
                c = ex.get("content", "")
                return {"text": (t + " " + c).strip()}

            for split in dataset_dict:
                ds = dataset_dict[split]

                # 1) Fuse into "text"
                ds = ds.map(fuse)

                # 2) Remove original string columns
                to_drop = [c for c in ["title", "content"] if c in ds.column_names]
                if to_drop:
                    ds = ds.remove_columns(to_drop)

                # 3) Deterministic shuffle + sample
                ds = ds.shuffle(seed=42)
                take = min(number_of_samples, len(ds))
                ds = ds.select(range(take))

                dataset_dict[split] = ds

            return dataset_dict


        def preprocess_yahoo_answers(dataset_dict):

            def fuse(ex):
                t = ex.get("question_title", "")
                c = ex.get("question_content", "")
                return {"text": (t + " " + c).strip()}

            # # ----------------------------
            #     title = ex.get("question_title", "")
            #     content = ex.get("question_content", "")

            #     # Shorten content safely
            #     content = content[:256]

            #     # Combine into a single text field
            #     text = (title + " " + content).strip()

            #     return {"text": text}

            for split in dataset_dict:
                ds = dataset_dict[split]

                ds = ds.map(fuse)

                # rename topic → label
                if "topic" in ds.column_names:
                    ds = ds.rename_column("topic", "label")

                # drop unused columns
                drop_cols = [c for c in ["id", "question_title", "question_content", "best_answer"] 
                            if c in ds.column_names]
                ds = ds.remove_columns(drop_cols)

                ds = ds.shuffle(seed=42)
                take = min(number_of_samples, len(ds))
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

                ds = _select_deterministic_subset(ds, number_of_samples, seed=42)
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

                ds = _select_deterministic_subset(ds, number_of_samples, seed=42)
                dataset_dict[split] = ds
            return dataset_dict
        # ------------------------------------------------------------------------------ #


        # --------------------------------------------------------------------------------------------- #
        # -------------------------------------- Dataset Loading -------------------------------------- # 
        if dataset_name == "IMDB":
            fds = FederatedDataset(dataset="stanfordnlp/imdb", partitioners={"train": partitioner}, preprocessor=preprocess_imdb, seed=42)
        elif dataset_name == "Yelp":
            fds = FederatedDataset(dataset="yelp_review_full", partitioners={"train": partitioner}, preprocessor=random_subset_preprocessor , seed=42)  #, load_dataset_kwargs={"split": "train"}
        elif dataset_name == "DBPedia":     # HF name: "dbpedia_14"  We keep the same partitioner; preprocessor will create a "text" field
            fds = FederatedDataset(dataset="dbpedia_14", partitioners={"train": partitioner}, preprocessor=preprocess_dbpedia, seed=42)
        elif dataset_name in ["SWAG", "swag"]:
            fds = FederatedDataset(dataset="swag", subset="regular", partitioners={"train": partitioner}, preprocessor=preprocess_swag, seed=42)
        elif dataset_name in ["PIQA", "piqa"]:
            fds = FederatedDataset(dataset="ybisk/piqa", partitioners={"train": partitioner}, preprocessor=preprocess_piqa, seed=42)



        # --------------------------------------------------------------------------------------------- #

    if problem_type == "seq_cls":
        partition = _get_seqcls_partition(
            partition_id=partition_id,
            num_partitions=num_partitions,
            dataset_name=dataset_name,
            partitioner_type=partitioner_type,
            partitioner_parameter=partitioner_parameter,
            number_of_samples=number_of_samples,
        )
    elif problem_type == "qa":
        partition = _get_newsqa_partition(
            partition_id=partition_id,
            num_partitions=num_partitions,
            partitioner_type=partitioner_type,
            partitioner_parameter=partitioner_parameter,
            number_of_samples=number_of_samples,
        )
    else:
        partition = fds.load_partition(partition_id)


    # ---------------------- Check if Partitions are Identical in different runs ----------------- #
    def partition_hash(partition_id):
        if problem_type == "seq_cls":
            partition_local = _get_seqcls_partition(
                partition_id=partition_id,
                num_partitions=num_partitions,
                dataset_name=dataset_name,
                partitioner_type=partitioner_type,
                partitioner_parameter=partitioner_parameter,
                number_of_samples=number_of_samples,
            )
            return hashlib.sha256(str(partition_local).encode()).hexdigest()
        if problem_type == "qa" and _is_newsqa_name(dataset_name):
            partition_local = _get_newsqa_partition(
                partition_id=partition_id,
                num_partitions=num_partitions,
                partitioner_type=partitioner_type,
                partitioner_parameter=partitioner_parameter,
                number_of_samples=number_of_samples,
            )
            return hashlib.sha256(str(partition_local).encode()).hexdigest()
        partition = fds.load_partition(partition_id)
        return hashlib.sha256(str(partition).encode()).hexdigest()

    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
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

        eval_ds_raw = _get_mcqa_eval_dataset(dataset_name)

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
        generator.manual_seed(42 + int(partition_id))
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

    partition_train_test = partition.train_test_split(test_size=0.4 if problem_type == "qa" else 0.2, seed=42)

    # -------------------------------------------------------------------------------------------- #



    # Split each client's local dataset into train/test.

    test_size = 0.4 if problem_type == "qa" else 0.2
    partition_train_test = partition.train_test_split(test_size=test_size, seed=42)
    train_partition = partition_train_test["train"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"


    # ------------------------------ Extractive QA -------------------------------- #
    if problem_type == "qa":
        # We create start/end token positions from character-level answer spans.
        # This stays local to this function to avoid touching the classification pipeline.

        max_length = 384
        doc_stride = 128

        def _prepare_train_features_qa(examples):
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

        def _prepare_eval_features_qa(examples):
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

        # Tokenize train split with labels (start/end positions)
        train_ds = train_partition.map(
            _prepare_train_features_qa,
            batched=True,
            remove_columns=train_partition.column_names,
        )

        # Tokenize eval split *without* labels and keep context/answers for post-processing
        eval_ds = eval_source.map(
            _prepare_eval_features_qa,
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
        generator.manual_seed(42 + int(partition_id))

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
    generator.manual_seed(42 + partition_id)  # Deterministic shuffle per client

    if "generator" not in locals():
        generator = torch.Generator()
        generator.manual_seed(42)

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
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)

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
        model = AutoModelForMultipleChoice.from_pretrained(model_name)
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
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
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

# def train(net, trainloader, epochs, device, use_ortho_loss, lambda_ortho: float) -> None:
def train(net, trainloader, lr , epochs, device, use_ortho_loss, lambda_ortho):


    # optimizer = AdamW((p for p in net.parameters() if p.requires_grad), lr=5e-5)
    optimizer = AdamW((p for p in net.parameters() if p.requires_grad), lr=lr)

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
    """Evaluate extractive QA (SQuAD-style) using *text* EM/F1.

    Requirements for `testloader` batches (from `load_data(..., problem_type="qa")`):
      - input tensors: input_ids, attention_mask, (token_type_ids if present), start_positions, end_positions
      - non-tensors preserved by the collate_fn: example_id, offset_mapping, context, answers

    Important: evaluation must be done once per original QA example, not once per
    overflowed feature/window. We therefore keep the best-scoring span across all
    windows that share the same example_id, then compute EM/F1 on that single
    prediction for the example.
    """

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
        # SQuAD uses max over all gold answers
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