"""LoRA-Sec: A Flower / Hugging Face app."""

import hashlib 

from typing import Any
from collections import OrderedDict

import torch
from evaluate import load as load_metric
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
)
from transformers import BertForSequenceClassification

from datasets.utils.logging import disable_progress_bar
from flwr_datasets import FederatedDataset
from peft import get_peft_model, LoraConfig, TaskType

from datasets import load_dataset   


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

# ====================================================================================================================================================== #




# ====================================================================================================================================================== #

def load_data(
    partition_id: int, num_partitions: int, model_name: str, partitioner_type: str, dataset_name: str , partitioner_parameter: float, number_of_samples: int
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Load IMDB data (training and eval)"""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:

        # Partition the IMDB dataset into N partitions
        # Types:   IID, LabelDis, QuantSkew, SplitByLabel, Dirichlet, SortBased

        # ============================================================================ #
        
        if partitioner_type == "IID":
            # ------------------------ IID Partitioning ------------------------ #

            partitioner = IidPartitioner(num_partitions = num_partitions, seed=42 )

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


        # def to_single_label(dataset_dict):
        #     # Convert each example's 'labels' field to a single label (e.g., first label)
        #     for split in dataset_dict:
        #         dataset_dict[split] = dataset_dict[split].map(
        #             lambda ex: {"labels": ex["labels"][0] if len(ex["labels"]) > 0 else -1}
        #         )
        #     return dataset_dict

        # --------------------------------------------------------------------------------------------- #
        # -------------------------------------- Dataset Loading -------------------------------------- # 
        if dataset_name == "IMDB":
            fds = FederatedDataset(dataset="stanfordnlp/imdb", partitioners={"train": partitioner}, preprocessor=preprocess_imdb, seed=42)
        elif dataset_name == "Yelp":
            fds = FederatedDataset(dataset="yelp_review_full", partitioners={"train": partitioner}, preprocessor=random_subset_preprocessor , seed=42)  #, load_dataset_kwargs={"split": "train"}
        elif dataset_name == "GoEmotions":
            fds = FederatedDataset(dataset="go_emotions", partitioners={"train": partitioner}, preprocessor=preprocess_go_emotions , seed=42 ) # preprocessor={"train": preprocess_go_emotions}
        elif dataset_name == "DBPedia":     # HF name: "dbpedia_14"  We keep the same partitioner; preprocessor will create a "text" field
            fds = FederatedDataset(dataset="dbpedia_14", partitioners={"train": partitioner}, preprocessor=preprocess_dbpedia, seed=42)

        # --------------------------------------------------------------------------------------------- #

    partition = fds.load_partition(partition_id)


    # ---------------------- Check if Partitions are Identical in different runs ----------------- #
    def partition_hash(partition_id):
        partition = fds.load_partition(partition_id)
        return hashlib.sha256(str(partition).encode()).hexdigest()

    # -------------------------------------------------------------------------------------------- #



    # Divide data: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

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

def get_model(model_name, num_labels):
    # Initialize the BERT model, wrap with LoRA, then UNFREEZE classifier head
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
    # Extract both model and LoRA parameters
    params = []
    for _, param in model.named_parameters():
        params.append(param.cpu().detach().numpy())
    return params

# ====================================================================================================================================================== #

def set_params(model, parameters) -> None:
    # Set both model and LoRA parameters
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

# ====================================================================================================================================================== #

def train(net, trainloader, epochs, device, use_ortho_loss, lambda_ortho: float) -> None:

    optimizer = AdamW((p for p in net.parameters() if p.requires_grad), lr=5e-5)

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