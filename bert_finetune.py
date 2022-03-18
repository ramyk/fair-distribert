import configs.bert_mrpc as config

import torch
import random
import numpy as np

from tqdm import tqdm
from os.path import join
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import (
    get_scheduler,
    AdamW,
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# REPRODUCIBILITY
random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
torch.use_deterministic_algorithms(True)

tokenizer = AutoTokenizer.from_pretrained(config.CHECKPOINT)
tokenized_data = (
    load_dataset("glue", "mrpc").map(
        lambda x: tokenizer(
            x["sentence1"],
            x["sentence2"],
            padding="max_length",
            truncation=True,
            max_length=config.MAX_LENGTH
        ),
        batched=True
    ).remove_columns(["idx", "sentence1", "sentence2"])
    .rename_column("label", "labels")
    .with_format("torch")
)

# DATA LOAD
train_dataload = DataLoader(
    tokenized_data["train"],
    shuffle=True,
    batch_size=config.TRAIN_BATCH_SIZE
)
eval_dataload = DataLoader(
    tokenized_data["validation"],
    batch_size=config.EVAL_BATCH_SIZE
)
test_dataload = DataLoader(
    tokenized_data["test"],
    batch_size=config.EVAL_BATCH_SIZE
)

# MODEL LOAD
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = AutoModelForSequenceClassification.from_pretrained(config.CHECKPOINT)
model.to(device)

# TRAINING SETUP
optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
lr_scheduler = get_scheduler(
    config.SCHEDULER,
    optimizer=optimizer,
    num_warmup_steps=config.WARMUP_STEPS,
    num_training_steps=config.EPOCHS * len(train_dataload)
)

# TRAIN & VALID LOOP
print(model, "\n")
for epoch in range(config.EPOCHS):
    model.train()
    for batch in tqdm(
        train_dataload,
        desc=f"TRAINING -- EPOCH n.{epoch+1}"
    ):
        batch = {k: v.to(device) for k, v in batch.items()}
        model(**batch).loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    
    torch.save(
        model.state_dict(),
        join(config.SNAP_REPO, f"{config.EXP_NAME}.EP{epoch+1}.bin")
    )
    
    metric = load_metric("glue", "mrpc")
    model.eval()
    for batch in tqdm(
        eval_dataload,
        desc=f"VALIDATION -- EPOCH n.{epoch+1}"
    ):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(**batch).logits
        
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    score = metric.compute()
    print(
        "EPOCH n.{epoch}\n\tACC = {acc}\n\tF1 = {f1}".format(
            epoch=epoch+1,
            acc=score["accuracy"],
            f1=score["f1"]
        ),
        "\n"
    )


# TEST MEASUREMENT
metric = load_metric("glue", "mrpc")
for batch in tqdm(
    test_dataload,
    desc=f"TESTING -- FINAL MODEL"
):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        logits = model(**batch).logits
    
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

torch.save(
    model.state_dict(),
    join(config.MODL_REPO, f"{config.EXP_NAME}.bin")
)
final_score = metric.compute()
print(
    "\nTEST SCORE\n\tACC = {acc}\n\tF1 = {f1}".format(
        acc=final_score["accuracy"],
        f1=final_score["f1"]
    )
)
