import utils
import configs.bert_mrpc as t_config
import configs.distilbert_mrpc as s_config

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
random.seed(s_config.SEED)
np.random.seed(s_config.SEED)
torch.manual_seed(s_config.SEED)

tokenizer = AutoTokenizer.from_pretrained(s_config.CHECKPOINT)
tokenized_data = (
    load_dataset("glue", "mrpc").map(
        lambda x: tokenizer(
            x["sentence1"],
            x["sentence2"],
            padding="max_length",
            truncation=True,
            max_length=s_config.MAX_LENGTH
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
    batch_size=s_config.TRAIN_BATCH_SIZE
)
eval_dataload = DataLoader(
    tokenized_data["validation"],
    batch_size=s_config.EVAL_BATCH_SIZE
)
test_dataload = DataLoader(
    tokenized_data["test"],
    batch_size=s_config.EVAL_BATCH_SIZE
)

# MODELS LOAD
device = torch.device("cuda:0")
teacher = AutoModelForSequenceClassification.from_pretrained(t_config.CHECKPOINT)
student = AutoModelForSequenceClassification.from_pretrained(s_config.CHECKPOINT)
teacher.load_state_dict(torch.load(join(t_config.MODL_REPO, f"{t_config.EXP_NAME}.bin")))
teacher.to(device)
student.to(device)

# TRAINING SETUP
optimizer = AdamW(student.parameters(), lr=s_config.LEARNING_RATE)
lr_scheduler = get_scheduler(
    s_config.SCHEDULER,
    optimizer=optimizer,
    num_warmup_steps=s_config.WARMUP_STEPS,
    num_training_steps=s_config.EPOCHS * len(train_dataload)
)

# TRAIN & VALID LOOP
teacher.eval()
print(student, "\n")
for epoch in range(s_config.EPOCHS):
    student.train()
    for batch in tqdm(
        train_dataload,
        desc=f"TRAINING -- EPOCH n.{epoch+1}"
    ):
        batch = {k: v.to(device) for k, v in batch.items()}
        s_outputs = student(**batch).logits
        with torch.no_grad():
            t_outputs = teacher(**batch).logits
        
        utils.distilloss(
            t_outputs,
            s_outputs,
            batch["labels"].double(),
            0.5
        ).backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    
    torch.save(
        student.state_dict(),
        join(s_config.SNAP_REPO, f"{s_config.EXP_NAME}.EP{epoch+1}.bin")
    )

    metric = load_metric("glue", "mrpc")
    student.eval()
    for batch in tqdm(
        eval_dataload,
        desc=f"VALIDATION -- EPOCH n.{epoch+1}"
    ):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits = student(**batch).logits
        
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
        logits = student(**batch).logits
    
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

torch.save(
    student.state_dict(),
    join(s_config.MODL_REPO, f"{s_config.EXP_NAME}.bin")
)
final_score = metric.compute()
print(
    "\nTEST SCORE\n\tACC = {acc}\n\tF1 = {f1}".format(
        acc=final_score["accuracy"],
        f1=final_score["f1"]
    )
)
