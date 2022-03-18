import utils
import configs.bert_mrpc as t_config
import configs.distilbert_mrpc as s_config

import random
import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
from os import environ
from os.path import join
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from datasets import (
    load_dataset,
    load_metric,
    logging as datalog
)
from transformers import (
    get_scheduler,
    logging as tflog,
    AutoTokenizer,
    AutoModelForSequenceClassification
)


def reproduce(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_loop(
    rank,
    epoch,
    dataloader,
    teacher,
    student,
    optim,
    scheduler
):
    if rank == 0:
        progress = tqdm(len(dataloader), desc=f"TRAINING -- EPOCH n.{epoch+1}")

    for batch in dataloader:
        batch = {k: v.to(rank) for k, v in batch.items()}
        s_outputs = student(**batch).logits
        with torch.no_grad():
            t_outputs = teacher(**batch).logits
        
        utils.distilloss(
            t_outputs,
            s_outputs,
            batch["labels"].double(),
            0.5
        ).backward()

        optim.step()
        scheduler.step()
        optim.zero_grad()
        if rank == 0: progress.update(1)


def train_setup(rank, world_size, dataset):
    environ['MASTER_ADDR'] = 'localhost'
    environ['MASTER_PORT'] = '12355'
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size
    )

    tflog.set_verbosity_error()
    reproduce(s_config.SEED)

    # DATA LOAD
    train_dataload = DataLoader(
        dataset=dataset["train"],
        batch_size=s_config.TRAIN_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=DistributedSampler(
            dataset=dataset["train"],
            num_replicas=world_size,
            rank=rank
        )
    )

    # MODELS LOAD
    teacher = AutoModelForSequenceClassification.from_pretrained(t_config.CHECKPOINT)
    teacher.load_state_dict(torch.load(join(t_config.MODL_REPO, f"{t_config.EXP_NAME}.bin")))
    teacher = teacher.to(rank)
    student = AutoModelForSequenceClassification.from_pretrained(s_config.CHECKPOINT)
    student_ddp = DistributedDataParallel(
        student.to(rank),
        device_ids=[rank]
    )

    # TRAINING SETUP
    teacher.eval()
    optimizer = AdamW(student_ddp.parameters(), lr=s_config.LEARNING_RATE)
    lr_scheduler = get_scheduler(
        s_config.SCHEDULER,
        optimizer=optimizer,
        num_warmup_steps=s_config.WARMUP_STEPS,
        num_training_steps=s_config.EPOCHS * len(train_dataload)
    )
    if rank == 0:
        print(student_ddp, "\n")

    for epoch in range(s_config.EPOCHS):
        student_ddp.train()
        train_loop(
            rank=rank,
            epoch=epoch,
            dataloader=train_dataload,
            teacher=teacher,
            student=student_ddp,
            optim=optimizer,
            scheduler=lr_scheduler
        )
        
        if rank == 0:
            torch.save(
                student_ddp.state_dict(),
                join(s_config.SNAP_REPO, f"{s_config.EXP_NAME}_DP.EP{epoch+1}.bin")
            )

        # VALID LOOP
        if rank == 0:
            metric = load_metric("glue", "mrpc")
            student_ddp.eval()
            eval_dataload = DataLoader(
                dataset["validation"],
                batch_size=s_config.EVAL_BATCH_SIZE
            )

            for batch in tqdm(
                eval_dataload,
                desc=f"VALIDATION -- EPOCH n.{epoch+1}"
            ):
                batch = {k: v.to(rank) for k, v in batch.items()}
                with torch.no_grad():
                    logits = student_ddp(**batch).logits
                
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
    if rank == 0:
        metric = load_metric("glue", "mrpc")
        test_dataload = DataLoader(
            dataset["test"],
            batch_size=s_config.EVAL_BATCH_SIZE
        )
        for batch in tqdm(
            test_dataload,
            desc=f"TESTING -- FINAL MODEL"
        ):
            batch = {k: v.to(rank) for k, v in batch.items()}
            with torch.no_grad():
                logits = student_ddp(**batch).logits
            
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        torch.save(
            student_ddp.state_dict(),
            join(s_config.MODL_REPO, f"{s_config.EXP_NAME}_DP.bin")
        )
        final_score = metric.compute()
        print(
            "\nTEST SCORE\n\tACC = {acc}\n\tF1 = {f1}".format(
                acc=final_score["accuracy"],
                f1=final_score["f1"]
            )
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    assert world_size >= 2, f"Requires at least 2 GPUs to run, but got {world_size}"
    
    datalog.set_verbosity_error()
    reproduce(s_config.SEED)

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

    mp.spawn(
        train_setup,
        args=(world_size, tokenized_data),
        nprocs=world_size,
        join=True
    )
