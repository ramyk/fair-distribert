import utils
import configs.bert_mrpc as t_config
import configs.distilbert_mrpc as s_config

import random
import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
from sys import stdout
from os import environ
from os.path import join
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from fairscale.optim.oss import OSS
from torch.utils.data import DataLoader
from fairscale.nn.data_parallel import ShardedDataParallel
from torch.utils.data.distributed import DistributedSampler
from datasets.dataset_dict import DatasetDict
from datasets import (
    load_dataset,
    load_metric,
    logging as datalog
)
from transformers import (
    get_scheduler,
    logging as tflog,
    AutoTokenizer,
    BertForSequenceClassification,
    AutoModelForSequenceClassification
)


def reproduce(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_loop(
    student: ShardedDataParallel,
    teacher: BertForSequenceClassification,
    dataloader: DataLoader,
    optim: OSS,
    scheduler: LambdaLR,
    epoch: int,
    rank: int,
) -> None:
    if rank == 0:
        progress = tqdm(
            len(dataloader),
            desc=f"EPOCH n.{epoch+1} -- TRAINING",
            file=stdout
        )

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
    
    if rank == 0: progress.close()
    
def eval_loop(
    dataloader: DataLoader,
    model: ShardedDataParallel,
    epoch: int,
    rank: int,
    world_size: int,
    testing=False
) -> None:
    if rank == 0:
        metric = load_metric("glue", "mrpc")
        progress = tqdm(
            len(dataloader),
            file=stdout,
            desc=(
                f"EPOCH n.{epoch+1} -- VALIDATION"
                if not testing
                else f"TESTING -- FINAL MODEL"
            )
        )
    
    for batch in dataloader:
        batch = {k: v.to(rank) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(**batch).logits
        
        predictions = torch.argmax(logits, dim=-1)
        if rank != 0:
            dist.gather(predictions, dst=0)
            dist.gather(batch["labels"], dst=0)
        else:
            all_preds = [torch.zeros_like(predictions) for _ in range(world_size)]
            all_lbls = [torch.zeros_like(batch["labels"]) for _ in range(world_size)]
            dist.gather(predictions, gather_list=all_preds)
            dist.gather(batch["labels"], gather_list=all_lbls)
            for preds, lbls in zip(all_preds, all_lbls):
                metric.add_batch(predictions=preds, references=lbls)
            progress.update(1)

    if rank == 0:
        progress.close()
        score = metric.compute()
        print(
            "\tACC = {acc}\n\tF1 = {f1}\n".format(
                acc=score["accuracy"],
                f1=score["f1"]
            )
        )


def train_setup(
    rank: int,
    world_size: int,
    dataset: DatasetDict
) -> None:
    environ['MASTER_ADDR'] = 'localhost'
    environ['MASTER_PORT'] = '12355'
    dist.init_process_group(
        backend="gloo",
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
    eval_dataload = DataLoader(
        dataset=dataset["validation"],
        batch_size=s_config.EVAL_BATCH_SIZE,
        num_workers=0,
        pin_memory=True,
        sampler=DistributedSampler(
            dataset=dataset["validation"],
            num_replicas=world_size,
            rank=rank
        )
    )
    test_dataload = DataLoader(
        dataset=dataset["test"],
        batch_size=s_config.EVAL_BATCH_SIZE,
        num_workers=0,
        pin_memory=True,
        sampler=DistributedSampler(
            dataset=dataset["test"],
            num_replicas=world_size,
            rank=rank
        )
    )

    # MODELS LOAD
    teacher = AutoModelForSequenceClassification.from_pretrained(t_config.CHECKPOINT)
    student = AutoModelForSequenceClassification.from_pretrained(s_config.CHECKPOINT)
    teacher.load_state_dict(torch.load(join(t_config.MODL_REPO, f"{t_config.EXP_NAME}.bin")))
    teacher = teacher.to(rank)
    student = student.to(rank)

    # TRAINING SETUP
    optim_args = {"lr": s_config.LEARNING_RATE}
    optimizer = OSS(
        student.parameters(),
        optim=AdamW,
        **optim_args
    )
    student = ShardedDataParallel(student, optimizer)
    lr_scheduler = get_scheduler(
        s_config.SCHEDULER,
        optimizer=optimizer,
        num_warmup_steps=s_config.WARMUP_STEPS,
        num_training_steps=s_config.EPOCHS * len(train_dataload)
    )
    if rank == 0: print(student, "\n")

    teacher.eval()
    for epoch in range(s_config.EPOCHS):
        student.train()
        train_loop(
            student=student,
            teacher=teacher,
            dataloader=train_dataload,
            optim=optimizer,
            scheduler=lr_scheduler,
            epoch=epoch,
            rank=rank
        )
        student.eval()
        eval_loop(
            dataloader=eval_dataload,
            model=student,
            epoch=epoch,
            rank=rank,
            world_size=world_size
        )
        
        if rank == 0:
            torch.save(
                student.state_dict(),
                join(s_config.SNAP_REPO, f"{s_config.EXP_NAME}_SDP.EP{epoch+1}.bin")
            )
    
    # TEST MEASUREMENT
    eval_loop(
        dataloader=test_dataload,
        model=student,
        epoch=epoch,
        rank=rank,
        world_size=world_size,
        testing=True
    )
    if rank == 0:
        torch.save(
            student.state_dict(),
            join(s_config.MODL_REPO, f"{s_config.EXP_NAME}_SDP.bin")
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
