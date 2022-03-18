import configs.bert_mrpc as t_config
import configs.distilbert_mrpc as s_config

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
from sys import stdout
from os import environ
from os.path import join
from utils import reproduce, distilloss
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from fairscale.nn.data_parallel import FullyShardedDataParallel
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
    PreTrainedModel,
    AutoModelForSequenceClassification
)


def train_loop(
    student: FullyShardedDataParallel,
    teacher: PreTrainedModel,
    t_dataloader: DataLoader,
    s_dataloader: DataLoader,
    optim: Optimizer,
    scheduler: LambdaLR,
    epoch: int,
    rank: int,
) -> None:
    if rank == 0:
        progress = tqdm(
            len(s_dataloader),
            desc=f"EPOCH n.{epoch+1} -- TRAINING",
            file=stdout
        )

    for t_batch, s_batch in zip(t_dataloader, s_dataloader):
        t_batch = {k: v.to(rank) for k, v in t_batch.items()}
        s_batch = {k: v.to(rank) for k, v in s_batch.items()}
        with torch.no_grad():
            t_outputs = teacher(**t_batch).logits
        
        distilloss(
            t_outputs,
            student(**s_batch).logits,
            s_batch["labels"].double(),
            temperature=s_config.TEMPERATURE,
            rank=rank
        ).backward()

        optim.step()
        scheduler.step()
        optim.zero_grad()
        if rank == 0: progress.update(1)
    
    if rank == 0: progress.close()
    
def eval_loop(
    dataloader: DataLoader,
    model: FullyShardedDataParallel,
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
        all_preds = [torch.zeros_like(predictions) for _ in range(world_size)]
        all_lbls = [torch.zeros_like(batch["labels"]) for _ in range(world_size)]
        dist.all_gather(tensor_list=all_preds, tensor=predictions)
        dist.all_gather(tensor_list=all_lbls, tensor=batch["labels"])
        if rank == 0:
            for preds, lbls in zip(all_preds, all_lbls):
                metric.add_batch(predictions=preds, references=lbls)
            progress.update(1)

        # NCCL BACKEND DOESN'T SUPPORT `torch.distributed.gather` OPERATOR
        # if rank != 0:
        #     dist.gather(predictions, dst=0)
        #     dist.gather(batch["labels"], dst=0)
        # else:
        #     all_preds = [torch.zeros_like(predictions) for _ in range(world_size)]
        #     all_lbls = [torch.zeros_like(batch["labels"]) for _ in range(world_size)]
        #     dist.gather(predictions, gather_list=all_preds)
        #     dist.gather(batch["labels"], gather_list=all_lbls)
        #     for preds, lbls in zip(all_preds, all_lbls):
        #         metric.add_batch(predictions=preds, references=lbls)
        #     progress.update(1)

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
    dataset: dict
) -> None:
    environ['MASTER_ADDR'] = 'localhost'
    environ['MASTER_PORT'] = '12355'
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

    tflog.set_verbosity_error()
    reproduce(s_config.SEED)

    # DATA LOAD
    t_train_dataload = DataLoader(
        dataset=dataset["teacher"]["train"],
        batch_size=s_config.TRAIN_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=DistributedSampler(
            dataset=dataset["teacher"]["train"],
            num_replicas=world_size,
            rank=rank
        )
    )
    s_train_dataload = DataLoader(
        dataset=dataset["student"]["train"],
        batch_size=s_config.TRAIN_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=DistributedSampler(
            dataset=dataset["student"]["train"],
            num_replicas=world_size,
            rank=rank
        )
    )
    eval_dataload = DataLoader(
        dataset=dataset["student"]["validation"],
        batch_size=s_config.EVAL_BATCH_SIZE,
        num_workers=0,
        pin_memory=True,
        sampler=DistributedSampler(
            dataset=dataset["student"]["validation"],
            num_replicas=world_size,
            rank=rank
        )
    )
    test_dataload = DataLoader(
        dataset=dataset["student"]["test"],
        batch_size=s_config.EVAL_BATCH_SIZE,
        num_workers=0,
        pin_memory=True,
        sampler=DistributedSampler(
            dataset=dataset["student"]["test"],
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

    # TRAIN SETUP
    student = FullyShardedDataParallel(student)
    optimizer = AdamW(student.parameters(), lr=s_config.LEARNING_RATE)
    lr_scheduler = get_scheduler(
        s_config.SCHEDULER,
        optimizer=optimizer,
        num_warmup_steps=s_config.WARMUP_STEPS,
        num_training_steps=s_config.EPOCHS * len(s_train_dataload)
    )
    if rank == 0: print(student, "\n")

    teacher.eval()
    mod_state = None
    for epoch in range(s_config.EPOCHS):
        student.train()
        train_loop(
            student=student,
            teacher=teacher,
            t_dataloader=t_train_dataload,
            s_dataloader=s_train_dataload,
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
        
        mod_state = student.state_dict()
        if rank == 0:
            torch.save(
                mod_state,
                join(s_config.SNAP_REPO, f"{s_config.EXP_NAME}_FSDP.EP{epoch+1}.bin")
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
            mod_state,
            join(s_config.MODL_REPO, f"{s_config.EXP_NAME}_FSDP.bin")
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    assert world_size >= 2, f"Requires at least 2 GPUs to run, but got {world_size}"
    
    datalog.set_verbosity_error()
    reproduce(s_config.SEED)

    t_tokenizer = AutoTokenizer.from_pretrained(t_config.CHECKPOINT)
    s_tokenizer = AutoTokenizer.from_pretrained(s_config.CHECKPOINT)
    tokenized_data = dict(
        teacher = (
            load_dataset("glue", "mrpc").map(
                lambda x: t_tokenizer(
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
        ),
        student = (           
            load_dataset("glue", "mrpc").map(
                lambda x: s_tokenizer(
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
    )

    mp.spawn(
        train_setup,
        args=(world_size, tokenized_data),
        nprocs=world_size,
        join=True
    )