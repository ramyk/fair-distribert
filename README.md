# fair-distribert
BERT distillation and quantization in a distributed setting using [Fairscale](https://fairscale.readthedocs.io/) library.
The repository contains different versions of the training code on the [GLUE](https://gluebenchmark.com) task using the [mRPC](https://paperswithcode.com/dataset/mrpc) dataset with different levels of parallelism using [PyTorch](https://pytorch.org/docs/stable/distributed.html) and FairScale's constructs:
- [ ] [Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [ ] [Sharded Data Parallel](https://fairscale.readthedocs.io/en/stable/api/nn/sharded_ddp.html)
- [ ] [Fully Sharded Data Parallel](https://fairscale.readthedocs.io/en/stable/api/nn/fsdp.html)
- [ ] [Model Off-loading](https://fairscale.readthedocs.io/en/stable/api/experimental/nn/offload_model.html)
- [ ] [Activation Checkpointing](https://fairscale.readthedocs.io/en/stable/api/nn/checkpoint/checkpoint_activations.html)
- [ ] [Pipeline Parallelism](https://fairscale.readthedocs.io/en/stable/api/nn/pipe.html)
- [ ] [SlowMo Distributed Data Parallel](https://fairscale.readthedocs.io/en/stable/api/experimental/nn/slowmo_ddp.html)
