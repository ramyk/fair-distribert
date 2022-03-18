from os.path import join


EXP_NAME    = "DISTILBERT_MRPC"
SEED        = 4321  # REPRODUCIBILITY

# REPOSITORIES
WORKSPACE   = "/home/ubuntu/interview/"
SNAP_REPO   = join(WORKSPACE, "snapshots/")
MODL_REPO   = join(WORKSPACE, "weights/")  
LOGS_REPO   = join(WORKSPACE, "logs/")

# MODEL PARAMS
CHECKPOINT  = "distilbert-base-uncased"
MAX_LENGTH  = 256

# TRAINING PARAMS
TRAIN_BATCH_SIZE    = 16
EVAL_BATCH_SIZE     = 32
EPOCHS              = 5
SCHEDULER           = "linear"
WARMUP_STEPS        = 0
LEARNING_RATE       = 5e-5
TEMPERATURE         = 0.5
