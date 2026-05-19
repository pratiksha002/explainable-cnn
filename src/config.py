import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
TRAIN_DIR = os.path.join(DATA_DIR, "seg_train", "seg_train")
VAL_DIR = os.path.join(DATA_DIR, "seg_test",  "seg_test")

OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")
MODEL_SAVE_DIR  = os.path.join(OUTPUT_DIR, "models")
LOG_DIR         = os.path.join(OUTPUT_DIR, "logs")

# Intel Image Classification — 6 classes
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE  = 150   
CHANNELS    = 3     

# ─── Training 

BATCH_SIZE    = 32
NUM_EPOCHS    = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4    # L2 regularisation

NUM_WORKERS   = 4       # DataLoader parallel workers (set 0 on Windows if issues)
PIN_MEMORY    = True    

# ─── Model 

MODEL_NAME        = "intel_cnn_v1"
CHECKPOINT_FNAME  = f"{MODEL_NAME}_best.pth"

# ─── Reproducibility

SEED = 42