import os

# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, 'newData')
TRAIN_DIR = os.path.join(DATA_DIR, 'train_split')
VALID_DIR = os.path.join(DATA_DIR, 'valid_split')
TEST_DIR = os.path.join(DATA_DIR, 'test_split')

MODELS_DIR = os.path.join(BASE_DIR, 'models')
CNN_MODEL_PATH = os.path.join(MODELS_DIR, 'cnn_model.keras')
MOBILENET_MODEL_PATH = os.path.join(MODELS_DIR, 'mobilenet_model.keras')

CLASS_NAMES_FILE = os.path.join(BASE_DIR, 'class_names.json')

# -------------------------
# Training Parameters
# -------------------------
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS_CNN = 50
EPOCHS_MOBILENET = 20
LEARNING_RATE = 1e-4
PATIENCE = 5

# -------------------------
# DVC Config (Optional)
# -------------------------
DVC_REMOTE = 'dvcc'  # Name of your remote
