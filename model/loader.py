import os
import torch

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel
)

from model.architecture import ElectraCNNClassifier


# -----------------------------
# Paths
# -----------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


LOCAL_HF_DIR = os.path.join(
    MODEL_DIR,
    "electra_baseline"
)


MODEL_PATH = os.path.join(
    MODEL_DIR,
    "electra_cnn_hybrid",
    "best_model.pt"
)


# -----------------------------
# Configuration
# -----------------------------

MODEL_NAME = "google/electra-base-discriminator"

NUM_CLASSES = 2


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# -----------------------------
# Load tokenizer
# -----------------------------

tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_HF_DIR,
    local_files_only=True
)


# -----------------------------
# Load ELECTRA configuration
# -----------------------------

electra_config = AutoConfig.from_pretrained(
    LOCAL_HF_DIR,
    local_files_only=True
)


# -----------------------------
# Rebuild model architecture
# -----------------------------

electra = AutoModel.from_config(
    electra_config
)


model = ElectraCNNClassifier(
    electra,
    num_classes=NUM_CLASSES
)


# -----------------------------
# Load trained checkpoint
# -----------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)


load_result = model.load_state_dict(
    checkpoint,
    strict=True
)


model.to(DEVICE)

model.eval()


print("=" * 80)
print("FINAL MODEL LOAD CHECK")
print("=" * 80)

print("Checkpoint tensors :", len(checkpoint))
print("Missing keys       :", load_result.missing_keys)
print("Unexpected keys    :", load_result.unexpected_keys)
print("Model mode         :", "evaluation" if not model.training else "training")
print("Device             :", DEVICE)


if load_result.missing_keys or load_result.unexpected_keys:
    raise RuntimeError(
        "Checkpoint did not load cleanly."
    )


print("\n✅ Final ELECTRA-CNN Hybrid loaded successfully")