import os

RANDOM_SEED = 0  # for reproducibility
CHALLENGE_ID = "defi-text-mine-2025"
CHALLENGE_DIR = f"data/{CHALLENGE_ID}"

EDA_DIR = os.path.join(CHALLENGE_DIR, "eda")
INTERIM_DIR = os.path.join(CHALLENGE_DIR, "interim")
LOGGING_DIR = os.path.join(CHALLENGE_DIR, "logs")
MODELS_DIR = os.path.join(CHALLENGE_DIR, "models")
OUTPUT_DIR = os.path.join(CHALLENGE_DIR, "output")
for dir_path in [EDA_DIR, INTERIM_DIR, LOGGING_DIR, MODELS_DIR, OUTPUT_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
