"""Global configuration for the gold trading model."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

SYMBOL = "XAUUSD"
TIMEFRAME = "H1"          # MT5 timeframe used for klines
BARS = 50_000             # number of H1 bars to pull for training
LABEL_HORIZON = 1         # predict direction of the next bar

# Optuna / LightGBM
N_TRIALS = 60
RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 100

# MCP server
MCP_HOST = "127.0.0.1"
MCP_PORT = 8000
MCP_PATH = "/mcp"

MODEL_PATH = MODEL_DIR / "gold_lgbm.pkl"
FEATURES_PATH = MODEL_DIR / "features.json"
