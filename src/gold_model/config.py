"""Global configuration for the gold trading model."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# 交易品种与周期
SYMBOL = "XAUUSD"           # 黄金 / 美元
TIMEFRAME = "H1"            # 1 小时 K 线

# 训练样本
BARS = 50_000               # 拉取的 H1 K 线根数（受 MT5 终端历史限制）

# 突破/扩张二分类：预测下一根 K 线振幅是否超过近 100 根中位数
BREAKOUT_HORIZON = 1
BREAKOUT_LOOKBACK = 100

# 方向三分类：未来 24 根（约 1 个交易日）收益率划分
DIRECTION3_HORIZON = 24
DIRECTION3_THRESHOLD = 0.003  # 0.3%（固定阈值的回退值）
ADAPTIVE_THRESHOLD_ATR_MULT = 1.0  # 自适应阈值 = 近24根 ATR% 中位数 × 该系数

# D1 方向三分类：未来 5 个交易日收益率划分（日线尺度，宏观特征咬合更好）
DIRECTION_D1_HORIZON = 5

# 训练数据快照与清洗
DATA_SNAPSHOT = DATA_DIR / "xauusd_h1_snapshot.parquet"
DATA_SNAPSHOT_D1 = DATA_DIR / "xauusd_d1_snapshot.parquet"
DENSE_HISTORY = True  # 训练前过滤伪装成 H1 的日线历史

# Optuna / LightGBM
N_TRIALS = 60
RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 100

# MCP 服务
MCP_HOST = "127.0.0.1"
MCP_PORT = 8000
MCP_PATH = "/mcp"

# 模型文件
BREAKOUT_MODEL_PATH = MODEL_DIR / "gold_lgbm_breakout.pkl"
DIRECTION3_MODEL_PATH = MODEL_DIR / "gold_lgbm_direction3.pkl"
DIRECTION_D1_MODEL_PATH = MODEL_DIR / "gold_lgbm_direction_d1.pkl"
FEATURES_PATH = MODEL_DIR / "features.json"
