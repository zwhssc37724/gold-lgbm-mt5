# 黄金交易模型（MT5 + Optuna + LightGBM + HTTP MCP）

基于本机 MetaTrader 5 终端的 XAUUSD（黄金）交易模型，Python 3.12 + uv 工程，LightGBM + Optuna（TPE）调参，通过 streamable-http MCP 对外提供**报价、K 线、四模型预测、CFTC/GLD 持仓、交易记录、漂移检查**等能力。

> **职责边界（2026-08-31 整改）**：本项目只做**模型计算 + 数据取数**，输出纯数据
> （概率/置信度/ATR/支撑阻力/持仓/成交记录）。**分析、解读、操作建议由上层 agent 负责**——
> MCP 工具不再输出止损止盈建议、Kelly 仓位、信号解读等任何建议性内容。
> 财经资讯/宏观数据类工具此前已剥离至外部 MCP（jin10、Alpha Vantage 等）。

---

## 快速开始

```bash
cd E:\Documents\PythonProjects\gold-lgbm-mt5
uv sync                                  # 首次：创建 .venv 并安装依赖
uv run gold-train --target breakout      # 训练波动扩张模型（H1 二分类）
uv run gold-train --target breakout_m15  # 训练 M15 波动扩张模型（提前 ~45 分钟预警）
uv run gold-train --target direction3    # 训练方向三分类模型
uv run gold-train --target direction_d1  # 训练日线方向模型
uv run gold-mcp                          # 启动 MCP 服务
```

服务启动后监听 `http://127.0.0.1:8000/mcp`，Hermes 的 `gold-trading` MCP 即指向此处。

---

## 评估协议（防泄漏）

1. **数据**：MT5 拉取 XAUUSD H1，经月度密度过滤剔除伪装成 H1 的日线历史后 ~20,000 根；
   存 parquet 快照保证可复现
2. **防泄漏切分**：末 20% 为不打扰的测试集；训练窗口内 5 折**扩展窗** CV，train 尾部
   **purge** horizon 根 + **embargo** 24 根，杜绝前瞻标签跨界
3. **调参**：Optuna TPE，60 trials，每 trial 5 折平均
4. **Walk-forward**：4 窗口滚动训练→滚动预测，报告均值±标准差
5. **朴素基线**：breakout 对比波动率动量基线，direction3 对比多数类基线
6. **宏观特征**：DXY / US10Y / VIX / GLD 日线特征，`shift(1)` + `merge_asof(backward)` 防泄漏对齐到 H1

---

## 置信度门控与多尺度过滤

- **direction3 置信度门控**：|p多−p空|>0.25 的多头腿命中率 72%（基线 39%）；空头腿较弱（40%）
- **D1 方向模型**：独立交易无效（=多头 beta），但作为 H1 高置信信号的**过滤器**有效
  - 叠加后命中率 ~75%（vs 不叠加 72%）

---

## 特征工程（81 维 = 65 价格 + 16 宏观）

| 类别 | 特征 |
|---|---|
| 多周期收益 | `ret_1/2/3/6/12/24/48` |
| K 线几何 | `body_ratio`、`upper_wick/lower_wick`、`hl_range` |
| 技术指标 | `rsi_7/14`、`macd` 族、`atr_7/14`、`bb_pos_20` |
| 均线族 | `ma_bias_10/20/50/100/200`、`ema_cross` |
| 波动/动量 | `vol_12/24/72/168`、`mom_12/24/72/168`、`skew_24/kurt_24` |
| 成交量 | `vol_ratio_24`、`vol_chg` |
| 状态指纹 | `autocorr_24` |
| 时段特征 | `hour_sin/cos`、`dow_sin/cos` |
| 支撑阻力 | `dist_to_high/low_10/20/50`、`range_position_10/20/50` |
| 斐波那契 | `fib_236/382/500/618` |
| 趋势强度 | `adx_plus/minus/diff` |
| 宏观驱动 | `macro_dxy/us10y/vix/gld` 各 4 项：`ret5/ret20/z20/ma20_bias` |

---

## 项目结构

```
E:\Documents\PythonProjects\gold-lgbm-mt5\
├── pyproject.toml
├── README.md
├── PREREGISTRATION.md
├── tests/test_pipeline.py     # 单元测试
├── src/gold_model/
│   ├── config.py             # 训练/评估/MCP 配置
│   ├── mt5_client.py         # MT5 报价 + K 线（线程锁/连接复用/密度过滤）
│   ├── features.py           # 65 维价格特征 + 二分类/三分类标签
│   ├── macro_features.py     # 宏观特征（DXY/US10Y/VIX/GLD，防泄漏对齐）
│   ├── ledger.py             # 预测台账（供 accuracy_check 对账）
│   ├── cftc.py               # CFTC 黄金持仓报告（独家，无外部 MCP 等价）
│   ├── gld_holdings.py       # SPDR GLD 黄金 ETF 持仓量（独家）
│   ├── train.py              # Optuna + LightGBM 训练
│   ├── backtest.py           # 向量化回测器（含随机对照）
│   ├── walkforward.py        # 共享 walk-forward 评估
│   ├── calibration.py        # 概率校准（isotonic）
│   ├── accuracy_check.py     # 预测对账（命中率/校准）
│   ├── drift.py              # 特征漂移监控（PSI）
│   └── serve_mcp.py          # HTTP MCP 服务端（11 个独家工具）
├── experiments/              # 实验脚本（D1 门控、H1×D1 交叉过滤等）
├── data/
│   ├── xauusd_h1_snapshot.parquet   # 训练数据快照
│   ├── xauusd_d1_snapshot.parquet   # 日线训练数据快照
│   └── prediction_ledger.jsonl      # 预测台账
└── models/
    ├── gold_lgbm_breakout.pkl       # 波动扩张模型
    ├── gold_lgbm_direction3.pkl     # 方向三分类模型
    ├── gold_lgbm_direction_d1.pkl   # 日线方向模型
    ├── train_report_*.json          # 训练报告
    └── features.json                # 特征列清单
```

### 常用命令

```bash
uv run gold-train --target breakout --use-snapshot   # 重训（复用数据快照）
uv run gold-backtest --model all                     # 回测：模型 vs 基线 vs 买入持有
uv run gold-accuracy --days 7                        # 预测对账（命中率/校准）
uv run gold-drift                                    # 特征漂移检查（PSI）
uv run --with pytest pytest tests/ -q                # 单元测试
```

---

## 运维与统计纪律

### 特征漂移监控（gold-drift）
每次训练自动保存训练特征的分位数参考（`models/drift_reference_<target>.json`），
`gold-drift` 对比近期真实行情的分布（PSI）。判读：PSI<0.10 稳定；0.10~0.25 轻度漂移；>0.25 显著漂移。

### 概率校准（isotonic）
多分类模型训练时在 WF 样本外概率上拟合 isotonic 校准器，1/4 Kelly 仓位建议基于校准概率计算。

### 随机对照回测
`gold-backtest` 对每个模型策略自动附带同频率随机信号对照（3 个种子）——模型夏普必须显著高于随机对照分布。

### 预注册验证（PREREGISTRATION.md）
置信度门控、H1×D1 交叉过滤的判定规则、有效阈值、失效标准全部预注册冻结。

---

## 已知限制

- MT5 终端必须已安装并登录。数据不可用时 K 线返回合成数据并明确标注
- 经纪商 H1 历史深度有限：2023-03 之前的"H1"实际是伪装的日线（已自动过滤）
- direction3 模型未超过多数类基线，方向信号仅作参考；breakout 模型已验证有效
- 模型预测的是条件概率，不构成投资建议