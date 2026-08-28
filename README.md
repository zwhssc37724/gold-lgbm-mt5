# 黄金交易模型（MT5 + Optuna + LightGBM + HTTP MCP）

基于本机 MetaTrader 5 终端的 XAUUSD（黄金）交易模型，Python 3.12 + uv 工程，LightGBM + Optuna（TPE）调参，通过 streamable-http MCP 对外提供**报价、K 线、波动扩张预测、方向三分类预测**四个能力。

---

## 快速开始

```bash
cd E:\Documents\PythonProjects\gold-lgbm-mt5
uv sync                                  # 首次：创建 .venv 并安装依赖
uv run gold-train --target breakout      # 训练波动扩张模型（二分类）
uv run gold-train --target direction3    # 训练方向三分类模型
uv run gold-mcp                          # 启动 MCP 服务
```

服务启动后监听 `http://127.0.0.1:8000/mcp`，MCP 客户端接入：

```json
{ "mcpServers": { "gold-model": { "url": "http://127.0.0.1:8000/mcp" } } }
```

---

## 预测能力

| 工具 | 任务 | 含义 | 真实数据样本外表现 |
|---|---|---|---|
| `predict` | 波动扩张（breakout，二分类） | 下一根 K 线振幅是否突破近 100 根中位数 | 测试集 AUC ≈ **0.765** |
| `predict_direction_3class` | 方向三分类（看空/观望/看多） | 未来 24 根 H1 K 线（约 1 个交易日）的方向 | 测试集宏平均 OvR AUC、见训练日志 |

### 方向三分类标签定义

按未来 24 根 H1 K 线对数收益率划分（阈值 ±0.3%）：

| 类别 | 名称 | 含义 |
|---|---|---|
| 0 | 看空 | 24 根对数收益 < -0.3% |
| 1 | 观望 | -0.3% ≤ 24 根对数收益 ≤ +0.3% |
| 2 | 看多 | 24 根对数收益 > +0.3% |

**为什么三分类里默认带观望档？** 真实黄金 H1 单根方向预测样本外 AUC 长期在 0.5x，强行做"非多即空"会被噪声拖垮；用 0.3% 阈值划出"不动"档可显著降低交易磨损，对应人工交易中"看不清就不开仓"的纪律。

### 为什么不默认用涨跌方向二分类

涨跌方向在有效市场下真实样本外 AUC ≈ 0.51——任何声称 0.7+ 的方向模型必然来自数据泄漏或过拟合。本项目提供波动扩张（AUC 0.77，可用于突破/止损定宽）和方向三分类（带观望）两个真实可学的目标。

---

## MCP 工具列表（http://127.0.0.1:8000/mcp）

| 工具名 | 说明 | 返回字段（中文） |
|---|---|---|
| `get_quote(symbol)` | MT5 实时报价 | 买价、卖价、最新价、时间、数据来源 |
| `get_klines(symbol, timeframe, bars)` | MT5 K 线 | time/open/high/low/close/tick_volume/spread |
| `predict(symbol, timeframe)` | 波动扩张预测 | 扩张概率、信号、最新收盘价、实时报价、K线时间 |
| `predict_direction_3class(symbol, timeframe)` | 方向三分类预测 | 看空概率、观望概率、看多概率、信号、预测类别、看多减看空置信度、最新收盘价、实时报价、K线时间 |

信号含义：
- 波动扩张：`预期扩张`（≥0.6）/ `预期收敛`（≤0.4）/ `中性`
- 方向三分类：`看空（开空仓）` / `观望（不操作）` / `看多（开多仓）`

---

## 评估协议（防泄漏）

1. **数据**：MT5 终端拉取 XAUUSD H1 K 线，23,328 根（2011-11 至今，受终端历史深度限制）
2. **时序切分**：末 20% 为不打扰的测试集；训练窗口内做 5 折**扩展窗** CV（训练永远早于验证）
3. **调参**：Optuna TPE，25–60 trials，每 trial 跑 5 折平均
4. **最终训练**：用最佳超参在训练窗口重训，仅在测试集上报告一次指标

---

## 特征工程（42 维）

| 类别 | 特征 |
|---|---|
| 多周期收益 | `ret_1/2/3/6/12/24/48` |
| K 线几何 | `body_ratio`（实体占比）、`upper_wick/lower_wick`（上下影线占比）、`hl_range` |
| 技术指标 | `rsi_7/14`、`macd/macd_signal/macd_hist`、`atr_7/14`（归一化）、`bb_pos_20`（布林带位置） |
| 均线族 | `ma_bias_10/20/50/100/200`、`ema_cross`（EMA20/EMA50 偏离） |
| 波动/动量 | `vol_12/24/72/168`、`mom_12/24/72/168`、`skew_24/kurt_24` |
| 成交量 | `vol_ratio_24`、`vol_chg`（仅在 tick_volume 可用时） |
| 状态指纹 | `autocorr_24`（24 根收益一阶自相关） |
| 时段特征 | `hour_sin/cos`、`dow_sin/cos` |

---

## 项目结构

```
E:\Documents\PythonProjects\gold-lgbm-mt5\
├── pyproject.toml
├── README.md
├── src/gold_model/
│   ├── config.py             # 周期/阈值/路径配置
│   ├── mt5_client.py         # MT5 报价 + K 线；终端离线时回退合成数据
│   ├── features.py           # 42 维特征 + 二分类/三分类标签
│   ├── train.py              # Optuna + LightGBM 训练管线（支持二/三分类）
│   └── serve_mcp.py          # HTTP MCP 服务（中文输出）
├── models/
│   ├── gold_lgbm_breakout.pkl     # 波动扩张模型
│   ├── gold_lgbm_direction3.pkl   # 方向三分类模型
│   └── features.json              # 特征列清单
├── train_full.log            # 突破模型训练日志
└── train_dir3.log            # 三分类模型训练日志
```

---

## 已知限制

- MT5 终端必须已安装并登录，否则服务自动回退到确定性合成数据（`来源` 字段会标明），模型预测会与合成数据相吻合，但不构成实盘可用信号
- 模型预测的是**条件概率**，不构成投资建议；实盘前需自行做含点差/滑光/手续费的策略级回测
- `predict` 工具在 `_load_model` 失败时返回明确错误指引，请按提示先训练模型
