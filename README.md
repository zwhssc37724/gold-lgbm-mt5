# Gold LightGBM Trading Model (MT5 + Optuna + HTTP MCP)

Python 3.12 + uv 工程：基于本机 MetaTrader 5 终端的 XAUUSD（黄金）交易模型，
LightGBM + Optuna（TPE）调参，通过 streamable-http MCP 对外提供报价 / K线 / 预测服务。

## 快速开始

```bash
cd gold-lgbm-mt5
uv sync                      # 创建 .venv 并安装依赖（Python 3.12）
uv run gold-train            # 训练（默认 target=breakout，60 trials）
uv run gold-mcp              # 启动 HTTP MCP 服务 http://127.0.0.1:8000/mcp
```

可选参数：

```bash
uv run gold-train --bars 23328 --trials 60 --target breakout   # 波动扩张（默认）
uv run gold-train --target direction                           # 涨跌方向（见下方说明）
```

## 预测目标（重要）

| target | 含义 | 真实数据样本外 AUC |
|---|---|---|
| `breakout`（默认） | 下一根K线振幅是否超过近100根中位数（波动扩张/突破） | **≈ 0.77** |
| `direction` | 下一根K线收涨还是收跌 | ≈ 0.51 |

`direction` 在真实黄金 H1 数据上无法达到 0.7——有效市场下技术特征对单根K线方向
的可预测上限就在 0.5x，任何声称 0.7+ 的结果都来自数据泄漏或过拟合。因此默认模型
采用可学习性真实存在的 `breakout` 目标（波动率聚类），服务于突破策略与止损定宽。
两个目标均可用 `--target` 切换，评估协议完全一致（时序切分、无泄漏）。

## 评估协议（防泄漏）

- MT5 真实 H1 数据 23,328 根（2011-11 至今），末 20% 为不打扰的测试集
- Optuna 在训练窗口内做 5 折扩展窗 CV（train 永远早于 validation）
- 最终模型在训练窗口重训，仅在测试集上报告一次 AUC

## MCP 工具（http://127.0.0.1:8000/mcp）

| 工具 | 说明 |
|---|---|
| `get_quote(symbol)` | MT5 实时报价（bid/ask/last） |
| `get_klines(symbol, timeframe, bars)` | OHLCV K线，M1、M15、M30、H1、H4、D1 |
| `predict(symbol, timeframe)` | LightGBM 预测（概率 + 信号 + 报价） |

连接示例（任何 MCP 客户端，如 Claude Desktop / Cursor）：

```json
{ "mcpServers": { "gold-model": { "url": "http://127.0.0.1:8000/mcp" } } }
```

## 项目结构

```
gold-lgbm-mt5/
├── pyproject.toml            # uv 工程（Python 3.12）
├── src/gold_model/
│   ├── config.py             # 符号/周期/超参配置
│   ├── mt5_client.py         # MT5 报价+K线（终端离线时自动切合成数据兜底）
│   ├── features.py           # 42 维特征工程 + 双目标标签
│   ├── train.py              # Optuna + LightGBM 训练管线
│   └── serve_mcp.py          # HTTP MCP 服务
├── models/gold_lgbm.pkl      # 训练产物
└── train_full.log            # 最近一次训练日志
```

## 已知限制

- MT5 终端必须已安装并登录，`copy_rates_from_pos` 才能取到真实数据；否则回退到
  确定性合成序列（仅用于管线验证，`source` 字段会标明数据来源）
- 模型预测波动扩张概率，不构成投资建议；实盘前需自行做交易成本敏感的回测
