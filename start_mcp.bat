@echo off
chcp 65001 >nul
title Gold MCP Server
cd /d E:\Documents\PythonProjects\gold-lgbm-mt5
echo 正在启动黄金模型 MCP 服务 ...
echo 服务地址: http://127.0.0.1:8000/mcp
echo 保持本窗口运行即可；关闭则服务停止。
echo.
.venv\Scripts\python.exe -c "from gold_model.serve_mcp import main; main()"
echo.
echo 服务已停止，按任意键退出。
pause >nul
