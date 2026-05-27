@echo off
cd /d %~dp0
echo ==========================================
echo   Agent Memory System - 记忆系统可视化
echo ==========================================
echo.
echo 安装依赖...
pip install -r requirements.txt -q
echo.
echo 启动服务...
echo 打开浏览器访问 http://localhost:8765
echo.
python server.py
pause
