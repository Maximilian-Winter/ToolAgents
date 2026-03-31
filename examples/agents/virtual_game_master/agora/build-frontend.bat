@echo off
echo === Building Agora Frontend ===
cd /d "%~dp0frontend"
call npm install
call npm run build
echo === Build complete. Restart the server to serve the new frontend. ===
pause
