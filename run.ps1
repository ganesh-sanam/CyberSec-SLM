# CyberSec-SLM PowerShell Launcher
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "          Starting CyberSec-SLM Application...              " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Start-Job -ScriptBlock {
    Start-Sleep -Seconds 3
    Start-Process "http://localhost:7860"
} | Out-Null

python app.py
