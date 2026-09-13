# =============================================================================
# Nap firmware + mo Serial Monitor cho board ESP32-S3 (mac dinh COM3)
# Dung:
#   powershell -ExecutionPolicy Bypass -File tools\flash_esp32.ps1
#   powershell -ExecutionPolicy Bypass -File tools\flash_esp32.ps1 -Port COM3
#   powershell -ExecutionPolicy Bypass -File tools\flash_esp32.ps1 -BuildOnly
# =============================================================================
param(
    [string]$Port = "COM3",
    [switch]$NoBuild,
    [switch]$BuildOnly
)

$IdfPath = "C:\Espressif\frameworks\esp-idf-v5.3.5"
$IdfPythonEnv = "C:\Espressif\python_env\idf5.3_py3.11_env"
$FwDir = Join-Path $PSScriptRoot "..\firmware_esp32" | Resolve-Path

# Thiet lap moi truong ESP-IDF
$env:PATH = (Join-Path $IdfPythonEnv "Scripts") + ";" + $env:PATH
$env:IDF_PYTHON_ENV_PATH = $IdfPythonEnv
. (Join-Path $IdfPath "export.ps1") *> $null

Push-Location $FwDir
try {
    if (-not $NoBuild) {
        Write-Host "==> Build firmware..." -ForegroundColor Cyan
        idf.py build
        if ($LASTEXITCODE -ne 0) { Write-Host "Build that bai!" -ForegroundColor Red; exit 1 }
    }

    if ($BuildOnly) { Write-Host "==> Build xong (khong nap)." -ForegroundColor Green; exit 0 }

    Write-Host "==> Nap firmware len $Port ..." -ForegroundColor Cyan
    idf.py -p $Port flash
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Nap that bai. Kiem tra cong $Port / cap / nguon." -ForegroundColor Red
        Write-Host "Goi y: powershell -ExecutionPolicy Bypass -File tools\find_esp_port.ps1" -ForegroundColor Yellow
        exit 1
    }

    Write-Host "==> Mo Serial Monitor ($Port). Thoat bang Ctrl+]" -ForegroundColor Green
    idf.py -p $Port monitor
}
finally {
    Pop-Location
}
