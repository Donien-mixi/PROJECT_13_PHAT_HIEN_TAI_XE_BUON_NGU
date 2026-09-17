# =============================================================================
# Nap firmware + mo Serial Monitor + (mac dinh) LUU LOG va MO MAN HINH TRUC QUAN
# Dung:
#   powershell -ExecutionPolicy Bypass -File tools\flash_esp32.ps1 -Port COM3
#   -BuildOnly   : chi build, khong nap
#   -NoBuild     : nap khong build lai
#   -SaveLog     : mo them cua so luu log telemetry UDP (output\esp32_log_<ts>.txt)
#   -NoDisplay   : KHONG tu mo esp_display_monitor.py
#   -NoSerialLog : KHONG luu log Serial ra file (chi hien thi)
# Mac dinh: luu log Serial -> output\esp32_serial_<ts>.txt  VA  mo man hinh truc quan.
# =============================================================================
param(
    [string]$Port = "COM3",
    [switch]$NoBuild,
    [switch]$BuildOnly,
    [switch]$SaveLog,
    [switch]$NoDisplay,
    [switch]$NoSerialLog
)

$IdfPath = "C:\Espressif\frameworks\esp-idf-v5.3.5"
$IdfPythonEnv = "C:\Espressif\python_env\idf5.3_py3.11_env"
$FwDir = Join-Path $PSScriptRoot "..\firmware_esp32" | Resolve-Path
$RootDir = Join-Path $PSScriptRoot ".." | Resolve-Path
$OutDir = Join-Path $RootDir "output"

# Tim python co cv2/numpy (moi truong conda cua do an)
function Find-CondaPython {
    $cands = @(
        "C:\Users\DONG NHIEN\.conda\envs\projet_13\python.exe",
        "$env:USERPROFILE\.conda\envs\projet_13\python.exe",
        "$env:USERPROFILE\miniconda3\envs\projet_13\python.exe",
        "$env:USERPROFILE\anaconda3\envs\projet_13\python.exe"
    )
    foreach ($c in $cands) { if (Test-Path $c) { return $c } }
    return "python"
}

# Thiet lap moi truong ESP-IDF
$env:PYTHONIOENCODING = "utf-8"
$env:PATH = (Join-Path $IdfPythonEnv "Scripts") + ";" + $env:PATH
$env:IDF_PYTHON_ENV_PATH = $IdfPythonEnv
. (Join-Path $IdfPath "export.ps1") *> $null

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

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

    # Cua so hien thi truc quan (display-only, UDP 8889)
    if (-not $NoDisplay) {
        $py = Find-CondaPython
        $mon = Join-Path $RootDir "host_laptop\esp_display_monitor.py"
        Write-Host "==> Mo cua so HIEN THI TRUC QUAN: esp_display_monitor.py ($py)..." -ForegroundColor Cyan
        Start-Process -FilePath $py -ArgumentList "`"$mon`"" -WorkingDirectory $RootDir
        Start-Sleep -Milliseconds 500
    }

    # Cua so luu log telemetry UDP (neu yeu cau)
    if ($SaveLog) {
        $Capture = Join-Path $PSScriptRoot "esp32_log_capture.py"
        Write-Host "==> Mo cua so luu log telemetry UDP (output\esp32_log_<ts>.txt)..." -ForegroundColor Cyan
        Start-Process -FilePath (Find-CondaPython) -ArgumentList "`"$Capture`""
    }

    # Serial Monitor: mac dinh vua hien thi vua GHI RA FILE
    if ($NoSerialLog) {
        Write-Host "==> Mo Serial Monitor ($Port). Thoat bang Ctrl+]" -ForegroundColor Green
        idf.py -p $Port monitor
    } else {
        $ts = Get-Date -Format "yyyyMMdd_HHmmss"
        $logPath = Join-Path $OutDir "esp32_serial_$ts.txt"
        Set-Content -Path (Join-Path $OutDir "latest_esp32_serial.txt") -Value $logPath
        Write-Host "==> Serial Monitor ($Port) + LUU LOG -> $logPath" -ForegroundColor Green
        Write-Host "    (Thoat: Ctrl+C | con tro: output\latest_esp32_serial.txt)" -ForegroundColor DarkGray
        # Ghi log bang UTF-8 (khong BOM) va loc bo ma mau ANSI -> de doc/phan tich
        $sw = New-Object System.IO.StreamWriter($logPath, $false, (New-Object System.Text.UTF8Encoding($false)))
        $sw.AutoFlush = $true
        try {
            idf.py -p $Port monitor 2>&1 | ForEach-Object {
                $line = "$_"
                $_                                   # hien thi tren console (giu mau)
                $sw.WriteLine(($line -replace "\x1b\[[0-9;]*m", ""))   # file: bo ma mau
            }
        } finally {
            $sw.Close()
        }
    }
}
finally {
    Pop-Location
}
