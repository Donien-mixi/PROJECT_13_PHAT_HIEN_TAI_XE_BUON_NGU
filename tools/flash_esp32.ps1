# =============================================================================
# Nạp firmware Project 13 (Edge AI Driver Drowsiness & Distraction Detection)
# Web Stream trực tiếp trên ESP32-S3 (xem qua IP từ điện thoại / laptop)
# Cách dùng:
#   powershell -ExecutionPolicy Bypass -File tools\flash_esp32.ps1 -Port COM3
#   -BuildOnly      : Chỉ build, không nạp
#   -NoBuild        : Nạp ngay firmware đã build (bỏ qua bước build)
#   -LaptopDisplay  : Mở thêm cửa sổ Python cũ trên laptop (esp_display_monitor.py)
#   -NoSerialLog    : Không lưu log Serial ra file (chỉ hiển thị)
# =============================================================================
param(
    [string]$Port = "COM3",
    [switch]$NoBuild,
    [switch]$BuildOnly,
    [switch]$LaptopDisplay,
    [switch]$NoDisplay,       # [v2] mac dinh da KHONG mo cua so laptop; co nay de tuong thich lenh cu
    [switch]$SaveLog,
    [switch]$NoSerialLog
)

$IdfPath = "C:\Espressif\frameworks\esp-idf-v5.3.5"
$IdfPythonEnv = "C:\Espressif\python_env\idf5.3_py3.11_env"
$FwDir = Join-Path $PSScriptRoot "..\firmware_esp32" | Resolve-Path
$RootDir = Join-Path $PSScriptRoot ".." | Resolve-Path
$OutDir = Join-Path $RootDir "output"
$PythonExe = Join-Path $IdfPythonEnv "Scripts\python.exe"

# Tìm python môi trường conda dự án (nếu chạy legacy display)
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

# Thiết lập môi trường ESP-IDF 5.3
$env:PYTHONIOENCODING = "utf-8"
$env:PATH = (Join-Path $IdfPythonEnv "Scripts") + ";" + $env:PATH
$env:IDF_PYTHON_ENV_PATH = $IdfPythonEnv
. (Join-Path $IdfPath "export.ps1") *> $null

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Push-Location $FwDir
try {
    $binPath = "build\tinydriver_esp32s3_edge_ai.bin"

    if (-not $NoBuild) {
        Write-Host "==> [Project 13] Biên dịch Firmware trên ESP-IDF 5.3..." -ForegroundColor Cyan
        idf.py build
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Build thất bại! Vui lòng kiểm tra lỗi." -ForegroundColor Red
            exit 1
        }
    }

    if ($BuildOnly) {
        Write-Host "==> Build thành công (không nạp)." -ForegroundColor Green
        exit 0
    }

    # Nạp nhanh qua esptool
    Write-Host "==> Đang nạp Firmware lên cổng $Port (tốc độ cao 460800 baud)..." -ForegroundColor Cyan
    & $PythonExe -m esptool --chip esp32s3 -p $Port -b 460800 --before default_reset --after hard_reset write_flash --flash_mode dio --flash_size 16MB --flash_freq 80m 0x0 build\bootloader\bootloader.bin 0x8000 build\partition_table\partition-table.bin 0x10000 $binPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Nạp thất bại! Lưu ý: Nếu có chương trình đang mở cổng Serial $Port, vui lòng đóng lại rồi thử lại." -ForegroundColor Red
        Write-Host "Kiểm tra cổng COM: powershell -ExecutionPolicy Bypass -File tools\find_esp_port.ps1" -ForegroundColor Yellow
        exit 1
    }

    Write-Host "==> Nạp firmware thành công!" -ForegroundColor Green
    Write-Host "========================================================================" -ForegroundColor Cyan
    Write-Host "🌐 GIAO DIỆN WEB STREAM TRỰC TIẾP (không cần phần mềm gì trên laptop):" -ForegroundColor Yellow
    Write-Host "   Sau khi khởi động, ESP32 in địa chỉ IP trên Serial Log, ví dụ:" -ForegroundColor White
    Write-Host "     WIFI_STREAM: Da nhan IP tu Router: 192.168.2.32" -ForegroundColor Gray
    Write-Host "     🌐 WEB DASHBOARD: http://192.168.2.32/" -ForegroundColor Gray
    Write-Host "   Mở trình duyệt (laptop/điện thoại CÙNG mạng WiFi) vào địa chỉ đó." -ForegroundColor Cyan
    Write-Host "   Xem Video Camera OV5640, lớp phủ 22 mốc khuôn mặt và HUD ADAS!" -ForegroundColor White
    Write-Host "========================================================================" -ForegroundColor Cyan

    # Mở màn hình Python laptop cũ nếu người dùng yêu cầu
    if ($LaptopDisplay) {
        $py = Find-CondaPython
        $mon = Join-Path $RootDir "host_laptop\esp_display_monitor.py"
        Write-Host "==> Mở cửa sổ Python Laptop: esp_display_monitor.py ($py)..." -ForegroundColor Cyan
        Start-Process -FilePath $py -ArgumentList "`"$mon`"" -WorkingDirectory $RootDir
        Start-Sleep -Milliseconds 500
    }

    # Cửa sổ lưu log telemetry UDP (nếu có yêu cầu)
    if ($SaveLog) {
        $Capture = Join-Path $PSScriptRoot "esp32_log_capture.py"
        Write-Host "==> Lưu log telemetry UDP -> output\esp32_log_<ts>.txt" -ForegroundColor Cyan
        Start-Process -FilePath (Find-CondaPython) -ArgumentList "`"$Capture`""
    }

    # Serial Monitor
    if ($NoSerialLog) {
        Write-Host "==> Mở Serial Monitor ($Port). Thoát bằng Ctrl+]" -ForegroundColor Green
        idf.py -p $Port monitor
    } else {
        $ts = Get-Date -Format "yyyyMMdd_HHmmss"
        $logPath = Join-Path $OutDir "esp32_serial_$ts.txt"
        Set-Content -Path (Join-Path $OutDir "latest_esp32_serial.txt") -Value $logPath
        Write-Host "==> Mở Serial Monitor ($Port) + LƯU LOG -> $logPath" -ForegroundColor Green
        Write-Host "    (Thoát: Ctrl+C | file log: output\latest_esp32_serial.txt)" -ForegroundColor DarkGray
        
        $sw = New-Object System.IO.StreamWriter($logPath, $false, (New-Object System.Text.UTF8Encoding($false)))
        $sw.AutoFlush = $true
        try {
            idf.py -p $Port monitor 2>&1 | ForEach-Object {
                $line = "$_"
                if ($line -match "WEB DASHBOARD URL" -or $line -match "sta ip:") {
                    Write-Host $line -ForegroundColor Yellow
                } else {
                    $_
                }
                $sw.WriteLine(($line -replace '\x1b\[[0-9;]*m', ''))
            }
        } finally {
            $sw.Close()
        }
    }
}
finally {
    Pop-Location
}
