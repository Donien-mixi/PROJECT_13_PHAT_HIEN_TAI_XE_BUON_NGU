# Nạp toàn bộ hệ thống Project 13 vào ESP32-S3 từ thư mục release/
#
# Cách dùng (PowerShell, tại thư mục gốc repo):
#   powershell -ExecutionPolicy Bypass -File tools\flash_release.ps1 -Port COM3
#
# Board: ESP32-S3 N16R8 (16 MB flash, 8 MB PSRAM octal) + camera OV5640.
# Lần nạp đầu tiên phải ghi cả 4 phần (bootloader, phân vùng, app, model PIPNet).
# Những lần sau chỉ cần app:  powershell -File tools\flash_release.ps1 -Port COM3 -AppOnly

param(
    [string]$Port = "COM3",
    [switch]$AppOnly
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$rel  = Join-Path $root "release"
$py   = "C:\Espressif\python_env\idf5.3_py3.11_env\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

function Assert-Hash($file, $sha) {
    if (-not (Test-Path $file)) { throw "thieu file: $file" }
    $h = (Get-FileHash $file -Algorithm SHA256).Hash
    if ($h -ne $sha) { throw "SAI HASH: $file`n  mong doi $sha`n  thuc te  $h" }
    Write-Host "  OK $((Split-Path $file -Leaf))  $($h.Substring(0,16))..." -ForegroundColor Green
}

Write-Host "== Kiem tra hash ==" -ForegroundColor Cyan
Assert-Hash (Join-Path $rel "app.bin")               "02972E91F1B60C57098B65282EB632952222305521140D73D6E4070BEF86BED8"
Assert-Hash (Join-Path $rel "pipnet_partition.bin")  "3E0690DDBF145BDE5DAD58CAEB2A98DFCFCE8C39AEC74834A32454E376713E15"

if (-not $AppOnly) {
    Assert-Hash (Join-Path $rel "bootloader.bin")       "54C23A2E48EE354D64D465EB5F5C6622357AA4FCA0F86C5A607264DCEC0B8DF1"
    Assert-Hash (Join-Path $rel "partition-table.bin")  "31C796E02D427EE582A3ECA7C06A7DAEFEF07480C9BB4B6AF04DF061401D3C33"
}

Write-Host "== Nap (dong Serial Monitor truoc khi chay) ==" -ForegroundColor Cyan
if ($AppOnly) {
    & $py -m esptool --chip esp32s3 -p $Port -b 921600 write_flash `
        0x10000 (Join-Path $rel "app.bin")
} else {
    & $py -m esptool --chip esp32s3 -p $Port -b 921600 write_flash `
        0x0      (Join-Path $rel "bootloader.bin") `
        0x8000   (Join-Path $rel "partition-table.bin") `
        0x10000  (Join-Path $rel "app.bin") `
        0x300000 (Join-Path $rel "pipnet_partition.bin")
}
if ($LASTEXITCODE -ne 0) { throw "esptool loi (exit $LASTEXITCODE)" }

Write-Host "== Xong. Mo Serial o 115200 de xem log: ==" -ForegroundColor Cyan
Write-Host "   & $py tools\watch_live.py --port $Port"
Write-Host "   roi mo http://<IP in ra trong log>/ tren dien thoai cung Wi-Fi"
