# =============================================================================
# Tìm cổng COM của board ESP32 đang cắm (CH343 / CH340 / CP210x / USB-JTAG)
# Dùng:  powershell -ExecutionPolicy Bypass -File tools\find_esp_port.ps1
# =============================================================================

$cands = Get-PnpDevice -Class Ports -PresentOnly -ErrorAction SilentlyContinue |
    Where-Object { $_.FriendlyName -match '\(COM\d+\)' } |
    ForEach-Object {
        if ($_.FriendlyName -match '\((COM\d+)\)') {
            [PSCustomObject]@{ Port = $Matches[1]; Name = $_.FriendlyName }
        }
    }

if (-not $cands) {
    Write-Host "X Khong tim thay cong COM nao dang cam. Kiem tra cap USB / nguon board." -ForegroundColor Red
    Write-Host "  (OV5640 an dong lon -> thu cap USB tot, cong USB 3.0, hoac hub co nguon)" -ForegroundColor Yellow
    exit 1
}

Write-Host "Cac cong COM dang hoat dong:" -ForegroundColor Cyan
$cands | Format-Table -AutoSize

$esp = $cands |
    Where-Object { $_.Name -match 'CH34|CP210|USB-Enhanced|USB-Serial|JTAG|Espressif' } |
    Select-Object -First 1

if ($esp) {
    Write-Host "=> Board ESP32 co the o cong: $($esp.Port)   ($($esp.Name))" -ForegroundColor Green
    Write-Host "   Nap firmware bang:  idf.py -p $($esp.Port) flash monitor" -ForegroundColor Yellow
} else {
    Write-Host "! Khong thay chip USB-Serial quen thuoc. Kiem tra driver CH343/CP210x." -ForegroundColor Yellow
}
