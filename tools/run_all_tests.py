#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
🧪 BỘ CHẠY KIỂM CHUẨN TỰ ĐỘNG TOÀN DIỆN 1-CLICK (MASTER TEST RUNNER)
=============================================================================
Đồ án 13: Hệ thống phát hiện tài xế ngủ gật & mất tập trung (Edge AI ESP32-S3)

Chạy tuần tự toàn bộ các bài kiểm thử tự động của cả 4 giai đoạn:
  1. Kiểm tra tính đồng bộ cấu hình (project_config.json)
  2. Kiểm thử Pipeline Giai đoạn 2 (Square crop, TCP/UDP stream, HUD)
  3. Kiểm thử Thuật toán Giai đoạn 3 (POSIT PnP C++, ADAS FSM State Machine)
  4. Kiểm chuẩn Định lượng Giai đoạn 4 (Sai số hình học, Độ trễ, 1.200 Mẫu biên)

Xuất bảng tổng kết nghiệm thu chất lượng kỹ thuật toàn diện.
=============================================================================
"""

import sys
import os
import time
import subprocess
from pathlib import Path

# Đảm bảo in tiếng Việt UTF-8
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

ROOT_DIR = Path(__file__).resolve().parent.parent


def run_command(cmd, desc):
    """Thực thi một lệnh và trả về (status, duration, output)."""
    print(f"\n▶️ ĐANG CHẠY: {desc}...")
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=120
        )
        duration = time.time() - t0
        passed = (proc.returncode == 0)
        return passed, duration, proc.stdout
    except Exception as e:
        duration = time.time() - t0
        return False, duration, str(e)


def main():
    print("=" * 75)
    print("🚀 BẮT ĐẦU CHẠY BỘ KIỂM CHUẨN TOÀN DIỆN 1-CLICK CHO TOÀN BỘ PROJECT 13")
    print("=" * 75)

    py_exe = sys.executable

    cpp_test_exe = ROOT_DIR / "firmware_esp32" / "test_embedded_algorithms.exe"

    test_suites = [
        (
            "Cấu hình & Đồng bộ",
            [py_exe, "tools/project_manager.py", "--sync"],
            "Kiểm tra tính đồng bộ cấu hình từ project_config.json"
        ),
        (
            "Giai đoạn 2 (Host IP Cam & HUD)",
            [py_exe, "host_laptop/test_phase2_pipeline.py"],
            "Kiểm thử 5 bài test TCP stream, Isomorphic crop, HUD Telemetry"
        ),
        (
            "Giai đoạn 3.1 (PnP & FaceTracker)",
            [py_exe, "tools/verify_pnp_and_aug.py"],
            "Kiểm thử giải thuật POSIT PnP, FaceTracker Hysteresis, Eyeglasses Aug"
        ),
        (
            "Giai đoạn 3.2 (Cơ học & 3-Way Sampling)",
            [py_exe, "tools/verify_fixes.py"],
            "Kiểm thử giải phẫu cằm ngáp, cân bằng 3 trạng thái (33/33/33) & Focal Loss"
        ),
        (
            "Giai đoạn 4.1 (Kiểm chuẩn Hình học)",
            [py_exe, "evaluation/verify_geometric_distortion.py"],
            "Xác nhận độ méo hình học Isomorphic = 0.00% so với Naive"
        ),
        (
            "Giai đoạn 4.2 (Độ trễ & FPS)",
            [py_exe, "evaluation/benchmark_latency_profile.py"],
            "Đo đạc độ trễ từng khâu pipeline và thông lượng 36.4 FPS"
        ),
        (
            "Giai đoạn 4.3 (1.200 Mẫu biên)",
            [py_exe, "evaluation/test_edge_cases.py"],
            "Kiểm thử độ chính xác trên 1.200 mẫu môi trường khắc nghiệt"
        ),
    ]

    if cpp_test_exe.exists():
        test_suites.insert(4, (
            "Giai đoạn 3.3 (Thuật toán C++ ESP32)",
            [str(cpp_test_exe)],
            "Kiểm thử POSIT PnP (<2us) & ADAS FSM viết bằng pure C++"
        ))

    results = []
    total_start = time.time()

    for category, cmd, desc in test_suites:
        passed, dur, output = run_command(cmd, desc)
        results.append({
            "category": category,
            "desc": desc,
            "passed": passed,
            "duration": dur,
            "output": output
        })
        status_str = "✅ PASS" if passed else "❌ FAILED"
        print(f"   {status_str} ({dur:.2f}s)")
        if not passed:
            print(f"\n--- ERROR LOG for {category} ---\n{output}\n--- END ERROR LOG ---\n")

    total_time = time.time() - total_start

    print("\n" + "=" * 75)
    print("📊 BẢNG TỔNG KẾT NGHIỆM THU CHẤT LƯỢNG TOÀN HỆ THỐNG (TEST SUMMARY)")
    print("=" * 75)
    print(f"{'Hạng mục kiểm thử':<35} | {'Trạng thái':<12} | {'Thời gian':<10}")
    print("-" * 75)

    all_passed = True
    for r in results:
        status_text = "✅ PASSED" if r["passed"] else "❌ FAILED"
        if not r["passed"]:
            all_passed = False
        print(f"{r['category']:<35} | {status_text:<12} | {r['duration']:>6.2f}s")

    print("-" * 75)
    total_status = "🎉 TẤT CẢ BÀI TEST ĐÃ VƯỢT QUA 100%!" if all_passed else "⚠️ CÓ BÀI TEST CHƯA VƯỢT QUA!"
    print(f"Tổng kết: {total_status} (Tổng thời gian: {total_time:.2f}s)")
    print("=" * 75)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
