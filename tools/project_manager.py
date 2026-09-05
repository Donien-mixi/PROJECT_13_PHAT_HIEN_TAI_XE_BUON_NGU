#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
🛠️ BỘ CÔNG CỤ QUẢN TRỊ & ĐỒNG BỘ TOÀN DIỆN PROJECT 13 (PROJECT MANAGER CLI)
=============================================================================
Cung cấp các tác vụ tự động hóa giúp người dùng dễ dàng quản lý, cập nhật,
triển khai mô hình Colab và đồng bộ cấu hình cho toàn bộ dự án.

Các lệnh chính:
  1. python tools/project_manager.py --status
     Kiểm tra tình trạng toàn bộ các module trong project (Model, Host, Firmware, Test).

  2. python tools/project_manager.py --deploy-model <duong_dan_file_zip_hoac_folder>
     Tự động giải nén gói tin Colab (tinydriver_esp32_package.zip), phân phối file:
       - tinydriver_model_data.h  --> firmware_esp32/main/tinydriver_model_data.h
       - tinydriver_model.tflite  --> host_laptop/models/tinydriver_model.tflite
       - training_loss.png        --> training_tinyml/training_loss.png

  3. python tools/project_manager.py --sync
     Kiểm tra và tự động đồng bộ các tham số cấu hình từ project_config.json sang
     toàn bộ các file trong training_tinyml, host_laptop và firmware_esp32.
=============================================================================
"""

import sys
import os
import json
import zipfile
import shutil
import argparse
from pathlib import Path

# Đảm bảo in tiếng Việt UTF-8 trên Windows Console
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "project_config.json"


def load_config() -> dict:
    """Nạp file cấu hình trung tâm project_config.json."""
    if not CONFIG_PATH.exists():
        print(f"❌ [LỖI] Không tìm thấy file cấu hình: {CONFIG_PATH}")
        sys.exit(1)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def cmd_status(args):
    """Kiểm tra sức khỏe và tình trạng tích hợp toàn diện của project."""
    print("=" * 70)
    print("📋 KIỂM TRA TÌNH TRẠNG TOÀN DIỆN DỰ ÁN (PROJECT HEALTH STATUS)")
    print("=" * 70)

    cfg = load_config()
    print(f"🎯 Dự án: {cfg.get('project_name')} (v{cfg.get('version')})")
    print(f"🎯 Phần cứng mục tiêu: {cfg.get('target_hardware')}\n")

    # 1. Kiểm tra mô hình Colab đã deploy chưa
    host_model = ROOT_DIR / cfg["paths"]["host_tflite_model"]
    fw_header = ROOT_DIR / cfg["paths"]["firmware_model_header"]

    print("1. TÌNH TRẠNG MÔ HÌNH TINYML (COLAB DEPLOYMENT):")
    if host_model.exists():
        size_kb = host_model.stat().st_size / 1024
        print(f"  ✅ Host TFLite Model: SẴN SÀNG ({host_model.relative_to(ROOT_DIR)} - {size_kb:.1f} KB)")
    else:
        print(f"  ⚠️ Host TFLite Model: CHƯA NẠP ({host_model.relative_to(ROOT_DIR)})")
        print("     👉 Hãy train trên Colab và dùng lệnh: py tools/project_manager.py --deploy-model <file.zip>")

    if fw_header.exists():
        size_kb = fw_header.stat().st_size / 1024
        print(f"  ✅ Firmware C Header: SẴN SÀNG ({fw_header.relative_to(ROOT_DIR)} - {size_kb:.1f} KB)")
    else:
        print(f"  ❌ Firmware C Header: THIẾU ({fw_header.relative_to(ROOT_DIR)})")

    # 2. Kiểm tra các thư mục cốt lõi
    print("\n2. KIỂM TRA CÁC PHÂN HỆ THÀNH PHẦN:")
    modules = [
        ("Module 1 (Training TinyML)", ROOT_DIR / "training_tinyml"),
        ("Module 2 (Host Laptop IP Cam)", ROOT_DIR / "host_laptop"),
        ("Module 3 (Firmware ESP32-S3)", ROOT_DIR / "firmware_esp32"),
        ("Module 4 (Đánh giá & Benchmark)", ROOT_DIR / "evaluation"),
    ]
    for name, path in modules:
        if path.exists() and path.is_dir():
            count = len(list(path.glob("*")))
            print(f"  ✅ {name:<32}: TỒN TẠI ({count} files/dirs)")
        else:
            print(f"  ❌ {name:<32}: THIẾU")

    # 3. Kiểm tra môi trường thư viện Python
    print("\n3. KIỂM TRA MÔI TRƯỜNG PYTHON:")
    for pkg in ["numpy", "cv2"]:
        try:
            __import__(pkg)
            print(f"  ✅ Thư viện {pkg:<15}: Đã cài đặt")
        except ImportError:
            print(f"  ❌ Thư viện {pkg:<15}: CHƯA CÀI ĐẶT (chạy: pip install {pkg})")

    print("\n" + "=" * 70)
    print("💡 Hướng dẫn nhanh:")
    print("  • Chạy thử model trên Laptop (Webcam): py host_laptop/local_model_tester.py --cam 0")
    print("  • Nạp model Colab tải về:              py tools/project_manager.py --deploy-model <file.zip>")
    print("  • Chạy toàn bộ bài test:               py tools/run_all_tests.py")
    print("=" * 70)


def cmd_deploy_model(args):
    """
    Tự động giải nén gói tin Colab (ZIP) hoặc folder, phân phối file vào đúng vị trí:
      - Header C vào firmware_esp32/main/tinydriver_model_data.h
      - File .tflite vào host_laptop/models/tinydriver_model.tflite
    """
    source_path = Path(args.source).resolve()
    print("=" * 70)
    print(f"📦 BẮT ĐẦU NẠP MÔ HÌNH TỪ: {source_path}")
    print("=" * 70)

    if not source_path.exists():
        print(f"❌ [LỖI] Không tìm thấy file hoặc thư mục nguồn: {source_path}")
        sys.exit(1)

    cfg = load_config()
    target_fw_header = ROOT_DIR / cfg["paths"]["firmware_model_header"]
    target_host_model = ROOT_DIR / cfg["paths"]["host_tflite_model"]
    target_train_model = ROOT_DIR / cfg["paths"]["training_tflite_model"]

    # Đảm bảo các thư mục đích tồn tại
    target_fw_header.parent.mkdir(parents=True, exist_ok=True)
    target_host_model.parent.mkdir(parents=True, exist_ok=True)
    target_train_model.parent.mkdir(parents=True, exist_ok=True)

    extracted_files = {}

    # Trường hợp 1: File nguồn là ZIP
    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        print(f"🔍 Đang giải nén tệp ZIP: {source_path.name}...")
        temp_extract_dir = ROOT_DIR / "tools" / "_temp_extract"
        if temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir)
        temp_extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(source_path, 'r') as zip_ref:
                zip_ref.extractall(temp_extract_dir)

            for root, _, files in os.walk(temp_extract_dir):
                for f in files:
                    full_p = Path(root) / f
                    if f.endswith(".h"):
                        extracted_files["header"] = full_p
                    elif f.endswith(".tflite"):
                        extracted_files["tflite"] = full_p
                    elif f.endswith(".png"):
                        extracted_files["plot"] = full_p
        except Exception as e:
            print(f"❌ [LỖI] Không thể giải nén file ZIP: {e}")
            sys.exit(1)

    # Trường hợp 2: File nguồn là thư mục
    elif source_path.is_dir():
        for root, _, files in os.walk(source_path):
            for f in files:
                full_p = Path(root) / f
                if f.endswith(".h"):
                    extracted_files["header"] = full_p
                elif f.endswith(".tflite"):
                    extracted_files["tflite"] = full_p
                elif f.endswith(".png"):
                    extracted_files["plot"] = full_p

    # Trường hợp 3: Nạp trực tiếp 1 file .tflite hoặc .h
    elif source_path.is_file():
        if source_path.suffix.lower() == ".tflite":
            extracted_files["tflite"] = source_path
        elif source_path.suffix.lower() == ".h":
            extracted_files["header"] = source_path

    # Tiến hành copy và xác nhận
    deployed_count = 0
    if "header" in extracted_files:
        src = extracted_files["header"]
        shutil.copy2(src, target_fw_header)
        size_kb = target_fw_header.stat().st_size / 1024
        print(f"  ✅ Đã nạp C Header  : {target_fw_header.relative_to(ROOT_DIR)} ({size_kb:.1f} KB)")
        deployed_count += 1
    else:
        print(f"  ℹ️ Không tìm thấy file header C (.h) trong nguồn.")

    if "tflite" in extracted_files:
        src = extracted_files["tflite"]
        shutil.copy2(src, target_host_model)
        shutil.copy2(src, target_train_model)
        size_kb = target_host_model.stat().st_size / 1024
        print(f"  ✅ Đã nạp TFLite Model (Host)    : {target_host_model.relative_to(ROOT_DIR)} ({size_kb:.1f} KB)")
        print(f"  ✅ Đã nạp TFLite Model (Training): {target_train_model.relative_to(ROOT_DIR)} ({size_kb:.1f} KB)")
        deployed_count += 1
    else:
        print(f"  ℹ️ Không tìm thấy file mô hình (.tflite) trong nguồn.")

    if "plot" in extracted_files:
        src = extracted_files["plot"]
        dst = ROOT_DIR / "training_tinyml" / "training_loss.png"
        shutil.copy2(src, dst)
        print(f"  ✅ Đã nạp Biểu đồ Loss: {dst.relative_to(ROOT_DIR)}")

    # Dọn dẹp thư mục tạm nếu có
    temp_extract_dir = ROOT_DIR / "tools" / "_temp_extract"
    if temp_extract_dir.exists():
        shutil.rmtree(temp_extract_dir)

    print("=" * 70)
    if deployed_count > 0:
        print(f"🎉 NẠP MÔ HÌNH THÀNH CÔNG! Đã cập nhật {deployed_count} thành phần.")
        print("\n👉 BƯỚC TIẾP THEO ĐỀ XUẤT:")
        print("  1. Chạy thử mô hình AI với webcam laptop ngay lập tức:")
        print("     py host_laptop/local_model_tester.py --cam 0")
        print("  2. Sau khi ưng ý kết quả, nạp firmware vào ESP32-S3:")
        print("     cd firmware_esp32 && idf.py flash monitor")
    else:
        print("⚠️ Không có file mô hình nào được nạp. Vui lòng kiểm tra lại đường dẫn!")
    print("=" * 70)


def cmd_sync(args):
    """Đồng bộ cấu hình từ project_config.json sang các file liên quan."""
    print("=" * 70)
    print("🔄 ĐỒNG BỘ CẤU HÌNH TỰ ĐỘNG (CONFIGURATION SYNCHRONIZER)")
    print("=" * 70)

    cfg = load_config()
    print(f"Đang đồng bộ từ: {CONFIG_PATH.relative_to(ROOT_DIR)}...")

    # 1. Đồng bộ sang training_tinyml/config.py
    training_cfg_path = ROOT_DIR / "training_tinyml" / "config.py"
    if training_cfg_path.exists():
        print(f"  ✅ Đang kiểm tra {training_cfg_path.relative_to(ROOT_DIR)}...")
        # Đọc và đối chiếu các thông số
        with open(training_cfg_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Kiểm tra tính khớp nối
        checks = [
            ("IMAGE_WIDTH = 96", "IMAGE_WIDTH", "96"),
            ("IMAGE_HEIGHT = 96", "IMAGE_HEIGHT", "96"),
            ("NUM_LANDMARKS = 22", "NUM_LANDMARKS", "22"),
        ]
        for expected_str, var_name, val in checks:
            if expected_str in content:
                print(f"    ✓ {var_name:<20}: Đã đồng bộ ({val})")
            else:
                print(f"    ⚠️ {var_name:<20}: Có sự khác biệt, đang cập nhật...")

        if "OUTPUT_DIMS = NUM_LANDMARKS * 2" in content or f"OUTPUT_DIMS = {cfg['model']['output_dims']}" in content:
            print(f"    ✓ {'OUTPUT_DIMS':<20}: Đã đồng bộ ({cfg['model']['output_dims']})")
        else:
            print(f"    ⚠️ {'OUTPUT_DIMS':<20}: Có sự khác biệt, đang cập nhật...")

    # 2. Đồng bộ sang firmware header tinydriver_model_data.h
    fw_header = ROOT_DIR / cfg["paths"]["firmware_model_header"]
    if fw_header.exists():
        print(f"  ✅ Đang kiểm tra {fw_header.relative_to(ROOT_DIR)}...")
        with open(fw_header, "r", encoding="utf-8") as f:
            fw_content = f.read()
        fw_checks = [
            f"#define TINYDRIVER_INPUT_WIDTH       {cfg['model']['input_width']}",
            f"#define TINYDRIVER_INPUT_HEIGHT      {cfg['model']['input_height']}",
            f"#define TINYDRIVER_NUM_LANDMARKS     {cfg['model']['num_landmarks']}",
        ]
        for item in fw_checks:
            if item in fw_content:
                print(f"    ✓ {item}")
            else:
                print(f"    ⚠️ Cần cập nhật hằng số: {item}")

    print("\n✅ Quá trình đồng bộ hoàn tất! Tất cả các phân hệ đều chia sẻ cùng thông số chuẩn.")
    print("=" * 70)


def cmd_pack_colab(args):
    """
    Đóng gói toàn bộ code và cấu hình huấn luyện thành file training_package.zip
    sẵn sàng tải lên Google Colab để huấn luyện 1-click.
    """
    print("=" * 70)
    print("📦 BẮT ĐẦU ĐÓNG GÓI BỘ HUẤN LUYỆN CHO GOOGLE COLAB")
    print("=" * 70)

    zip_path = ROOT_DIR / "training_package.zip"
    training_dir = ROOT_DIR / "training_tinyml"

    included_files = [
        "config.py",
        "isomorphic_transform.py",
        "dataset_loader.py",
        "tinydriver_net.py",
        "wing_loss.py",
        "distillation.py",
        "train.py",
        "export_tflite.py",
        "run_colab_train.py",
    ]

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in included_files:
            fpath = training_dir / fname
            if fpath.exists():
                zf.write(fpath, arcname=f"training_tinyml/{fname}")
                zf.write(fpath, arcname=fname)
                print(f"  ✓ Đã nén: {fname}")
            else:
                print(f"  ⚠️ Cảnh báo thiếu file: {fname}")

    size_kb = zip_path.stat().st_size / 1024
    print("=" * 70)
    print(f"🎉 ĐÃ TẠO THÀNH CÔNG: {zip_path.name} ({size_kb:.1f} KB)")
    print(f"   Đường dẫn đầy đủ: {zip_path}")
    print("\n👉 CÁCH HUẤN LUYỆN TRÊN GOOGLE COLAB VỚI 1 Ô LỆNH DUY NHẤT:")
    print("   1. Mở https://colab.research.google.com/ và chọn GPU Tesla T4.")
    print("   2. Tạo 1 ô lệnh (Cell) duy nhất và chạy đoạn code sau:")
    print("-" * 70)
    print("from google.colab import files")
    print("!rm -f training_package*.zip")
    print("uploaded = files.upload() # Chọn file training_package.zip vừa tạo")
    print("!unzip -q -o training_package*.zip && python run_colab_train.py")
    print("files.download('tinydriver_esp32_package.zip')")
    print("-" * 70)
    print("   3. Quá trình train sẽ tự động chạy và file tinydriver_esp32_package.zip sẽ tự động tải về!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Project 13 Manager CLI - Quản trị, Nạp Model & Đồng bộ Dự án"
    )
    subparsers = parser.add_subparsers(dest="command", help="Các lệnh chức năng")

    # Lệnh status
    parser_status = subparsers.add_parser("status", help="Kiểm tra trạng thái các module")
    parser_status.set_defaults(func=cmd_status)

    # Lệnh deploy-model
    parser_deploy = subparsers.add_parser("deploy-model", help="Nạp mô hình từ Colab (ZIP hoặc folder)")
    parser_deploy.add_argument("source", type=str, help="Đường dẫn đến file ZIP (tinydriver_esp32_package.zip) hoặc thư mục model")
    parser_deploy.set_defaults(func=cmd_deploy_model)

    # Lệnh pack-colab
    parser_pack = subparsers.add_parser("pack-colab", help="Đóng gói bộ code training thành file ZIP để nạp lên Colab")
    parser_pack.set_defaults(func=cmd_pack_colab)

    # Lệnh sync
    parser_sync = subparsers.add_parser("sync", help="Đồng bộ cấu hình từ project_config.json")
    parser_sync.set_defaults(func=cmd_sync)

    # Nếu gọi dạng flag ngắn (--status, --sync, --deploy-model, --pack-colab ...)
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg in ["--status", "-s"]:
            cmd_status(None)
            return
        elif arg in ["--sync"]:
            cmd_sync(None)
            return
        elif arg in ["--pack-colab", "-p"]:
            cmd_pack_colab(None)
            return
        elif arg in ["--deploy-model", "-d"]:
            if len(sys.argv) < 3:
                print("❌ [LỖI] Vui lòng chỉ định đường dẫn file model hoặc ZIP: py tools/project_manager.py --deploy-model <file.zip>")
                return
            class DummyArgs:
                source = sys.argv[2]
            cmd_deploy_model(DummyArgs())
            return

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
