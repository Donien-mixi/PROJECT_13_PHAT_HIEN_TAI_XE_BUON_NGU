"""Convert a .tflite model into a C header (rodata) for the ESP32 firmware.

Usage:
  py -3.11 tools/tflite_to_c_header.py <model.tflite> <out.h> <symbol> [guard]

Emits:
  const unsigned char <symbol>[] = { ... };
  const int <symbol>_len = <N>;
and stamps the SHA-256 of the source model in the header comment.
"""
import hashlib
import os
import sys


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 1
    src, dst, sym = sys.argv[1], sys.argv[2], sys.argv[3]
    guard = sys.argv[4] if len(sys.argv) > 4 else "BLAZE_IQ_MODEL_DATA_H"
    with open(src, "rb") as f:
        data = f.read()
    sha = hashlib.sha256(data).hexdigest().upper()
    lines = []
    lines.append("// AUTO-GENERATED - do not edit.")
    lines.append(f"// source : {os.path.basename(src)}")
    lines.append(f"// bytes  : {len(data)}")
    lines.append(f"// sha256 : {sha}")
    lines.append(f"#ifndef {guard}")
    lines.append(f"#define {guard}")
    lines.append("")
    lines.append(f"extern const unsigned char {sym}[];")
    lines.append(f"extern const int {sym}_len;")
    lines.append("")
    lines.append(f"const unsigned char {sym}[] = {{")
    for i in range(0, len(data), 20):
        chunk = ",".join(f"0x{b:02x}" for b in data[i:i + 20])
        lines.append("  " + chunk + ",")
    lines.append("};")
    lines.append(f"const int {sym}_len = {len(data)};")
    lines.append("")
    lines.append(f"#endif  // {guard}")
    with open(dst, "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{dst}: {len(data)} bytes sha256={sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
