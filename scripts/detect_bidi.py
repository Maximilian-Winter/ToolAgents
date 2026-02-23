import pathlib

TARGET_ROOT = pathlib.Path("examples/execution_integrity")
SUFFIXES = {".py", ".md", ".txt"}

RANGES = [
    (0x202A, 0x202E),
    (0x2066, 0x2069),
]
SINGLES = {0x200E, 0x200F, 0x061C}


def is_control(cp):
    if cp in SINGLES:
        return True
    for start, end in RANGES:
        if start <= cp <= end:
            return True
    return False


def scan_file(path):
    hits = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    for lineno, line in enumerate(text.splitlines(), start=1):
        for ch in line:
            cp = ord(ch)
            if is_control(cp):
                hits.append((lineno, cp))
    return hits


def main():
    hit_count = 0
    for path in sorted(TARGET_ROOT.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUFFIXES:
            continue
        hits = scan_file(path)
        for lineno, cp in hits:
            hit_count += 1
            print(f"{path}:{lineno}: U+{cp:04X}")

    if hit_count == 0:
        print("NO_BIDI_CONTROL_CHARS")
        return 0

    print(f"TOTAL_HITS: {hit_count}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
