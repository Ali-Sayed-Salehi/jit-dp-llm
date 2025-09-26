#!/usr/bin/env python3
"""
Patch DeepSpeed's op_builder/builder.py so that:

    def cpu_arch(self):

always returns "-march=skylake-avx512".

Usage:
  python patch_cpu_arch.py [--path /custom/site-packages] [--dry-run]

- By default, searches sys.path for:
    deepspeed/ops/op_builder/builder.py
- You can point directly at a site-packages root via --path (optional).
- Creates a backup next to builder.py (e.g., builder.py.bak-2025-09-25-2030).
"""

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

TARGET_RET = '-march=skylake-avx512'
RELATIVE_BUILDER = Path('deepspeed/ops/op_builder/builder.py')

def find_builder_py(explicit_site_packages: str | None) -> Path:
    if explicit_site_packages:
        p = Path(explicit_site_packages).expanduser().resolve() / RELATIVE_BUILDER
        if p.exists():
            return p
        print(f"ERROR: Not found at --path: {p}", file=sys.stderr)
        sys.exit(1)

    # Search every entry in sys.path (covers venvs, different Python versions, lib vs lib64, etc.)
    candidates = []
    for entry in sys.path:
        try:
            root = Path(entry)
        except Exception:
            continue
        if not root.exists() or not root.is_dir():
            continue
        p = (root / RELATIVE_BUILDER)
        if p.exists():
            candidates.append(p.resolve())

    if not candidates:
        print("ERROR: Could not locate deepspeed/ops/op_builder/builder.py via sys.path.", file=sys.stderr)
        sys.exit(1)

    # If multiple are found, pick the first deterministically (sorted for stability)
    candidates = sorted(candidates)
    return candidates[0]

def already_patched(text: str) -> bool:
    # Look for a cpu_arch(self) that immediately returns our target
    pat = r"""def\s+cpu_arch\s*\(\s*self\s*\)\s*:\s*return\s+['"]""" + re.escape(TARGET_RET) + r"""['"]"""
    return re.search(pat, text, flags=re.DOTALL) is not None

def replace_method(text: str) -> tuple[str, bool]:
    """
    Replace the entire body of def cpu_arch(self): ... with a single return.
    We look ahead to the next top-level def/class/decorator or EOF.
    """
    func_pat = re.compile(
        r"(^def\s+cpu_arch\s*\(\s*self\s*\)\s*:\s*[\s\S]*?)(?=^def\s|^class\s|^@|\Z)",
        flags=re.MULTILINE
    )
    m = func_pat.search(text)
    if not m:
        return text, False

    replacement = (
        "def cpu_arch(self):\n"
        "    return '-march=skylake-avx512'\n\n"
    )
    new_text = text[:m.start()] + replacement + text[m.end():]
    return new_text, True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", help="Optional site-packages root to search within (directory that contains 'deepspeed/')", default=None)
    ap.add_argument("--dry-run", action="store_true", help="Show diff but do not write changes")
    args = ap.parse_args()

    builder_py = find_builder_py(args.path)
    print(f"[INFO] builder.py: {builder_py}")

    original = builder_py.read_text(encoding="utf-8")

    if already_patched(original):
        print("[INFO] Already patched; no changes needed.")
        return

    patched, replaced = replace_method(original)
    if not replaced:
        print("[ERROR] Could not locate 'def cpu_arch(self):' in the file.", file=sys.stderr)
        sys.exit(2)

    if args.dry_run:
        import difflib
        diff = difflib.unified_diff(
            original.splitlines(True),
            patched.splitlines(True),
            fromfile=str(builder_py),
            tofile=str(builder_py) + " (patched)"
        )
        print("[DRY-RUN] Showing diff:")
        sys.stdout.writelines(diff)
        return

    # Backup
    ts = dt.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    backup = builder_py.with_suffix(builder_py.suffix + f".bak-{ts}")
    backup.write_text(original, encoding="utf-8")
    print(f"[INFO] Backup created: {backup}")

    # Write
    builder_py.write_text(patched, encoding="utf-8")
    print(f"[INFO] Patch applied. cpu_arch(self) now returns: {TARGET_RET}")

if __name__ == "__main__":
    main()
