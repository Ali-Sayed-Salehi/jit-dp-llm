#!/usr/bin/env python3
"""
Patch DeepSpeed's parameter_offload.py:
Replace
    # post backward hook
    if not hasattr(module, "post_bwd_fn"):
with
    # post backward hook
    if "post_bwd_fn" not in module.__dict__:
"""

import os
import re
import site
import sys
from pathlib import Path
from typing import Optional

def find_target() -> Optional[Path]:
    # Prefer venv site-packages if running inside the image’s venv
    candidates = []
    try:
        candidates.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        usr = site.getusersitepackages()
        if usr:
            candidates.append(usr)
    except Exception:
        pass

    # Fallback: probe common venv path if site-packages not helpful
    candidates.extend([
        "/opt/venv/lib/python3.12/site-packages",
        "/opt/venv/lib/python3.11/site-packages",
        "/opt/venv/lib/python3.10/site-packages",
    ])

    seen = set()
    for base in candidates:
        if not base or base in seen:
            continue
        seen.add(base)
        p = Path(base) / "deepspeed" / "runtime" / "zero" / "parameter_offload.py"
        if p.is_file():
            return p
    return None

def main() -> int:
    target = find_target()
    if not target:
        print("ERROR: parameter_offload.py not found under site-packages/deepspeed/runtime/zero/", file=sys.stderr)
        return 2

    print(f"Patching: {target}")
    src = target.read_text(encoding="utf-8")

    pattern = re.compile(
        r'(#\s*post backward hook\s*\n\s*)'
        r'if\s+not\s+hasattr\(\s*module\s*,\s*"post_bwd_fn"\s*\)\s*:',
        flags=re.M
    )

    new_src, n = pattern.subn(r'\1if "post_bwd_fn" not in module.__dict__:', src, count=1)
    if n == 0:
        print("ERROR: Expected pattern not found; no changes made.", file=sys.stderr)
        return 3

    # Backup then write
    backup = target.with_suffix(".py.bak")
    backup.write_text(src, encoding="utf-8")
    target.write_text(new_src, encoding="utf-8")
    print(f"Patch applied successfully. Backup at: {backup}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
