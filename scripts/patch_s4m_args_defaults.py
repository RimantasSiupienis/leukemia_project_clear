#!/usr/bin/env python3
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNPY = REPO_ROOT / "models" / "s4m_model" / "s4m_official" / "s4m" / "run.py"

def add_arg_block(text: str, anchor_pattern: str, block: str) -> str:
    m = re.search(anchor_pattern, text)
    if not m:
        raise RuntimeError(f"Anchor not found: {anchor_pattern}")
    idx = m.end()
    return text[:idx] + "\n" + block + "\n" + text[idx:]

s = RUNPY.read_text()

# If already present, do nothing
if "d_var" in s:
    print("ℹ d_var already appears in run.py; not patching.")
    raise SystemExit(0)

# Find a good anchor near other model-dim args
# We'll insert after the d_model argument definition (exists in your args printout).
anchor = r"parser\.add_argument\(\s*'--d_model'[\s\S]*?\)\s*"

block = "\n".join([
    "parser.add_argument('--d_var', type=int, default=None, help='(S4M) feature dimension override; if None uses enc_in')",
])

s2 = add_arg_block(s, anchor, block)

RUNPY.write_text(s2)
print(f" Patched: {RUNPY}")
print("Added argparse: --d_var (default None)")
