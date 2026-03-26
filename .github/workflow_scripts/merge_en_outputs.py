#!/usr/bin/env python3
"""Merge pre-executed outputs from d2l-en release notebooks into d2l-jp eval cache.

Downloads d2l-en-1.0.3.zip from GitHub releases, extracts the PyTorch notebooks,
and copies their execution outputs (plots, text, tables) into the matching
d2l-jp notebooks in _build/eval/.

The code cells are identical between EN and JP — only markdown text differs.
Outputs are matched by code cell index position.
"""
import json
import os
import sys
import zipfile
import urllib.request
from pathlib import Path

EN_ZIP_URL = "https://github.com/d2l-ai/d2l-en/releases/download/v1.0.3/d2l-en-1.0.3.zip"
EN_ZIP_PATH = "/tmp/d2l-en-1.0.3.zip"
EN_EXTRACT_DIR = "/tmp/d2l-en"


def download_en_release():
    """Download d2l-en release zip if not already cached."""
    if os.path.exists(EN_ZIP_PATH):
        print(f"Using cached {EN_ZIP_PATH}")
        return
    print(f"Downloading {EN_ZIP_URL}...")
    urllib.request.urlretrieve(EN_ZIP_URL, EN_ZIP_PATH)
    print(f"Downloaded ({os.path.getsize(EN_ZIP_PATH) // 1024 // 1024} MB)")


def extract_en_release():
    """Extract the zip file."""
    if os.path.exists(os.path.join(EN_EXTRACT_DIR, "pytorch")):
        print(f"Using cached extraction at {EN_EXTRACT_DIR}")
        return
    print("Extracting...")
    with zipfile.ZipFile(EN_ZIP_PATH, 'r') as z:
        z.extractall(EN_EXTRACT_DIR)
    print("Extracted")


def merge_outputs(jp_eval_dir, tab="pytorch"):
    """Merge EN outputs into JP notebooks."""
    en_dir = Path(EN_EXTRACT_DIR) / tab
    jp_dir = Path(jp_eval_dir)

    if not en_dir.exists():
        print(f"ERROR: {en_dir} not found")
        sys.exit(1)
    if not jp_dir.exists():
        print(f"ERROR: {jp_dir} not found")
        sys.exit(1)

    merged = 0
    output_cells = 0
    skipped = 0

    for en_chapter in sorted(en_dir.iterdir()):
        if not en_chapter.is_dir() or not en_chapter.name.startswith("chapter_"):
            continue

        jp_chapter = jp_dir / en_chapter.name
        if not jp_chapter.exists():
            continue

        for en_nb_path in sorted(en_chapter.glob("*.ipynb")):
            jp_nb_path = jp_chapter / en_nb_path.name
            if not jp_nb_path.exists():
                continue

            try:
                en_nb = json.loads(en_nb_path.read_text())
                jp_nb = json.loads(jp_nb_path.read_text())
            except (json.JSONDecodeError, Exception):
                skipped += 1
                continue

            en_code = [c for c in en_nb["cells"] if c["cell_type"] == "code"]
            jp_code = [c for c in jp_nb["cells"] if c["cell_type"] == "code"]

            if len(en_code) != len(jp_code):
                skipped += 1
                continue

            cells_added = 0
            for en_cell, jp_cell in zip(en_code, jp_code):
                if en_cell.get("outputs"):
                    jp_cell["outputs"] = en_cell["outputs"]
                    jp_cell["execution_count"] = en_cell.get("execution_count")
                    cells_added += 1

            if cells_added > 0:
                jp_nb_path.write_text(json.dumps(jp_nb, indent=1, ensure_ascii=False))
                output_cells += cells_added
                merged += 1

    print(f"Merged {merged} notebooks, {output_cells} output cells ({skipped} skipped)")
    return merged


def main():
    # Usage: merge_en_outputs.py <base_eval_dir> <tab1> [tab2] [tab3] [tab4]
    base_eval_dir = sys.argv[1] if len(sys.argv) > 1 else "_build/eval"
    tabs = sys.argv[2:] if len(sys.argv) > 2 else ["pytorch"]

    download_en_release()
    extract_en_release()

    # d2lbook eval directory naming:
    # default tab (first tab = pytorch) → _build/eval/
    # other tabs → _build/eval_<tab>/
    default_tab = tabs[0] if tabs else "pytorch"

    for tab in tabs:
        if tab == default_tab:
            eval_dir = base_eval_dir
        else:
            # Convert _build/eval → _build/eval_<tab>
            eval_dir = f"{base_eval_dir}_{tab}"

        if not os.path.exists(eval_dir):
            print(f"Skipping {tab}: {eval_dir} not found")
            continue

        print(f"\n--- Merging {tab} outputs into {eval_dir} ---")
        merge_outputs(eval_dir, tab)

    # MXNet fallback for MXNet-only chapters
    MXNET_ONLY_CHAPTERS = ["chapter_recommender-systems"]
    if "mxnet" in tabs:
        mxnet_dir = f"{base_eval_dir}_mxnet"
        if os.path.exists(mxnet_dir):
            merge_mxnet_fallback(mxnet_dir, MXNET_ONLY_CHAPTERS)
    # Also apply to default eval dir for chapters with no pytorch code
    merge_mxnet_fallback(base_eval_dir, MXNET_ONLY_CHAPTERS)


def merge_mxnet_fallback(jp_eval_dir, chapters):
    """For MXNet-only chapters, copy EN MXNet notebooks with JP markdown."""
    en_mx_dir = Path(EN_EXTRACT_DIR) / "mxnet"
    jp_dir = Path(jp_eval_dir)

    if not en_mx_dir.exists():
        return

    copied = 0
    for ch_name in chapters:
        en_ch = en_mx_dir / ch_name
        jp_ch = jp_dir / ch_name
        if not en_ch.exists() or not jp_ch.exists():
            continue

        for en_nb_path in sorted(en_ch.glob("*.ipynb")):
            jp_nb_path = jp_ch / en_nb_path.name
            try:
                en_nb = json.loads(en_nb_path.read_text())
            except Exception:
                continue

            has_outputs = any(
                c.get("outputs") for c in en_nb["cells"] if c["cell_type"] == "code"
            )
            if not has_outputs:
                continue

            # Overlay JP markdown onto EN notebook
            if jp_nb_path.exists():
                try:
                    jp_nb = json.loads(jp_nb_path.read_text())
                    jp_md = [c for c in jp_nb["cells"] if c["cell_type"] == "markdown"]
                    md_idx = 0
                    for cell in en_nb["cells"]:
                        if cell["cell_type"] == "markdown" and md_idx < len(jp_md):
                            cell["source"] = jp_md[md_idx]["source"]
                            md_idx += 1
                except Exception:
                    pass

            jp_nb_path.write_text(json.dumps(en_nb, indent=1, ensure_ascii=False))
            copied += 1

    if copied:
        print(f"MXNet fallback: {copied} notebooks updated")


if __name__ == "__main__":
    main()
