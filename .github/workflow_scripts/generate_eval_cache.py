#!/usr/bin/env python3
"""Generate stub ipynb files from markdown sources to pre-populate d2lbook's eval cache.

d2lbook's eval() function skips files that already exist in _build/eval/.
Without cached ipynb files, eval() tries to convert md→ipynb via notedown
which crashes on {.python .input} code fences.

This script creates minimal valid ipynb files so d2lbook skips the conversion.
"""
import os
import json
import glob
import re

def md_to_stub_ipynb(md_path):
    """Convert a markdown file to a minimal ipynb with markdown-only cells."""
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Strip code blocks entirely - we can't evaluate them anyway
    # Keep only markdown content
    lines = content.split('\n')
    cells = []
    current_md = []
    in_code_block = False

    for line in lines:
        if line.startswith('```') and not in_code_block:
            # Start of code block - flush markdown
            if current_md:
                cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [l + '\n' for l in current_md]
                })
                current_md = []
            in_code_block = True
            # Create empty code cell
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": []
            })
        elif line.startswith('```') and in_code_block:
            in_code_block = False
        elif not in_code_block:
            current_md.append(line)

    if current_md:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [l + '\n' for l in current_md]
        })

    if not cells:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [""]
        })

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.9.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    return nb


def main():
    import configparser
    config = configparser.ConfigParser()
    config.read('config.ini')

    notebooks_pattern = config.get('build', 'notebooks', fallback='*.md */*.md')
    exclusions = config.get('build', 'exclusions', fallback='').split()

    eval_dir = '_build/eval'
    os.makedirs(eval_dir, exist_ok=True)

    # Find all markdown files
    md_files = []
    for pattern in notebooks_pattern.split():
        md_files.extend(glob.glob(pattern))

    # Filter exclusions
    excluded = set()
    for excl in exclusions:
        excluded.update(glob.glob(excl))

    md_files = [f for f in md_files if f not in excluded]

    count = 0
    for md_file in sorted(set(md_files)):
        ipynb_path = os.path.join(eval_dir, md_file.replace('.md', '.ipynb'))
        os.makedirs(os.path.dirname(ipynb_path), exist_ok=True)

        nb = md_to_stub_ipynb(md_file)
        with open(ipynb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        count += 1

    print(f"Generated {count} stub ipynb files in {eval_dir}/")


if __name__ == '__main__':
    main()
