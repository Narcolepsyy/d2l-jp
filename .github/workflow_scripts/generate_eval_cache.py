#!/usr/bin/env python3
"""Generate stub ipynb files from markdown sources to pre-populate d2lbook's eval cache.

d2lbook's eval() function skips files that already exist in _build/eval/.
Without cached ipynb files, eval() tries to convert md→ipynb via notedown
which crashes on {.python .input} code fences without S3 cache.

This script creates minimal valid ipynb files (with empty cells) so
d2lbook skips the conversion. Cells are empty to avoid triggering
remove_slide_marks() assertions on unmatched bold markers.
"""
import os
import json
import glob
import configparser


def create_stub_notebook():
    """Create a minimal valid ipynb with a single empty markdown cell."""
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [""]
            }
        ],
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


def main():
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
    nb = create_stub_notebook()
    for md_file in sorted(set(md_files)):
        ipynb_path = os.path.join(eval_dir, md_file.replace('.md', '.ipynb'))
        os.makedirs(os.path.dirname(ipynb_path), exist_ok=True)

        with open(ipynb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        count += 1

    print(f"Generated {count} stub ipynb files in {eval_dir}/")


if __name__ == '__main__':
    main()
