#!/usr/bin/env python3
"""Generate proper ipynb files from markdown sources for d2lbook's eval cache.

d2lbook's eval() crashes when converting md→ipynb via notedown because
{.python .input} code fences produce invalid notebook cells. This script
replaces notedown with a simple parser that:
1. Splits markdown into text and code cells
2. Handles {.python .input} code fences
3. Filters to only keep the specified tab (e.g., pytorch)
4. Strips d2lbook-specific markers that crash later pipeline steps
"""
import os
import json
import glob
import re
import sys
import configparser


def parse_md_to_cells(content, tab='pytorch'):
    """Parse d2l markdown into notebook cells.

    Handles:
    - ```{.python .input} code fences
    - #@tab markers for framework-specific code
    - :begin_tab: / :end_tab: blocks
    - Regular markdown content
    """
    lines = content.split('\n')
    cells = []
    current_text = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check for code block start
        if line.startswith('```'):
            # Flush accumulated markdown
            if current_text:
                src = '\n'.join(current_text)
                if src.strip():
                    cells.append(make_md_cell(src))
                current_text = []

            # Parse code block
            code_lines = []
            cell_tab = None
            i += 1

            while i < len(lines) and not lines[i].startswith('```'):
                code_line = lines[i]

                # Check for #@tab directive
                if code_line.strip().startswith('#@tab'):
                    tab_value = code_line.strip().replace('#@tab', '').strip()
                    cell_tab = tab_value
                    i += 1
                    continue

                code_lines.append(code_line)
                i += 1

            # Skip the closing ```
            if i < len(lines):
                i += 1

            # Decide whether to include this code cell
            code_src = '\n'.join(code_lines)
            if cell_tab is None or cell_tab == 'all' or cell_tab == tab:
                if code_src.strip():
                    cells.append(make_code_cell(code_src))

        # Check for :begin_tab: / :end_tab: blocks
        elif line.strip().startswith(':begin_tab:'):
            tab_match = re.search(r':begin_tab:`(\w+)`', line)
            block_tab = tab_match.group(1) if tab_match else None
            block_lines = []
            i += 1

            while i < len(lines) and not lines[i].strip().startswith(':end_tab:'):
                block_lines.append(lines[i])
                i += 1

            # Skip :end_tab:
            if i < len(lines):
                i += 1

            # Only include content for matching tab or 'all'
            if block_tab is None or block_tab == 'all' or block_tab == tab:
                block_content = '\n'.join(block_lines)
                if block_content.strip():
                    current_text.append(block_content)

        else:
            current_text.append(line)
            i += 1

    # Flush remaining markdown
    if current_text:
        src = '\n'.join(current_text)
        if src.strip():
            cells.append(make_md_cell(src))

    # Ensure at least one cell
    if not cells:
        cells.append(make_md_cell(' '))

    return cells


def make_md_cell(source):
    """Create a markdown cell, stripping slide marks."""
    # Remove slide mark patterns that crash d2lbook's remove_slide_marks
    # These are (**  **) pairs used for slide highlighting
    source = re.sub(r'\(\*\*', '', source)
    source = re.sub(r'\*\*\)', '', source)
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.split('\n') if source else ['']
    }


def make_code_cell(source):
    """Create a code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.split('\n') if source else ['']
    }


def create_notebook(cells):
    """Wrap cells in a notebook structure."""
    return {
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


def main():
    tab = sys.argv[1] if len(sys.argv) > 1 else 'pytorch'

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

        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()

        cells = parse_md_to_cells(content, tab=tab)
        nb = create_notebook(cells)

        with open(ipynb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        count += 1

    print(f"Generated {count} ipynb files (tab={tab}) in {eval_dir}/")


if __name__ == '__main__':
    main()
