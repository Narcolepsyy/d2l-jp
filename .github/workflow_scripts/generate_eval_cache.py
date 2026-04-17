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
    - %%tab and #@tab markers for framework-specific code
    - :begin_tab: / :end_tab: blocks
    - Files starting with bare {.python .input} (missing leading ```)
    - Regular markdown content
    """
    # Fix files that start with bare {.python .input} without leading ```
    # These are d2lbook setup blocks that should be skipped entirely
    content = _fix_malformed_opening(content)

    lines = content.split('\n')
    cells = []
    current_text = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check for code block start
        if line.startswith('```'):
            # Determine the fence type
            fence_type = line[3:].strip()

            # d2lbook-specific fences: eval_rst, toc
            # These must be preserved as RST directives in markdown cells
            if fence_type in ('eval_rst', 'toc'):
                # Flush accumulated markdown
                if current_text:
                    src = '\n'.join(current_text)
                    if src.strip():
                        cells.append(make_md_cell(src))
                    current_text = []

                rst_lines = []
                i += 1
                while i < len(lines) and not lines[i].startswith('```'):
                    rst_lines.append(lines[i])
                    i += 1
                # Skip closing ```
                if i < len(lines):
                    i += 1

                rst_content = '\n'.join(rst_lines)
                if fence_type == 'toc':
                    # Convert ```toc to .. toctree:: directive
                    rst_content = '.. toctree::\n' + rst_content
                    # Embed as raw RST in a markdown cell (toc is special case)
                    if rst_content.strip():
                        cells.append(make_md_cell(rst_content))
                else:
                    # For eval_rst, preserve the fences so d2lbook's ipynb2rst
                    # can properly extract it as raw RST without pandoc escaping it!
                    if rst_content.strip():
                        fenced_rst = f"```{fence_type}\n{rst_content}\n```"
                        cells.append(make_md_cell(fenced_rst))
                continue

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

                # Check for %%tab directive (d2l format)
                if code_line.strip().startswith('%%tab'):
                    tab_value = code_line.strip().replace('%%tab', '').strip()
                    # %%tab can list multiple tabs: "%%tab mxnet, pytorch"
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
            if _tab_matches(cell_tab, tab):
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


def _fix_malformed_opening(content):
    """Fix files that start with bare {.python .input} without leading ```.

    Some d2l files start with:
        {.python .input}
        %load_ext d2lbook.tab
        tab.interact_select([...])
        ```

    This is a d2lbook setup block that should be completely removed.
    """
    lines = content.split('\n')
    if lines and lines[0].strip() in ('{.python .input}',
                                       '{.python .input  n=1}',
                                       '{.python .input  n=2}'):
        # Find the closing ``` and skip everything up to and including it
        for j in range(1, len(lines)):
            if lines[j].startswith('```'):
                content = '\n'.join(lines[j + 1:])
                break
    return content


def _tab_matches(cell_tab, target_tab):
    """Check if a cell's tab specification matches the target tab.

    Handles:
    - None (no tab specified) -> include
    - 'all' -> include
    - Single tab like 'pytorch' -> exact match
    - Multiple tabs like 'mxnet, pytorch' -> check if target is in list
    - 'pytorch, mxnet, tensorflow' -> check if target is in list
    """
    if cell_tab is None or cell_tab == 'all':
        return True
    # Split by comma and strip whitespace
    tabs = [t.strip() for t in cell_tab.split(',')]
    return target_tab in tabs


def make_md_cell(source):
    """Create a markdown cell, stripping slide marks."""
    # Remove slide mark patterns that crash d2lbook's remove_slide_marks
    # These are (**  **) pairs used for slide highlighting
    source = re.sub(r'\(\*\*', '', source)
    source = re.sub(r'\*\*\)', '', source)
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _lines_with_newlines(source)
    }


def make_code_cell(source):
    """Create a code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines_with_newlines(source)
    }


def _lines_with_newlines(source):
    """Split source into lines with trailing newlines (nbformat spec)."""
    if not source:
        return ['']
    lines = source.split('\n')
    # Each line except the last gets a trailing newline
    return [line + '\n' for line in lines[:-1]] + [lines[-1]]


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

    # d2lbook uses tab-based eval directories:
    # default tab (pytorch) → _build/eval/
    # other tabs → _build/eval_<tab>/
    tabs_str = config.get('build', 'tabs', fallback='pytorch')
    default_tab = tabs_str.split(',')[0].strip()
    if tab == default_tab:
        eval_dir = '_build/eval'
    else:
        eval_dir = f'_build/eval_{tab}'
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
