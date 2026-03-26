#!/usr/bin/env python3
"""Merge per-tab eval notebooks into a single notebook with tab UI.

d2lbook's internal merge requires `origin_pos` metadata we don't have.
This script does the same job: takes per-tab eval directories and produces
merged notebooks in the default eval dir with Material Design tab selectors.

Tab HTML format matches d2lbook's add_html_tab() output exactly.
"""
import json
import os
import sys
import uuid
from pathlib import Path


def get_tab_bar_html(tabs, tab_id, default_tab):
    """Generate MDL tab bar HTML (matches d2lbook's format)."""
    bar = '<div class="mdl-tabs mdl-js-tabs mdl-js-ripple-effect"><div class="mdl-tabs__tab-bar ">'
    for i, tab in enumerate(tabs):
        active = "is-active" if tab == default_tab else ""
        bar += f'<a href="#{tab}-{tab_id}-{i}" onclick="tagClick(\'{tab}\'); return false;" class="mdl-tabs__tab {active}">{tab}</a>'
    bar += "</div>"
    return bar


def get_tab_panel_begin(tab, tab_id, default_tab):
    """Generate tab panel opening div."""
    active = "is-active" if tab == default_tab else ""
    return f'<div class="mdl-tabs__panel {active}" id="{tab}-{tab_id}">'


def make_raw_html_md_cell(html):
    """Create a markdown cell with raw RST/HTML."""
    source = f"```eval_rst\n\n.. raw:: html\n\n    {html}\n```"
    # nbformat requires each line except the last to end with \n
    lines = source.split("\n")
    source_list = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return {"cell_type": "markdown", "metadata": {}, "source": source_list}


def merge_chapter_notebook(default_nb_path, tab_nb_paths, tabs, default_tab):
    """Merge tab notebooks into the default notebook with tab UI.

    Strategy:
    1. Use the default (pytorch) notebook as the base
    2. For each code cell, check if other tabs have different code
    3. If tabs differ, wrap in tab panels with tab bar
    4. If identical across all tabs, keep as single cell (no tabs needed)
    """
    # Load all tab notebooks
    tab_nbs = {}
    for tab, path in tab_nb_paths.items():
        try:
            tab_nbs[tab] = json.loads(path.read_text())
        except Exception:
            continue

    if not tab_nbs:
        return False

    # Load default notebook
    try:
        default_nb = json.loads(default_nb_path.read_text())
    except Exception:
        return False

    # Get code cells for each tab
    tab_code_cells = {}
    for tab, nb in tab_nbs.items():
        tab_code_cells[tab] = [c for c in nb["cells"] if c["cell_type"] == "code"]

    # Get the available tabs that have code cells
    available_tabs = [t for t in tabs if t in tab_code_cells and tab_code_cells[t]]
    if len(available_tabs) <= 1:
        return False  # Only one tab, no need for tab UI

    # Build new cells list with tab UI injected
    new_cells = []
    default_code_idx = 0
    default_code_cells = [c for c in default_nb["cells"] if c["cell_type"] == "code"]

    for cell in default_nb["cells"]:
        if cell["cell_type"] != "code":
            new_cells.append(cell)
            continue

        # This is a code cell - check if we should add tabs
        tab_id = uuid.uuid4().hex[:8]

        # Collect this code cell from each tab (by index)
        tab_cells_for_pos = {}
        for tab in available_tabs:
            cells = tab_code_cells[tab]
            if default_code_idx < len(cells):
                tab_cells_for_pos[tab] = cells[default_code_idx]

        default_code_idx += 1

        if len(tab_cells_for_pos) <= 1:
            # Only one tab has this cell, just keep it
            new_cells.append(cell)
            continue

        # Check if all tabs have identical source
        sources = set()
        for t, c in tab_cells_for_pos.items():
            src = "".join(c.get("source", []))
            sources.add(src.strip())

        if len(sources) == 1:
            # All tabs identical - no need for tab UI
            # But use the cell with outputs if any tab has them
            best_cell = cell
            for t, c in tab_cells_for_pos.items():
                if c.get("outputs"):
                    best_cell = c
                    break
            new_cells.append(best_cell)
            continue

        # Different code per tab - inject tab UI
        active_tabs = [t for t in available_tabs if t in tab_cells_for_pos]

        # Tab bar
        new_cells.append(make_raw_html_md_cell(
            get_tab_bar_html(active_tabs, tab_id, default_tab)
        ))

        # Tab panels
        for i, tab in enumerate(active_tabs):
            tab_cell = tab_cells_for_pos[tab]
            # Panel begin
            new_cells.append(make_raw_html_md_cell(
                get_tab_panel_begin(tab, f"{tab_id}-{i}", default_tab)
            ))
            # The code cell itself
            new_cells.append(tab_cell)
            # Panel end
            new_cells.append(make_raw_html_md_cell("</div>"))

        # Close the tab container
        new_cells.append(make_raw_html_md_cell("</div>"))

    # Write merged notebook
    default_nb["cells"] = new_cells
    default_nb_path.write_text(json.dumps(default_nb, indent=1, ensure_ascii=False))
    return True


def main():
    base_eval_dir = sys.argv[1] if len(sys.argv) > 1 else "_build/eval"
    tabs = sys.argv[2:] if len(sys.argv) > 2 else ["pytorch", "mxnet", "jax", "tensorflow"]

    default_tab = tabs[0]
    default_dir = Path(base_eval_dir)

    if not default_dir.exists():
        print(f"ERROR: {default_dir} not found")
        sys.exit(1)

    # Map tabs to their eval directories
    tab_dirs = {}
    for tab in tabs:
        if tab == default_tab:
            tab_dirs[tab] = default_dir
        else:
            d = Path(f"{base_eval_dir}_{tab}")
            if d.exists():
                tab_dirs[tab] = d

    print(f"Tabs: {list(tab_dirs.keys())}")
    merged = 0

    # Process each notebook in the default eval dir
    for chapter_dir in sorted(default_dir.iterdir()):
        if not chapter_dir.is_dir() or not chapter_dir.name.startswith("chapter_"):
            continue

        for nb_path in sorted(chapter_dir.glob("*.ipynb")):
            # Find this notebook in other tab dirs
            tab_nb_paths = {}
            for tab, tdir in tab_dirs.items():
                other_nb = tdir / chapter_dir.name / nb_path.name
                if other_nb.exists():
                    tab_nb_paths[tab] = other_nb

            if len(tab_nb_paths) > 1:
                if merge_chapter_notebook(nb_path, tab_nb_paths, tabs, default_tab):
                    merged += 1

    print(f"Merged {merged} notebooks with tab UI")


if __name__ == "__main__":
    main()
