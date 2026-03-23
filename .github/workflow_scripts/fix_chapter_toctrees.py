#!/usr/bin/env python3
"""Fix broken toctree directives in chapter index RST files.

When d2lbook converts ipynb → RST via pandoc, toctree directives get mangled:
  - Options like :maxdepth: end up on the same line as .. toctree::
  - Entries get collapsed onto single lines separated by spaces

This script fixes them to valid RST format:
  .. toctree::
     :maxdepth: 2

     entry1
     entry2
"""
import os
import re
import glob


def fix_toctree(rst_content):
    """Fix mangled toctree directives in RST content."""
    # Match pattern: .. toctree:: :options\n\nentries on single lines
    # The mangled format looks like:
    #   .. toctree:: :maxdepth: 2
    #
    #   entry1 entry2 entry3
    #   entry4 entry5

    def fix_match(match):
        directive_line = match.group(1)  # e.g. ".. toctree:: :maxdepth: 2"
        body = match.group(2)  # entries block

        # Extract options from the directive line
        # ".. toctree:: :maxdepth: 2" -> options = ":maxdepth: 2"
        options_str = directive_line.replace('.. toctree::', '').strip()

        # Parse individual options (e.g., ":maxdepth: 2")
        options = re.findall(r'(:\w+:\s*\S+)', options_str)

        # Parse entries - they may be space-separated on lines
        entries = []
        for line in body.strip().split('\n'):
            line = line.strip()
            if line:
                entries.extend(line.split())

        # Rebuild proper RST toctree
        result = '.. toctree::\n'
        for opt in options:
            result += f'   {opt}\n'
        result += '\n'
        for entry in entries:
            result += f'   {entry}\n'

        return result

    # Match toctree directive with everything until next directive or end of content
    pattern = r'(\.\.\ toctree::\ [^\n]+)\n\n((?:(?!\.\.)[\s\S])*?)(?=\n\.\.|$)'
    fixed = re.sub(pattern, fix_match, rst_content)

    return fixed


def main():
    rst_dir = '_build/rst'

    # Find all chapter index.rst files
    index_files = glob.glob(os.path.join(rst_dir, 'chapter_*/index.rst'))

    fixed_count = 0
    for rst_file in sorted(index_files):
        with open(rst_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if it has mangled toctree
        if '.. toctree:: :' in content:
            fixed_content = fix_toctree(content)
            with open(rst_file, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            print(f"Fixed: {rst_file}")
            fixed_count += 1
        else:
            print(f"OK:    {rst_file}")

    print(f"\nFixed {fixed_count} files out of {len(index_files)} total")


if __name__ == '__main__':
    main()
