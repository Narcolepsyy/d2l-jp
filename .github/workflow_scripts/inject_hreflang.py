#!/usr/bin/env python3
"""Post-process built HTML files to inject per-page hreflang tags.

Each page gets hreflang pointing to its equivalent URL on the EN/ZH versions,
rather than all pages pointing to the root URL (which was the previous bug).
"""

import os
import sys
from pathlib import Path


BUILD_DIR = Path("_build/html")

# Language version root URLs
LANG_URLS = {
    "ja": "https://d2l-jp.me/",
    "en": "https://d2l.ai/",
    "zh": "https://zh.d2l.ai/",
}


def process_file(filepath, build_dir):
    """Inject per-page hreflang tags into a single HTML file."""
    rel = os.path.relpath(filepath, build_dir).replace(os.sep, "/")

    with open(filepath, "r", encoding="utf-8") as f:
        html = f.read()

    if 'hreflang=' in html:
        return False  # Already has hreflang tags

    ja_url = LANG_URLS["ja"] + rel
    en_url = LANG_URLS["en"] + rel
    zh_url = LANG_URLS["zh"] + rel

    hreflang_block = (
        f'  <link rel="alternate" hreflang="ja" href="{ja_url}" />\n'
        f'<link rel="alternate" hreflang="en" href="{en_url}" />\n'
        f'<link rel="alternate" hreflang="zh" href="{zh_url}" />\n'
        f'<link rel="alternate" hreflang="x-default" href="{en_url}" />\n'
    )

    html = html.replace("</head>", hreflang_block + "</head>")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)
    return True


def main():
    build_dir = BUILD_DIR
    if len(sys.argv) > 1:
        build_dir = Path(sys.argv[1])

    if not build_dir.exists():
        print(f"Build directory not found: {build_dir}")
        sys.exit(1)

    count = 0
    total = 0
    for root, _, files in os.walk(build_dir):
        for fname in files:
            if not fname.endswith(".html"):
                continue
            total += 1
            filepath = os.path.join(root, fname)
            if process_file(filepath, build_dir):
                count += 1

    print(f"Hreflang tags injected: {count}/{total} HTML files modified")


if __name__ == "__main__":
    main()
