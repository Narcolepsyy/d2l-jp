#!/usr/bin/env python3
"""Post-process built HTML files to inject SEO tags.

Adds:
  1. <link rel="canonical"> based on file path
  2. Per-page <meta name="description"> extracted from page content
  3. Cleans up title tag format (removes "ドキュメント" suffix)
"""

import os
import re
import sys
from html.parser import HTMLParser
from pathlib import Path


BASE_URL = "https://d2l-jp.me/"
BUILD_DIR = Path("_build/html")
MAX_DESC_LEN = 155

# Global description fallback
FALLBACK_DESC = (
    "ディープラーニングを深く学ぶ - コード、数式、議論を通じた"
    "インタラクティブな深層学習の日本語教科書。PyTorch対応。"
)


class ParagraphExtractor(HTMLParser):
    """Extract text from the first meaningful <p> tag in the page content."""

    def __init__(self):
        super().__init__()
        self.in_p = False
        self.p_depth = 0
        self.paragraphs = []
        self.current_text = []
        self.in_nav = False
        self.in_sidebar = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        cls = attrs_dict.get("class", "")

        # Skip navigation and sidebar elements
        if tag in ("nav",) or "sidebar" in cls or "toctree" in cls:
            self.in_nav = True
            return

        if tag == "p" and not self.in_nav:
            self.in_p = True
            self.p_depth += 1
            self.current_text = []

    def handle_endtag(self, tag):
        if tag in ("nav",):
            self.in_nav = False

        if tag == "p" and self.in_p:
            self.p_depth -= 1
            if self.p_depth == 0:
                self.in_p = False
                text = "".join(self.current_text).strip()
                # Filter out very short or navigation-like text
                if len(text) > 30:
                    self.paragraphs.append(text)

    def handle_data(self, data):
        if self.in_p:
            self.current_text.append(data)


def extract_description(html_content):
    """Extract a description from the page's first paragraph."""
    # Try to find the main content area
    # Sphinx uses <div class="body" role="main"> or <div class="document">
    main_match = re.search(
        r'<div\s+class="body"\s+role="main">(.*?)</div>\s*<div',
        html_content,
        re.DOTALL,
    )
    if not main_match:
        main_match = re.search(
            r'role="main">(.*?)</div>\s*</div>',
            html_content,
            re.DOTALL,
        )

    content = main_match.group(1) if main_match else html_content

    parser = ParagraphExtractor()
    try:
        parser.feed(content)
    except Exception:
        return None

    for para in parser.paragraphs:
        # Clean up the text
        text = re.sub(r"\s+", " ", para).strip()
        # Remove RST/Sphinx artifacts
        text = re.sub(r"\[.*?\]", "", text).strip()
        text = re.sub(r"¶", "", text).strip()

        if len(text) < 30:
            continue

        # Truncate at word/sentence boundary
        if len(text) > MAX_DESC_LEN:
            # Try to cut at a sentence boundary
            cut = text[:MAX_DESC_LEN].rfind("。")
            if cut > 80:
                text = text[: cut + 1]
            else:
                # Cut at last space or punctuation
                cut = text[:MAX_DESC_LEN].rfind(" ")
                if cut < 80:
                    cut = MAX_DESC_LEN - 3
                text = text[:cut] + "…"

        return text

    return None


def get_canonical_url(filepath, build_dir):
    """Generate canonical URL from file path."""
    rel = os.path.relpath(filepath, build_dir)
    # Normalize: index.html stays, others keep their path
    url = BASE_URL + rel.replace(os.sep, "/")
    return url


def clean_title(html_content):
    """Remove version and 'ドキュメント' from the title tag."""
    def replace_title(m):
        title = m.group(1)
        # Remove " 1.0.3 ドキュメント" or similar version+doc suffix
        title = re.sub(r"\s+[\d.]+\s+ドキュメント$", "", title)
        # Also remove standalone " — ディープラーニングを深く学ぶ 1.0.3 ドキュメント"
        # We want: "前提知識 — ディープラーニングを深く学ぶ"
        title = re.sub(
            r"(.*? — .*?)\s+[\d.]+\s+ドキュメント",
            r"\1",
            title,
        )
        return f"<title>{title}</title>"

    return re.sub(r"<title>(.*?)</title>", replace_title, html_content)


def process_file(filepath, build_dir):
    """Process a single HTML file."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    modified = False

    # 1. Clean title format
    new_content = clean_title(content)
    if new_content != content:
        content = new_content
        modified = True

    # 2. Add canonical URL if not present
    if 'rel="canonical"' not in content:
        canonical_url = get_canonical_url(filepath, build_dir)
        canonical_tag = f'<link rel="canonical" href="{canonical_url}" />'
        content = content.replace("</head>", f"{canonical_tag}\n</head>")
        modified = True

    # 3. Add per-page meta description
    desc = extract_description(content)
    if desc:
        desc_escaped = desc.replace('"', "&quot;")
        new_meta = f'<meta name="description" content="{desc_escaped}" />'

        # Check if there's already a global meta description
        existing = re.search(
            r'<meta\s+name="description"\s+content="[^"]*"\s*/?>',
            content,
        )
        if existing:
            # Replace only if the existing one is the global fallback
            old_desc = existing.group(0)
            if "インタラクティブな深層学習の日本語教科書" in old_desc:
                content = content.replace(old_desc, new_meta)
                modified = True
        else:
            content = content.replace("</head>", f"{new_meta}\n</head>")
            modified = True
    elif 'name="description"' not in content:
        # Use fallback for pages where we can't extract content
        fallback_meta = (
            f'<meta name="description" content="{FALLBACK_DESC}" />'
        )
        content = content.replace("</head>", f"{fallback_meta}\n</head>")
        modified = True

    if modified:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    # 4. Add favicon if not present
    if 'rel="icon"' not in content and 'rel="shortcut icon"' not in content:
        favicon_tag = '<link rel="icon" type="image/png" href="/_static/favicon.png" />'
        content = content.replace("</head>", f"{favicon_tag}\n</head>")
        modified = True

    if modified:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    return modified


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

    print(f"SEO tags injected: {count}/{total} HTML files modified")


if __name__ == "__main__":
    main()
