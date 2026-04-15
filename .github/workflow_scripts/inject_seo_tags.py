#!/usr/bin/env python3
"""Post-process built HTML files to inject SEO tags.

Adds:
  1. <link rel="canonical"> based on file path
  2. Per-page <meta name="description"> extracted from page content
  3. Cleans up title tag format (removes "ドキュメント" suffix)
  4. JSON-LD structured data (WebSite + Book on homepage, BreadcrumbList on inner pages)
  5. Cleans up duplicate viewport/charset meta tags
  6. Overrides homepage OG description
"""

import json
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

# Custom homepage meta description (replaces auto-generated TOC dump)
HOMEPAGE_DESC = (
    "「Dive into Deep Learning」の日本語版。PyTorch・TensorFlow・JAX対応の"
    "実装コード付きで、深層学習の理論と実践を体系的に学べる無料オンライン教科書。"
    "500以上の大学で採用。"
)

# Custom homepage OG description
HOMEPAGE_OG_DESC = (
    "「Dive into Deep Learning」日本語版 — コード・数式・実行結果を交えながら"
    "ディープラーニングの基礎から最新手法までを体系的に解説する、"
    "無料のインタラクティブ教科書。"
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


def clean_title(html_content, is_homepage=False):
    """Remove version and 'ドキュメント' from the title tag.

    Also fixes the homepage title which redundantly repeats the same phrase.
    """
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
        # Fix homepage: "X — X" → "X | D2L日本語版"
        if is_homepage:
            title = "ディープラーニングを深く学ぶ | D2L日本語版 - 無料DLテキストブック"
        return f"<title>{title}</title>"

    return re.sub(r"<title>(.*?)</title>", replace_title, html_content)


def clean_duplicate_meta(content):
    """Remove duplicate viewport and charset meta tags."""
    # Keep only the first charset declaration
    charset_count = [0]
    def dedup_charset(m):
        charset_count[0] += 1
        return "" if charset_count[0] > 1 else m.group(0)
    content = re.sub(r'<meta\s+charset="utf-8"\s*/?>', dedup_charset, content)

    # Keep only the last viewport (most specific), remove earlier ones
    viewport_matches = list(re.finditer(
        r'<meta\s+name="viewport"\s+content="[^"]*"\s*/?>', content
    ))
    if len(viewport_matches) > 1:
        # Remove all but the last
        for m in viewport_matches[:-1]:
            content = content.replace(m.group(0), "", 1)

    return content


def generate_homepage_jsonld():
    """Generate JSON-LD structured data for the homepage."""
    website = {
        "@context": "https://schema.org",
        "@type": "WebSite",
        "name": "ディープラーニングを深く学ぶ",
        "alternateName": "D2L日本語版",
        "url": "https://d2l-jp.me/",
        "inLanguage": "ja",
        "description": HOMEPAGE_DESC,
    }
    book = {
        "@context": "https://schema.org",
        "@type": "Book",
        "name": "ディープラーニングを深く学ぶ",
        "alternateName": ["Dive into Deep Learning", "D2L"],
        "url": "https://d2l-jp.me/",
        "inLanguage": "ja",
        "author": [
            {"@type": "Person", "name": "Aston Zhang"},
            {"@type": "Person", "name": "Zachary C. Lipton"},
            {"@type": "Person", "name": "Mu Li"},
            {"@type": "Person", "name": "Alexander J. Smola"},
        ],
        "bookFormat": "https://schema.org/EBook",
        "isAccessibleForFree": True,
        "license": "https://creativecommons.org/licenses/by-sa/4.0/",
        "image": "https://d2l-jp.me/_images/front-cup.jpg",
        "about": {
            "@type": "Thing",
            "name": "深層学習",
            "sameAs": "https://ja.wikipedia.org/wiki/ディープラーニング",
        },
        "translationOfWork": {
            "@type": "Book",
            "name": "Dive into Deep Learning",
            "url": "https://d2l.ai/",
            "inLanguage": "en",
        },
    }
    return (
        '<script type="application/ld+json">'
        + json.dumps(website, ensure_ascii=False)
        + "</script>\n"
        '<script type="application/ld+json">'
        + json.dumps(book, ensure_ascii=False)
        + "</script>"
    )


def generate_breadcrumb_jsonld(filepath, build_dir):
    """Generate BreadcrumbList JSON-LD for inner pages."""
    rel = os.path.relpath(filepath, build_dir).replace(os.sep, "/")
    parts = rel.split("/")

    if len(parts) < 2:
        return ""

    # Build breadcrumb: Home > Chapter > Page
    items = [
        {
            "@type": "ListItem",
            "position": 1,
            "name": "ホーム",
            "item": "https://d2l-jp.me/",
        }
    ]

    # Chapter level
    chapter_dir = parts[0]
    chapter_name = chapter_dir.replace("chapter_", "").replace("-", " ").title()
    items.append({
        "@type": "ListItem",
        "position": 2,
        "name": chapter_name,
        "item": f"https://d2l-jp.me/{chapter_dir}/index.html",
    })

    # Page level (if not index)
    if parts[-1] != "index.html":
        page_name = parts[-1].replace(".html", "").replace("-", " ").title()
        items.append({
            "@type": "ListItem",
            "position": 3,
            "name": page_name,
            "item": f"https://d2l-jp.me/{rel}",
        })

    breadcrumb = {
        "@context": "https://schema.org",
        "@type": "BreadcrumbList",
        "itemListElement": items,
    }
    return (
        '<script type="application/ld+json">'
        + json.dumps(breadcrumb, ensure_ascii=False)
        + "</script>"
    )


def process_file(filepath, build_dir):
    """Process a single HTML file."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    modified = False
    rel = os.path.relpath(filepath, build_dir).replace(os.sep, "/")
    is_homepage = rel == "index.html"

    # Ensure logo image in sidebar (Sphinx/mxtheme may omit html_logo and use text instead)
    new_content = re.sub(
        r'<span\s+class="title-text">\s*ディープラーニングを深く学ぶ\s*</span>',
        r'<img class="logo" src="/_static/logo-with-text.png" alt="ディープラーニングを深く学ぶ"/>',
        content
    )
    if new_content != content:
        content = new_content
        modified = True

    # 1. Clean title format
    new_content = clean_title(content, is_homepage=is_homepage)
    if new_content != content:
        content = new_content
        modified = True

    # 2. Add canonical URL if not present
    if 'rel="canonical"' not in content:
        canonical_url = get_canonical_url(filepath, build_dir)
        canonical_tag = f'<link rel="canonical" href="{canonical_url}" />'
        content = content.replace("</head>", f"{canonical_tag}\n</head>")
        modified = True

    # 3. Add/fix per-page meta description
    if is_homepage:
        # Homepage: replace auto-generated TOC description with custom one
        existing = re.search(
            r'<meta\s+name="description"\s+content="[^"]*"\s*/?>',
            content,
        )
        homepage_meta = f'<meta name="description" content="{HOMEPAGE_DESC}" />'
        if existing:
            content = content.replace(existing.group(0), homepage_meta)
        else:
            content = content.replace("</head>", f"{homepage_meta}\n</head>")
        modified = True

        # Also fix OG description on homepage
        og_existing = re.search(
            r'<meta\s+property="og:description"\s+content="[^"]*"\s*/?>',
            content,
        )
        if og_existing:
            og_new = f'<meta property="og:description" content="{HOMEPAGE_OG_DESC}" />'
            content = content.replace(og_existing.group(0), og_new)
            modified = True
    else:
        desc = extract_description(content)
        if desc:
            desc_escaped = desc.replace('"', "&quot;")
            new_meta = f'<meta name="description" content="{desc_escaped}" />'

            existing = re.search(
                r'<meta\s+name="description"\s+content="[^"]*"\s*/?>',
                content,
            )
            if existing:
                old_desc = existing.group(0)
                if "インタラクティブな深層学習の日本語教科書" in old_desc:
                    content = content.replace(old_desc, new_meta)
                    modified = True
            else:
                content = content.replace("</head>", f"{new_meta}\n</head>")
                modified = True
        elif 'name="description"' not in content:
            fallback_meta = (
                f'<meta name="description" content="{FALLBACK_DESC}" />'
            )
            content = content.replace("</head>", f"{fallback_meta}\n</head>")
            modified = True

    # 4. Add favicon if not present
    if 'rel="icon"' not in content and 'rel="shortcut icon"' not in content:
        favicon_tag = '<link rel="icon" type="image/png" href="/_static/favicon.png" />'
        content = content.replace("</head>", f"{favicon_tag}\n</head>")
        modified = True

    # 5. Inject JSON-LD structured data
    if 'application/ld+json' not in content:
        if is_homepage:
            jsonld = generate_homepage_jsonld()
        else:
            jsonld = generate_breadcrumb_jsonld(filepath, build_dir)
        if jsonld:
            content = content.replace("</head>", f"{jsonld}\n</head>")
            modified = True

    # 6. Clean up duplicate viewport/charset meta tags
    new_content = clean_duplicate_meta(content)
    if new_content != content:
        content = new_content
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
