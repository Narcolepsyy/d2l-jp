#!/usr/bin/env python3
"""Post-process built HTML files to improve page speed.

Optimizations:
  1. Add lazy loading to images below the fold
  2. Defer non-critical JavaScript
  3. Preload critical fonts + preconnect to external origins
  4. Add font-display: swap to font CSS
  5. Add explicit width/height to known images
"""

import os
import re
import sys
from pathlib import Path


BUILD_DIR = Path("_build/html")


def add_lazy_loading(html):
    """Add loading='lazy' to images that are not in the viewport initially.

    Skip:
    - Images with loading= already set
    - The first image on the page (likely above the fold / LCP)
    - Logo images in the header
    """
    img_count = 0

    def replace_img(m):
        nonlocal img_count
        tag = m.group(0)

        # Skip if already has loading attribute
        if "loading=" in tag:
            return tag

        # Skip logo/favicon images (always above fold)
        if 'class="logo"' in tag or "favicon" in tag:
            return tag

        img_count += 1

        # Skip first image (likely above-the-fold / LCP candidate)
        if img_count <= 1:
            return tag

        # Add lazy loading and decoding
        tag = tag.replace("<img ", '<img loading="lazy" decoding="async" ')
        return tag

    return re.sub(r"<img\s[^>]+>", replace_img, html)


def defer_scripts(html):
    """Add defer to synchronous scripts that aren't critical for first paint.

    Skip:
    - Scripts that already have async/defer
    - Inline scripts (no src)
    - Critical scripts (jQuery needed by theme)
    """
    def replace_script(m):
        tag = m.group(0)

        # Skip if already has async or defer
        if " async" in tag or " defer" in tag:
            return tag

        # Skip inline scripts
        if 'src="' not in tag and "src='" not in tag:
            return tag

        # Skip jQuery (theme depends on it synchronously)
        if "jquery.js" in tag:
            return tag

        # Skip compat layer (needed before other scripts)
        if "frameworks_compat" in tag:
            return tag

        # Defer everything else
        tag = tag.replace("<script ", "<script defer ", 1)
        return tag

    return re.sub(r"<script\s[^>]*>", replace_script, html)


def add_resource_hints(html):
    """Add preconnect and preload hints for critical external resources."""
    hints = []

    # Preconnect to Google Fonts (used via @import in d2l.css)
    if "fonts.googleapis.com" in html or "d2l.css" in html:
        hints.append(
            '<link rel="preconnect" href="https://fonts.googleapis.com" />'
        )
        hints.append(
            '<link rel="preconnect" href="https://fonts.gstatic.com"'
            ' crossorigin />'
        )

    # Preconnect to MathJax CDN
    if "mathjax" in html:
        hints.append(
            '<link rel="preconnect" href="https://cdn.jsdelivr.net" />'
        )

    # DNS prefetch for external widgets
    if "buttons.github.io" in html:
        hints.append(
            '<link rel="dns-prefetch" href="https://buttons.github.io" />'
        )
    if "platform.twitter.com" in html:
        hints.append(
            '<link rel="dns-prefetch" href="https://platform.twitter.com" />'
        )

    if hints:
        hint_block = "\n".join(hints) + "\n"
        # Insert right after <head> or after charset meta
        if '<meta charset=' in html:
            html = re.sub(
                r'(<meta charset=[^>]+>)',
                r'\1\n' + hint_block,
                html,
                count=1,
            )
        else:
            html = html.replace("<head>", "<head>\n" + hint_block, 1)

    return html


def fix_font_display(html):
    """Replace Google Fonts @import with preload link for better performance.

    The @import in d2l.css is render-blocking. We inject a <link> with
    display=swap in the HTML head instead, which is non-blocking.
    """
    # Add font preload link if the page uses d2l.css
    if "d2l.css" in html:
        font_link = (
            '<link rel="stylesheet"'
            ' href="https://fonts.googleapis.com/css2'
            '?family=Noto+Sans+JP:wght@300;400;500;700'
            '&display=swap"'
            ' media="print" onload="this.media=\'all\'" />\n'
            '<noscript><link rel="stylesheet"'
            ' href="https://fonts.googleapis.com/css2'
            '?family=Noto+Sans+JP:wght@300;400;500;700'
            '&display=swap" /></noscript>'
        )
        html = html.replace("</head>", font_link + "\n</head>")

    return html


def add_image_dimensions(html):
    """Add width/height to known frontpage images to prevent CLS."""
    known_images = {
        "front-cup.jpg": ('width="300"', 'height="400"'),
        "notebook.gif": ('width="700"', 'height="400"'),
        "eq.jpg": ('width="350"', 'height="200"'),
        "figure.jpg": ('width="350"', 'height="200"'),
        "code.jpg": ('width="350"', 'height="200"'),
    }

    for img_name, (w, h) in known_images.items():
        # Only add if not already present
        pattern = f'src="[^"]*{re.escape(img_name)}"'
        match = re.search(pattern, html)
        if match and "width=" not in html[
            max(0, match.start() - 50):match.end() + 50
        ]:
            html = html.replace(
                match.group(0),
                f'{match.group(0)} {w} {h}',
            )

    return html


def process_file(filepath):
    """Apply all page speed optimizations to a single HTML file."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    original = content

    content = add_resource_hints(content)
    content = add_lazy_loading(content)
    content = defer_scripts(content)
    content = fix_font_display(content)
    content = add_image_dimensions(content)

    if content != original:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    return False


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
            if process_file(filepath):
                count += 1

    print(f"Page speed optimized: {count}/{total} HTML files modified")


if __name__ == "__main__":
    main()
