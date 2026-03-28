#!/usr/bin/env python3
"""Post-process built HTML files to improve page speed.

Optimizations:
  1. Add lazy loading to images below the fold
  2. Defer ALL JavaScript (including jQuery)
  3. Preload critical fonts + preconnect to external origins
  4. Add font-display: swap to font CSS
  5. Add explicit width/height to known images
  6. Defer non-critical CSS (fonts.css, fontawesome, pygments)
  7. Inline critical CSS for fast first paint
  8. Add fetchpriority="high" to critical CSS files
"""

import os
import re
import sys
from pathlib import Path


BUILD_DIR = Path("_build/html")

# ---------------------------------------------------------------------------
# Critical CSS – inlined into <head> so initial paint doesn't depend on any
# external stylesheet.  Covers: background, header bar, drawer skeleton,
# content area, and the PJAX fade-in gate.
# ---------------------------------------------------------------------------
CRITICAL_CSS = """
<style>
/* Critical CSS – inlined for fast first paint */
html{background-color:#fafafa}
body{margin:0;font-family:Roboto,'Noto Sans JP',sans-serif;font-size:17px;color:rgba(0,0,0,.87)}
/* Header skeleton */
.mdl-layout__header{display:flex;flex-direction:column;background-color:rgb(25,118,210);color:#fff;min-height:64px;z-index:3;box-shadow:0 2px 2px 0 rgba(0,0,0,.14),0 3px 1px -2px rgba(0,0,0,.12),0 1px 5px 0 rgba(0,0,0,.2)}
.mdl-layout__header-row{display:flex;align-items:center;height:64px;padding:0 40px 0 80px}
/* Drawer skeleton */
.mdl-layout__drawer{display:flex;flex-direction:column;width:250px;background:#fff;border-right:1px solid #e0e0e0;overflow-y:auto;overflow-x:hidden;position:fixed;top:0;left:0;height:100%;z-index:5;transform:translateX(-250px)}
.mdl-layout--fixed-drawer>.mdl-layout__drawer{transform:translateX(0)}
/* Content area */
.mdl-layout__content{display:inline-block;flex-grow:1;overflow:auto;order:1}
.mdl-layout--fixed-drawer>.mdl-layout__content{margin-left:250px}
@media(max-width:1024px){.mdl-layout--fixed-drawer>.mdl-layout__content{margin-left:0}.mdl-layout__drawer{transform:translateX(-250px)}}
</style>
""".strip()

# CSS files that are non-critical and can be loaded asynchronously.
# Pattern fragments matched against the href attribute.
NON_CRITICAL_CSS = [
    "fontawesome/all.css", # Icons – tolerable FOUT
    "pygments.css",       # Syntax highlighting – below fold
]

# CSS files to strip entirely from HTML (truly unused on this site).
REMOVE_CSS = [
    "fonts.css",  # 288KB Noto Sans SC Sliced (Chinese font) – unnecessary
                  # for this Japanese site; Noto Sans JP from Google Fonts
                  # is the primary font and covers all needed characters.
]

# CSS files that are critical and should keep blocking but get fetchpriority.
CRITICAL_CSS_FILES = [
    "material-design-lite",           # MDL layout framework
    "sphinx_materialdesign_theme.css", # Theme layout
    "basic.css",                       # Sphinx base
    "d2l.css",                         # Our custom styles
]


def add_lazy_loading(html):
    """Add loading='lazy' to images that are not in the viewport initially.

    Skip:
    - Images with loading= already set
    - The first image on the page (likely above the fold / LCP)
    - Logo images in the header
    """
    img_count = [0]

    def replace_img(m):
        tag = m.group(0)

        # Skip if already has loading attribute
        if "loading=" in tag:
            return tag

        # Skip logo/favicon images (always above fold)
        if 'class="logo"' in tag or "favicon" in tag:
            return tag

        img_count[0] += 1

        # Skip first image (likely above-the-fold / LCP candidate)
        if img_count[0] <= 1:
            return tag

        # Add lazy loading and decoding
        tag = tag.replace("<img ", '<img loading="lazy" decoding="async" ')
        return tag

    return re.sub(r"<img\s[^>]+>", replace_img, html)


def defer_scripts(html):
    """Add defer to ALL synchronous scripts.

    Skip:
    - Scripts that already have async/defer
    - Inline scripts (no src)
    """
    def replace_script(m):
        tag = m.group(0)

        # Skip if already has async or defer
        if " async" in tag or " defer" in tag:
            return tag

        # Skip inline scripts
        if 'src="' not in tag and "src='" not in tag:
            return tag

        # Defer everything – including jQuery.
        # The theme JS is a self-contained bundle and doesn't depend on
        # jQuery being loaded first.  The only inline $() call is in
        # search.html which has its own script block (deferred scripts
        # execute in order, so jQuery will be available).
        tag = tag.replace("<script ", "<script defer ", 1)
        return tag

    return re.sub(r"<script\s[^>]*>", replace_script, html)


def strip_unused_css(html):
    """Remove CSS links that are completely unused on this site."""
    def replace_link(m):
        tag = m.group(0)
        for pattern in REMOVE_CSS:
            if pattern in tag:
                return ""  # Strip entirely
        return tag

    return re.sub(r"<link\s[^>]*/?>\s*", replace_link, html)


def defer_css(html):
    """Convert non-critical CSS to async loading.

    Uses the standard media="print" + onload="this.media='all'" pattern
    with a <noscript> fallback.
    """
    def replace_link(m):
        tag = m.group(0)

        # Only process stylesheet links
        if 'rel="stylesheet"' not in tag:
            return tag

        # Skip if already has media override (e.g. Google Fonts async)
        if 'media="print"' in tag:
            return tag

        # Check if this is a non-critical CSS file
        is_non_critical = False
        for pattern in NON_CRITICAL_CSS:
            if pattern in tag:
                is_non_critical = True
                break

        if not is_non_critical:
            return tag

        # Extract href for noscript fallback
        href_match = re.search(r'href="([^"]*)"', tag)
        if not href_match:
            return tag

        href = href_match.group(1)

        # Convert to async: media="print" onload="this.media='all'"
        async_tag = tag.replace(
            'rel="stylesheet"',
            'rel="stylesheet" media="print" onload="this.media=\'all\'"',
        )
        noscript = f'<noscript><link rel="stylesheet" href="{href}" /></noscript>'

        return async_tag + "\n" + noscript

    return re.sub(r"<link\s[^>]*/?>", replace_link, html)


def add_fetchpriority(html):
    """Add fetchpriority="high" to critical CSS files still loaded as blocking."""
    def replace_link(m):
        tag = m.group(0)

        if 'rel="stylesheet"' not in tag:
            return tag

        # Skip already-deferred CSS
        if 'media="print"' in tag:
            return tag

        # Skip if already has fetchpriority
        if "fetchpriority" in tag:
            return tag

        # Check if this is a critical CSS file
        is_critical = False
        for pattern in CRITICAL_CSS_FILES:
            if pattern in tag:
                is_critical = True
                break

        if not is_critical:
            return tag

        tag = tag.replace("<link ", '<link fetchpriority="high" ', 1)
        return tag

    return re.sub(r"<link\s[^>]*/?>", replace_link, html)


def inline_critical_css(html):
    """Inject critical inline CSS immediately after <head> for fast first paint."""
    # Don't add if already present
    if "Critical CSS" in html:
        return html

    # Insert right after the charset meta tag (placed by preconnect hints)
    # or after <head> as fallback
    if '<meta charset=' in html:
        html = re.sub(
            r'(<meta charset=[^>]+>)',
            r'\1\n' + CRITICAL_CSS,
            html,
            count=1,
        )
    else:
        html = html.replace("<head>", "<head>\n" + CRITICAL_CSS, 1)

    return html


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
    """Inject async Google Fonts link for Noto Sans JP.

    Only load weights 400 (body) and 700 (headings). The CSS2 API
    automatically uses unicode-range splitting for CJK fonts, so no
    &subset= parameter is needed (it's not a valid CSS2 API param).
    """
    if "d2l.css" in html:
        gf_url = (
            "https://fonts.googleapis.com/css2"
            "?family=Noto+Sans+JP:wght@400;700"
            "&display=swap"
        )
        font_link = (
            f'<link rel="preload" as="style" href="{gf_url}" />\n'
            f'<link rel="stylesheet" href="{gf_url}"'
            ' media="print" onload="this.media=\'all\'" />\n'
            f'<noscript><link rel="stylesheet" href="{gf_url}" /></noscript>'
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
    content = inline_critical_css(content)
    content = strip_unused_css(content)
    content = add_lazy_loading(content)
    content = defer_scripts(content)
    content = defer_css(content)
    content = add_fetchpriority(content)
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
