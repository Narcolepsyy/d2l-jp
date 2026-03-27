// Instant page navigation for d2l-jp
// Combines instant.page prefetching with smooth page transitions
// to eliminate navbar flicker and improve perceived performance.
(function () {
  'use strict';

  // =========================================================================
  // 1. Page Transition Effect (fade out on navigate, fade in on load)
  // =========================================================================

  // On page load: fade in
  document.documentElement.classList.add('page-ready');

  // Intercept internal link clicks for a smooth fade-out transition
  document.addEventListener('click', function (e) {
    var link = e.target.closest('a[href]');
    if (!link) return;

    var href = link.getAttribute('href');
    // Skip external links, anchors, and special links
    if (!href ||
        href.startsWith('#') ||
        href.startsWith('http') ||
        href.startsWith('mailto:') ||
        href.startsWith('javascript:') ||
        link.hasAttribute('download') ||
        link.target === '_blank' ||
        e.ctrlKey || e.metaKey || e.shiftKey || e.altKey) {
      return;
    }

    // It's an internal navigation — apply fade-out
    e.preventDefault();
    document.documentElement.classList.add('page-leaving');

    setTimeout(function () {
      window.location.href = link.href;
    }, 120);
  });

  // =========================================================================
  // 2. Preserve Drawer Scroll Position
  // =========================================================================

  var DRAWER_SCROLL_KEY = 'd2l-drawer-scroll';

  function saveDrawerScroll() {
    var drawer = document.querySelector('.mdl-layout__drawer');
    if (drawer) {
      try {
        sessionStorage.setItem(DRAWER_SCROLL_KEY, drawer.scrollTop);
      } catch (e) { /* sessionStorage unavailable */ }
    }
  }

  function restoreDrawerScroll() {
    var drawer = document.querySelector('.mdl-layout__drawer');
    if (drawer) {
      try {
        var pos = sessionStorage.getItem(DRAWER_SCROLL_KEY);
        if (pos !== null) {
          drawer.scrollTop = parseInt(pos, 10);
        }
      } catch (e) { /* sessionStorage unavailable */ }
    }
  }

  // Save scroll before navigating away
  window.addEventListener('beforeunload', saveDrawerScroll);

  // Restore scroll on load
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', restoreDrawerScroll);
  } else {
    restoreDrawerScroll();
  }

  // =========================================================================
  // 3. Mobile Floating ToC Button
  // On mobile/tablet (≤1199px) the side-doc-outline is hidden.
  // This button opens the drawer for quick chapter navigation.
  // =========================================================================

  function createMobileTocButton() {
    // Don't add on index page
    var path = window.location.pathname;
    if (path === '/' || path === '/index.html') return;

    var btn = document.createElement('button');
    btn.className = 'mobile-toc-btn';
    btn.setAttribute('aria-label', 'Open table of contents');
    btn.setAttribute('title', '目次を開く');
    btn.innerHTML = '☰';

    btn.addEventListener('click', function () {
      // Use MDL's drawer toggle
      var drawer = document.querySelector('.mdl-layout__drawer');
      var obfuscator = document.querySelector('.mdl-layout__obfuscator');
      if (drawer) {
        drawer.classList.add('is-visible');
        if (obfuscator) obfuscator.classList.add('is-visible');
      }
    });

    document.body.appendChild(btn);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', createMobileTocButton);
  } else {
    createMobileTocButton();
  }

  // =========================================================================
  // 4. SEO Enhancements
  // Inject meta tags, structured data, hreflang, canonical URLs.
  // These help search engines understand the JP site's relationship to d2l.ai.
  // =========================================================================

  function injectSEO() {
    var head = document.head;
    var path = window.location.pathname;

    // --- 4a. Set html lang attribute ---
    document.documentElement.setAttribute('lang', 'ja');

    // --- 4b. Canonical URL ---
    if (!document.querySelector('link[rel="canonical"]')) {
      var canonical = document.createElement('link');
      canonical.rel = 'canonical';
      canonical.href = 'https://d2l-jp.me' + path;
      head.appendChild(canonical);
    }

    // --- 4c. Hreflang tags (JP ↔ EN) ---
    // Map JP URL paths to d2l.ai equivalents (same structure)
    var hreflangJa = document.createElement('link');
    hreflangJa.rel = 'alternate';
    hreflangJa.hreflang = 'ja';
    hreflangJa.href = 'https://d2l-jp.me' + path;
    head.appendChild(hreflangJa);

    var hreflangEn = document.createElement('link');
    hreflangEn.rel = 'alternate';
    hreflangEn.hreflang = 'en';
    hreflangEn.href = 'https://d2l.ai' + path;
    head.appendChild(hreflangEn);

    var hreflangDefault = document.createElement('link');
    hreflangDefault.rel = 'alternate';
    hreflangDefault.hreflang = 'x-default';
    hreflangDefault.href = 'https://d2l.ai' + path;
    head.appendChild(hreflangDefault);

    // --- 4d. Open Graph meta tags ---
    function setMeta(property, content) {
      if (!document.querySelector('meta[property="' + property + '"]')) {
        var meta = document.createElement('meta');
        meta.setAttribute('property', property);
        meta.content = content;
        head.appendChild(meta);
      }
    }

    setMeta('og:type', 'website');
    setMeta('og:locale', 'ja_JP');
    setMeta('og:site_name', 'ディープラーニングを深く学ぶ');
    setMeta('og:url', 'https://d2l-jp.me' + path);
    setMeta('og:image', 'https://d2l-jp.me/_static/logo-with-text.png');

    // --- 4e. Additional meta ---
    if (!document.querySelector('meta[name="robots"]')) {
      var robots = document.createElement('meta');
      robots.name = 'robots';
      robots.content = 'index, follow';
      head.appendChild(robots);
    }

    if (!document.querySelector('meta[name="author"]')) {
      var author = document.createElement('meta');
      author.name = 'author';
      author.content = 'Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola';
      head.appendChild(author);
    }

    // --- 4f. JSON-LD Structured Data ---
    var pageTitle = document.title || '';
    var pageDesc = '';
    var descMeta = document.querySelector('meta[name="description"]') ||
                   document.querySelector('meta[property="og:description"]');
    if (descMeta) pageDesc = descMeta.content;

    // WebPage schema
    var jsonLd = {
      '@context': 'https://schema.org',
      '@type': 'WebPage',
      'name': pageTitle,
      'description': pageDesc,
      'url': 'https://d2l-jp.me' + path,
      'inLanguage': 'ja',
      'isPartOf': {
        '@type': 'WebSite',
        'name': 'ディープラーニングを深く学ぶ',
        'url': 'https://d2l-jp.me/',
        'inLanguage': 'ja'
      },
      'translationOfWork': {
        '@type': 'WebPage',
        'url': 'https://d2l.ai' + path,
        'inLanguage': 'en'
      }
    };

    var scriptTag = document.createElement('script');
    scriptTag.type = 'application/ld+json';
    scriptTag.textContent = JSON.stringify(jsonLd);
    head.appendChild(scriptTag);

    // BreadcrumbList schema (from URL path)
    var pathParts = path.replace(/^\//, '').replace(/\.html$/, '').split('/');
    if (pathParts.length > 0 && pathParts[0] !== '' && pathParts[0] !== 'index') {
      var breadcrumbs = [{
        '@type': 'ListItem',
        'position': 1,
        'name': 'ホーム',
        'item': 'https://d2l-jp.me/'
      }];

      var currentPath = '';
      for (var i = 0; i < pathParts.length; i++) {
        currentPath += '/' + pathParts[i];
        var name = pathParts[i]
          .replace(/^chapter_/, '')
          .replace(/-/g, ' ');
        // Capitalize first letter
        name = name.charAt(0).toUpperCase() + name.slice(1);

        breadcrumbs.push({
          '@type': 'ListItem',
          'position': i + 2,
          'name': name,
          'item': 'https://d2l-jp.me' + currentPath + (i === pathParts.length - 1 ? '.html' : '/index.html')
        });
      }

      var breadcrumbLd = {
        '@context': 'https://schema.org',
        '@type': 'BreadcrumbList',
        'itemListElement': breadcrumbs
      };

      var breadcrumbScript = document.createElement('script');
      breadcrumbScript.type = 'application/ld+json';
      breadcrumbScript.textContent = JSON.stringify(breadcrumbLd);
      head.appendChild(breadcrumbScript);
    }
  }

  injectSEO();

  // =========================================================================
  // 5. Load instant.page for Speculative Prefetching
  // =========================================================================

  var script = document.createElement('script');
  script.type = 'module';
  script.src = 'https://cdn.jsdelivr.net/npm/instant.page@5.2.0/instantpage.js';
  script.integrity = 'sha384-jnZyxPjiipYXnSU0ez8Mcp8KlGnWw5JhEAXS2nNPahJJEFPEn2kpV30cGZBRSiN';
  script.crossOrigin = 'anonymous';
  document.head.appendChild(script);
})();
