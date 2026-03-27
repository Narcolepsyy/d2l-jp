// PJAX-style instant navigation for d2l-jp
// Fetches pages via AJAX and swaps only the main content area,
// keeping sidebar & navbar persistent for zero-flash navigation.
(function () {
  'use strict';

  // =========================================================================
  // Configuration
  // =========================================================================
  var CONTENT_SELECTOR = '.mdl-layout__content';
  var DRAWER_SELECTOR = '.mdl-layout__drawer';
  var TRANSITION_MS = 150;

  // =========================================================================
  // 0. Ensure sidebar logo image is displayed
  // =========================================================================
  // The Sphinx build may render only a <span class="title-text"> instead of
  // the <img class="logo"> when the html_logo variable isn't passed through.
  // This injects the logo image if only text is found.
  (function ensureSidebarLogo() {
    var titleLink = document.querySelector('.mdl-layout__drawer > .mdl-layout-title .title');
    if (!titleLink) return;
    // Already has an img.logo — nothing to do
    if (titleLink.querySelector('img.logo')) return;
    var textSpan = titleLink.querySelector('.title-text');
    if (!textSpan) return;

    var img = document.createElement('img');
    img.className = 'logo';
    img.src = '/_static/logo-with-text.png';
    img.alt = textSpan.textContent.trim();
    titleLink.replaceChild(img, textSpan);
  })();

  // Cache for prefetched pages
  var pageCache = {};
  var MAX_CACHE = 30;

  // =========================================================================
  // Fix: Normalize all sidebar & content links to absolute paths on init.
  // Relative hrefs break after pushState changes the base URL.
  // =========================================================================
  function normalizeSidebarLinks() {
    var drawer = document.querySelector(DRAWER_SELECTOR);
    if (!drawer) return;
    var links = drawer.querySelectorAll('a[href]');
    for (var i = 0; i < links.length; i++) {
      // link.href (property) resolves to absolute based on current base URL.
      // On initial page load, the base URL is correct, so we capture it now.
      var absoluteUrl = links[i].href; // already resolved to absolute
      var attr = links[i].getAttribute('href');
      // Only fix if the attribute is relative (doesn't start with / or http)
      if (attr && !attr.startsWith('/') && !attr.startsWith('http') && !attr.startsWith('#')) {
        links[i].setAttribute('href', new URL(absoluteUrl).pathname);
      }
    }
  }
  normalizeSidebarLinks();

  // =========================================================================
  // 1. PJAX Navigation Core
  // =========================================================================

  function isInternalLink(link) {
    if (!link) return false;
    var href = link.getAttribute('href');
    if (!href) return false;
    if (href.startsWith('#') ||
        href.startsWith('mailto:') ||
        href.startsWith('javascript:') ||
        link.hasAttribute('download') ||
        link.target === '_blank') {
      return false;
    }
    // Must be same origin
    try {
      var url = new URL(href, window.location.origin);
      return url.origin === window.location.origin;
    } catch (e) {
      return false;
    }
  }

  function fetchPage(url) {
    if (pageCache[url]) {
      return Promise.resolve(pageCache[url]);
    }
    return fetch(url)
      .then(function (response) {
        if (!response.ok) throw new Error('HTTP ' + response.status);
        return response.text();
      })
      .then(function (html) {
        // Cache it
        var keys = Object.keys(pageCache);
        if (keys.length >= MAX_CACHE) {
          delete pageCache[keys[0]];
        }
        pageCache[url] = html;
        return html;
      });
  }

  function parseHTML(html) {
    return new DOMParser().parseFromString(html, 'text/html');
  }

  function swapContent(doc, url) {
    var currentContent = document.querySelector(CONTENT_SELECTOR);
    var newContent = doc.querySelector(CONTENT_SELECTOR);
    if (!currentContent || !newContent) {
      // Fallback: something unexpected, do full navigation
      window.location.href = url;
      return;
    }

    // Swap inner HTML
    currentContent.innerHTML = newContent.innerHTML;

    // Scroll to top of content
    currentContent.scrollTop = 0;

    // Update document title
    document.title = doc.title;

    // Update meta description
    var newDesc = doc.querySelector('meta[name="description"]');
    var oldDesc = document.querySelector('meta[name="description"]');
    if (newDesc && oldDesc) {
      oldDesc.setAttribute('content', newDesc.getAttribute('content'));
    } else if (newDesc && !oldDesc) {
      document.head.appendChild(newDesc.cloneNode(true));
    }
  }

  function updateSidebarActive(url) {
    var drawer = document.querySelector(DRAWER_SELECTOR);
    if (!drawer) return;

    // Remove current active state
    var active = drawer.querySelectorAll('.current');
    for (var i = 0; i < active.length; i++) {
      active[i].classList.remove('current');
    }

    // Find and activate the new link
    var path = new URL(url, window.location.origin).pathname;
    var links = drawer.querySelectorAll('a[href]');
    for (var j = 0; j < links.length; j++) {
      var linkPath;
      try {
        linkPath = new URL(links[j].href).pathname;
      } catch (e) {
        continue;
      }
      if (linkPath === path) {
        links[j].classList.add('current');
        // Also mark parent li elements
        var parentLi = links[j].closest('li');
        if (parentLi) {
          parentLi.classList.add('current');
        }
        // Scroll sidebar to show active link
        var linkRect = links[j].getBoundingClientRect();
        var drawerRect = drawer.getBoundingClientRect();
        if (linkRect.top < drawerRect.top || linkRect.bottom > drawerRect.bottom) {
          links[j].scrollIntoView({ block: 'center', behavior: 'smooth' });
        }
      }
    }
  }

  function reinitPageScripts() {
    // Re-render MathJax
    if (window.MathJax) {
      if (MathJax.typesetPromise) {
        MathJax.typesetPromise().catch(function () {});
      } else if (MathJax.Hub) {
        MathJax.Hub.Queue(['Typeset', MathJax.Hub]);
      }
    }

    // Re-init Giscus comments
    initGiscusForPjax();

    // Re-init code block copy buttons or similar enhancements
    if (window.componentHandler) {
      componentHandler.upgradeDom();
    }
  }

  function initGiscusForPjax() {
    // Remove any existing giscus container
    var existing = document.querySelector('.giscus-container');
    if (existing) existing.remove();

    // Don't load on index pages
    var path = window.location.pathname;
    if (path === '/' || path === '/index.html') return;

    var container = document.querySelector('.page-content');
    if (!container) return;

    var wrapper = document.createElement('div');
    wrapper.className = 'giscus-container';
    wrapper.style.marginTop = '2rem';
    wrapper.style.paddingTop = '1.5rem';
    wrapper.style.borderTop = '1px solid #e0e0e0';

    var heading = document.createElement('h2');
    heading.textContent = 'コメント';
    heading.style.marginBottom = '1rem';
    wrapper.appendChild(heading);

    var script = document.createElement('script');
    script.src = 'https://giscus.app/client.js';
    script.setAttribute('data-repo', 'Narcolepsyy/d2l-jp');
    script.setAttribute('data-repo-id', 'R_kgDORumNZQ');
    script.setAttribute('data-category', 'Announcements');
    script.setAttribute('data-category-id', 'DIC_kwDORumNZc4C5WfX');
    script.setAttribute('data-mapping', 'pathname');
    script.setAttribute('data-strict', '0');
    script.setAttribute('data-reactions-enabled', '1');
    script.setAttribute('data-emit-metadata', '0');
    script.setAttribute('data-input-position', 'top');
    script.setAttribute('data-theme', 'preferred_color_scheme');
    script.setAttribute('data-lang', 'ja');
    script.setAttribute('data-loading', 'lazy');
    script.crossOrigin = 'anonymous';
    script.async = true;

    wrapper.appendChild(script);
    container.appendChild(wrapper);
  }

  function updateSEO() {
    var path = window.location.pathname;

    // Update canonical
    var canonical = document.querySelector('link[rel="canonical"]');
    if (canonical) {
      canonical.href = 'https://d2l-jp.me' + path;
    }

    // Update OG URL
    var ogUrl = document.querySelector('meta[property="og:url"]');
    if (ogUrl) {
      ogUrl.setAttribute('content', 'https://d2l-jp.me' + path);
    }

    // Update hreflang
    var hreflangJa = document.querySelector('link[hreflang="ja"]');
    if (hreflangJa) hreflangJa.href = 'https://d2l-jp.me' + path;
    var hreflangEn = document.querySelector('link[hreflang="en"]');
    if (hreflangEn) hreflangEn.href = 'https://d2l.ai' + path;
    var hreflangDefault = document.querySelector('link[hreflang="x-default"]');
    if (hreflangDefault) hreflangDefault.href = 'https://d2l.ai' + path;
  }

  // The main PJAX navigation function
  function navigateTo(url, pushState) {
    var content = document.querySelector(CONTENT_SELECTOR);
    if (!content) {
      window.location.href = url;
      return;
    }

    // Fade out
    content.classList.add('pjax-leaving');

    // Small delay for the fade-out animation
    setTimeout(function () {
      fetchPage(url)
        .then(function (html) {
          var doc = parseHTML(html);
          swapContent(doc, url);
          updateSidebarActive(url);
          updateSEO();

          if (pushState !== false) {
            history.pushState({ pjax: true, url: url }, '', url);
          }

          // Fade in
          content.classList.remove('pjax-leaving');
          // Force reflow
          void content.offsetHeight;
          content.classList.add('pjax-entering');

          setTimeout(function () {
            content.classList.remove('pjax-entering');
          }, TRANSITION_MS);

          // Re-initialize scripts after content is swapped
          reinitPageScripts();
        })
        .catch(function () {
          // On any error, fall back to normal navigation
          window.location.href = url;
        });
    }, TRANSITION_MS);
  }

  // Intercept internal link clicks
  document.addEventListener('click', function (e) {
    // Don't intercept if modifier keys are held (open in new tab)
    if (e.ctrlKey || e.metaKey || e.shiftKey || e.altKey) return;

    var link = e.target.closest('a[href]');
    if (!isInternalLink(link)) return;

    // Don't intercept anchor-only links on the same page
    var targetUrl = new URL(link.href);
    if (targetUrl.pathname === window.location.pathname && targetUrl.hash) {
      return;
    }

    e.preventDefault();
    navigateTo(link.href);
  });

  // Handle browser back/forward
  window.addEventListener('popstate', function (e) {
    if (e.state && e.state.pjax) {
      navigateTo(e.state.url, false);
    } else {
      // For non-PJAX history entries, do normal navigation
      navigateTo(location.href, false);
    }
  });

  // Replace the current history entry so popstate works on first back
  history.replaceState({ pjax: true, url: location.href }, '', location.href);

  // =========================================================================
  // 2. Speculative Prefetching on Hover
  // =========================================================================

  var prefetchTimeout = null;

  document.addEventListener('mouseover', function (e) {
    var link = e.target.closest('a[href]');
    if (!isInternalLink(link)) return;

    var url = link.href;
    if (pageCache[url]) return; // Already cached

    prefetchTimeout = setTimeout(function () {
      fetchPage(url).catch(function () {}); // Silent prefetch
    }, 80); // Short delay to avoid prefetching on accidental hover
  });

  document.addEventListener('mouseout', function (e) {
    if (prefetchTimeout) {
      clearTimeout(prefetchTimeout);
      prefetchTimeout = null;
    }
  });

  // Also prefetch on touchstart for mobile
  document.addEventListener('touchstart', function (e) {
    var link = e.target.closest('a[href]');
    if (!isInternalLink(link)) return;
    fetchPage(link.href).catch(function () {});
  }, { passive: true });

  // =========================================================================
  // 3. Preserve Drawer Scroll Position
  // =========================================================================
  // With PJAX, the drawer is never destroyed so scroll is naturally preserved.
  // This section only handles the initial page load from a cold start.

  var DRAWER_SCROLL_KEY = 'd2l-drawer-scroll';

  function restoreDrawerScroll() {
    var drawer = document.querySelector(DRAWER_SELECTOR);
    if (drawer) {
      try {
        var pos = sessionStorage.getItem(DRAWER_SCROLL_KEY);
        if (pos !== null) {
          drawer.scrollTop = parseInt(pos, 10);
        }
      } catch (e) { /* sessionStorage unavailable */ }
    }
  }

  window.addEventListener('beforeunload', function () {
    var drawer = document.querySelector(DRAWER_SELECTOR);
    if (drawer) {
      try {
        sessionStorage.setItem(DRAWER_SCROLL_KEY, drawer.scrollTop);
      } catch (e) { /* sessionStorage unavailable */ }
    }
  });

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', restoreDrawerScroll);
  } else {
    restoreDrawerScroll();
  }

  // =========================================================================
  // 4. Mobile Floating ToC Button
  // =========================================================================

  function createMobileTocButton() {
    var path = window.location.pathname;
    if (path === '/' || path === '/index.html') return;

    var btn = document.createElement('button');
    btn.className = 'mobile-toc-btn';
    btn.setAttribute('aria-label', 'Open table of contents');
    btn.setAttribute('title', '目次を開く');
    btn.innerHTML = '☰';

    btn.addEventListener('click', function () {
      var drawer = document.querySelector(DRAWER_SELECTOR);
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
  // 5. SEO Enhancements (runs once on initial load)
  // =========================================================================

  function injectSEO() {
    var head = document.head;
    var path = window.location.pathname;

    document.documentElement.setAttribute('lang', 'ja');

    // Ensure favicon is always present with an absolute path
    var existingFavicon = document.querySelector('link[rel="icon"], link[rel="shortcut icon"]');
    if (existingFavicon) {
      // Normalize to absolute path so it works on every subpage
      existingFavicon.href = '/_static/favicon.png';
    } else {
      var faviconLink = document.createElement('link');
      faviconLink.rel = 'icon';
      faviconLink.type = 'image/png';
      faviconLink.href = '/_static/favicon.png';
      head.appendChild(faviconLink);
    }

    if (!document.querySelector('link[rel="canonical"]')) {
      var canonical = document.createElement('link');
      canonical.rel = 'canonical';
      canonical.href = 'https://d2l-jp.me' + path;
      head.appendChild(canonical);
    }

    // Hreflang tags
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

    // Open Graph meta
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

    // JSON-LD
    var pageTitle = document.title || '';
    var pageDesc = '';
    var descMeta = document.querySelector('meta[name="description"]') ||
                   document.querySelector('meta[property="og:description"]');
    if (descMeta) pageDesc = descMeta.content;

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

    // BreadcrumbList
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

  // Mark initial page as ready (fade in on first load)
  document.documentElement.classList.add('page-ready');

})();
