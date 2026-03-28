// Stub for tagClick (used by d2l-book tab bars, normally provided by GA)
if (typeof tagClick === 'undefined') {
  var tagClick = function(tab) {};
}

// Giscus comment system integration for d2l-jp
// Lazy-loads via Intersection Observer - only when user scrolls near comments section
(function () {
  'use strict';

  // Don't load on index/table-of-contents pages
  var path = window.location.pathname;
  if (path === '/' || path === '/index.html') return;

  let giscusLoaded = false;

  function initGiscus() {
    if (giscusLoaded) return;
    giscusLoaded = true;

    // Find the main content area
    var container = document.querySelector('.page-content');
    if (!container) return;

    // Create a wrapper div for Giscus
    var wrapper = document.createElement('div');
    wrapper.className = 'giscus-container';
    wrapper.style.marginTop = '2rem';
    wrapper.style.paddingTop = '1.5rem';
    wrapper.style.borderTop = '1px solid #e0e0e0';

    // Add a heading
    var heading = document.createElement('h2');
    heading.textContent = 'コメント';
    heading.style.marginBottom = '1rem';
    wrapper.appendChild(heading);

    // Create the Giscus script element
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

  // Use Intersection Observer for true lazy loading
  // Only load Giscus when user scrolls near the bottom of the page
  function setupObserver() {
    var container = document.querySelector('.page-content');
    if (!container) return;

    // Create a sentinel element at the bottom
    var sentinel = document.createElement('div');
    sentinel.className = 'giscus-sentinel';
    
    var observer = new IntersectionObserver(function(entries) {
      entries.forEach(function(entry) {
        if (entry.isIntersecting) {
          initGiscus();
          observer.unobserve(entry.target);
        }
      });
    }, {
      rootMargin: '50px' // Start loading 50px before user reaches bottom
    });

    container.appendChild(sentinel);
    observer.observe(sentinel);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupObserver);
  } else {
    setupObserver();
  }
})();
