// Stub for tagClick (used by d2l-book tab bars, normally provided by GA)
if (typeof tagClick === 'undefined') {
  var tagClick = function(tab) {};
}

// Giscus comment system integration for d2l-jp
// Automatically appends a Giscus widget to every page
(function () {
  'use strict';

  // Don't load on index/table-of-contents pages
  var path = window.location.pathname;
  if (path === '/' || path === '/index.html') return;

  // Wait for DOM to be ready
  function initGiscus() {
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

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initGiscus);
  } else {
    initGiscus();
  }
})();
