/**
 * Shared utilities — loaded as a classic script from base.html so helpers
 * are available globally to all templates.
 */

/**
 * Append a cache-busting timestamp query parameter to url.
 * @param {string} url
 * @returns {string}
 */
function bust(url) {
    return url + (url.includes("?") ? "&" : "?") + "t=" + Date.now();
}

/**
 * Return a log(text, level) function bound to the given console element.
 * Appends timestamped lines and sets the element's CSS class on error/success.
 *
 * @param {HTMLElement} el - the console div to write into
 * @returns {function(string, string=): void}
 */
function makeConsoleLogger(el) {
    return function log(text, level = "info") {
        const stamp = new Date().toLocaleTimeString();
        el.textContent += `[${stamp}] ${text}\n`;
        el.scrollTop = el.scrollHeight;
        if (level === "error") el.className = "console error";
        else if (level === "success") el.className = "console success";
    };
}
