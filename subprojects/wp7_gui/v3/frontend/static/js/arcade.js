/* Arcade cover screen — inactivity / attract-mode logic. */
(function () {
    'use strict';

    const INACTIVITY_MS = 180 * 1000; // 120 seconds → cover reappears
    const ACTIVITY_EVENTS = ['mousemove', 'mousedown', 'keydown', 'scroll', 'touchstart'];

    const cover = document.getElementById('arcadeCover');
    const idleNote = document.getElementById('arcadeIdleNote');
    const SESSION_KEY = 'arcade-dismissed';
    let timer = null;

    function showCover() {
        idleNote.textContent = sessionStorage.getItem(SESSION_KEY)
            ? 'Session idle \u2014 attract mode restored'
            : '';
        cover.classList.remove('arcade-hidden');
        cover.focus();
    }

    function hideCover() {
        sessionStorage.setItem(SESSION_KEY, '1');
        idleNote.textContent = '';
        cover.classList.add('arcade-hidden');
        resetTimer();
    }

    function resetTimer() {
        clearTimeout(timer);
        timer = setTimeout(showCover, INACTIVITY_MS);
    }

    /* Activity outside the cover resets the inactivity countdown.
     * Throttled: at most one timer reset per 500 ms to avoid
     * clearTimeout/setTimeout on every mousemove event. */
    let lastActivity = 0;
    function onActivity() {
        if (!cover.classList.contains('arcade-hidden')) return;
        const now = Date.now();
        if (now - lastActivity < 500) return;
        lastActivity = now;
        resetTimer();
    }

    /* Dismiss on click anywhere on the cover */
    cover.addEventListener('click', hideCover);

    /* Dismiss on any key while cover has focus */
    cover.addEventListener('keydown', function (e) {
        if ([' ', 'Enter', 'Escape', 'ArrowRight', 'ArrowDown'].includes(e.key)) {
            e.preventDefault();
            hideCover();
        }
    });

    /* Track user activity to reset inactivity timer */
    ACTIVITY_EVENTS.forEach(function (evt) {
        document.addEventListener(evt, onActivity, { passive: true });
    });

    /* Show on first-ever load; on subsequent page navigations just restart the timer */
    if (!sessionStorage.getItem(SESSION_KEY)) {
        showCover();
    } else {
        resetTimer();
    }
})();
