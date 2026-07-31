/* Service worker — caches the app shell so it opens instantly and works as an
   installed PWA. Live data (/status, /video, /goals, /scan) is NEVER cached. */
const CACHE = 'handball-shell-v8';
const SHELL = ['/', '/index.html', '/style.css', '/app.js',
               '/manifest.webmanifest', '/icon-192.png', '/icon-512.png'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(keys =>
    Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
  ).then(() => self.clients.claim()));
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  // Always go to network for live/data + cross-origin (e.g. a typed server).
  // calib covers both /calib.html and the /calib/* API — always network-fresh.
  const livePath = /^\/(status|video|goals|goal|scan|cmd|open|stop|ping|calib|analyze)/.test(url.pathname);
  if (livePath || url.origin !== location.origin || e.request.method !== 'GET') {
    return;   // default: let the browser fetch from network
  }
  e.respondWith(
    caches.match(e.request).then(hit => hit || fetch(e.request).then(resp => {
      const copy = resp.clone();
      caches.open(CACHE).then(c => c.put(e.request, copy)).catch(()=>{});
      return resp;
    }).catch(() => caches.match('/index.html')))
  );
});
