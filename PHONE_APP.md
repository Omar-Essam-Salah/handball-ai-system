# Handball Studio — Phone App

Watch and control the whole match from your phone. The AI runs on your PC; the
phone is a fast, good-looking remote view (live video + score + players +
tactics + all controls). Nothing in the app has a hardcoded IP — it all routes
to whichever server the phone connects to.

There are **two ways to get it on your phone**. Both use the same app UI (the
files in `webapp/`), so anything you change there shows up in both.

---

## Option A — Use it now (no build, recommended for testing)

1. On the PC, double-click **`Handball_Mobile.bat`** (or run
   `\.venv\Scripts\python.exe src\mobile_server.py --source handball.mp4`).
2. It prints a line like:
   ```
   http://192.168.1.20:8000
   ```
3. On your phone (**same WiFi**), open that address in **Chrome**.
4. Tap the browser menu → **Add to Home Screen**. Now it has an app icon and
   opens full-screen like a real app (it's an installed PWA).

That's the whole app — Match / Players / Tactics / Controls tabs, live video,
camera scan, manual goals, etc.

> The PC's IP can change (DHCP). If it does, just open the new address the
> server prints. Tip: give the PC a reserved/static lease in your router, or use
> the auto-find in the real APK (Option B).

---

## Option B — Real installable `.apk` (built in the cloud, no tools to install)

The native app is a thin shell (`android/`) that finds your PC on WiFi and loads
the same UI. You build the APK once in GitHub's cloud — you don't need Android
Studio, the SDK, or Java on your PC.

1. Put this project on GitHub (one time):
   ```bash
   git init
   git add .
   git commit -m "Handball Studio"
   git branch -M main
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
2. On GitHub → **Actions** tab → **Build Android APK** → **Run workflow**.
3. When it finishes (~3–5 min), open the run → **Artifacts** →
   download **handball-studio-apk** → unzip → `app-debug.apk`.
4. Copy the APK to your phone and install it (allow "install from unknown
   sources"). Open it, tap **Auto-find on WiFi** (or type the PC address once).

To change the app later, just edit the files in **`webapp/`** — no rebuild
needed; the installed app loads the live UI from your PC.

---

## What you can do from the phone
- **Match:** live video, score, clock, possession, shots, fast breaks, passes, events
- **Players:** live on-court list + jersey numbers + persistent identities
- **Tactics:** every auto-saved pre-goal freeze — tap to enlarge
- **Controls:** Recalibrate · Flip Sides · Goal Home · Goal Away · Debug · Resume
- **Source:** open a video file, paste an RTSP URL, or **Scan cameras** on the network

## Files
| Path | What |
|---|---|
| `webapp/` | the app UI (HTML/CSS/JS) — edit here to change anything |
| `src/mobile_server.py` | PC server: runs the pipeline, streams video + data |
| `src/camera_discovery.py` | network camera auto-detect |
| `android/` | thin WebView shell (compiled to the APK) |
| `.github/workflows/build-apk.yml` | cloud APK build |
| `Handball_Mobile.bat` | start the server |
