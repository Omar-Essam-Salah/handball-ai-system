# 🚀 Master Handball AI: Performance & Advanced Tactical Analytics Plan

## 📌 Context & Vision
The goal is to upgrade the current pipeline to match international tier-1 analysis software (like Once Sport / Handball.ai). The system must extract deep tactical data (jump shots, pass networks, 3x3 goal grids, 2D court mapping) from dynamic, multi-angle broadcast videos.
**Current Blocker:** The pipeline is too heavy. FPS has dropped significantly due to synchronous heavy processing and rendering on every frame.

---

## ⚡ Phase 1: Pipeline Optimization & FPS Recovery
**Goal:** Restructure the pipeline to run at real-time or near-real-time speeds before adding more tactical weight.
1. **Frame Skipping & Interpolation:**
   - Do not run YOLO Pose + Object Detection on every single frame. Process every Nth frame (e.g., every 3rd frame) and use lightweight optical flow or bounding box interpolation (via BoT-SORT) for the frames in between.
2. **Asynchronous Rendering:**
   - Decouple the analysis engine from the video rendering engine. Tactical overlays should be drawn asynchronously or saved as metadata to be rendered post-process, rather than blocking the main loop.
3. **Region of Interest (ROI) Dynamic Cropping:**
   - Only run the high-res ball detector on the quadrant of the screen where the ball was last seen or where player density is highest.

---

## 🔍 Phase 2: Unbreakable Identity (Cross-Angle Tracking)
**Goal:** Maintain Player IDs across camera cuts to accurately build passing networks.
1. **OCR Jersey Number Anchoring:**
   - Apply lightweight OCR (like EasyOCR) strictly to the upper torso bounding boxes of detected players.
   - Use OCR + Team Color HSV to build a definitive ID ledger that survives camera angle changes.
2. **Deep Re-ID Fallback:**
   - Use a lightweight feature extractor to match players when their backs (numbers) aren't visible.

---

## 🧠 Phase 3: Tactical Action Engine (The "IDEA" Features)
**Goal:** Translate YOLO tracks into tactical handball events.
1. **Pass Network & Collaboration:**
   - Track ball-to-hands intersections.
   - Record `[Player A -> Ball in Air -> Player B]`. If both are the same team = Pass. If different = Turnover.
2. **Technique Analysis (Jump Shots & Angles):**
   - Utilize YOLO-Pose keypoints to analyze technique. 
   - **Jump Shot Detection:** Compare the Y-coordinates of the player's ankle keypoints against the detected court lines/homography ground plane at the moment of ball release.
3. **Shot Origin & Distance:**
   - Use homography (2D court mapping) to calculate the exact distance (e.g., 6m, 9m, wing) of the shot origin.

---

## 📊 Phase 4: Professional Dashboards & Output
**Goal:** Generate the visual outputs seen in standard tactical software.
1. **3x3 Goal-Mouth Grid:**
   - Map the ball's final coordinates relative to the goalpost bounding box to classify shots into 9 zones (Top-Left, Bottom-Right, etc.) for Goalkeeper vs. Shooter analysis.
2. **2D Court Scatter Plots:**
   - Translate player/event coordinates from the video perspective into a top-down 2D court diagram showing team connections and shooting locations.
   

---

## 🎯 Immediate Execution Task for Claude
1. Read this `MASTER_TACTICAL_PLAN.md` document.
2. Start strictly with **Phase 1 (Optimization)**. 
3. Review `pipeline.py` and propose a code update to implement Frame Skipping/Interpolation and decouple the rendering to fix the FPS drop immediately. Do not move to Phase 2 until FPS is stable.
