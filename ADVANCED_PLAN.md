# 🚀 Advanced Handball AI: Cross-Angle Tracking & Tactical Analysis Plan

## 📌 Context
The project is upgrading from a static-camera setup to handling **dynamic, multi-angle broadcast videos**. The goal is to accurately track player identities across camera cuts, map passes (who passed to whom), and generate deep tactical insights matching the "IDEA folder" standard.

## 🛠️ Phase 1: Rock-Solid Player Identification (Cross-Angle Re-ID)
**Goal:** Prevent ID switching when the camera angle changes.
1. **Deep Feature Extraction (Upgrade from HSV):** 
   - Replace or augment the current color-based histogram (HSV) with a lightweight Deep Re-ID model (e.g., OSNet or a ResNet-based feature extractor). 
   - Store embeddings in a rolling memory bank per player.
2. **Jersey Number OCR (Hard Anchor):**
   - Integrate `PaddleOCR` or `EasyOCR` localized to the player's bounding box (specifically the upper body/back).
   - Use the jersey number + team color as the absolute source of truth to merge broken tracks across camera cuts.

## ⚽ Phase 2: Tactical Action Graph (Passing & Cooperation)
**Goal:** Understand "Who did what" and "Who collaborated with whom."
1. **Possession State Machine:**
   - Track ball-to-player proximity over time.
   - Define strict states: `Player A Possession` -> `Ball in Flight` -> `Player B Possession`.
2. **Event Ledger:**
   - Create an event log: `[Time, Action, Player_ID, Zone]`.
   - Actions to detect: `Pass` (if Player B is same team), `Turnover` (if Player B is opposite team), `Shot` (if ball trajectory heads to goal frame).
3. **Assist Mapping:** Backtrack the event ledger to link the last pass to the goal scorer.

## 🗺️ Phase 3: Dynamic Context & Visualization (IDEA Folder Level)
**Goal:** Adapt analysis to moving cameras and output professional overlays.
1. **Dynamic Court Mapping:** If homography points (court lines) are lost due to a tight camera angle, fallback to relative positioning (using the nearest D-line or sideline).
2. **Advanced Overlays:** Update `goal_replay.py` to draw:
   - Player movement paths (trails).
   - Pass trajectory arrows connecting Player A to Player B.
   - Distance indicators (e.g., "9m Shot").

## 🎯 Immediate Execution Task for Claude
**Do not jump to UI/PDF generation.**
Start exclusively with **Phase 1 (Jersey Number OCR + Deep Re-ID)**. 
- Propose a lightweight way to extract Jersey numbers from existing bounding boxes without killing the FPS.
- How can we update `multi_session_reid.py` to use OCR as the primary key for player IDs?