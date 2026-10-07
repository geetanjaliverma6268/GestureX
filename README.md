# AI-Powered Touchless Human-Computer Interaction System

**Using Hand Tracking and Gesture Recognition**

A real-time, camera-driven system that lets a user control computer functions —
media, volume, brightness, mouse, scrolling, screenshots, presentation and more —
without touching a mouse or keyboard. Built on MediaPipe hand tracking, a modular
geometry-based gesture engine (ML-ready), temporal smoothing, a centralised
action controller with cooldown/debounce, and a control-lock safety system.

---

## 1. Problem Statement

Conventional human-computer interaction depends on physical contact with a mouse
and keyboard. This is unhygienic in shared/public spaces, inaccessible for users
with limited mobility, and inefficient in contexts like presentations, surgery,
kiosks or kitchens where hands are occupied or must stay clean. A touchless,
vision-based control channel addresses these limitations using only a webcam.

## 2. Motivation

Modern commodity hardware (a laptop CPU + webcam) is now powerful enough to run
real-time hand-landmark inference. MediaPipe provides robust 21-point hand
tracking, which makes a practical, low-cost touchless control interface feasible
without special sensors, gloves or depth cameras.

## 3. Objectives

- Detect up to two hands and track 21 landmarks each, in real time (30+ FPS target).
- Recognise 11 static gestures plus dynamic (movement) gestures, each with a
  geometric confidence score.
- Map gestures to concrete computer-control actions across three operating modes.
- Guarantee safety: no action executes while the system is **CONTROL LOCKED**,
  and no action repeats without a cooldown.
- Remain stable and never crash when hands leave the frame, lighting changes, or
  tracking confidence drops.
- Use a modular, ML-ready architecture and a professional test suite.

## 4. Features

- MediaPipe **Tasks API** hand landmarker (VIDEO mode), up to 2 hands, handedness.
- **11 gestures**: Thumbs Up, Thumbs Down, Closed Fist, Pointing Up, Victory,
  OK Sign, I Love You, Open Palm, Pinch, Three Finger, Four Finger.
- **Geometric confidence** per gesture (never random); below threshold -> `UNKNOWN`.
- **Temporal smoothing**: EMA landmark smoothing + majority-vote gesture window.
- **Control Lock/Unlock**: system starts locked; hold Open Palm to toggle.
- **Cooldown / debounce**: a centralised `ActionController` gates every action.
- **Two-finger directional scroll** (motion-based, jitter-resistant).
- **Virtual mouse**: index-finger cursor, pinch click, pinch-hold drag, double click.
- **Media / volume / brightness** control via native Windows input.
- **Three-finger screenshot** (timestamped PNG) and **four-finger show desktop**.
- **Presentation mode** using generic arrow/F5/Esc keys (software-agnostic).
- **Three modes** (Normal / Mouse / Presentation) with clear on-screen status.
- **Optional web dashboard** (FastAPI) for live monitoring.
- **Logging**, **JSON configuration**, and a **53-test** headless suite.

## 5. System Architecture

```
            +------------------------------------------------------+
            |                       main.py                        |
            |  (camera loop, mode manager, key handling, routing)  |
            +------------------------------------------------------+
                 |                |                 |            |
                 v                v                 v            v
   +-------------------+  +----------------+  +-------------+  +-----------+
   |  hand_tracking/   |  |    gesture/    |  |  control/   |  |   ui/     |
   | detector          |  | finger_state   |  | platform_io |  | overlay   |
   | landmarks (EMA)   |->| features (ML)  |->| action_ctrl |->| dashboard |
   | handedness        |  | classifier     |  | mouse_ctrl  |  +-----------+
   +-------------------+  | confidence     |  | media/sys   |
            ^             | smoothing      |  | presentation|
            |             | direction_det  |  +-------------+
       MediaPipe          | events         |        |
     HandLandmarker       +----------------+   ctypes SendInput
        (VIDEO)                  |             / Pillow capture
                          GeometryClassifier
                          (or MLClassifier)
```

Data flow per frame:

```
Camera -> MediaPipe 21-landmark detection -> per-hand EMA smoothing
      -> gesture classification + confidence -> majority-vote smoothing
      -> CONTROL LOCK gate -> stability + confidence + cooldown gate
      -> action dispatch (media / volume / brightness / mouse / scroll /
         screenshot / desktop / presentation) -> OpenCV overlay (+ dashboard)
```

The classifier is an interface (`BaseGestureClassifier`). `GeometryClassifier`
ships by default; `MLGestureClassifier` can load a trained scikit-learn model
over the 48-dim feature vector from `calculate_landmark_features` without any
other change to the pipeline.

## 6. Technologies

| Technology | Purpose |
|---|---|
| Python 3 | Implementation language |
| MediaPipe (Tasks API) | 21-point hand landmark detection + handedness |
| OpenCV | Camera capture, rendering, real-time overlay UI |
| NumPy | Landmark geometry / vectorised math |
| Pillow | Full-screen capture for the screenshot gesture |
| ctypes (Win32 `SendInput`) | Keyboard/mouse/media/volume injection |
| screen-brightness-control | Display brightness (WMI/DDC-CI) |
| FastAPI + Uvicorn (optional) | Web monitoring dashboard |
| scikit-learn + joblib (optional) | Future ML gesture classifier |

> **Why not PyAutoGUI?** All input is delivered through the native Windows
> `SendInput` API. It is lower-latency, has no built-in `PAUSE`/failsafe that
> interferes with real-time control, and adds no dependency. PyAutoGUI's
> screenshot capability is replaced by Pillow's `ImageGrab`.

## 7. Installation

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the MediaPipe hand landmarker model into the project root
curl -L -o hand_landmarker.task ^
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## 8. How to Run

```bash
python main.py
```

Enable the optional web dashboard by setting `"dashboard": {"enabled": true}` in
`config.json`, then open `http://127.0.0.1:8000`.

**Runtime keys**

| Key | Action |
|---|---|
| `N` | Normal mode |
| `M` | Mouse mode |
| `P` | Presentation mode |
| `C` | Toggle computer control ON/OFF |
| `R` | Reset smoothing / history / cooldowns |
| `Q` / `ESC` | Quit |

The system starts **CONTROL LOCKED**. Hold **Open Palm** steadily (~12 frames) to
unlock; hold it again to re-lock. While locked, gestures are still recognised and
displayed, but **no computer action executes and the mouse never moves**.

## 9. Gesture-Command Table

**Normal mode**

| Gesture | Command |
|---|---|
| Thumbs Up | Play media |
| Thumbs Down | Stop media |
| Closed Fist | Previous track |
| Pointing Up | Volume up |
| Pinch | Volume down |
| OK Sign | Brightness up |
| I Love You | Brightness down |
| Open Palm (held) | Toggle Control Lock / Unlock |
| Three Finger (held) | Screenshot |
| Four Finger (held) | Show desktop |
| Victory + move up/down | Scroll up / down (dynamic) |

**Mouse mode** (left hand)

| Gesture | Command |
|---|---|
| Index finger move | Move cursor |
| Pinch | Left click |
| Pinch hold + move | Drag |
| Pinch release | Drop |
| Two quick pinches | Double click |
| Victory + move | Scroll |
| Open Palm (held) | Toggle Control Lock |

**Presentation mode** (left hand)

| Gesture | Command |
|---|---|
| Victory | Next slide (Right arrow) |
| Closed Fist | Previous slide (Left arrow) |
| Open Palm | Start/Stop slideshow (F5 / Esc) |

> **Two-hand operation:** the right hand is the media/system hand; the left hand
> drives the mouse (Mouse mode) or slides (Presentation mode). Each hand's
> gesture is tracked and gated independently so one hand cannot trigger the
> other's command.

## 10. Project Screenshots

Run the app and capture the OpenCV window; saved screenshots from the
three-finger gesture land in `screenshots/` as
`screenshot_YYYY-MM-DD_HH-MM-SS.png`.

## 11. Folder Structure

```
project/
├── main.py                     # application entry point
├── config.json                 # all tunables (no hard-coded thresholds)
├── requirements.txt
├── README.md
├── .gitignore
├── hand_landmarker.task        # MediaPipe model (downloaded, git-ignored)
├── hand_tracking/
│   ├── detector.py             # MediaPipe HandLandmarker (VIDEO mode)
│   ├── landmarks.py            # indices, geometry helpers, EMA smoother
│   └── handedness.py           # label normalisation + hand roles
├── gesture/
│   ├── finger_state.py         # calculate_finger_states / count_fingers
│   ├── gesture_features.py     # calculate_landmark_features (ML-ready, 48-dim)
│   ├── geometry.py             # shared geometry for classify + confidence
│   ├── confidence.py           # calculate_gesture_confidence
│   ├── gesture_classifier.py   # Geometry / ML classifiers + GestureEngine
│   ├── gesture_smoothing.py    # majority-vote temporal smoothing
│   ├── direction_detector.py   # detect_direction (dynamic gestures)
│   ├── events.py               # process_gesture_event (gesture -> intent)
│   └── thresholds.py           # config-backed geometry thresholds
├── control/
│   ├── platform_input.py       # Win32 SendInput + Pillow screen capture
│   ├── action_controller.py    # ActionController + ControlLock (safety gate)
│   ├── mouse_controller.py     # MouseController + ScrollController
│   ├── media_controller.py     # play / stop / next / prev
│   ├── system_controller.py    # volume / brightness / screenshot / desktop
│   └── presentation_controller.py
├── ui/
│   ├── overlay.py              # OpenCV status panel + hand skeletons
│   └── dashboard.py            # optional FastAPI monitoring dashboard
├── utils/
│   ├── config.py               # config.json loader (deep-merged defaults)
│   ├── logger.py               # event-based logging (+ throttle helper)
│   ├── performance.py          # smoothed FPS counter
│   └── constants.py            # gesture / mode / hand-label names
├── screenshots/                # captured screenshots (git-ignored)
├── tests/                      # 53-test headless suite
│   ├── synthetic_hands.py      # landmark generator for tests
│   ├── helpers.py              # frame driver + OS-input kill switch
│   ├── test_gestures.py
│   ├── test_smoothing_direction.py
│   ├── test_control.py
│   ├── test_pipeline.py
│   └── test_integration.py
└── legacy/
    └── hand_tracking.py        # original single-file prototype (reference)
```

## 12. Configuration (`config.json`)

Every threshold is configurable: camera index, model path, detection/tracking
confidence, gesture confidence threshold, smoothing window, hold frames,
landmark smoothing alpha, all cooldowns, scroll threshold/steps, cursor region
and sensitivity, click/double-click timing, brightness step, volume presses,
screenshot directory, UI window, dashboard host/port and logging.

## 13. Testing

```bash
python -m unittest discover -s tests -t .
```

53 tests run fully headless — no webcam, and **all real OS input is disabled**
(`platform_input.set_enabled(False)`), so tests never fire media keys, move the
mouse or write screenshots. Coverage:

- All 11 gestures classify correctly with confidence above threshold.
- Finger counting, confidence bounds, scale invariance.
- Majority-vote smoothing (noise frames do not flip the label).
- Direction detection (up/down/left/right, threshold, jitter rejection).
- `ActionController`: lock gate, disabled gate, stability gate, confidence gate,
  cooldown, cooldown override, reset.
- `ControlLock`: starts locked, hold-to-toggle, edge-triggered release.
- Action mapping for all three modes.
- Pipeline: no-hand / one-hand / two-hand routing, two-finger scroll,
  three-finger screenshot, four-finger desktop, static-two-finger does not
  scroll, locked pipeline never acts, mouse never moves while locked,
  cooldown-bounded noise, rapid gesture changes never settle.
- ML-ready features: dimension, translation/scale invariance, separability.
- `main.run()` integration with a mocked camera + landmarker (full loop, clean shutdown).

## 14. Limitations

- Input injection (media/volume/mouse/keys, Win+D) is **Windows-only**
  (`SendInput`); other OSes degrade gracefully to no-ops. Brightness needs
  WMI/DDC-CI support.
- Windows exposes a single combined Play/Pause key; there is no discrete
  "Play" vs "Pause" virtual key, so Play sends the toggle and Stop is separate.
- The live webcam view cannot be verified in a headless/CI environment; the
  integration test mocks the camera. Real-world lighting and hand appearance
  vary.
- The geometry classifier is rule-based; unusual hand poses may read as `UNKNOWN`
  (which is the safe outcome — no action fires).
- No ML model is shipped/trained (no dataset); `MLGestureClassifier` is a
  ready integration point only.

## 15. Future Enhancements

- Train the ML classifier (Random Forest / SVM / MLP) on collected landmark
  feature vectors; swap it in via `GestureEngine`.
- Add dynamic gestures: swipe left/right, circle, wave, pinch-zoom in/out.
- Continuous brightness/volume while a gesture is held (rate-limited).
- Per-application gesture profiles and a gesture remapping UI.
- Multi-monitor and DPI-aware cursor mapping.
- macOS/Linux input backends (e.g. `pyobjc` / `xdotool`).

## 16. Research Possibilities

- Temporal models (LSTM / Transformer) over landmark sequences for dynamic
  gesture and continuous-command recognition.
- Person-independent calibration and domain adaptation for robustness across
  users, skin tones and lighting.
- Multimodal fusion (hand + gaze + head pose) for richer hands-free HCI.
- Accessibility studies: usability of touchless control for motor-impaired users.
- Latency/accuracy trade-offs of on-device vs. edge inference.

---

### Notes

- The original single-file prototype is preserved under `legacy/hand_tracking.py`
  for reference; the modular package is the maintained implementation.
- Logging is event-based (camera ready, gesture fired, action executed, control
  locked, warnings) — it does not spam per frame.
