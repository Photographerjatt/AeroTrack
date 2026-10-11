# AeroTrack

**Point-tracking video stabilizer for aircraft footage.**

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-early%20release-orange)

Shooting a plane with a handheld camera or a long lens? AeroTrack locks onto the
aircraft and keeps it fixed in the centre of the frame, so the **aircraft stays
still and the world moves around it**. It works like the point tracker in Premiere
Pro, After Effects or DaVinci Resolve, in a small desktop app.

<!-- Add a screenshot or GIF here:  ![AeroTrack preview](docs/screenshot.png) -->

## Features

- **Click-to-track:** click the aircraft once, press *Track Forward*.
- **Stabilised live preview:** the dot, circle and crop box stay fixed at the centre
  while the frame slides underneath, so you see exactly what the export will look like.
- **Robust tracker:** sub-pixel template matching with drift correction, so it keeps
  following the aircraft as it banks, changes size or the camera shakes.
- **Honest failure handling:** if the track is lost (cloud, glare, fast motion) it stops
  on the frame where it failed. Re-click the aircraft and carry on, or step back and re-track.
- **Non-destructive:** scrubbing the timeline never deletes tracking data.
- **Export to MP4:** H.264 with the original audio (via FFmpeg), choice of aspect ratio
  (16:9, 9:16, 4:3, 1:1, 3:2, original) and resolution (720p to 4K).
- **Controls for tuning:** patch size, search range, minimum match, zoom, smoothing,
  edge handling.

## Installation

Requires **Python 3.11 or newer**.

```powershell
git clone https://github.com/<your-username>/AeroTrack.git
cd AeroTrack

python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS / Linux: source .venv/bin/activate

pip install -r requirements.txt
python app/tracker.py
```

> **OpenCV note:** install only **one** OpenCV package (`opencv-contrib-python` is what
> `requirements.txt` uses). Having both `opencv-python` and `opencv-contrib-python`
> in the same environment causes import problems. If you hit that, run
> `pip uninstall -y opencv-python opencv-contrib-python` and then
> `pip install -r requirements.txt`.

## Quick start

1. **Open Video**: pick your clip.
2. **Select Tracking Point**, then click on a distinctive part of the aircraft
   (fuselage / cockpit / wing root, not empty sky).
3. **Track Forward**: tracking runs until the end of the video or until it loses the aircraft.
4. Check the result with **Play Preview** or by dragging the timeline.
5. Pick *Aspect*, *Zoom* and *Output*, then press **Generate Output Video**.

If tracking is lost, the app stops on the failing frame: click **Select Tracking Point**,
click the aircraft again, and press **Track Forward**. To redo a section, step back with
the arrow keys, re-select the aircraft there and track forward again.

### Keyboard shortcuts

| Key | Action |
| --- | --- |
| `←` / `→` | Previous / next frame |
| `Shift` + `←` / `→` | 5 frames back / forward |

### Settings

| Setting | What it does |
| --- | --- |
| **Patch** | Size of the image patch that is followed. Cover the aircraft with a little background around it. |
| **Search range** | How far (px) the aircraft can move between two frames. Raise it for shaky footage. |
| **Min match** | Minimum match confidence before tracking is declared lost. Lower = more tolerant, higher = stops earlier. |
| **Zoom** | How tightly the output is cropped. Higher zoom gives the crop more room to move, so fewer black borders. |
| **Smoothing** | `off` locks the aircraft exactly. Higher values smooth the camera path and leave some of the aircraft's motion in. |
| **Edges** | *Exact lock* keeps the aircraft perfectly centred (black borders if the crop leaves the video). *Clamp* never shows black but lets the aircraft drift off-centre near the edges. |
| **Aspect / Output** | Shape and resolution of the exported video. |

## How it works

1. A small patch around your click is stored as a template.
2. For every new frame, the template is matched (normalised cross-correlation) inside a
   search window around the previous position, with sub-pixel refinement.
3. The original patch is used as a drift anchor; a second, slowly updated patch adapts to
   banking and lighting changes.
4. The crop window is centred on the tracked point every frame, rendered with sub-pixel
   accuracy, and resized to the output resolution.

## Audio and encoding

If FFmpeg is available (system `ffmpeg` on your PATH, or the bundled one from
`imageio-ffmpeg`), the export is H.264 (CRF 17) with the original audio for the tracked
range. Without FFmpeg it falls back to OpenCV's writer, which has **no audio** and lower
quality.

## Project structure

```
AeroTrack/
├── app/
│   └── tracker.py          # the application
├── requirements.txt
├── CHANGELOG.md
├── LICENSE
└── README.md
```

## Known limitations (v0.1.0)

- Tracks a single point and stabilises translation only (no rotation / horizon levelling).
- A long occlusion needs a manual re-select.
- Export runs on the UI thread (a progress dialog with *Cancel* is shown).
- Only the tracked range is exported.

## Roadmap

- [ ] Save / load tracking data
- [ ] Automatic re-acquire after short occlusions
- [ ] Rotation / horizon stabilisation
- [ ] Background-thread export
- [ ] Batch processing

## Versioning

AeroTrack follows [Semantic Versioning](https://semver.org/). The current version is
**0.1.0**. See [CHANGELOG.md](CHANGELOG.md) for the release history.

## License

Released under the [MIT License](LICENSE).