# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-11

First public release.

### Added
- Point tracker based on template matching with sub-pixel precision, drift
  correction against the original patch, and slow adaptation to appearance changes.
- Stabilised live preview: the tracking dot, circle and crop box stay fixed at the
  centre while the video frame moves underneath.
- Export of the stabilised crop to MP4 (H.264 + original audio when FFmpeg is
  available, OpenCV writer fallback otherwise).
- Adjustable patch size, search range, minimum match confidence, zoom, smoothing,
  edge handling (exact lock / clamp), aspect ratio and output resolution.
- Automatic stop on lost tracking, with re-select and resume from any tracked frame.
- Preview playback, frame stepping (buttons and arrow keys) and Clear Track.

### Fixed (vs. the pre-release prototype)
- Tracking no longer dies after a few seconds (the old optical-flow feature cloud
  was never replenished).
- Scrubbing the timeline no longer deletes tracking data.
- `self.width` / `self.height` no longer shadow `QWidget.width()` / `height()`.