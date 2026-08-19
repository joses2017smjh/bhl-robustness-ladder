#!/bin/bash
set -euo pipefail
cd "$REPO"
ffmpeg -y -loglevel error -i docs/gifs/multi_race.mp4 -vf "select=eq(n\,80)" -vframes 1 docs/gifs/_frame_early.png
ffmpeg -y -loglevel error -i docs/gifs/multi_race.mp4 -vf "select=eq(n\,180)" -vframes 1 docs/gifs/_frame_late.png
ls -lh docs/gifs/_frame_*.png
