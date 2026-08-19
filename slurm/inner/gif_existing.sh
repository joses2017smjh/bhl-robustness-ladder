#!/bin/bash
set -euo pipefail
cd "$REPO"
ffmpeg -y -loglevel error -i docs/gifs/multi_race.mp4 \
  -vf "fps=10,scale=880:-2:flags=lanczos,trim=duration=9,setpts=PTS-STARTPTS,split[s0][s1];[s0]palettegen=max_colors=128[pal];[s1][pal]paletteuse=dither=bayer:bayer_scale=3" \
  -loop 0 docs/gifs/multi_race.gif
ffmpeg -y -loglevel error -i docs/gifs/multi_race.mp4 -vf "select=eq(n\,150)" -vframes 1 /tmp/multi_race_frame.png
ls -lh docs/gifs/multi_race.gif /tmp/multi_race_frame.png
