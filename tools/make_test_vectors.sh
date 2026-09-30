#!/usr/bin/env bash
# PS5 AT9 Converter - Generate the synthetic test vectors.
# Copyright (C) 2026 BlackBearReloaded
# SPDX-License-Identifier: GPL-3.0-or-later

# Regenerate the small synthetic test vectors in tests/data (needs ffmpeg with libmp3lame).
# The signals are generated here, so the files contain no third-party audio.
set -euo pipefail
cd "$(dirname "$0")/../tests/data"

tone="aevalsrc='0.5*sin(2*PI*440*t)|0.5*sin(2*PI*660*t)':s=44100:d=1"
clicks="aevalsrc='0.8*sin(2*PI*880*t)*exp(-40*mod(t,0.25))':s=44100:d=1"

ffmpeg -v error -y -f lavfi -i "$tone" -c:a libmp3lame -b:a 128k tone-44k-stereo.mp3
ffmpeg -v error -y -f lavfi -i "$clicks" -ac 2 -c:a libmp3lame -b:a 96k clicks-44k.mp3
ffmpeg -v error -y -f lavfi -i "$tone" -ar 22050 -ac 1 -c:a libmp3lame -b:a 32k tone-22k-mono.mp3
ffmpeg -v error -y -f lavfi -i "$tone" -ar 8000 -ac 1 -c:a libmp3lame -b:a 16k tone-8k-mono.mp3
ffmpeg -v error -y -f lavfi -i "$tone" -sample_fmt s16 -compression_level 8 -c:a flac tone-44k.flac
ls -l
