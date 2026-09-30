#!/usr/bin/env python3
# PS5 AT9 Converter - Command-line entry point.
# Copyright (C) 2026 BlackBearReloaded
# SPDX-License-Identifier: GPL-3.0-or-later

"""Convert an audio file into a PlayStation 5 snd0.at9 selection-music asset.

Example: ./ps5_at9.py --input song.mp3 --output snd0.at9
Requires Python 3.8+ and numpy. Run with --help for all options.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import numpy  # noqa: F401
except ImportError:
    sys.exit("error: this tool needs numpy. Install it with:  python -m pip install numpy")

from ps5at9.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
