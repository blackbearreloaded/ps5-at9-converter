# Third-party notices

PS5 AT9 Converter depends on, interoperates with, or was developed using the
projects below. Inclusion here does not imply endorsement.

## Tables converted into the source

### LibAtrac9

[LibAtrac9](https://github.com/Thealexbarney/LibAtrac9) by Alex Barney
documents the ATRAC9 bitstream. The encoder and the reference decoder follow its
decoder rules, and [`tools/gen_atrac9_tables.py`](tools/gen_atrac9_tables.py)
converts its constant tables (Huffman codebooks, scale factor weights, gradient
curve and band extension tables) into
[`ps5at9/atrac9_tables.py`](ps5at9/atrac9_tables.py). LibAtrac9 is distributed
under the MIT License, Copyright (c) 2018 Alex Barney. The full text is in
[`third_party/LibAtrac9.LICENSE.txt`](third_party/LibAtrac9.LICENSE.txt).

### PDMP3

[PDMP3](https://github.com/technosaurus/PDMP3) by Krister Lagerström supplies
the MP3 Huffman code trees and the synthesis window. They are converted into
[`ps5at9/mp3_tables.py`](ps5at9/mp3_tables.py) by
[`tools/gen_mp3_tables.py`](tools/gen_mp3_tables.py). PDMP3 is released into
the public domain under the Unlicense.

### minimp3

[minimp3](https://github.com/lieff/minimp3) by lieff supplies the MP3 scale
factor band widths for every MPEG-1, MPEG-2 and MPEG-2.5 sample rate. They are
converted by the same script. minimp3 is dedicated to the public domain under
CC0 1.0.

## Runtime dependency

[numpy](https://numpy.org/) is the only runtime dependency. Users install it
themselves, and it is not redistributed. numpy is distributed under the BSD
3-Clause License.

## Specifications

The MP3, FLAC, loudness and true-peak code is written from public
specifications:

- ISO/IEC 11172-3 and ISO/IEC 13818-3;
- RFC 9639;
- ITU-R BS.1770-4.

## Development references

These tools were used only during development. They are not build or runtime
dependencies, and nothing from them is included or redistributed:

- Sony's `at9tool` served as the reference decoder for conformance checks and
  as the quality baseline. It is never invoked by this project, and it was not
  decompiled.
- [FFmpeg](https://ffmpeg.org/) was used to cross-check the MP3 and FLAC
  decoders. With [LAME](https://lame.sourceforge.io/), it also generated the
  synthetic MP3 and FLAC test vectors in `tests/data` from tones defined in
  [`tools/make_test_vectors.sh`](tools/make_test_vectors.sh).
- [EAQUAL](https://github.com/godock/eaqual) measured PEAQ quality scores.

## Trademarks

ATRAC9 and PlayStation are associated with Sony Interactive Entertainment and
Sony Group Corporation. Their use here describes file compatibility only and
does not imply endorsement.
