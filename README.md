<div align="center">

# PS5 AT9 Converter

Turn any MP3, WAV, AIFF or FLAC file into the `snd0.at9` selection music of a
PlayStation 5 app with one command and no external tools.

[![Build](https://github.com/blackbearreloaded/ps5-at9-converter/actions/workflows/ci.yml/badge.svg)](https://github.com/blackbearreloaded/ps5-at9-converter/actions/workflows/ci.yml)
[![Latest release](https://img.shields.io/github/v/release/blackbearreloaded/ps5-at9-converter)](https://github.com/blackbearreloaded/ps5-at9-converter/releases/latest)
[![License: GPL v3+](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)

</div>

> [!IMPORTANT]
> The encoder's output is validated against Sony's reference ATRAC9 decoder,
> but this release has not been played on PS5 hardware yet. Please report
> results through [GitHub issues](https://github.com/blackbearreloaded/ps5-at9-converter/issues).

## Project foundations

PS5 AT9 Converter contains its own ATRAC9 encoder, written from the bitstream
documented by [LibAtrac9](https://github.com/Thealexbarney/LibAtrac9), so it
needs neither Sony's `at9tool` nor FFmpeg. Its MP3, FLAC and WAV/AIFF decoders
follow ISO/IEC 11172-3, ISO/IEC 13818-3 and RFC 9639. Loudness is measured to
ITU-R BS.1770-4. The only dependency is [numpy](https://numpy.org/).

## Features

- One command turns an MP3, WAV, AIFF or FLAC file into a finished `snd0.at9`.
- Native ATRAC9 encoder: 48 kHz stereo at 72 to 192 kb/s (192 by default).
- Excerpt selection with `--start` and `--duration`, or `--duration max` for the
  longest excerpt that fits.
- Loudness normalisation to −28 LUFS and a true-peak limiter at −2 dBTP.
- Whole-track loop metadata with short fades, so the loop does not click.
- Refuses to write files above the 2 MiB PS5 limit.
- Writes `loudnessSnd0` into `param.json`, and optionally a WAV preview of the
  decoded result.
- Runs on Windows, Linux and macOS with Python 3.9 or newer. The output is
  byte-identical on every platform.

## How it works

1. **Decode.** The input is decoded: MP3 (MPEG-1/2/2.5 layer III, with gapless
   LAME/Xing trimming), WAV/RF64, AIFF/AIFF-C or FLAC. Only the selected
   excerpt is decoded.
2. **Format.** The excerpt is converted to 48 kHz stereo with a windowed-sinc
   resampler. Mono is duplicated and 5.1 is downmixed.
3. **Loudness.** Integrated loudness is set to −28 LUFS. A smooth look-ahead
   limiter keeps the 4× oversampled true peak at or below −2 dBTP, and 10 ms
   fades are added at both ends.
4. **Encode.** For every 256-sample frame, the encoder computes the exact bit
   cost and noise of each allocation the decoder can derive. A psychoacoustic
   masking model picks the one that sounds best, and each 512-byte superframe
   is shared between its four frames.
5. **Write.** The RIFF container is written with the same `fmt`, `fact`,
   `smpl` and `data` layout as `at9tool -e -br 192 -wholeloop`, and the size is
   checked against the 2,097,152-byte limit.

The format details are in [the ATRAC9 profile](docs/atrac9-profile.md) and the
encoder design in [how the encoder works](docs/encoder.md).

## Requirements

- Python 3.9 or newer on Windows, Linux or macOS.
- numpy 1.20 or newer:

  ```bash
  python -m pip install numpy
  ```

## Download and run

Download `ps5-at9-converter.pyz` and `SHA256SUMS.txt` from the
[latest release](https://github.com/blackbearreloaded/ps5-at9-converter/releases/latest),
verify the checksum, then run:

```bash
python ps5-at9-converter.pyz --input song.mp3 --output snd0.at9
```

From a clone of this repository, `./ps5_at9.py` does the same. On Windows, use
`py` instead of `python`.

```text
Input     song.mp3 (44100 Hz, 2 channels)
Excerpt   0:00.000 - 0:15.000 (15.000 s)
Loudness  -17.3 LUFS -> -28.0 LUFS (gain -10.7 dB), true peak -10.8 dBTP
Encoding  ATRAC9 192 kb/s, 48 kHz stereo
Wrote     snd0.at9 (360,616 bytes, 17.2% of the 2 MiB limit, whole-track loop)
param     set "loudnessSnd0": "-28.00" under "pubtools" in sce_sys/param.json
```

Copy the file to `sce_sys/snd0.at9` in the app image. Then set the loudness
under the existing `pubtools` object of `sce_sys/param.json`, or pass
`--param-json path/to/param.json` to let the tool write it:

```json
{
  "pubtools": {
    "loudnessSnd0": "-28.00"
  }
}
```

Rebuild or redeploy the app. Then fully restart ShadowMountPlus, or the
console, before checking updated presentation assets.

## Options

| Option | Default | Purpose |
| --- | --- | --- |
| `-i`, `--input FILE` | required | Source audio: MP3, WAV, AIFF or FLAC. |
| `-o`, `--output FILE` | `snd0.at9` | Output file. |
| `--start TIME` | `0` | Excerpt start, in seconds or `[hh:]mm:ss`. |
| `--duration TIME` | `15` | Excerpt length, or `max` for the longest excerpt that fits 2 MiB. |
| `--loudness LUFS` | `-28` | Target integrated loudness. |
| `--true-peak DBTP` | `-2` | True-peak ceiling. |
| `--no-normalize` | off | Keep the source level and only limit peaks. |
| `--fade-in SEC`, `--fade-out SEC` | `0.01` | Fade lengths. |
| `--bitrate KBPS` | `192` | 72, 96, 120, 144, 168 or 192. |
| `--no-loop` | off | Omit the whole-track loop metadata. |
| `--param-json PATH` | none | Write the measured loudness to `pubtools.loudnessSnd0`. |
| `--preview WAV` | none | Also write the decoded result, which is what the console plays. |
| `--allow-oversize` | off | Write the file even above the 2 MiB limit. |
| `--workers N` | all CPUs | Number of encoder processes. |
| `-q`, `--quiet` | off | Print errors only. |

```bash
# 30 seconds starting at 1:05, and update param.json
./ps5_at9.py -i song.flac -o sce_sys/snd0.at9 --start 1:05 --duration 30 --param-json sce_sys/param.json

# the longest excerpt that fits, with gentle fades, plus a WAV to listen to
./ps5_at9.py -i song.mp3 --duration max --fade-in 1 --fade-out 3 --preview preview.wav
```

At 192 kb/s the longest excerpt that fits 2 MiB is 87.35 seconds. Lower bit
rates allow longer excerpts, for example 116.47 seconds at 144 kb/s.

## Validation

- **Decoder conformance.** Sony's reference decoder (`at9tool -d`) decodes the
  encoder's output bit-exactly the same as the built-in decoder. The test set
  has 64 files: all six bit rates, silence, very short input, full-scale square
  and sine waves, noise, sweeps, clicks and ten music excerpts. The check also
  showed that Sony's decoder only accepts superframes whose unused tail is
  filled with `0x01` bytes, so the encoder does that.
- **Container.** The header is byte-identical to `at9tool -e -br 192 -wholeloop`
  output for the same sample count.
- **Quality.** Ten 15-second music excerpts were prepared like the manual
  `at9tool` workflow. On them, the objective PEAQ score averages −0.34 ODG,
  against −0.61 for `at9tool` at the same 192 kb/s (EAQUAL, basic model;
  0 is transparent).
- **Input decoders.** MP3 decoding matches FFmpeg to about 120 dB SNR with the
  same gapless length, and FLAC decoding is bit-exact.
- **Speed.** A 15-second excerpt takes about 6 to 15 seconds on a desktop PC,
  depending on the number of CPU cores.

The validation workflow is described in [development and validation](docs/development.md).

## Source layout

| Path | Purpose |
| --- | --- |
| `ps5_at9.py` | Command-line entry point. |
| `ps5at9/cli.py` | Conversion pipeline and options. |
| `ps5at9/atrac9_encoder.py` | ATRAC9 encoder. |
| `ps5at9/atrac9.py` | ATRAC9 definitions and reference decoder, used by `--preview` and the tests. |
| `ps5at9/at9file.py` | `.at9` RIFF reader and writer. |
| `ps5at9/mp3.py`, `flac.py`, `wav.py`, `audio_io.py` | Input decoders and format detection. |
| `ps5at9/dsp.py` | Resampling, loudness, true peak, limiter and fades. |
| `ps5at9/*_tables.py` | Constant tables generated by `tools/gen_*.py`. |
| `tests/` | Unit tests; `tests/data` holds synthetic vectors made by `tools/make_test_vectors.sh`. |
| `docs/` | Format, encoder and validation notes. |

## Development

```bash
python -m pip install numpy
python -m unittest discover -s tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## Releases

Every push and pull request runs the unit tests and a conversion on Windows,
Linux and macOS in GitHub Actions. Version tags matching `v*` publish a GitHub
Release with `ps5-at9-converter.pyz`, a source ZIP and `SHA256SUMS.txt`.

## Limitations

- The output is fixed to the PS5 selection-music profile: ATRAC9, 48 kHz,
  stereo.
- OGG, AAC, M4A and other formats need to be converted to WAV or FLAC first.
- The encoder uses neither intensity stereo nor band extension. At 192 kb/s it
  codes up to 18 kHz, like Sony's encoder at the same rate.
- The output has not yet been tested on PS5 hardware.

<!-- bbr-footer:start -->
<!-- Generated by ps5-homebrew-dev-protocol/scripts/readme-footer. Edit the template there, not here. -->

## Credits

Thanks to John Törnblom (ps5-payload-dev) for the [PS5 Payload SDK](https://github.com/ps5-payload-dev/sdk), which much of the PS5 homebrew scene is built on.
Third-party components, authors and licenses are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

Copyright © 2026 BlackBearReloaded. Licensed under GPL-3.0-or-later; see [LICENSE](LICENSE). Third-party components keep their own licenses. Binary releases are built from the tagged source in this repository.

## Disclaimer

- **No affiliation.** This is an independent homebrew project. It is not
  affiliated with, endorsed by, or sponsored by Sony Interactive Entertainment.
  "PlayStation", "PS5" and related marks are trademarks of Sony Interactive
  Entertainment Inc. ATRAC9 is a trademark of Sony Group Corporation.
- **No proprietary material.** No Sony SDK, firmware, encryption keys or
  decrypted system modules are included.
- **No warranty.** This project is provided "as is", without warranty of any
  kind, to the extent permitted by law. See sections 15 and 16 of the GPL.
- **Legal use only.** Use it only with hardware, accounts and content you own.
  This project does not support or enable piracy.

## AI assistance

This project was developed with AI assistance from OpenAI and/or Anthropic tools.
<!-- bbr-footer:end -->
