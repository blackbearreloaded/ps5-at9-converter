# Development and validation

## Local checks

```bash
python -m pip install numpy
python -m unittest discover -s tests
```

The tests cover:

- the codebook rules the encoder relies on;
- the container header, byte for byte against `at9tool` output;
- the 2 MiB size boundary;
- round trips at every bit rate, including silence and tiny input, the `0x01` superframe padding, and identical parallel and serial output;
- the MP3, FLAC, WAV and AIFF decoders, using the synthetic vectors in `tests/data`;
- loudness and true-peak reference levels, resampling and the limiter;
- the full command-line pipeline, including `param.json` updates.

## Regenerating generated files

| File | Command | Source |
| --- | --- | --- |
| `ps5at9/atrac9_tables.py` | `tools/gen_atrac9_tables.py /path/to/LibAtrac9 > ps5at9/atrac9_tables.py` | [LibAtrac9](https://github.com/Thealexbarney/LibAtrac9) source tree |
| `ps5at9/mp3_tables.py` | `tools/gen_mp3_tables.py /path/to/PDMP3/pdmp3.c /path/to/minimp3/minimp3.h > ps5at9/mp3_tables.py` | [PDMP3](https://github.com/technosaurus/PDMP3) and [minimp3](https://github.com/lieff/minimp3) |
| `tests/data/*` | `tools/make_test_vectors.sh` | Synthetic tones, encoded with FFmpeg and LAME |

## Checks against external references

These checks use tools that are not part of the project and are never run by it:

- **Sony decoder conformance.** Decode an encoded file with `at9tool -d -repeat 1 file.at9 out.wav`. Then compare it with the built-in decoder, rounded to 16 bits: `atrac9.Decoder(info.config).decode(info.data)`, skipping the 256-sample delay. They must match exactly. Include low bit rates and synthetic edge cases: silence, very short input, full-scale square waves, noise, sweeps and clicks.
- **Input decoders.** Compare MP3 and FLAC decoding with `ffmpeg -i file -f f32le -`. MP3 should match to about 120 dB SNR with the same gapless length, and FLAC should match exactly.
- **Quality.** Compare with `at9tool -e -br 192` on the same prepared WAV files using a PEAQ implementation such as EAQUAL or GstPEAQ.

When changing the bitstream writer or the container, repeat the Sony decoder conformance check. Record in the pull request whether the result was also checked on PS5 hardware.

## Releases

Pushing a tag such as `v1.0.0` runs the tests on Linux, Windows and macOS. It then publishes:

- a source ZIP with the notices and licenses;
- `ps5-at9-converter.pyz`, a single-file zipapp that runs with `python ps5-at9-converter.pyz ...`;
- SHA-256 checksums.
