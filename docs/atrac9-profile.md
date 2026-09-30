# PS5 snd0.at9 profile

`snd0.at9` is the selection music a PlayStation 5 plays while an app's tile is focused. It is an ATRAC9 stream in a RIFF/WAVE container, placed at `sce_sys/snd0.at9`, and its loudness is declared in `sce_sys/param.json` as `pubtools.loudnessSnd0`.

## Stream configuration

| Field | Value at 192 kb/s |
| --- | --- |
| Sample rate | 48,000 Hz |
| Channels | 2 (one stereo block, channel configuration 2) |
| Average byte rate | 24,000 bytes/s |
| ATRAC9 frame size | 128 bytes on average |
| Frames per superframe | 4 |
| Superframe size | 512 bytes |
| Samples per frame | 256 |
| Samples per superframe | 1,024 |
| ATRAC9 configuration | `FE 74 0F F0` |
| Encoder delay | 256 samples |

Lower bit rates keep the layout and shrink the frames: 168 kb/s is `FE 74 0D F0` (448-byte superframes), 144 kb/s is `FE 74 0B F0` (384), 120 kb/s is `FE 74 09 F0` (320), 96 kb/s is `FE 74 07 F0` (256) and 72 kb/s is `FE 74 05 F0` (192).

The four-byte configuration packs a `0xFE` marker, the sample rate index (7 = 48 kHz), the channel configuration index, a zero validation bit, the average frame size minus one (11 bits) and the superframe index (2 = four frames).

## Container

The file has these chunks, in the same order and with the same field values that Sony's `at9tool -e -br 192 -wholeloop` writes:

| Chunk | Size | Contents |
| --- | --- | --- |
| `fmt ` | 52 | `WAVE_FORMAT_EXTENSIBLE`: 2 channels, 48 kHz, byte rate, block align = superframe size, 0 bits per sample, 1,024 samples per block, speaker mask 3, ATRAC9 sub-format GUID `47E142D2-36BA-4D8D-88FC-61654F8C836C`, version 1, configuration word, reserved 0 |
| `fact` | 12 | Sample count, then 256 and 256 (overlap and encoder delay) |
| `smpl` | 60 | One forward loop from 256 to `sample_count + 255` (both include the encoder delay), infinite play count, sample period 20,833 ns, `cbSamplerData` 24 as in Sony's output |
| `data` | n × 512 | Whole superframes |

`--no-loop` omits the `smpl` chunk.

## Size

The header is 168 bytes with the loop chunk. Audio is rounded up to whole superframes and includes the 256-sample encoder delay:

```text
superframes = ceil((sample_count + 256) / 1024)
file_size   = 168 + superframes * 512            (192 kb/s)
```

The PS5 limit is 2,097,152 bytes. At 192 kb/s the largest excerpt that fits is 4,193,024 samples (87.354666 s), which makes a 2,096,808-byte file. One more sample needs another superframe and gives 2,097,320 bytes. Lower bit rates allow longer excerpts, for example 116.47 s at 144 kb/s.

## Superframe padding

Frames inside a superframe are packed back to back and each ends on a byte boundary. The encoder may give some frames more than the average and others less. Any bytes left after the fourth frame must be `0x01`: Sony's decoder rejects a superframe whose tail is zero or any other value (API error 514, detail 577). FFmpeg and LibAtrac9 do not check this.

## Decoder rules the encoder must respect

Several rules are not visible from the format description alone, and a stream that breaks them may still decode in some decoders:

- The gradient start unit is at least 1 and smaller than the end unit, and the end unit is at most 31. The start and end values differ, because decoders disagree when they are equal.
- Precisions stay at or below 15, so no fine-precision data is needed. Decoders disagree when a precision exceeds 30.
- Scale factors stay within 0..31 without relying on wrap-around.
- Some spectrum symbols have no Huffman code. The most negative value of each word length is never coded. In quantisation units 0–7 at word lengths 2–4, the largest magnitude in each pair must be at least 2^(word length − 2). In units 8–11 at word length 2, a group cannot be all zero. Tight scale factors satisfy this naturally; silent units get a tiny non-zero value.
- Band parameters may be reused only after the first frame of a superframe, and scale factors are predicted from the previous frame only within a superframe. Every superframe can therefore be decoded on its own, which makes looping safe.
