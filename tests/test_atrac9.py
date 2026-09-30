# PS5 AT9 Converter - ATRAC9 codec and container tests.
# Copyright (C) 2026 BlackBearReloaded
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ps5at9 import at9file, atrac9, atrac9_encoder  # noqa: E402

# First 168 bytes of a 1-second, 48,000-sample file written by Sony's at9tool with
# "-e -br 192 -wholeloop": RIFF, fmt (WAVE_FORMAT_EXTENSIBLE + ATRAC9), fact, smpl, data.
AT9TOOL_HEADER = bytes.fromhex(
    "52494646 a0600000 57415645 666d7420"
    "34000000 feff0200 80bb0000 c05d0000"
    "00020000 22000004 03000000 d242e147"
    "ba368d4d 88fc6165 4f8c836c 01000000"
    "fe740ff0 00000000 66616374 0c000000"
    "80bb0000 00010000 00010000 736d706c"
    "3c000000 00000000 00000000 61510000"
    "3c000000 00000000 00000000 00000000"
    "01000000 18000000 00000000 00000000"
    "00010000 7fbc0000 00000000 00000000"
    "64617461 00600000")


def sine(freq, seconds, amp=0.5, rate=48000):
    t = np.arange(int(seconds * rate)) / rate
    return np.vstack([amp * np.sin(2 * np.pi * freq * t), amp * np.cos(2 * np.pi * freq * t)])


def roundtrip(pcm, bitrate=192, workers=1):
    enc = atrac9_encoder.Encoder(2, bitrate)
    data = enc.encode(pcm * 32768.0, workers=workers)
    blob = at9file.build_at9(enc.config, 2, pcm.shape[1], data)
    info = at9file.read_at9(blob)
    out = atrac9.Decoder(info.config).decode(info.data)
    return blob, out[:, 256:256 + pcm.shape[1]] / 32768.0


def snr(ref, test):
    return 10 * np.log10(np.sum(ref ** 2) / np.sum((ref - test) ** 2))


class CodebookRules(unittest.TestCase):
    """The encoder relies on these rules for which spectrum symbols are codable."""

    def test_valid_symbols(self):
        for (s, wl, cbi), book in atrac9.SPECTRUM.items():
            for sym, length in enumerate(book.bits):
                vals = [atrac9.sign_extend((sym >> (j * book.value_bits)) & (book.value_max - 1),
                                           book.value_bits) for j in range(book.value_count)]
                valid = min(vals) > -(1 << (book.value_bits - 1))
                if cbi == 0 and wl <= 4:
                    valid = valid and max(abs(v) for v in vals) >= 1 << (wl - 2)
                if cbi == 1 and wl == 2:
                    valid = valid and any(vals)
                self.assertEqual(valid, length != 0, (s, wl, cbi, vals))


class Container(unittest.TestCase):
    def test_config_word(self):
        enc = atrac9_encoder.Encoder(2, 192)
        self.assertEqual(enc.config.data.hex(), "fe740ff0")
        self.assertEqual(atrac9_encoder.Encoder(2, 144).config.data.hex(), "fe740bf0")

    def test_header_matches_at9tool(self):
        cfg = atrac9_encoder.Encoder(2, 192).config
        blob = at9file.build_at9(cfg, 2, 48000, b"\x01" * (48 * 512), loop=True)
        self.assertEqual(blob[:168], AT9TOOL_HEADER)
        self.assertEqual(len(blob), at9file.file_size(48000))

    def test_size_limit_boundary(self):
        self.assertEqual(at9file.file_size(4193024), 2096808)
        self.assertEqual(at9file.file_size(4193025), 2097320)


class RoundTrip(unittest.TestCase):
    def test_sine_quality(self):
        pcm = sine(1000, 1.0)
        blob, out = roundtrip(pcm)
        self.assertEqual(out.shape, pcm.shape)
        self.assertGreater(snr(pcm, out), 45.0)

    def test_every_bitrate_decodes(self):
        rng = np.random.default_rng(1)
        pcm = np.clip(rng.standard_normal((2, 24000)) * 0.2, -1, 1)
        for rate in atrac9_encoder.STEREO_BITRATES:
            blob, out = roundtrip(pcm, rate)
            self.assertEqual(len(blob), at9file.file_size(pcm.shape[1], atrac9_encoder.frame_bytes_for(rate) * 4))
            self.assertGreater(snr(pcm, out), 0.5, rate)

    def test_silence_and_tiny_input(self):
        _blob, out = roundtrip(np.zeros((2, 5000)))
        self.assertLess(np.max(np.abs(out)), 1e-3)
        _blob, out = roundtrip(np.full((2, 7), 0.25))
        self.assertEqual(out.shape, (2, 7))

    def test_superframe_padding_is_0x01(self):
        blob, _out = roundtrip(sine(440, 0.5, 0.01))
        info = at9file.read_at9(blob)
        trace = []
        atrac9.Decoder(info.config, trace=trace).decode(info.data)
        size = info.config.superframe_bytes
        for s in range(len(info.data) // size):
            end = (trace[s * 4 + 3]["bits_end"] + 7) // 8
            tail = info.data[s * size + end:(s + 1) * size]
            self.assertEqual(tail, b"\x01" * len(tail))

    def test_parallel_output_is_identical(self):
        pcm = sine(3000, 0.6, 0.3) + sine(200, 0.6, 0.3)
        enc = atrac9_encoder.Encoder(2, 192)
        one = enc.encode(pcm * 32768.0, workers=1)
        two = enc.encode(pcm * 32768.0, workers=2)
        self.assertEqual(one, two)


if __name__ == "__main__":
    unittest.main()
