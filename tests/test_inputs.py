# PS5 AT9 Converter - Input decoder tests.
# Copyright (C) 2026 BlackBearReloaded
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import struct
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from ps5at9 import audio_io, flac, mp3, wav  # noqa: E402

DATA = os.path.join(HERE, "data")


def read(name):
    with open(os.path.join(DATA, name), "rb") as f:
        return f.read()


def tone(freq, rate, n, amp=0.5):
    return amp * np.sin(2 * np.pi * freq * np.arange(n) / rate)


def correlation(a, b):
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


class Mp3(unittest.TestCase):
    """Vectors from tools/make_test_vectors.sh (LAME, with gapless LAME tags)."""

    def check(self, name, rate, channels, freqs):
        pcm, got_rate = mp3.decode_mp3(read(name))
        self.assertEqual(got_rate, rate)
        self.assertEqual(pcm.shape, (channels, rate))       # one second, trimmed gaplessly
        for ch, parts in enumerate(freqs):
            ideal = sum(tone(f, rate, rate) for f in parts)
            self.assertGreater(correlation(pcm[ch, 2000:-2000], ideal[2000:-2000]), 0.999, (name, ch))

    def test_mpeg1_joint_stereo(self):
        self.check("tone-44k-stereo.mp3", 44100, 2, ((440,), (660,)))

    def test_mpeg2_mono(self):
        self.check("tone-22k-mono.mp3", 22050, 1, ((440, 660),))     # mono downmix of both tones

    def test_mpeg25_mono(self):
        self.check("tone-8k-mono.mp3", 8000, 1, ((440, 660),))

    def test_short_blocks(self):
        pcm, rate = mp3.decode_mp3(read("clicks-44k.mp3"))
        t = np.arange(rate) / rate
        ideal = 0.8 * np.sin(2 * np.pi * 880 * t) * np.exp(-40 * np.mod(t, 0.25))
        self.assertGreater(correlation(pcm[0], ideal), 0.995)

    def test_excerpt(self):
        whole, rate = mp3.decode_mp3(read("tone-44k-stereo.mp3"))
        part, _ = mp3.decode_mp3(read("tone-44k-stereo.mp3"), 0.5, 0.25)
        a = int(0.5 * rate)
        np.testing.assert_allclose(part, whole[:, a:a + part.shape[1]], atol=1e-9)
        self.assertEqual(part.shape[1], int(0.25 * rate))


class Flac(unittest.TestCase):
    def test_decode(self):
        pcm, rate = flac.decode_flac(read("tone-44k.flac"))
        self.assertEqual((rate, pcm.shape), (44100, (2, 44100)))
        self.assertLess(np.max(np.abs(pcm[0] - tone(440, 44100, 44100))), 2.0 / 32768)
        self.assertLess(np.max(np.abs(pcm[1] - tone(660, 44100, 44100))), 2.0 / 32768)

    def test_excerpt_matches_full_decode(self):
        whole, rate = flac.decode_flac(read("tone-44k.flac"))
        part, _ = flac.decode_flac(read("tone-44k.flac"), 0.3, 0.4)
        a = int(round(0.3 * rate))
        np.testing.assert_array_equal(part, whole[:, a:a + part.shape[1]])


class Pcm(unittest.TestCase):
    def test_wav_roundtrip_formats(self):
        src = np.vstack([tone(440, 48000, 4800), tone(880, 48000, 4800, 0.25)])
        with tempfile.TemporaryDirectory() as tmp:
            for bits, tolerance in ((16, 1.0 / 32768), (24, 1.0 / 8388608), (32, 1e-7)):   # 32 = float
                path = os.path.join(tmp, "t%d.wav" % bits)
                wav.write_wav(path, src, 48000, bits)
                back, rate = wav.read_wav(path)
                self.assertEqual(rate, 48000)
                self.assertLess(np.max(np.abs(back - src)), tolerance)

    def test_aiff(self):
        src = (tone(440, 44100, 1000) * 32767).astype(">i2")
        comm = struct.pack(">hIh", 1, 1000, 16) + bytes.fromhex("400EAC44000000000000")
        ssnd = struct.pack(">II", 0, 0) + src.tobytes()
        body = b"AIFF" + b"COMM" + struct.pack(">I", len(comm)) + comm + b"SSND" + struct.pack(">I", len(ssnd)) + ssnd
        pcm, rate = audio_io.read_aiff(b"FORM" + struct.pack(">I", len(body)) + body)
        self.assertEqual((rate, pcm.shape), (44100, (1, 1000)))
        self.assertLess(np.max(np.abs(pcm[0] - src / 32768.0)), 1e-9)

    def test_detection(self):
        self.assertEqual(audio_io.detect(read("tone-44k.flac")), "flac")
        self.assertEqual(audio_io.detect(read("tone-8k-mono.mp3")), "mp3")
        self.assertEqual(audio_io.detect(b"OggS" + b"\0" * 20), "ogg")


if __name__ == "__main__":
    unittest.main()
