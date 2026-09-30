# PS5 AT9 Converter - Signal processing tests.
# Copyright (C) 2026 BlackBearReloaded
# SPDX-License-Identifier: GPL-3.0-or-later

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ps5at9 import dsp  # noqa: E402


class Loudness(unittest.TestCase):
    def test_sine_reference_levels(self):
        # BS.1770: a 997 Hz sine at full scale in one channel reads -3.01 LKFS
        t = np.arange(48000 * 5) / 48000.0
        one = np.vstack([np.sin(2 * np.pi * 997 * t), np.zeros_like(t)])
        self.assertAlmostEqual(dsp.integrated_loudness(one), -3.01, delta=0.05)
        both = np.vstack([np.sin(2 * np.pi * 997 * t)] * 2) * 0.1
        self.assertAlmostEqual(dsp.integrated_loudness(both), -20.0, delta=0.05)

    def test_silence(self):
        self.assertEqual(dsp.integrated_loudness(np.zeros((2, 48000))), -math.inf)


class Resampling(unittest.TestCase):
    def test_44100_to_48000(self):
        t = np.arange(44100) / 44100.0
        out = dsp.resample(np.vstack([np.sin(2 * np.pi * 1000 * t)] * 2), 44100, 48000)
        self.assertEqual(out.shape, (2, 48000))
        ideal = np.sin(2 * np.pi * 1000 * np.arange(48000) / 48000.0)
        self.assertLess(np.max(np.abs(out[0, 200:-200] - ideal[200:-200])), 1e-4)

    def test_image_rejection(self):
        t = np.arange(96000) / 96000.0
        out = dsp.resample(np.vstack([np.sin(2 * np.pi * 30000 * t)]), 96000, 48000)
        self.assertLess(np.max(np.abs(out[0, 500:-500])), 1e-4)       # above the new Nyquist


class Peaks(unittest.TestCase):
    def test_intersample_peak(self):
        n = np.arange(48000)
        x = np.sin(2 * np.pi * 12000 * n / 48000 + np.pi / 4)[None, :]  # samples at +-0.707
        self.assertAlmostEqual(np.max(np.abs(x)), math.sqrt(0.5), places=6)
        self.assertAlmostEqual(dsp.true_peak_db(x), 0.0, delta=0.2)

    def test_limiter_meets_ceiling(self):
        rng = np.random.default_rng(3)
        x = np.clip(rng.standard_normal((2, 48000)) * 0.3, -1, 1)
        x[:, 24000] = 0.999
        y, limited = dsp.limit_true_peak(x, -2.0)
        self.assertTrue(limited)
        self.assertLessEqual(dsp.true_peak_db(y), -2.0)
        z, limited = dsp.limit_true_peak(x * 0.1, -2.0)
        self.assertFalse(limited)


class Misc(unittest.TestCase):
    def test_channel_mapping(self):
        mono = np.ones((1, 10))
        self.assertEqual(dsp.to_stereo(mono).shape, (2, 10))
        surround = np.ones((6, 10))
        self.assertTrue(np.allclose(dsp.to_stereo(surround), 1.0))

    def test_fades(self):
        y = dsp.apply_fades(np.ones((2, 48000)), 48000, 0.1, 0.1)
        self.assertLess(y[0, 0], 0.01)
        self.assertLess(y[0, -1], 0.01)
        self.assertEqual(y[0, 24000], 1.0)


if __name__ == "__main__":
    unittest.main()
