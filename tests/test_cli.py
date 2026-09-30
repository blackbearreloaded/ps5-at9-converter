# PS5 AT9 Converter - Command-line pipeline tests.
# Copyright (C) 2026 BlackBearReloaded
# SPDX-License-Identifier: GPL-3.0-or-later

import json
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from ps5at9 import at9file, cli, dsp, wav  # noqa: E402


class Cli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        t = np.arange(44100 * 4) / 44100.0
        music = 0.6 * np.vstack([np.sin(2 * np.pi * 330 * t), np.sin(2 * np.pi * 495 * t)]) * (0.6 + 0.4 * np.sin(2 * np.pi * t))
        self.src = os.path.join(self.dir, "in.wav")
        wav.write_wav(self.src, music, 44100)

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *args):
        return cli.main(["--quiet", "--workers", "1", "--input", self.src] + list(args))

    def test_full_pipeline(self):
        out = os.path.join(self.dir, "snd0.at9")
        param = os.path.join(self.dir, "param.json")
        preview = os.path.join(self.dir, "preview.wav")
        with open(param, "w") as f:
            json.dump({"titleId": "PPSA00000", "pubtools": {"creationDate": "2026-01-01"}}, f)
        self.assertEqual(self.run_cli("--output", out, "--start", "0:00.5", "--duration", "2",
                                      "--param-json", param, "--preview", preview), 0)
        info = at9file.read_at9(out)
        self.assertEqual(info.sample_count, 96000)
        self.assertEqual(info.loops, [(0, 256, 96255, 0)])
        self.assertEqual(os.path.getsize(out), at9file.file_size(96000))
        with open(param) as f:
            data = json.load(f)
        self.assertEqual(data["titleId"], "PPSA00000")
        self.assertAlmostEqual(float(data["pubtools"]["loudnessSnd0"]), -28.0, delta=0.2)
        decoded, rate = wav.read_wav(preview)
        self.assertEqual((rate, decoded.shape), (48000, (2, 96000)))
        self.assertAlmostEqual(dsp.integrated_loudness(decoded), -28.0, delta=0.3)

    def test_no_loop_and_bitrate(self):
        out = os.path.join(self.dir, "a.at9")
        self.assertEqual(self.run_cli("--output", out, "--duration", "1", "--no-loop", "--bitrate", "144"), 0)
        info = at9file.read_at9(out)
        self.assertEqual(info.loops, [])
        self.assertEqual(info.config.data.hex(), "fe740bf0")

    def test_short_input_is_used_whole(self):
        out = os.path.join(self.dir, "b.at9")
        self.assertEqual(self.run_cli("--output", out, "--duration", "max"), 0)
        self.assertEqual(at9file.read_at9(out).sample_count, 4 * 48000)

    def test_oversize_is_refused(self):
        self.assertEqual(cli.max_samples_for(192, True), 4193024)
        long = os.path.join(self.dir, "long.wav")
        wav.write_wav(long, np.zeros((2, 48000 * 90)), 48000)
        with self.assertRaises(SystemExit):
            cli.main(["--quiet", "--input", long, "--output", os.path.join(self.dir, "c.at9"), "--duration", "90"])

    def test_time_parsing(self):
        self.assertEqual(cli.parse_time("1:30"), 90.0)
        self.assertEqual(cli.parse_time("1:02:03.5"), 3723.5)
        self.assertEqual(cli.parse_duration("max"), "max")


if __name__ == "__main__":
    unittest.main()
