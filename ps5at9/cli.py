# PS5 AT9 Converter - Command-line conversion pipeline.
# Copyright (C) 2026 BlackBearReloaded
# SPDX-License-Identifier: GPL-3.0-or-later

"""Command-line interface: prepare audio and write a PS5 snd0.at9 asset."""

import argparse
import json
import math
import os
import sys
import tempfile
import time

import numpy as np

from . import __version__, at9file, atrac9, audio_io, dsp
from .atrac9_encoder import Encoder, STEREO_BITRATES, frame_bytes_for

PS5_SIZE_LIMIT = 2 * 1024 * 1024
RATE = 48000
SUPERFRAME_SAMPLES = 1024


def parse_time(text):
    """Seconds from "90", "90.5", "1:30" or "1:02:03.5"."""
    parts = str(text).strip().split(":")
    if not 1 <= len(parts) <= 3:
        raise argparse.ArgumentTypeError("invalid time %r" % text)
    try:
        value = 0.0
        for part in parts:
            value = value * 60.0 + float(part)
    except ValueError:
        raise argparse.ArgumentTypeError("invalid time %r" % text)
    if value < 0 or not math.isfinite(value):
        raise argparse.ArgumentTypeError("time must be positive")
    return value


def parse_duration(text):
    if str(text).strip().lower() == "max":
        return "max"
    value = parse_time(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("duration must be greater than zero")
    return value


def format_time(seconds):
    minutes, sec = divmod(seconds, 60.0)
    return "%d:%06.3f" % (int(minutes), sec)


def max_samples_for(bitrate, loop):
    superframe_bytes = frame_bytes_for(bitrate) * 4
    superframes = (PS5_SIZE_LIMIT - at9file.header_size(loop)) // superframe_bytes
    return superframes * SUPERFRAME_SAMPLES - at9file.ENCODER_DELAY


def build_parser():
    p = argparse.ArgumentParser(
        prog="ps5_at9.py",
        description="Turn an audio file into a PlayStation 5 snd0.at9 selection-music asset: "
                    "excerpt, 48 kHz stereo, loudness normalisation, true-peak limiting and "
                    "ATRAC9 encoding with a whole-track loop. No external tools are needed.")
    p.add_argument("-i", "--input", required=True, help="source audio: MP3, WAV, AIFF or FLAC")
    p.add_argument("-o", "--output", default="snd0.at9", help="output file (default: snd0.at9)")
    p.add_argument("--start", type=parse_time, default=0.0, metavar="TIME",
                   help="excerpt start in seconds or [hh:]mm:ss (default: 0)")
    p.add_argument("--duration", type=parse_duration, default=15.0, metavar="TIME",
                   help='excerpt length in seconds or [hh:]mm:ss, or "max" for the longest '
                        "excerpt that fits the 2 MiB PS5 limit (default: 15)")
    p.add_argument("--loudness", type=float, default=-28.0, metavar="LUFS",
                   help="target integrated loudness (default: -28, matching loudnessSnd0)")
    p.add_argument("--true-peak", type=float, default=-2.0, metavar="DBTP",
                   help="true-peak ceiling (default: -2)")
    p.add_argument("--no-normalize", action="store_true", help="keep the source level; only limit peaks")
    p.add_argument("--fade-in", type=float, default=0.01, metavar="SEC",
                   help="fade-in length (default: 0.01, avoids a click at the loop point)")
    p.add_argument("--fade-out", type=float, default=0.01, metavar="SEC",
                   help="fade-out length (default: 0.01)")
    p.add_argument("--bitrate", type=int, default=192, choices=STEREO_BITRATES, metavar="KBPS",
                   help="ATRAC9 bit rate: %s (default: 192)" % ", ".join(map(str, STEREO_BITRATES)))
    p.add_argument("--no-loop", action="store_true", help="omit the whole-track loop metadata")
    p.add_argument("--param-json", metavar="PATH",
                   help="also write the measured loudness to pubtools.loudnessSnd0 in this param.json")
    p.add_argument("--preview", metavar="WAV", help="also write the decoded result (what the PS5 plays) as WAV")
    p.add_argument("--allow-oversize", action="store_true",
                   help="write the file even if it exceeds the 2 MiB PS5 limit")
    p.add_argument("--workers", type=int, default=None, metavar="N",
                   help="encoder processes (default: one per CPU)")
    p.add_argument("-q", "--quiet", action="store_true", help="only print errors")
    p.add_argument("--version", action="version", version="%(prog)s " + __version__)
    return p


class _Log:
    def __init__(self, quiet):
        self.quiet = quiet
        self.tty = sys.stderr.isatty()

    def __call__(self, text=""):
        if not self.quiet:
            print(text, file=sys.stderr, flush=True)

    def progress(self, done, total):
        if self.quiet or not self.tty:
            return
        width = 30
        filled = int(width * done / total)
        sys.stderr.write("\r  encoding [%s%s] %3d%%" % ("#" * filled, "." * (width - filled), 100 * done // total))
        if done == total:
            sys.stderr.write("\n")
        sys.stderr.flush()


def _write_atomic(path, blob):
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".snd0-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(blob)
        try:
            mode = os.stat(path).st_mode & 0o777
        except FileNotFoundError:
            umask = os.umask(0)
            os.umask(umask)
            mode = 0o666 & ~umask
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def update_param_json(path, loudness):
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("param.json does not contain a JSON object")
    pubtools = data.setdefault("pubtools", {})
    if not isinstance(pubtools, dict):
        raise ValueError('param.json "pubtools" is not an object')
    pubtools["loudnessSnd0"] = "%.2f" % loudness
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    _write_atomic(path, text.encode("utf-8"))


def run(args, log):
    if not os.path.isfile(args.input):
        raise SystemExit("error: input file not found: %s" % args.input)
    for path in (args.output, args.preview):
        if path and (os.path.isdir(path) or not os.path.isdir(os.path.dirname(os.path.abspath(path)))):
            raise SystemExit("error: cannot write %s (missing folder, or a folder of that name exists)" % path)
    loop = not args.no_loop
    max_samples = max_samples_for(args.bitrate, loop)
    started = time.time()

    decode_duration = None if args.duration == "max" else args.duration
    pcm, rate, offset = audio_io.load(args.input, args.start, decode_duration)
    if pcm.shape[1] == 0:
        raise SystemExit("error: --start is beyond the end of the input")
    log("Input     %s (%d Hz, %d channel%s)" % (os.path.basename(args.input), rate, pcm.shape[0],
                                              "" if pcm.shape[0] == 1 else "s"))
    pcm = dsp.resample(dsp.to_stereo(pcm), rate, RATE)
    skip = int(round(offset * RATE))
    available = max(0, pcm.shape[1] - skip)
    if args.duration == "max":
        count = min(available, max_samples)
    else:
        count = min(available, int(round(args.duration * RATE)))
        if count < int(round(args.duration * RATE)):
            log("Note      the input ends %.3f s after --start; using %.3f s" % (available / RATE, available / RATE))
    if count <= 0:
        raise SystemExit("error: the selected excerpt is empty")
    pcm = np.ascontiguousarray(pcm[:, skip:skip + count])
    log("Excerpt   %s - %s (%.3f s)" % (format_time(args.start), format_time(args.start + count / RATE), count / RATE))

    size = at9file.file_size(count, frame_bytes_for(args.bitrate) * 4, SUPERFRAME_SAMPLES, loop)
    if size > PS5_SIZE_LIMIT and not args.allow_oversize:
        raise SystemExit("error: %.3f s at %d kb/s makes a %d-byte file, above the 2 MiB PS5 limit; "
                         "use at most %.3f s (--duration max) or a lower --bitrate"
                         % (count / RATE, args.bitrate, size, max_samples / RATE))

    source_lufs = dsp.integrated_loudness(pcm)
    gain_db = 0.0
    if not args.no_normalize:
        if math.isfinite(source_lufs):
            gain_db = args.loudness - source_lufs
            pcm = pcm * 10.0 ** (gain_db / 20.0)
        else:
            log("Note      the excerpt is silent; loudness normalisation skipped")
    pcm, limited = dsp.limit_true_peak(pcm, args.true_peak, RATE)
    pcm = dsp.apply_fades(pcm, RATE, args.fade_in, args.fade_out)
    final_lufs = dsp.integrated_loudness(pcm)
    peak_db = dsp.true_peak_db(pcm)
    log("Loudness  %s LUFS -> %s LUFS (gain %+.1f dB), true peak %.1f dBTP%s"
        % (_fmt_db(source_lufs), _fmt_db(final_lufs), gain_db, peak_db,
           ", peaks limited" if limited else ""))

    encoder = Encoder(channels=2, bitrate=args.bitrate)
    log("Encoding  ATRAC9 %d kb/s, 48 kHz stereo" % args.bitrate)
    data = encoder.encode(pcm * 32768.0, progress=log.progress, workers=args.workers)
    blob = at9file.build_at9(encoder.config, 2, count, data, loop=loop)
    if len(blob) != size:
        raise RuntimeError("internal error: unexpected file size")
    _write_atomic(args.output, blob)
    log("Wrote     %s (%s bytes, %.1f%% of the 2 MiB limit%s)"
        % (args.output, format(len(blob), ","), 100.0 * len(blob) / PS5_SIZE_LIMIT,
           ", whole-track loop" if loop else ""))

    if args.preview:
        from . import wav
        info = at9file.read_at9(blob)
        decoded = atrac9.Decoder(info.config).decode(info.data)
        decoded = np.clip(np.floor(decoded[:, at9file.ENCODER_DELAY:at9file.ENCODER_DELAY + count] + 0.5),
                          -32768, 32767) / 32768.0
        wav.write_wav(args.preview, decoded, RATE)
        err = decoded - pcm
        snr = 10.0 * math.log10(np.sum(pcm ** 2) / max(np.sum(err ** 2), 1e-20))
        log("Preview   %s (decoded output, SNR %.1f dB)" % (args.preview, snr))

    if args.param_json:
        update_param_json(args.param_json, final_lufs)
        log('param     %s: pubtools.loudnessSnd0 = "%.2f"' % (args.param_json, final_lufs))
    else:
        log('param     set "loudnessSnd0": "%.2f" under "pubtools" in sce_sys/param.json' % final_lufs)
    log("Done      %.1f s" % (time.time() - started))
    return 0


def _fmt_db(value):
    return "%.1f" % value if math.isfinite(value) else "-inf"


def main(argv=None):
    args = build_parser().parse_args(argv)
    log = _Log(args.quiet)
    if args.workers is not None and args.workers < 1:
        raise SystemExit("error: --workers must be at least 1")
    try:
        return run(args, log)
    except audio_io.InputError as exc:
        raise SystemExit("error: %s" % exc)
    except KeyboardInterrupt:
        raise SystemExit("interrupted")
