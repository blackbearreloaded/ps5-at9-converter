# PS5 AT9 Converter - ATRAC9 RIFF (.at9) reader and writer.
# Copyright (C) 2026 BlackBearReloaded
# SPDX-License-Identifier: GPL-3.0-or-later

"""RIFF/WAVE container for ATRAC9 (.at9), laid out like Sony's at9tool output."""

import struct

from .atrac9 import ATRAC9_GUID, Config, FormatError

ENCODER_DELAY = 256
SPEAKER_MASKS = {1: 0x4, 2: 0x3}


class At9Info:
    def __init__(self):
        self.config = None
        self.channels = 0
        self.sample_rate = 0
        self.byte_rate = 0
        self.block_align = 0
        self.samples_per_block = 0
        self.version = 0
        self.sample_count = 0
        self.overlap_delay = 0
        self.encoder_delay = 0
        self.loops = []
        self.data = b""


def read_at9(path_or_bytes):
    if isinstance(path_or_bytes, (bytes, bytearray)):
        raw = path_or_bytes
    else:
        with open(path_or_bytes, "rb") as f:
            raw = f.read()
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise FormatError("not a RIFF/WAVE file")
    info = At9Info()
    pos = 12
    have_fmt = False
    while pos + 8 <= len(raw):
        cid, size = raw[pos:pos + 4], struct.unpack_from("<I", raw, pos + 4)[0]
        body = raw[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            tag, info.channels, info.sample_rate, info.byte_rate, info.block_align, _bits = \
                struct.unpack_from("<HHIIHH", body, 0)
            if tag != 0xFFFE or len(body) < 52 or body[24:40] != ATRAC9_GUID:
                raise FormatError("fmt chunk is not ATRAC9 WAVE_FORMAT_EXTENSIBLE")
            info.samples_per_block = struct.unpack_from("<H", body, 18)[0]
            info.version = struct.unpack_from("<I", body, 40)[0]
            info.config = Config(body[44:48])
            have_fmt = True
        elif cid == b"fact":
            info.sample_count = struct.unpack_from("<I", body, 0)[0]
            if size >= 12:
                info.overlap_delay, info.encoder_delay = struct.unpack_from("<II", body, 4)
        elif cid == b"smpl" and size >= 36:
            count = struct.unpack_from("<I", body, 28)[0]
            for i in range(count):
                off = 36 + 24 * i
                if off + 24 <= len(body):
                    _cue, kind, start, end, _frac, plays = struct.unpack_from("<6I", body, off)
                    info.loops.append((kind, start, end, plays))
        elif cid == b"data":
            info.data = bytes(body)
        pos += 8 + size + (size & 1)
    if not have_fmt:
        raise FormatError("missing fmt chunk")
    return info


def build_at9(config, channels, sample_count, data, loop=True, encoder_delay=ENCODER_DELAY):
    """Return the complete .at9 file bytes.

    `loop=True` writes the whole-track loop that at9tool's -wholeloop produces:
    start = encoder delay, end = sample_count + encoder delay - 1.
    """
    if len(data) % config.superframe_bytes:
        raise ValueError("data must contain whole superframes")
    byte_rate = config.superframe_bytes * config.sample_rate // config.superframe_samples
    fmt = struct.pack("<HHIIHHHHI", 0xFFFE, channels, config.sample_rate, byte_rate,
                      config.superframe_bytes, 0, 34, config.superframe_samples,
                      SPEAKER_MASKS.get(channels, 0))
    fmt += ATRAC9_GUID + struct.pack("<I", 1) + config.data + struct.pack("<I", 0)
    chunks = [(b"fmt ", fmt), (b"fact", struct.pack("<III", sample_count, encoder_delay, encoder_delay))]
    if loop:
        period = round(1e9 / config.sample_rate)
        smpl = struct.pack("<9I", 0, 0, period, 60, 0, 0, 0, 1, 24)
        smpl += struct.pack("<6I", 0, 0, encoder_delay, sample_count + encoder_delay - 1, 0, 0)
        chunks.append((b"smpl", smpl))
    chunks.append((b"data", bytes(data)))
    body = b"WAVE" + b"".join(cid + struct.pack("<I", len(payload)) + payload for cid, payload in chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def header_size(loop=True):
    return 12 + 60 + 20 + (68 if loop else 0) + 8


def file_size(sample_count, superframe_bytes=512, superframe_samples=1024, loop=True,
              encoder_delay=ENCODER_DELAY):
    superframes = -(-(sample_count + encoder_delay) // superframe_samples)
    return header_size(loop) + superframes * superframe_bytes
