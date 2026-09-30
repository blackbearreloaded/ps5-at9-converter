# PS5 AT9 Converter - Input format detection and loading.
# Copyright (C) 2026 BlackBearReloaded
# SPDX-License-Identifier: GPL-3.0-or-later

"""Input decoding: format detection and PCM loading."""

import math
import struct

import numpy as np

from . import mp3, wav


class InputError(ValueError):
    pass


def _extended_to_float(b):
    """IEEE 754 80-bit extended (AIFF sample rate) to float."""
    exponent = ((b[0] & 0x7F) << 8) | b[1]
    mantissa = int.from_bytes(b[2:10], "big")
    if exponent == 0 and mantissa == 0:
        return 0.0
    sign = -1.0 if b[0] & 0x80 else 1.0
    return sign * mantissa * 2.0 ** (exponent - 16383 - 63)


def read_aiff(data):
    if data[:4] != b"FORM" or data[8:12] not in (b"AIFF", b"AIFC"):
        raise InputError("not an AIFF file")
    aifc = data[8:12] == b"AIFC"
    pos = 12
    comm = ssnd = None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack(">I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"COMM":
            comm = body
        elif cid == b"SSND":
            offset = struct.unpack(">I", body[:4])[0]
            ssnd = body[8 + offset:]
        pos += 8 + size + (size & 1)
    if comm is None or ssnd is None:
        raise InputError("AIFF file is missing COMM or SSND")
    channels, frames, bits = struct.unpack(">hIh", comm[:8])
    rate = _extended_to_float(comm[8:18])
    kind = comm[18:22] if aifc else b"NONE"
    width = (bits + 7) // 8
    if kind in (b"NONE", b"twos", b"sowt"):
        raw = np.frombuffer(ssnd[:frames * channels * width], dtype=np.uint8).reshape(-1, channels, width)
        if kind == b"sowt":
            raw = raw[:, :, ::-1]
        padded = np.zeros((raw.shape[0], channels, 4), dtype=np.uint8)
        padded[:, :, :width] = raw
        ints = padded.view(">i4").reshape(-1, channels)
        pcm = ints.astype(np.float64) / 2147483648.0
    elif kind in (b"fl32", b"FL32"):
        pcm = np.frombuffer(ssnd[:frames * channels * 4], dtype=">f4").reshape(-1, channels).astype(np.float64)
    elif kind in (b"fl64", b"FL64"):
        pcm = np.frombuffer(ssnd[:frames * channels * 8], dtype=">f8").reshape(-1, channels).astype(np.float64)
    else:
        raise InputError("unsupported AIFF-C compression %r" % kind.decode("latin-1"))
    return np.ascontiguousarray(pcm.T), int(round(rate))


def detect(data):
    head = data[:12]
    if head[:4] in (b"RIFF", b"RF64") and head[8:12] == b"WAVE":
        return "wav"
    if head[:4] == b"FORM" and head[8:12] in (b"AIFF", b"AIFC"):
        return "aiff"
    if head[:4] == b"fLaC":
        return "flac"
    if head[:4] == b"OggS":
        return "ogg"
    if head[:3] == b"ID3" or (len(head) > 1 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return "mp3"
    if head[4:8] == b"ftyp":
        return "mp4"
    # MP3 without ID3 may start with junk; let the MP3 parser search for frames
    return "mp3"


def load(path, start=0.0, duration=None, margin=0.25):
    """Return (pcm[channels, samples] in [-1, 1], rate, offset).

    The returned audio starts `offset` seconds before `start` (at most `margin`)
    so that resampling can be trimmed cleanly afterwards.
    """
    with open(path, "rb") as f:
        data = f.read()
    kind = detect(data)
    begin = max(0.0, start - margin)
    offset = start - begin
    length = None if duration is None else duration + offset + margin
    if kind == "mp3":
        try:
            pcm, rate = mp3.decode_mp3(data, begin, length)
        except mp3.Mp3Error as exc:
            raise InputError("could not decode MP3: %s" % exc)
        return pcm, rate, offset
    if kind == "wav":
        try:
            pcm, rate = wav.read_wav(data)
        except wav.WavError as exc:
            raise InputError("could not read WAV: %s" % exc)
    elif kind == "aiff":
        pcm, rate = read_aiff(data)
    elif kind == "flac":
        from . import flac
        try:
            pcm, rate = flac.decode_flac(data, begin, length)
        except flac.FlacError as exc:
            raise InputError("could not decode FLAC: %s" % exc)
        return pcm, rate, offset
    else:
        raise InputError("unsupported input format (%s); use MP3, WAV, AIFF or FLAC" % kind.upper())
    a = int(math.floor(begin * rate))
    b = pcm.shape[1] if length is None else min(pcm.shape[1], a + int(math.ceil(length * rate)))
    return pcm[:, a:b], rate, offset
