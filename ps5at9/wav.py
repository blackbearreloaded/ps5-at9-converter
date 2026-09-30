# PS5 AT9 Converter - WAV reader and writer.
# Copyright (C) 2026 BlackBearReloaded
# SPDX-License-Identifier: GPL-3.0-or-later

"""Minimal RIFF/WAVE (and RF64) PCM reader and writer."""

import struct

import numpy as np

PCM, IEEE_FLOAT, EXTENSIBLE = 0x0001, 0x0003, 0xFFFE


class WavError(ValueError):
    pass


def read_wav(path_or_bytes):
    """Return (samples[channels, frames] float64 in [-1, 1), sample_rate)."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        raw = path_or_bytes
    else:
        with open(path_or_bytes, "rb") as f:
            raw = f.read()
    if len(raw) < 12 or raw[:4] not in (b"RIFF", b"RF64") or raw[8:12] != b"WAVE":
        raise WavError("not a RIFF/WAVE file")
    pos = 12
    fmt = None
    data = None
    ds64_data_size = None
    while pos + 8 <= len(raw):
        cid = raw[pos:pos + 4]
        size = struct.unpack_from("<I", raw, pos + 4)[0]
        if cid == b"ds64" and size >= 16:
            ds64_data_size = struct.unpack_from("<Q", raw, pos + 16)[0]
        if cid == b"data" and size == 0xFFFFFFFF:
            size = ds64_data_size if ds64_data_size is not None else len(raw) - pos - 8
        body = raw[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = body
        elif cid == b"data":
            data = body
            break
        pos += 8 + size + (size & 1)
    if fmt is None or data is None or len(fmt) < 16:
        raise WavError("missing fmt or data chunk")
    tag, channels, rate, _byte_rate, align, bits = struct.unpack_from("<HHIIHH", fmt, 0)
    if tag == EXTENSIBLE and len(fmt) >= 40:
        tag = struct.unpack_from("<H", fmt, 24)[0]
    if channels < 1 or rate < 1 or bits == 0:
        raise WavError("unsupported WAVE format")
    width = bits // 8 if tag == IEEE_FLOAT else (bits + 7) // 8
    if align < channels * width:
        align = channels * width
    frames = len(data) // align
    buf = np.frombuffer(data[:frames * align], dtype=np.uint8).reshape(frames, align)
    buf = buf[:, :channels * width].reshape(frames, channels, width)
    if tag == IEEE_FLOAT:
        dtype = {4: "<f4", 8: "<f8"}.get(width)
        if dtype is None:
            raise WavError("unsupported float width")
        pcm = np.ascontiguousarray(buf).view(dtype).reshape(frames, channels).astype(np.float64)
    elif tag == PCM:
        if width == 1:
            pcm = (buf[:, :, 0].astype(np.float64) - 128.0) / 128.0
        elif width in (2, 3, 4):
            padded = np.zeros((frames, channels, 4), dtype=np.uint8)
            padded[:, :, 4 - width:] = buf
            pcm = padded.view("<i4").reshape(frames, channels).astype(np.float64) / 2147483648.0
        else:
            raise WavError("unsupported PCM width")
    else:
        raise WavError("unsupported WAVE encoding 0x%04X" % tag)
    return np.ascontiguousarray(pcm.T), rate


def write_wav(path, samples, rate, bits=16):
    """Write float samples[channels, frames] as PCM (16/24 bit) or float (32 bit)."""
    samples = np.atleast_2d(np.asarray(samples, dtype=np.float64))
    channels, frames = samples.shape
    inter = samples.T
    if bits == 32:
        payload = inter.astype("<f4").tobytes()
        tag = IEEE_FLOAT
    elif bits in (16, 24):
        scale = float(1 << (bits - 1))
        ints = np.ascontiguousarray(np.clip(np.round(inter * scale), -scale, scale - 1).astype("<i4"))
        if bits == 16:
            payload = ints.astype("<i2").tobytes()
        else:
            payload = ints.view(np.uint8).reshape(frames, channels, 4)[:, :, :3].tobytes()
        tag = PCM
    else:
        raise ValueError("bits must be 16, 24 or 32")
    width = bits // 8
    fmt = struct.pack("<HHIIHH", tag, channels, rate, rate * channels * width, channels * width, bits)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(payload)) + payload
    if len(payload) & 1:
        body += b"\0"
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", len(body)) + body)
