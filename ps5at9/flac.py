# PS5 AT9 Converter - FLAC decoder.
# Copyright (C) 2026 BlackBearReloaded
# SPDX-License-Identifier: GPL-3.0-or-later

"""FLAC decoder in Python and numpy (RFC 9639).

Supports fixed and variable block sizes, all channel decorrelation modes,
wasted bits, verbatim/constant/fixed/LPC subframes and both Rice coding
methods. Frames before the requested start are skipped without decoding.
"""

import numpy as np


class FlacError(ValueError):
    pass


_BLOCK_SIZES = {1: 192, 2: 576, 3: 1152, 4: 2304, 5: 4608, 8: 256, 9: 512, 10: 1024, 11: 2048,
                12: 4096, 13: 8192, 14: 16384, 15: 32768}
_RATES = {1: 88200, 2: 176400, 3: 192000, 4: 8000, 5: 16000, 6: 22050, 7: 24000, 8: 32000,
          9: 44100, 10: 48000, 11: 96000}
_DEPTHS = {1: 8, 2: 12, 4: 16, 5: 20, 6: 24, 7: 32}


def _crc8(data):
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


class _Reader:
    """MSB-first bit reader over a bytes object with a small integer cache."""

    __slots__ = ("data", "byte", "cache", "bits")

    def __init__(self, data, byte):
        self.data = data
        self.byte = byte
        self.cache = 0
        self.bits = 0

    def _fill(self):
        chunk = self.data[self.byte:self.byte + 8]
        if not chunk:
            raise FlacError("unexpected end of FLAC data")
        self.cache = (self.cache << (8 * len(chunk))) | int.from_bytes(chunk, "big")
        self.bits += 8 * len(chunk)
        self.byte += len(chunk)

    def read(self, n):
        while self.bits < n:
            self._fill()
        self.bits -= n
        value = self.cache >> self.bits
        self.cache &= (1 << self.bits) - 1
        return value

    def read_signed(self, n):
        v = self.read(n)
        return v - (1 << n) if n and v >> (n - 1) else v

    def unary(self):
        count = 0
        while True:
            if self.cache == 0:
                count += self.bits
                self.bits = 0
                self._fill()
                continue
            zeros = self.bits - self.cache.bit_length()
            count += zeros
            self.bits -= zeros + 1
            self.cache &= (1 << self.bits) - 1
            return count

    def align(self):
        self.bits -= self.bits % 8
        self.cache &= (1 << self.bits) - 1

    def tell(self):
        return self.byte - self.bits // 8


def _rice(r, count, param, out):
    read, unary = r.read, r.unary
    for _ in range(count):
        u = (unary() << param) | read(param) if param else unary()
        out.append((u >> 1) ^ -(u & 1))


def _residual(r, block, order):
    method = r.read(2)
    if method > 1:
        raise FlacError("reserved residual coding method")
    pbits = 4 if method == 0 else 5
    escape = (1 << pbits) - 1
    porder = r.read(4)
    parts = 1 << porder
    out = []
    for p in range(parts):
        count = (block >> porder) - (order if p == 0 else 0)
        if count < 0:
            raise FlacError("invalid residual partition")
        param = r.read(pbits)
        if param == escape:
            nbits = r.read(5)
            out.extend(r.read_signed(nbits) if nbits else 0 for _ in range(count))
        else:
            _rice(r, count, param, out)
    return out


_LPC_CACHE = {}


def _lpc_function(order):
    """Build an unrolled integer LPC reconstruction loop for one predictor order."""
    fn = _LPC_CACHE.get(order)
    if fn is None:
        hist = ", ".join("s%d" % i for i in range(1, order + 1))
        coefs = ", ".join("c%d" % i for i in range(1, order + 1))
        dot = " + ".join("c%d * s%d" % (i, i) for i in range(1, order + 1))
        shift = "; ".join("s%d = s%d" % (i, i - 1) for i in range(order, 1, -1))
        src = ("def restore(res, hist, coefs, shift):\n"
               "    %s, = hist\n"
               "    %s, = coefs\n"
               "    out = []\n"
               "    append = out.append\n"
               "    for e in res:\n"
               "        v = e + ((%s) >> shift)\n"
               "        append(v)\n"
               "        %s%ss1 = v\n"
               "    return out\n") % (hist, coefs, dot, shift, "; " if shift else "")
        scope = {}
        exec(compile(src, "<flac-lpc-%d>" % order, "exec"), scope)
        fn = _LPC_CACHE[order] = scope["restore"]
    return fn


def _fixed_restore(warm, res, order):
    s = np.array(warm + res, dtype=np.int64)
    if order == 0:
        return s
    # the residual is the order-th difference; integrate it back from the warm-up samples
    diffs = [s[:order]]
    for _ in range(order - 1):
        diffs.append(np.diff(diffs[-1]))
    level = s[order:]
    for j in range(order - 1, -1, -1):
        level = diffs[j][-1] + np.cumsum(level)
    s[order:] = level
    return s


def _subframe(r, block, bps):
    if r.read(1):
        raise FlacError("invalid subframe padding")
    kind = r.read(6)
    wasted = 0
    if r.read(1):
        wasted = r.unary() + 1
        bps -= wasted
    if kind == 0:
        samples = np.full(block, r.read_signed(bps), dtype=np.int64)
    elif kind == 1:
        samples = np.array([r.read_signed(bps) for _ in range(block)], dtype=np.int64)
    elif 8 <= kind <= 12:
        order = kind - 8
        warm = [r.read_signed(bps) for _ in range(order)]
        samples = _fixed_restore(warm, _residual(r, block, order), order)
    elif kind >= 32:
        order = kind - 31
        warm = [r.read_signed(bps) for _ in range(order)]
        precision = r.read(4) + 1
        if precision == 16:
            raise FlacError("invalid LPC precision")
        shift = r.read_signed(5)
        if shift < 0:
            raise FlacError("negative LPC shift")
        coefs = [r.read_signed(precision) for _ in range(order)]
        res = _residual(r, block, order)
        restored = _lpc_function(order)(res, warm[::-1], coefs, shift)
        samples = np.array(warm + restored, dtype=np.int64)
    else:
        raise FlacError("reserved subframe type")
    if wasted:
        samples <<= wasted
    return samples


def _utf8_number(data, pos):
    first = data[pos]
    if first < 0x80:
        return first, pos + 1
    extra = 0
    mask = 0x40
    while first & mask:
        extra += 1
        mask >>= 1
    if extra == 0 or extra > 6:
        raise FlacError("invalid frame number")
    value = first & (mask - 1)
    for i in range(1, extra + 1):
        b = data[pos + i]
        if b & 0xC0 != 0x80:
            raise FlacError("invalid frame number")
        value = (value << 6) | (b & 0x3F)
    return value, pos + extra + 1


class _FrameHeader:
    __slots__ = ("variable", "block", "rate", "assignment", "channels", "bps", "number", "end")


def _frame_header(data, pos, info):
    """Parse a frame header at pos; return _FrameHeader or None if it is not a valid header."""
    if pos + 6 > len(data) or data[pos] != 0xFF or data[pos + 1] & 0xFE != 0xF8:
        return None
    h = _FrameHeader()
    h.variable = data[pos + 1] & 1
    bs_code, rate_code = data[pos + 2] >> 4, data[pos + 2] & 15
    h.assignment, depth_code = data[pos + 3] >> 4, (data[pos + 3] >> 1) & 7
    if bs_code == 0 or rate_code == 15 or h.assignment > 10 or depth_code == 3 or data[pos + 3] & 1:
        return None
    try:
        h.number, p = _utf8_number(data, pos + 4)
        if bs_code == 6:
            h.block = data[p] + 1
            p += 1
        elif bs_code == 7:
            h.block = int.from_bytes(data[p:p + 2], "big") + 1
            p += 2
        else:
            h.block = _BLOCK_SIZES[bs_code]
        if rate_code == 0:
            h.rate = info["rate"]
        elif rate_code == 12:
            h.rate = data[p] * 1000
            p += 1
        elif rate_code == 13:
            h.rate = int.from_bytes(data[p:p + 2], "big")
            p += 2
        elif rate_code == 14:
            h.rate = int.from_bytes(data[p:p + 2], "big") * 10
            p += 2
        else:
            h.rate = _RATES[rate_code]
        if p >= len(data) or _crc8(data[pos:p]) != data[p]:
            return None
    except (IndexError, KeyError, FlacError):
        return None
    h.bps = info["bps"] if depth_code == 0 else _DEPTHS[depth_code]
    h.channels = h.assignment + 1 if h.assignment < 8 else 2
    h.end = p + 1
    return h


def _first_sample(h, info):
    return h.number if h.variable else h.number * info["block"]


def _next_header(data, pos, info, expected):
    """Find the next frame header after pos whose first sample equals `expected`."""
    while True:
        pos = data.find(b"\xff", pos)
        if pos < 0:
            return -1, None
        h = _frame_header(data, pos, info)
        if h is not None and _first_sample(h, info) == expected:
            return pos, h
        pos += 1


def decode_flac(data, start=0.0, duration=None):
    """Decode FLAC bytes. Returns (pcm[channels, samples] float64 in [-1, 1], sample_rate)."""
    data = bytes(data)
    if data[:3] == b"ID3":
        size = (data[6] << 21) | (data[7] << 14) | (data[8] << 7) | data[9]
        data = data[10 + size:]
    if data[:4] != b"fLaC":
        raise FlacError("not a FLAC stream")
    pos = 4
    info = None
    while True:
        if pos + 4 > len(data):
            raise FlacError("truncated metadata")
        last, kind = data[pos] >> 7, data[pos] & 0x7F
        length = int.from_bytes(data[pos + 1:pos + 4], "big")
        body = data[pos + 4:pos + 4 + length]
        if kind == 0:
            min_block = int.from_bytes(body[0:2], "big")
            max_block = int.from_bytes(body[2:4], "big")
            packed = int.from_bytes(body[10:18], "big")
            info = {
                "rate": packed >> 44,
                "channels": ((packed >> 41) & 7) + 1,
                "bps": ((packed >> 36) & 31) + 1,
                "total": packed & ((1 << 36) - 1),
                "block": max_block if min_block == max_block else 0,
            }
        pos += 4 + length
        if last:
            break
    if info is None or info["rate"] == 0:
        raise FlacError("missing STREAMINFO")
    rate = info["rate"]
    first = int(round(start * rate))
    last_sample = None if duration is None else first + int(round(duration * rate))

    chunks = []
    produced_start = None
    sample = 0
    h = _frame_header(data, pos, info)
    if h is None:
        pos, h = _next_header(data, pos, info, 0)
    while h is not None:
        if last_sample is not None and sample >= last_sample:
            break
        if info["block"] and sample + h.block <= first:
            nxt_pos, nxt = _next_header(data, h.end, info, sample + h.block)
            if nxt is not None:
                sample += h.block
                pos, h = nxt_pos, nxt
                continue
        r = _Reader(data, h.end)
        chans = []
        for ch in range(h.channels):
            extra = 1 if (h.assignment == 8 and ch == 1) or (h.assignment == 9 and ch == 0) or \
                (h.assignment == 10 and ch == 1) else 0
            chans.append(_subframe(r, h.block, h.bps + extra))
        if h.assignment == 8:
            chans[1] = chans[0] - chans[1]
        elif h.assignment == 9:
            chans[0] = chans[0] + chans[1]
        elif h.assignment == 10:
            side = chans[1]
            mid = (chans[0] << 1) | (side & 1)
            chans = [(mid + side) >> 1, (mid - side) >> 1]
        r.align()
        end = r.tell() + 2                      # CRC-16 footer
        if produced_start is None:
            produced_start = sample
        chunks.append(np.stack(chans).astype(np.float64) / float(1 << (h.bps - 1)))
        sample += h.block
        pos = end
        h = _frame_header(data, pos, info)
        if h is None and pos < len(data):
            pos, h = _next_header(data, pos, info, sample)
    if not chunks:
        return np.zeros((info["channels"], 0)), rate
    pcm = np.concatenate(chunks, axis=1)
    a = max(0, first - produced_start)
    b = pcm.shape[1] if last_sample is None else max(a, last_sample - produced_start)
    return pcm[:, a:b], rate
