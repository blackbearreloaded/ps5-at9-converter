# PS5 AT9 Converter - MP3 decoder.
# Copyright (C) 2026 BlackBearReloaded
# SPDX-License-Identifier: GPL-3.0-or-later

"""MPEG-1/2/2.5 audio layer III (MP3) decoder in Python and numpy.

Follows ISO/IEC 11172-3 and 13818-3: bit reservoir, MPEG-1 and LSF scale
factors, M/S and intensity stereo, long/short/mixed blocks, alias reduction,
hybrid IMDCT and the polyphase synthesis filterbank (batched with numpy).
LAME/Xing gapless information is honoured when present.
"""

import math

import numpy as np

from . import mp3_tables as T


class Mp3Error(ValueError):
    pass


_BITRATES = {1: [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
             2: [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160]}
_RATES = {3: [44100, 48000, 32000], 2: [22050, 24000, 16000], 0: [11025, 12000, 8000]}
_SLEN = [(0, 0), (0, 1), (0, 2), (0, 3), (3, 0), (1, 1), (1, 2), (1, 3),
         (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3), (4, 2), (4, 3)]
_PRETAB = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 2, 0], dtype=np.int64)
_LSF_NSF = [
    [[6, 5, 5, 5], [9, 9, 9, 9], [6, 9, 9, 9]],
    [[6, 5, 7, 3], [9, 9, 12, 6], [6, 9, 12, 6]],
    [[11, 10, 0, 0], [18, 18, 0, 0], [15, 18, 0, 0]],
    [[7, 7, 7, 0], [12, 12, 12, 0], [6, 15, 12, 0]],
    [[6, 6, 6, 3], [12, 9, 9, 6], [6, 12, 9, 6]],
    [[8, 8, 5, 0], [15, 12, 9, 0], [6, 18, 9, 0]],
]
_POW43 = np.arange(8207, dtype=np.float64) ** (4.0 / 3.0)
_CI = np.array([-0.6, -0.535, -0.33, -0.185, -0.095, -0.041, -0.0142, -0.0037])
_CS = 1.0 / np.sqrt(1.0 + _CI * _CI)
_CA = _CI / np.sqrt(1.0 + _CI * _CI)


# -- Huffman -------------------------------------------------------------------

class _Huff:
    """Two-level lookup: (length, x, y) entries, or (0, subtable) for long codes."""

    def __init__(self, linbits, codes, primary=8):
        self.linbits = linbits
        maxlen = max(c[1] for c in codes)
        k = min(primary, maxlen)
        r = maxlen - k
        table = [None] * (1 << k)
        subs = {}
        for code, length, x, y in codes:
            if length <= k:
                base = code << (k - length)
                for i in range(1 << (k - length)):
                    table[base + i] = (length, x, y)
            else:
                prefix = code >> (length - k)
                sub = subs.setdefault(prefix, [(maxlen, 0, 0)] * (1 << r))
                rest = length - k
                base = (code & ((1 << rest) - 1)) << (r - rest)
                for i in range(1 << (r - rest)):
                    sub[base + i] = (length, x, y)
        for prefix, sub in subs.items():
            table[prefix] = (0, sub)
        for i, e in enumerate(table):
            if e is None:
                table[i] = (k, 0, 0)
        self.table = table
        self.k = k
        self.r = r


_HUFF = {num: _Huff(lin, codes) for num, (lin, codes) in T.HUFFMAN.items()}


class _Reader:
    __slots__ = ("val", "n", "pos")

    def __init__(self, val, n):
        self.val = val
        self.n = n
        self.pos = 0

    def read(self, k):
        if k == 0:
            return 0
        self.pos += k
        s = self.n - self.pos
        if s >= 0:
            return (self.val >> s) & ((1 << k) - 1)
        return (self.val << -s) & ((1 << k) - 1)


def _huffman_pairs(r, out, start, end, huff):
    """Decode big_values pairs into out[start:end]."""
    if huff is None:
        return
    val, n, pos = r.val, r.n, r.pos
    table, k, rbits, linbits = huff.table, huff.k, huff.r, huff.linbits
    kmask = (1 << k) - 1
    rmask = (1 << rbits) - 1
    for i in range(start, end, 2):
        s = n - pos - k
        e = table[((val >> s) if s >= 0 else (val << -s)) & kmask]
        if e[0]:
            length, x, y = e
        else:
            s -= rbits
            length, x, y = e[1][((val >> s) if s >= 0 else (val << -s)) & rmask]
        pos += length
        if x:
            if linbits and x == 15:
                pos += linbits
                s = n - pos
                x += ((val >> s) if s >= 0 else (val << -s)) & ((1 << linbits) - 1)
            pos += 1
            s = n - pos
            if (((val >> s) if s >= 0 else (val << -s)) & 1):
                x = -x
        if y:
            if linbits and y == 15:
                pos += linbits
                s = n - pos
                y += ((val >> s) if s >= 0 else (val << -s)) & ((1 << linbits) - 1)
            pos += 1
            s = n - pos
            if (((val >> s) if s >= 0 else (val << -s)) & 1):
                y = -y
        out[i] = x
        out[i + 1] = y
    r.pos = pos


def _huffman_quads(r, out, start, end_bits, huff):
    """Decode the count1 region; returns the index after the last value."""
    val, n, pos = r.val, r.n, r.pos
    table, k = huff.table, huff.k
    kmask = (1 << k) - 1
    i = start
    while pos < end_bits and i + 4 <= 576:
        s = n - pos - k
        length, _x, v = table[((val >> s) if s >= 0 else (val << -s)) & kmask]
        p = pos + length
        quad = [(v >> 3) & 1, (v >> 2) & 1, (v >> 1) & 1, v & 1]
        for j in range(4):
            if quad[j]:
                p += 1
                s = n - p
                if (((val >> s) if s >= 0 else (val << -s)) & 1):
                    quad[j] = -1
        if p > end_bits:
            break
        pos = p
        out[i:i + 4] = quad
        i += 4
    r.pos = pos
    return i


# -- band layout -----------------------------------------------------------------

class _Layout:
    """Per sample-rate maps from spectral line to scale factor band and window."""

    _cache = {}

    def __init__(self, rate, mpeg1):
        long_b, short_b = T.SFB[rate]
        self.long_b = long_b
        self.short_b = short_b
        self.long_sfb = np.zeros(576, dtype=np.int64)
        for b in range(22):
            self.long_sfb[long_b[b]:long_b[b + 1]] = b
        self.short_sfb = np.zeros(576, dtype=np.int64)
        self.short_win = np.zeros(576, dtype=np.int64)
        order = []
        for b in range(13):
            w = short_b[b + 1] - short_b[b]
            base = 3 * short_b[b]
            for win in range(3):
                self.short_sfb[base + win * w:base + (win + 1) * w] = b
                self.short_win[base + win * w:base + (win + 1) * w] = win
            for f in range(w):
                for win in range(3):
                    order.append(base + win * w + f)
        self.reorder = np.array(order, dtype=np.int64)       # reordered[j] = raw[reorder[j]]
        # mixed blocks: long bands covering the first 36 lines, then short bands
        self.mixed_long = max(b for b in range(23) if long_b[b] <= 36)
        self.mixed_short = min(b for b in range(14) if 3 * short_b[b] >= 36)
        self.mpeg1 = mpeg1

    @classmethod
    def get(cls, rate, mpeg1):
        key = (rate, mpeg1)
        if key not in cls._cache:
            cls._cache[key] = cls(rate, mpeg1)
        return cls._cache[key]


# -- transforms ----------------------------------------------------------------

def _imdct_matrix(n):
    half = n // 2
    i = np.arange(n)[:, None]
    k = np.arange(half)[None, :]
    return np.cos(np.pi / (2 * n) * (2 * i + 1 + half) * (2 * k + 1)).T      # (half, n)


_IMDCT36 = _imdct_matrix(36)
_IMDCT12 = _imdct_matrix(12)
_WIN = np.zeros((4, 36))
_i36 = np.arange(36)
_WIN[0] = np.sin(np.pi / 36 * (_i36 + 0.5))
_WIN[1, :18] = _WIN[0, :18]
_WIN[1, 18:24] = 1.0
_WIN[1, 24:30] = np.sin(np.pi / 12 * (np.arange(24, 30) - 18 + 0.5))
_WIN[3, 6:12] = np.sin(np.pi / 12 * (np.arange(6, 12) - 6 + 0.5))
_WIN[3, 12:18] = 1.0
_WIN[3, 18:] = _WIN[0, 18:]
_WIN_SHORT = np.sin(np.pi / 12 * (np.arange(12) + 0.5))
_SYNTH_N = np.cos((16 + np.arange(64))[:, None] * (2 * np.arange(32) + 1)[None, :] * np.pi / 64)
_SYNTH_D = np.array(T.SYNTH_WINDOW)


def _synthesis(sub):
    """sub: (slots, 32) subband samples -> (slots * 32,) PCM."""
    slots = sub.shape[0]
    v = np.zeros((slots + 16, 64))
    v[16:] = sub @ _SYNTH_N.T
    out = np.zeros((slots, 32))
    for m in range(8):
        a = v[16 - 2 * m:16 - 2 * m + slots, :32]
        b = v[15 - 2 * m:15 - 2 * m + slots, 32:]
        out += a * _SYNTH_D[64 * m:64 * m + 32] + b * _SYNTH_D[64 * m + 32:64 * m + 64]
    return out.reshape(-1)


# -- frame parsing -------------------------------------------------------------

class _Header:
    __slots__ = ("version", "mpeg1", "protection", "bitrate", "rate", "padding", "mode", "mode_ext",
                 "channels", "length", "side_size", "granules")

    def __init__(self, h):
        self.version = (h >> 19) & 3
        layer = (h >> 17) & 3
        if self.version == 1 or layer != 1:
            raise Mp3Error("not an MPEG audio layer III frame")
        self.mpeg1 = self.version == 3
        self.protection = not ((h >> 16) & 1)
        br = (h >> 12) & 15
        sr = (h >> 10) & 3
        if br in (0, 15) or sr == 3:
            raise Mp3Error("unsupported bit rate or sample rate")
        self.bitrate = _BITRATES[1 if self.mpeg1 else 2][br] * 1000
        self.rate = _RATES[self.version][sr]
        self.padding = (h >> 9) & 1
        self.mode = (h >> 6) & 3
        self.mode_ext = (h >> 4) & 3
        self.channels = 1 if self.mode == 3 else 2
        self.length = (144 if self.mpeg1 else 72) * self.bitrate // self.rate + self.padding
        self.side_size = (17 if self.channels == 1 else 32) if self.mpeg1 else (9 if self.channels == 1 else 17)
        self.granules = 2 if self.mpeg1 else 1


def _skip_id3v2(data):
    pos = 0
    while data[pos:pos + 3] == b"ID3" and len(data) >= pos + 10:
        size = (data[pos + 6] << 21) | (data[pos + 7] << 14) | (data[pos + 8] << 7) | data[pos + 9]
        pos += 10 + size + (10 if data[pos + 5] & 0x10 else 0)
    return pos


def _header_at(data, pos, stream):
    if pos + 4 > len(data) or data[pos] != 0xFF or (data[pos + 1] & 0xE0) != 0xE0:
        return None
    try:
        hdr = _Header(int.from_bytes(data[pos:pos + 4], "big"))
    except Mp3Error:
        return None
    if stream is not None and (hdr.version, hdr.rate) != stream:
        return None
    return hdr


def _frames(data):
    """Yield (offset, header) for consecutive valid layer III frames."""
    pos = _skip_id3v2(data)
    end = len(data)
    stream = None
    synced = False
    while pos + 4 <= end:
        hdr = _header_at(data, pos, stream)
        if hdr is not None and pos + hdr.length > end:
            return
        if hdr is not None and not synced:
            # after a gap, trust a sync word only if another frame (or the end) follows it
            nxt = pos + hdr.length
            if nxt + 4 <= end and _header_at(data, nxt, (hdr.version, hdr.rate)) is None:
                hdr = None
        if hdr is None:
            synced = False
            pos = data.find(b"\xff", pos + 1)
            if pos < 0:
                return
            continue
        stream = (hdr.version, hdr.rate)
        synced = True
        yield pos, hdr
        pos += hdr.length


def _gapless_info(frame, hdr):
    """Return (frame_count, delay, padding) from a Xing/Info/LAME or VBRI header, or None."""
    off = 4 + (2 if hdr.protection else 0) + hdr.side_size
    tag = frame[off:off + 4]
    if tag in (b"Xing", b"Info"):
        flags = int.from_bytes(frame[off + 4:off + 8], "big")
        p = off + 8
        frames = None
        if flags & 1:
            frames = int.from_bytes(frame[p:p + 4], "big")
            p += 4
        if flags & 2:
            p += 4
        if flags & 4:
            p += 100
        if flags & 8:
            p += 4
        lame = frame[p:p + 24]
        delay = padding = None
        if len(lame) == 24 and lame[:4] in (b"LAME", b"Lavc", b"Lavf", b"GOGO"):
            delay = (lame[21] << 4) | (lame[22] >> 4)
            padding = ((lame[22] & 15) << 8) | lame[23]
        return frames, delay, padding
    if frame[36:40] == b"VBRI":
        delay = int.from_bytes(frame[42:44], "big")
        frames = int.from_bytes(frame[50:54], "big")
        return frames, delay, None
    return None


# -- decoder -------------------------------------------------------------------

class _Granule:
    __slots__ = ("part23", "big_values", "global_gain", "sf_compress", "block_type", "mixed",
                 "tables", "subblock_gain", "region0", "region1", "preflag", "sf_scale", "count1_table")


def _side_info(r, hdr):
    mpeg1 = hdr.mpeg1
    nch = hdr.channels
    main_begin = r.read(9 if mpeg1 else 8)
    r.read((5 if nch == 1 else 3) if mpeg1 else (1 if nch == 1 else 2))
    scfsi = [[r.read(1) for _ in range(4)] for _ in range(nch)] if mpeg1 else [[0] * 4 for _ in range(nch)]
    grans = []
    for _gr in range(hdr.granules):
        chans = []
        for _ch in range(nch):
            g = _Granule()
            g.part23 = r.read(12)
            g.big_values = min(r.read(9), 288)
            g.global_gain = r.read(8)
            g.sf_compress = r.read(4 if mpeg1 else 9)
            if r.read(1):
                g.block_type = r.read(2)
                g.mixed = r.read(1)
                g.tables = [r.read(5), r.read(5), 0]
                g.subblock_gain = [r.read(3), r.read(3), r.read(3)]
                g.region0 = None
                g.region1 = None
            else:
                g.block_type = 0
                g.mixed = 0
                g.tables = [r.read(5), r.read(5), r.read(5)]
                g.subblock_gain = [0, 0, 0]
                g.region0 = r.read(4)
                g.region1 = r.read(3)
            g.preflag = r.read(1) if mpeg1 else 0
            g.sf_scale = r.read(1)
            g.count1_table = r.read(1)
            chans.append(g)
        grans.append(chans)
    return main_begin, scfsi, grans


class Mp3Decoder:
    def __init__(self):
        self.reservoir = b""
        self.overlap = None
        self.prev_sf = None

    def decode(self, data, first_frame=0, last_frame=None):
        """Decode frames [first_frame, last_frame] (indices among audio frames).

        Returns (pcm[channels, samples] float64, sample_rate, info dict).
        """
        frames = list(_frames(data))
        if not frames:
            raise Mp3Error("no MPEG audio layer III frames found")
        info = {"delay": None, "padding": None, "frames": None}
        off0, hdr0 = frames[0]
        tag = _gapless_info(data[off0:off0 + hdr0.length], hdr0)
        if tag is not None:
            info["frames"], info["delay"], info["padding"] = tag
            frames = frames[1:]
        if not frames:
            raise Mp3Error("MP3 contains no audio frames")
        rate = frames[0][1].rate
        nch = max(h.channels for _o, h in frames)
        spf = 576 * frames[0][1].granules
        info["samples_per_frame"] = spf
        info["total_frames"] = len(frames)
        if last_frame is None or last_frame >= len(frames):
            last_frame = len(frames) - 1
        first_frame = max(0, first_frame)
        start_decode = max(0, first_frame - 2)
        self.overlap = np.zeros((nch, 32, 18))
        subbands = [[] for _ in range(nch)]
        for index in range(len(frames)):
            if index > last_frame:
                break
            off, hdr = frames[index]
            frame = data[off:off + hdr.length]
            pos = 4 + (2 if hdr.protection else 0)
            side = frame[pos:pos + hdr.side_size]
            main = frame[pos + hdr.side_size:]
            if index < start_decode:
                self._keep(main)
                continue
            sr = _Reader(int.from_bytes(side, "big"), len(side) * 8)
            main_begin, scfsi, grans = _side_info(sr, hdr)
            if main_begin > len(self.reservoir):
                self._keep(main)
                for ch in range(nch):
                    for _ in range(hdr.granules):
                        subbands[ch].append(np.zeros((18, 32)))
                continue
            buf = self.reservoir[len(self.reservoir) - main_begin:] + main
            self._keep(main)
            self._frame(hdr, buf, scfsi, grans, subbands, nch)
        pcm = np.stack([_synthesis(np.concatenate(s, axis=0)) for s in subbands])
        info["first_sample"] = start_decode * spf
        return pcm, rate, info

    def _keep(self, main):
        self.reservoir = (self.reservoir + main)[-4096:]

    def _frame(self, hdr, buf, scfsi, grans, subbands, nch_out):
        total = len(buf) * 8
        val = int.from_bytes(buf, "big")
        layout = _Layout.get(hdr.rate, hdr.mpeg1)
        pos = 0
        nch = hdr.channels
        if self.prev_sf is None or len(self.prev_sf) != nch:
            self.prev_sf = [np.zeros(22, dtype=np.int64) for _ in range(nch)]
        for gr in range(hdr.granules):
            spectra = []
            infos = []
            for ch in range(nch):
                g = grans[gr][ch]
                length = g.part23
                if pos + length > total:
                    seg, seg_len = 0, 0
                    length = 0
                else:
                    seg = (val >> (total - pos - length)) & ((1 << length) - 1) if length else 0
                    seg_len = length
                pos += length
                r = _Reader(seg, seg_len)
                sf = self._scalefactors(r, hdr, g, ch, gr, scfsi)
                values = [0] * 576
                nz = self._huffman(r, g, layout, values, seg_len)
                spectra.append(self._requantize(np.array(values, dtype=np.int64), g, sf, layout, hdr))
                infos.append((g, sf, nz, np.array(values, dtype=np.int64)))
            if nch == 2 and hdr.mode == 1:
                self._stereo(spectra, infos, hdr, layout)
            for ch in range(nch):
                out = self._hybrid(spectra[ch], infos[ch][0], layout, ch)
                subbands[ch].append(out)
            for ch in range(nch, nch_out):
                subbands[ch].append(subbands[0][-1])

    # scale factors -----------------------------------------------------------

    def _scalefactors(self, r, hdr, g, ch, gr, scfsi):
        """Return dict with long sf (22,), short sf (13, 3) and LSF intensity illegal markers."""
        long_sf = np.zeros(22, dtype=np.int64)
        short_sf = np.zeros((13, 3), dtype=np.int64)
        illegal_long = np.zeros(22, dtype=bool)
        illegal_short = np.zeros((13, 3), dtype=bool)
        short = g.block_type == 2
        if hdr.mpeg1:
            slen1, slen2 = _SLEN[g.sf_compress]
            if short:
                if g.mixed:
                    for b in range(8):
                        long_sf[b] = r.read(slen1)
                    start = 3
                else:
                    start = 0
                for b in range(start, 12):
                    n = slen1 if b < 6 else slen2
                    for w in range(3):
                        short_sf[b, w] = r.read(n)
            else:
                groups = ((0, 6), (6, 11), (11, 16), (16, 21))
                for i, (a, b) in enumerate(groups):
                    n = slen1 if i < 2 else slen2
                    if gr == 1 and scfsi[ch][i]:
                        long_sf[a:b] = self.prev_sf[ch][a:b]
                    else:
                        for j in range(a, b):
                            long_sf[j] = r.read(n)
                self.prev_sf[ch] = long_sf.copy()
            illegal = 7
            return {"long": long_sf, "short": short_sf, "ill_long": long_sf == illegal,
                    "ill_short": short_sf == illegal, "preflag": g.preflag}
        # MPEG-2 / 2.5 (LSF)
        sfc = g.sf_compress
        intensity_right = ch == 1 and (hdr.mode_ext & 1) and hdr.mode == 1
        preflag = 0
        if not intensity_right:
            if sfc < 400:
                slen = [(sfc >> 4) // 5, (sfc >> 4) % 5, (sfc & 15) >> 2, sfc & 3]
                tab = 0
            elif sfc < 500:
                s = sfc - 400
                slen = [(s >> 2) // 5, (s >> 2) % 5, s & 3, 0]
                tab = 1
            else:
                s = sfc - 500
                slen = [s // 3, s % 3, 0, 0]
                tab = 2
                preflag = 1
        else:
            s = sfc >> 1
            if s < 180:
                slen = [s // 36, (s % 36) // 6, (s % 36) % 6, 0]
                tab = 3
            elif s < 244:
                s -= 180
                slen = [(s % 64) >> 4, (s % 16) >> 2, s % 4, 0]
                tab = 4
            else:
                s -= 244
                slen = [s // 3, s % 3, 0, 0]
                tab = 5
        kind = 0 if not short else (2 if g.mixed else 1)
        counts = _LSF_NSF[tab][kind]
        values = []
        limits = []
        for n, count in zip(slen, counts):
            for _ in range(count):
                values.append(r.read(n))
                limits.append((1 << n) - 1)
        idx = 0
        if not short:
            for b in range(min(21, len(values))):
                long_sf[b] = values[b]
                illegal_long[b] = values[b] == limits[b]
        else:
            start = 0
            if g.mixed:
                for b in range(6):
                    if idx < len(values):
                        long_sf[b] = values[idx]
                        illegal_long[b] = values[idx] == limits[idx]
                    idx += 1
                start = 3
            for b in range(start, 12):
                for w in range(3):
                    if idx < len(values):
                        short_sf[b, w] = values[idx]
                        illegal_short[b, w] = values[idx] == limits[idx]
                    idx += 1
        return {"long": long_sf, "short": short_sf, "ill_long": illegal_long, "ill_short": illegal_short,
                "preflag": preflag, "is_mode": g.sf_compress & 1}

    # Huffman -------------------------------------------------------------------

    def _huffman(self, r, g, layout, values, end_bits):
        big = g.big_values * 2
        if g.block_type == 2 or g.region0 is None:
            if g.block_type == 2:
                r1 = 3 * layout.short_b[3]
            else:
                r1 = layout.long_b[8]
            r2 = 576
        else:
            r1 = layout.long_b[min(g.region0 + 1, 22)]
            r2 = layout.long_b[min(g.region0 + g.region1 + 2, 22)]
        r1 = min(r1, big)
        r2 = min(r2, big)
        bounds = ((0, r1, g.tables[0]), (r1, r2, g.tables[1]), (r2, big, g.tables[2]))
        for a, b, t in bounds:
            if b > a:
                _huffman_pairs(r, values, a, b, _HUFF.get(t))
        return _huffman_quads(r, values, big, end_bits, _HUFF[32 + g.count1_table])

    # requantisation ------------------------------------------------------------

    def _requantize(self, values, g, sf, layout, hdr):
        mult = 2 * (1 + g.sf_scale)
        base = g.global_gain - 210
        if g.block_type == 2:
            exp = (base - 8 * np.array(g.subblock_gain)[layout.short_win]
                   - mult * sf["short"][layout.short_sfb, layout.short_win])
            if g.mixed:
                lines = layout.long_b[layout.mixed_long]
                pre = _PRETAB * g.preflag if sf["preflag"] else 0
                long_exp = base - mult * (sf["long"] + pre)[layout.long_sfb]
                exp[:lines] = long_exp[:lines]
        else:
            pre = _PRETAB if sf["preflag"] else 0
            exp = base - mult * (sf["long"] + pre)[layout.long_sfb]
        mag = _POW43[np.minimum(np.abs(values), 8206)]
        return np.sign(values) * mag * np.exp2(exp / 4.0)

    # stereo --------------------------------------------------------------------

    def _stereo(self, spectra, infos, hdr, layout):
        left, right = spectra
        ms = bool(hdr.mode_ext & 2)
        intensity = bool(hdr.mode_ext & 1)
        is_mask = np.zeros(576, dtype=bool)
        if intensity:
            g_r, sf_r, _nz, raw_r = infos[1]
            ratio_l = np.ones(576)
            ratio_r = np.ones(576)
            nonzero = np.flatnonzero(raw_r)
            if g_r.block_type == 2:
                for w in range(3):
                    lines = np.flatnonzero((layout.short_win == w) & (raw_r != 0))
                    last_b = layout.short_sfb[lines[-1]] if len(lines) else -1
                    for b in range(last_b + 1, 13):
                        if g_r.mixed and 3 * layout.short_b[b] < 36:
                            continue
                        sel = (layout.short_sfb == b) & (layout.short_win == w)
                        src = min(b, 11)
                        pos_i = sf_r["short"][src, w]
                        illegal = sf_r["ill_short"][src, w]
                        self._apply_is(sel, pos_i, illegal, hdr, sf_r, is_mask, ratio_l, ratio_r)
            else:
                last_b = layout.long_sfb[nonzero[-1]] if len(nonzero) else -1
                for b in range(last_b + 1, 22):
                    sel = layout.long_sfb == b
                    src = min(b, 20)
                    pos_i = sf_r["long"][src]
                    illegal = sf_r["ill_long"][src]
                    self._apply_is(sel, pos_i, illegal, hdr, sf_r, is_mask, ratio_l, ratio_r)
        if ms:
            m = ~is_mask
            lm, rm = left[m], right[m]
            left[m] = (lm + rm) * (1.0 / math.sqrt(2.0))
            right[m] = (lm - rm) * (1.0 / math.sqrt(2.0))
        if intensity and is_mask.any():
            src = left[is_mask].copy()
            left[is_mask] = src * ratio_l[is_mask]
            right[is_mask] = src * ratio_r[is_mask]

    @staticmethod
    def _apply_is(sel, pos_i, illegal, hdr, sf_r, is_mask, ratio_l, ratio_r):
        if illegal:
            return
        if hdr.mpeg1:
            if pos_i >= 7:
                return
            t = math.tan(pos_i * math.pi / 12.0)
            ratio_l[sel] = t / (1.0 + t)
            ratio_r[sel] = 1.0 / (1.0 + t)
        else:
            io = 2.0 ** -0.25 if sf_r.get("is_mode", 0) == 0 else 2.0 ** -0.5
            if pos_i == 0:
                ratio_l[sel] = 1.0
                ratio_r[sel] = 1.0
            elif pos_i & 1:
                ratio_l[sel] = io ** ((pos_i + 1) // 2)
                ratio_r[sel] = 1.0
            else:
                ratio_l[sel] = 1.0
                ratio_r[sel] = io ** (pos_i // 2)
        is_mask[sel] = True

    # hybrid filterbank ---------------------------------------------------------

    def _hybrid(self, xr, g, layout, ch):
        if g.block_type == 2:
            if g.mixed:
                lines = 36
                x = xr.copy()
                raw = xr
                reordered = raw[layout.reorder]
                x[lines:] = reordered[lines:]
            else:
                x = xr[layout.reorder]
        else:
            x = xr.copy()
        sb = x.reshape(32, 18)
        # alias reduction
        if g.block_type != 2 or g.mixed:
            limit = 2 if g.block_type == 2 else 32
            for k in range(1, limit):
                lo = sb[k - 1, 17::-1][:8].copy()     # sb[k-1][17-i]
                hi = sb[k, :8].copy()
                sb[k - 1, 17 - np.arange(8)] = lo * _CS - hi * _CA
                sb[k, :8] = hi * _CS + lo * _CA
        out = np.empty((32, 36))
        if g.block_type != 2:
            out[:] = (sb @ _IMDCT36) * _WIN[g.block_type]
        else:
            long_sb = 2 if g.mixed else 0
            if long_sb:
                out[:long_sb] = (sb[:long_sb] @ _IMDCT36) * _WIN[0]
            s = sb[long_sb:].reshape(-1, 6, 3)
            y = np.zeros((32 - long_sb, 36))
            for w in range(3):
                yw = (s[:, :, w] @ _IMDCT12) * _WIN_SHORT
                y[:, 6 + 6 * w:18 + 6 * w] += yw
            out[long_sb:] = y
        res = out[:, :18] + self.overlap[ch]
        self.overlap[ch] = out[:, 18:]
        res[1::2, 1::2] *= -1.0
        return res.T.copy()                      # (18 time slots, 32 subbands)


def decode_mp3(data, start=0.0, duration=None):
    """Decode MP3 bytes. Returns (pcm[channels, samples] float64 in [-1, 1], sample_rate).

    `start`/`duration` (seconds) select a range of the gapless-trimmed stream;
    frames outside it are skipped rather than decoded.
    """
    data = bytes(data)
    frames = list(_frames(data))
    if not frames:
        raise Mp3Error("no MPEG audio layer III frames found")
    off0, hdr0 = frames[0]
    tag = _gapless_info(data[off0:off0 + hdr0.length], hdr0)
    rate = hdr0.rate
    spf = 576 * hdr0.granules
    skip = 0
    total = None
    audio_frames = len(frames) - (1 if tag else 0)
    if tag is not None and tag[1] is not None:
        skip = tag[1] + 529
        if tag[2] is not None:
            count = tag[0] if tag[0] else audio_frames
            total = max(0, count * spf - tag[1] - tag[2])
    first = skip + int(round(start * rate))
    last = None
    if duration is not None:
        last = first + int(round(duration * rate))
        if total is not None:
            last = min(last, skip + total)
    elif total is not None:
        last = skip + total
    first_frame = first // spf
    last_frame = None if last is None else (last + spf - 1) // spf
    pcm, rate, info = Mp3Decoder().decode(data, first_frame, last_frame)
    a = first - info["first_sample"]
    b = pcm.shape[1] if last is None else last - info["first_sample"]
    return pcm[:, max(a, 0):max(min(b, pcm.shape[1]), 0)], rate
