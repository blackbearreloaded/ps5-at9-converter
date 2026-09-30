# PS5 AT9 Converter - ATRAC9 definitions and reference decoder.
# Copyright (C) 2026 BlackBearReloaded
# SPDX-License-Identifier: GPL-3.0-or-later

"""ATRAC9 bitstream definitions and a reference decoder.

The decoder follows the behaviour of LibAtrac9 by Alex Barney (MIT License),
which documents the format. It is used to verify encoder output and to
analyse existing streams; it is not needed to encode.
"""

import math

import numpy as np

from . import atrac9_tables as T

SAMPLE_RATES = [11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000,
                44100, 48000, 64000, 88200, 96000, 128000, 176400, 192000]
FRAME_SAMPLES_POWER = [6, 6, 7, 7, 7, 8, 8, 8, 6, 6, 7, 7, 7, 8, 8, 8]
MAX_BAND_COUNT = [8, 8, 12, 12, 12, 18, 18, 18, 8, 8, 12, 12, 12, 16, 16, 16]
BAND_TO_QUANT_UNITS = [0, 4, 8, 10, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 24, 25, 26, 28, 30]
QUANT_UNIT_TO_COEFF_COUNT = [2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8,
                             8, 8, 8, 8, 8, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]
QUANT_UNIT_TO_COEFF_INDEX = [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56,
                             64, 72, 80, 88, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256]
QUANT_UNIT_TO_CODEBOOK_INDEX = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

MONO, STEREO, LFE = 0, 1, 2
CHANNEL_CONFIGS = [
    (MONO,),
    (MONO, MONO),
    (STEREO,),
    (STEREO, MONO, LFE, STEREO),
    (STEREO, MONO, LFE, STEREO, STEREO),
    (STEREO, STEREO),
]
BLOCK_CHANNELS = {MONO: 1, STEREO: 2, LFE: 1}

SPECTRUM_SCALE = [2.0 ** (i - 15) for i in range(32)]
QUANT_STEP = [2.0 / ((1 << (i + 1)) - 1) for i in range(16)]
QUANT_STEP_FINE = [QUANT_STEP[i] / 65535 for i in range(16)]

GRADIENT_CURVES = [[T.GRADIENT_MAIN[i * len(T.GRADIENT_MAIN) // n] for i in range(n)]
                   for n in range(1, len(T.GRADIENT_MAIN) + 1)]

ATRAC9_GUID = bytes.fromhex("D242E147BA368D4D88FC61654F8C836C")


class FormatError(ValueError):
    pass


def sign_extend(value, bits):
    sign = 1 << (bits - 1)
    return (value & (sign - 1)) - (value & sign)


class Codebook:
    """A Huffman codebook. Symbols pack `value_count` signed values of `value_bits` bits."""

    def __init__(self, bits, codes, value_count_power):
        self.bits = list(bits)
        self.codes = list(codes)
        self.value_count_power = value_count_power
        self.value_count = 1 << value_count_power
        self.value_bits = (len(codes).bit_length() - 1) >> value_count_power
        self.value_max = 1 << self.value_bits
        self.max_bits = max(bits)
        lookup = [0] * (1 << self.max_bits)
        for symbol, length in enumerate(bits):
            if length == 0:
                continue
            unused = self.max_bits - length
            start = codes[symbol] << unused
            lookup[start:start + (1 << unused)] = [symbol] * (1 << unused)
        self.lookup = lookup


SF_UNSIGNED = {n: Codebook(b, c, 0) for n, (b, c) in T.SF_UNSIGNED.items()}
SF_SIGNED = {n: Codebook(b, c, 0) for n, (b, c) in T.SF_SIGNED.items()}
SPECTRUM = {key: Codebook(b, c, p) for key, (p, b, c) in T.SPECTRUM.items()}


class Config:
    """The 4-byte ATRAC9 configuration word."""

    def __init__(self, data):
        data = bytes(data)
        if len(data) != 4:
            raise FormatError("ATRAC9 config data must be 4 bytes")
        word = int.from_bytes(data, "big")
        header = word >> 24
        self.sample_rate_index = (word >> 20) & 15
        self.channel_config_index = (word >> 17) & 7
        validation = (word >> 16) & 1
        self.frame_bytes = ((word >> 5) & 0x7FF) + 1
        self.superframe_index = (word >> 3) & 3
        if header != 0xFE or validation != 0 or self.channel_config_index > 5:
            raise FormatError("invalid ATRAC9 config data")
        self.data = data
        self.frames_per_superframe = 1 << self.superframe_index
        self.superframe_bytes = self.frame_bytes << self.superframe_index
        self.block_types = CHANNEL_CONFIGS[self.channel_config_index]
        self.channel_count = sum(BLOCK_CHANNELS[t] for t in self.block_types)
        self.sample_rate = SAMPLE_RATES[self.sample_rate_index]
        self.high_sample_rate = self.sample_rate_index > 7
        self.frame_samples_power = FRAME_SAMPLES_POWER[self.sample_rate_index]
        self.frame_samples = 1 << self.frame_samples_power
        self.superframe_samples = self.frame_samples * self.frames_per_superframe

    @staticmethod
    def build(sample_rate_index, channel_config_index, frame_bytes, superframe_index):
        word = (0xFE << 24) | (sample_rate_index << 20) | (channel_config_index << 17)
        word |= ((frame_bytes - 1) << 5) | (superframe_index << 3)
        return Config(word.to_bytes(4, "big"))


class BitReader:
    def __init__(self, data):
        self.value = int.from_bytes(data, "big")
        self.length = len(data) * 8
        self.pos = 0

    def peek(self, count):
        end = self.pos + count
        if end <= self.length:
            return (self.value >> (self.length - end)) & ((1 << count) - 1)
        if self.pos >= self.length:
            return 0
        avail = self.length - self.pos
        return (self.value & ((1 << avail) - 1)) << (count - avail)

    def read(self, count):
        value = self.peek(count)
        self.pos += count
        return value

    def read_signed(self, count):
        return sign_extend(self.read(count), count)

    def huffman(self, book, signed=False):
        symbol = book.lookup[self.peek(book.max_bits)]
        self.pos += book.bits[symbol]
        return sign_extend(symbol, book.value_bits) if signed else symbol

    def align(self):
        self.pos = (self.pos + 7) & ~7


def gradient_curve(mode, start_unit, end_unit, start_value, end_value, unit_count):
    """Return the 31-entry allocation gradient, exactly as the decoder derives it."""
    grad = [start_value] * 31
    for i in range(end_unit, min(unit_count, 30) + 1):
        grad[i] = end_value
    units = end_unit - start_unit
    values = end_value - start_value
    if units > 0 and values != 0:
        curve = GRADIENT_CURVES[units - 1]
        if values < 0:
            scale = (-values - 1) / 31.0
            for i in range(start_unit, end_unit):
                grad[i] = start_value - 1 - int(curve[i - start_unit] * scale)
        else:
            scale = (values - 1) / 31.0
            for i in range(start_unit, end_unit):
                grad[i] = start_value + 1 + int(curve[i - start_unit] * scale)
    return grad


def precision_mask(sf, unit_count):
    mask = [0] * 30
    for i in range(1, unit_count):
        delta = sf[i] - sf[i - 1]
        if delta > 1:
            mask[i] += min(delta - 1, 5)
        elif delta < -1:
            mask[i - 1] += min(-delta - 1, 5)
    return mask


def precisions(sf, grad, mode, boundary, unit_count):
    """Return (coarse, fine) precisions for one channel."""
    coarse = [0] * 30
    if mode:
        mask = precision_mask(sf, unit_count)
        for i in range(unit_count):
            p = sf[i] + mask[i] - grad[i]
            if p > 0:
                p = p // 2 if mode == 1 else (3 * p) // 8 if mode == 2 else p // 4
            coarse[i] = p
    else:
        for i in range(unit_count):
            coarse[i] = sf[i] - grad[i]
    fine = [0] * 30
    for i in range(unit_count):
        if coarse[i] < 1:
            coarse[i] = 1
        if i < boundary:
            coarse[i] += 1
        if coarse[i] > 15:
            fine[i] = coarse[i] - 15
            coarse[i] = 15
    return coarse, fine


def codebook_sets(sf, coded_units, high_sample_rate):
    sets = [0] * 30
    if coded_units <= 1 or high_sample_rate:
        return sets
    s = list(sf[:coded_units]) + [sf[coded_units - 1]]
    avg = 0
    if coded_units > 12:
        avg = (sum(s[:12]) + 6) // 12
    for i in range(8, coded_units):
        lo = min(s[i - 1], s[i + 1])
        if s[i] - lo >= 3 or 2 * s[i] - s[i - 1] - s[i + 1] >= 3:
            sets[i] = 1
    for i in range(12, coded_units):
        if not sets[i]:
            lo = min(s[i - 1], s[i + 1])
            if s[i] - lo >= 2 and s[i] >= avg - (1 if QUANT_UNIT_TO_COEFF_COUNT[i] == 16 else 0):
                sets[i] = 1
    return sets


def mdct_windows(frame_samples):
    """Analysis (encoder) and synthesis (decoder) half-windows of length N."""
    n = np.arange(frame_samples)
    analysis = (np.sin(((n + 0.5) / frame_samples - 0.5) * np.pi) + 1.0) * 0.5
    rev = analysis[::-1]
    synthesis = analysis / (rev * rev + analysis * analysis)
    return analysis, synthesis


def dct4_matrix(size):
    k = np.arange(size) + 0.5
    return np.cos(np.pi / size * np.outer(k, k))


class ChannelState:
    def __init__(self):
        self.sf = [0] * 31
        self.sf_prev = [0] * 31
        self.bex_mode = 0
        self.bex_value_count = 0
        self.bex_values = [0] * 4
        self.rng = None


class BlockState:
    def __init__(self, block_type):
        self.type = block_type
        self.channels = [ChannelState() for _ in range(BLOCK_CHANNELS[block_type])]
        self.band_count = 0
        self.stereo_band = 0
        self.unit_count = 0
        self.stereo_units = 0
        self.extension_unit = 0
        self.bex_enabled = False
        self.units_prev = 0
        self.have_band_params = False


class Rng:
    """Xorshift generator used for band-extension noise."""

    def __init__(self, seed):
        start = (0x4D93 * (seed ^ (seed >> 14))) & 0xFFFF
        self.a = (3 - start) & 0xFFFF
        self.b = (2 - start) & 0xFFFF
        self.c = (1 - start) & 0xFFFF
        self.d = (0 - start) & 0xFFFF

    def next(self):
        t = (self.d ^ (self.d << 5)) & 0xFFFF
        self.d, self.c, self.b = self.c, self.b, self.a
        self.a = (t ^ self.a ^ ((t ^ (self.a >> 5)) >> 4)) & 0xFFFF
        return self.a


class Decoder:
    """Decode ATRAC9 superframes to float PCM (16-bit scale)."""

    def __init__(self, config, trace=None):
        self.config = config if isinstance(config, Config) else Config(config)
        self.blocks = [BlockState(t) for t in self.config.block_types]
        self.trace = trace
        n = self.config.frame_samples
        _, self.window = mdct_windows(n)
        self.dct = dct4_matrix(n)
        self.overlap = [np.zeros(n) for _ in range(self.config.channel_count)]

    def decode(self, data):
        """Decode a whole stream (multiple of superframe_bytes). Returns (channels, samples)."""
        cfg = self.config
        count = len(data) // cfg.superframe_bytes
        spectra = np.zeros((cfg.channel_count, count * cfg.frames_per_superframe, cfg.frame_samples))
        for sf_index in range(count):
            chunk = data[sf_index * cfg.superframe_bytes:(sf_index + 1) * cfg.superframe_bytes]
            reader = BitReader(chunk)
            for frame in range(cfg.frames_per_superframe):
                row = sf_index * cfg.frames_per_superframe + frame
                ch = 0
                for block in self.blocks:
                    out = self._block(reader, block, frame, row)
                    for spec in out:
                        spectra[ch, row] = spec
                        ch += 1
                    reader.align()
        return self._imdct(spectra)

    def _imdct(self, spectra):
        n = self.config.frame_samples
        half = n // 2
        win = self.window
        pcm = np.zeros((spectra.shape[0], spectra.shape[1] * n))
        for ch in range(spectra.shape[0]):
            y = spectra[ch] @ self.dct  # (frames, n) DCT-IV
            cur = np.empty_like(y)
            nxt = np.empty_like(y)
            i = np.arange(half)
            cur[:, i] = win[i] * y[:, i + half]
            cur[:, i + half] = -win[i + half] * y[:, n - 1 - i]
            nxt[:, i] = -win[n - 1 - i] * y[:, half - 1 - i]
            nxt[:, i + half] = -win[half - 1 - i] * y[:, i]
            out = cur.copy()
            out[0] += self.overlap[ch]
            out[1:] += nxt[:-1]
            self.overlap[ch] = nxt[-1].copy()
            pcm[ch] = out.reshape(-1)
        return pcm

    # -- bitstream ---------------------------------------------------------

    def _block(self, r, b, frame, row):
        first = not r.read(1)
        reuse = r.read(1)
        if first != (frame == 0):
            raise FormatError("first-in-superframe flag mismatch")
        if b.type == LFE:
            return [self._lfe(r, b, reuse)]
        if first and reuse:
            raise FormatError("band parameters reused in first frame")
        cfg = self.config
        info = {"frame": row}
        if not reuse:
            self._band_params(r, b)
        elif not b.have_band_params:
            raise FormatError("band parameters reused before being set")
        mode = r.read(2)
        if mode:
            start_unit, end_unit = r.read(5), 31
            start_value, end_value = r.read(5), 31
        else:
            start_unit, end_unit = r.read(6), r.read(6) + 1
            start_value, end_value = r.read(5), r.read(5)
        boundary = r.read(4)
        if boundary > b.unit_count or not 1 <= start_unit <= end_unit <= 31:
            raise FormatError("invalid gradient parameters")
        grad = gradient_curve(mode, start_unit, end_unit, start_value, end_value, b.unit_count)
        primary = 0
        signs = [0] * 30
        if b.type == STEREO:
            primary = r.read(1)
            if r.read(1):
                for i in range(b.stereo_units, b.unit_count):
                    signs[i] = r.read(1)
        self._extension_params(r, b)
        info.update(reuse=reuse, bands=b.band_count, stereo_band=b.stereo_band, bex=b.bex_enabled,
                    ext_band=b.extension_band if b.bex_enabled else None, grad=(mode, start_unit, end_unit,
                    start_value, end_value, boundary), primary=primary, bits_start=r.pos)
        spectra = []
        sf_modes = []
        all_prec = []
        for index, ch in enumerate(b.channels):
            coded = b.unit_count if (b.type != STEREO or index == primary) else b.stereo_units
            sf_modes.append(self._scale_factors(r, b, ch, index, first))
            coarse, fine = precisions(ch.sf, grad, mode, boundary, b.unit_count)
            all_prec.append(coarse[:b.unit_count])
            sets = codebook_sets(ch.sf, coded, cfg.high_sample_rate)
            q = [0] * 256
            for i in range(coded):
                start, end = QUANT_UNIT_TO_COEFF_INDEX[i], QUANT_UNIT_TO_COEFF_INDEX[i + 1]
                wl = coarse[i] + 1
                if wl <= (1 if cfg.high_sample_rate else 7):
                    book = SPECTRUM[(sets[i], wl, QUANT_UNIT_TO_CODEBOOK_INDEX[i])]
                    pos = start
                    mask = book.value_max - 1
                    for _ in range((end - start) >> book.value_count_power):
                        sym = r.huffman(book)
                        for _ in range(book.value_count):
                            q[pos] = sign_extend(sym & mask, book.value_bits)
                            sym >>= book.value_bits
                            pos += 1
                else:
                    for j in range(start, end):
                        q[j] = r.read_signed(wl)
            spec = np.zeros(256)
            for i in range(coded):
                start, end = QUANT_UNIT_TO_COEFF_INDEX[i], QUANT_UNIT_TO_COEFF_INDEX[i + 1]
                spec[start:end] = np.array(q[start:end]) * QUANT_STEP[coarse[i]]
            for i in range(coded):
                if fine[i] > 0:
                    start, end = QUANT_UNIT_TO_COEFF_INDEX[i], QUANT_UNIT_TO_COEFF_INDEX[i + 1]
                    for j in range(start, end):
                        spec[j] += r.read_signed(fine[i] + 1) * QUANT_STEP_FINE[fine[i]]
            spectra.append(spec)
        b.units_prev = b.extension_unit if b.bex_enabled else b.unit_count
        if b.type == STEREO and b.stereo_units < b.unit_count:
            src = spectra[primary]
            dst = spectra[1 - primary]
            for i in range(b.stereo_units, b.unit_count):
                start, end = QUANT_UNIT_TO_COEFF_INDEX[i], QUANT_UNIT_TO_COEFF_INDEX[i + 1]
                dst[start:end] = -src[start:end] if signs[i] else src[start:end]
        for index, ch in enumerate(b.channels):
            spec = spectra[index]
            for i in range(b.unit_count):
                start, end = QUANT_UNIT_TO_COEFF_INDEX[i], QUANT_UNIT_TO_COEFF_INDEX[i + 1]
                spec[start:end] *= SPECTRUM_SCALE[ch.sf[i]]
            if b.bex_enabled and b.has_extension_data:
                self._apply_bex(b, ch, spec)
        if self.trace is not None:
            info.update(sf=[list(c.sf[:b.extension_unit]) for c in b.channels], sf_modes=sf_modes,
                        precisions=all_prec, bits_end=r.pos)
            self.trace.append(info)
        return spectra

    def _band_params(self, r, b):
        cfg = self.config
        min_bands = 1 if cfg.high_sample_rate else 3
        max_ext = 16 if cfg.high_sample_rate else 18
        b.band_count = r.read(4) + min_bands
        if b.band_count > MAX_BAND_COUNT[cfg.sample_rate_index]:
            raise FormatError("invalid band count")
        b.unit_count = BAND_TO_QUANT_UNITS[b.band_count]
        if b.type == STEREO:
            b.stereo_band = r.read(4) + min_bands
            if b.stereo_band > b.band_count:
                raise FormatError("invalid stereo band")
        else:
            b.stereo_band = b.band_count
        b.stereo_units = BAND_TO_QUANT_UNITS[b.stereo_band]
        b.bex_enabled = bool(r.read(1))
        if b.bex_enabled:
            b.extension_band = r.read(4) + min_bands
            if b.extension_band < b.band_count or b.extension_band > max_ext:
                raise FormatError("invalid extension band")
        else:
            b.extension_band = b.band_count
        b.extension_unit = BAND_TO_QUANT_UNITS[b.extension_band]
        b.have_band_params = True

    def _extension_params(self, r, b):
        bex_band = 0
        if b.bex_enabled:
            if not 13 <= b.unit_count <= 20:
                raise FormatError("invalid band extension unit count")
            bex_band = T.BEX_GROUP_INFO[b.unit_count - 13][2]
            if b.type == STEREO:
                self._bex_header(r, b.channels[1], bex_band)
            else:
                r.pos += 1
        b.has_extension_data = bool(r.read(1))
        if not b.has_extension_data:
            return
        if not b.bex_enabled:
            r.read(2)
            r.pos += r.read(5)
            return
        self._bex_header(r, b.channels[0], bex_band)
        length = r.read(5)
        if length <= 0:
            return
        end = r.pos + length
        for ch in b.channels:
            for i in range(ch.bex_value_count):
                ch.bex_values[i] = r.read(T.BEX_DATA_LENGTHS[ch.bex_mode][bex_band][i])
        if r.pos > end:
            raise FormatError("band extension data overrun")

    @staticmethod
    def _bex_header(r, ch, bex_band):
        mode = r.read(2)
        ch.bex_mode = mode if bex_band > 2 else 4
        ch.bex_value_count = T.BEX_VALUE_COUNTS[ch.bex_mode][bex_band]

    def _scale_factors(self, r, b, ch, index, first):
        sf = [0] * 31
        mode = r.read(2)
        units = b.extension_unit
        if index == 0:
            if mode == 0:
                self._sf_delta_offset(r, sf, units)
            elif mode == 1:
                self._sf_clc(r, sf, units)
            else:
                if first:
                    raise FormatError("scale factors reference previous frame in first frame")
                if mode == 2:
                    self._sf_distance(r, sf, units, ch.sf_prev, b.units_prev)
                else:
                    self._sf_delta_baseline(r, sf, units, ch.sf_prev, b.units_prev)
        else:
            base = b.channels[0].sf
            if mode == 0:
                self._sf_delta_offset(r, sf, units)
            elif mode == 1:
                self._sf_distance(r, sf, units, base, units)
            elif mode == 2:
                self._sf_delta_baseline(r, sf, units, base, units)
            else:
                if first:
                    raise FormatError("scale factors reference previous frame in first frame")
                self._sf_distance(r, sf, units, ch.sf_prev, b.units_prev)
        if any(v < 0 or v > 31 for v in sf[:units]):
            raise FormatError("scale factor out of range")
        ch.sf = sf
        ch.sf_prev = list(sf)
        return mode

    @staticmethod
    def _sf_clc(r, sf, units):
        length = r.read(2) + 2
        base = r.read(5) if length < 5 else 0
        for i in range(units):
            sf[i] = r.read(length) + base

    @staticmethod
    def _sf_delta_offset(r, sf, units):
        weights = T.SF_WEIGHTS[r.read(3)]
        base = r.read(5)
        length = r.read(2) + 3
        book = SF_UNSIGNED[length]
        sf[0] = r.read(length)
        for i in range(1, units):
            sf[i] = (sf[i - 1] + r.huffman(book)) & (book.value_max - 1)
        for i in range(units):
            sf[i] += base - weights[i]

    @staticmethod
    def _sf_distance(r, sf, units, baseline, baseline_len):
        length = r.read(2) + 2
        book = SF_SIGNED[length]
        count = min(units, baseline_len)
        for i in range(count):
            sf[i] = (baseline[i] + r.huffman(book, signed=True)) & 31
        for i in range(count, units):
            sf[i] = r.read(5)

    @staticmethod
    def _sf_delta_baseline(r, sf, units, baseline, baseline_len):
        base = r.read(5) - 16
        length = r.read(2) + 1
        book = SF_UNSIGNED[length]
        count = min(units, baseline_len)
        sf[0] = r.read(length)
        for i in range(1, count):
            sf[i] = (sf[i - 1] + r.huffman(book)) & (book.value_max - 1)
        for i in range(count):
            sf[i] += base + baseline[i]
        for i in range(count, units):
            sf[i] = r.read(5)

    def _lfe(self, r, b, reuse):
        ch = b.channels[0]
        b.unit_count = 2
        ch.sf = [0] * 31
        for i in range(2):
            ch.sf[i] = r.read(5)
        precision = 8 if reuse else 4
        spec = np.zeros(256)
        for i in range(2):
            for j in range(QUANT_UNIT_TO_COEFF_INDEX[i], QUANT_UNIT_TO_COEFF_INDEX[i + 1]):
                spec[j] = r.read_signed(precision + 1) * QUANT_STEP[precision] * SPECTRUM_SCALE[ch.sf[i]]
        return spec

    def _apply_bex(self, b, ch, spec):
        a_unit = b.unit_count
        b_unit, c_unit, band_count = T.BEX_GROUP_INFO[a_unit - 13]
        total_units = max(c_unit, 22)
        a_bin = QUANT_UNIT_TO_COEFF_INDEX[a_unit]
        b_bin = QUANT_UNIT_TO_COEFF_INDEX[b_unit]
        c_bin = QUANT_UNIT_TO_COEFF_INDEX[c_unit]
        total_bins = QUANT_UNIT_TO_COEFF_INDEX[total_units]
        for i in range(b_bin - a_bin):
            spec[a_bin + i] = spec[a_bin - i - 1]
        for i in range(c_bin - b_bin):
            spec[b_bin + i] = spec[b_bin - i - 1]
        for i in range(total_bins - c_bin):
            spec[c_bin + i] = spec[c_bin - i - 1]
        v = ch.bex_values
        mode = ch.bex_mode

        def add_noise(index, count):
            if ch.rng is None:
                ch.rng = Rng((543 * (ch.sf[8] + ch.sf[12] + ch.sf[15] + 1)) & 0xFFFF)
            for i in range(count):
                spec[index + i] = ch.rng.next() / 65535.0 * 2.0 - 1.0

        def scale_units(scales):
            for i in range(a_unit, total_units):
                start, end = QUANT_UNIT_TO_COEFF_INDEX[i], QUANT_UNIT_TO_COEFF_INDEX[i + 1]
                spec[start:end] *= scales[i - a_unit]

        if mode == 0:
            scales = [0.0] * 6
            if band_count == 3:
                t = T.BEX_MODE0_BANDS3
                scales[:5] = [t[0][v[0]], t[1][v[0]], t[2][v[1]], t[3][v[2]], t[4][v[3]]]
            elif band_count == 4:
                t = T.BEX_MODE0_BANDS4
                scales[:5] = [t[0][v[0]], t[1][v[0]], t[2][v[1]], t[3][v[2]], t[4][v[3]]]
            elif band_count == 5:
                t = T.BEX_MODE0_BANDS5
                scales[:3] = [t[0][v[0]], t[1][v[1]], t[2][v[1]]]
            scales[total_units - a_unit - 1] = SPECTRUM_SCALE[ch.sf[a_unit]]
            add_noise(QUANT_UNIT_TO_COEFF_INDEX[total_units - 1], QUANT_UNIT_TO_COEFF_COUNT[total_units - 1])
            scale_units(scales)
        elif mode == 1:
            scales = [SPECTRUM_SCALE[ch.sf[i]] for i in range(a_unit, total_units)]
            add_noise(a_bin, total_bins - a_bin)
            scale_units(scales)
        elif mode == 2:
            spec[a_bin:b_bin] *= T.BEX_MODE2_SCALE[v[0]]
            spec[b_bin:c_bin] *= T.BEX_MODE2_SCALE[v[1]]
        elif mode == 3:
            rate = 2.0 ** T.BEX_MODE3_RATE[v[1]]
            scale = T.BEX_MODE3_INITIAL[v[0]]
            for i in range(a_bin, total_bins):
                scale *= rate
                spec[i] *= scale
        elif mode == 4:
            mult = T.BEX_MODE4_MULTIPLIER[v[0]]
            spec[a_bin:b_bin] *= 0.7079468 * mult
            spec[b_bin:c_bin] *= 0.5011902 * mult
            spec[c_bin:total_bins] *= 0.3548279 * mult
