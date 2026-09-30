# PS5 AT9 Converter - ATRAC9 encoder.
# Copyright (C) 2026 BlackBearReloaded
# SPDX-License-Identifier: GPL-3.0-or-later

"""Native ATRAC9 encoder (48 kHz, mono or stereo).

The encoder produces streams that an ATRAC9 decoder reconstructs exactly as
modelled here: everything the decoder derives (precisions, codebook sets,
scale-factor predictions) is recomputed with the decoder's own integer rules.

Per 256-sample frame and channel it computes MDCT coefficients, tight scale
factors and, for every word length, the exact Huffman cost and quantisation
noise of each quantisation unit. Bit allocation then searches the decoder's
gradient parameters, weighting noise by a psychoacoustic masking threshold,
and a knapsack over the four frames of each superframe spends the fixed
superframe size where it lowers the weighted noise the most. Superframes are
independent, so they are encoded in parallel.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import atrac9 as A
from . import atrac9_tables as T

N = 256
UNITS = 30
QIDX = np.array(A.QUANT_UNIT_TO_COEFF_INDEX, dtype=np.int64)
QCNT = np.array(A.QUANT_UNIT_TO_COEFF_COUNT, dtype=np.int64)
COEFF_UNIT = np.repeat(np.arange(UNITS), QCNT)
CBI = np.array(A.QUANT_UNIT_TO_CODEBOOK_INDEX, dtype=np.int64)
STEP = np.array(A.QUANT_STEP)
INV_STEP = 1.0 / STEP
MAXQ = (1 << np.arange(16)) - 1
BAND_UNITS = A.BAND_TO_QUANT_UNITS
FRAMES_PER_SUPERFRAME = 4
SAMPLE_RATE = 48000
SAMPLE_RATE_INDEX = 7

STEREO_BITRATES = (72, 96, 120, 144, 168, 192)
MONO_BITRATES = (36, 48, 60, 72, 84, 96, 120, 144)

# Highest band coded at each per-channel bit rate (kb/s). Band 16 is 18 kHz.
_MAX_BAND_BY_CHANNEL_RATE = ((96, 16), (84, 16), (72, 15), (60, 14), (48, 13), (36, 12))

INF = float("inf")


class _Overflow(Exception):
    """The superframe does not fit at the requested bandwidth."""


def frame_bytes_for(bitrate_kbps):
    value = bitrate_kbps * 1000 * N // (8 * SAMPLE_RATE)
    if value * 8 * SAMPLE_RATE != bitrate_kbps * 1000 * N:
        raise ValueError("unsupported bit rate %d kb/s" % bitrate_kbps)
    return value


# -- Huffman tables ------------------------------------------------------------

class _SpectrumBooks:
    """Per word length: code/length lookup plus the grouping of coefficients."""

    def __init__(self, wl):
        self.wl = wl
        self.mask = (1 << wl) - 1
        size = 1 << (4 * wl if wl == 2 else 2 * wl if wl <= 4 else wl)
        self.bits = np.zeros((2, 4, size), dtype=np.int64)
        self.codes = np.zeros((2, 4, size), dtype=np.int64)
        vcp = [0, 0, 0, 0]
        for (s, w, c), book in A.SPECTRUM.items():
            if w != wl:
                continue
            n = len(book.bits)
            self.bits[s, c, :n] = book.bits
            self.codes[s, c, :n] = book.codes
            vcp[c] = book.value_count_power
        rows, units = [], []
        for u in range(UNITS):
            vc = 1 << vcp[CBI[u]]
            for g in range(int(QCNT[u]) // vc):
                start = int(QIDX[u]) + g * vc
                rows.append(list(range(start, start + vc)) + [N] * (4 - vc))
                units.append(u)
        self.idx = np.array(rows, dtype=np.int64)
        self.unit = np.array(units, dtype=np.int64)
        self.cbi = CBI[self.unit]
        self.shift = np.arange(4, dtype=np.int64) * wl
        self.group_start = np.searchsorted(self.unit, np.arange(UNITS + 1))

    def symbols(self, q_ext):
        """q_ext: (..., N + 1) quantised values with a trailing zero."""
        vals = np.take(q_ext, self.idx, axis=-1) & self.mask
        return (vals << self.shift).sum(axis=-1)


_BOOKS = {wl: _SpectrumBooks(wl) for wl in range(2, 8)}


def _sf_book(book):
    return np.array(book.bits, dtype=np.int64), np.array(book.codes, dtype=np.int64)


_SF_U = {n: _sf_book(b) for n, b in A.SF_UNSIGNED.items()}
_SF_S = {n: _sf_book(b) for n, b in A.SF_SIGNED.items()}
_SF_WEIGHTS = np.array(T.SF_WEIGHTS, dtype=np.int64)


# -- gradient candidates -------------------------------------------------------

D_MIN, D_MAX = -40, 50
D_SPAN = D_MAX - D_MIN + 1
P_CAP = 17                      # precisions above 15 would need fine bits: never chosen


def _mode_precision(mode, d):
    if mode == 1:
        d = np.where(d > 0, d // 2, d)
    elif mode == 2:
        d = np.where(d > 0, (3 * d) // 8, d)
    elif mode == 3:
        d = np.where(d > 0, d // 4, d)
    return np.clip(d, 1, P_CAP)


_P_OF_D = np.stack([_mode_precision(m, np.arange(D_MIN, D_MAX + 1)) for m in range(4)])
_P1_OF_D = np.minimum(_P_OF_D + 1, P_CAP)
_GRAD_ROW = {}
for _m in range(4):
    for _s in range(1, 31):
        for _a in range(31):
            _GRAD_ROW[(_m, _s, _a)] = A.gradient_curve(_m, _s, 31, _a, 31, 30)[:UNITS]


class _Candidates:
    """Gradient parameter sets searched for a given number of coded units."""

    _cache = {}

    def __init__(self, units):
        keys = set()
        for m in (1, 2, 3):
            for s in range(max(1, units - 12), min(30, units + 2) + 1):
                for a in range(27):
                    keys.add((m, s, a))
        for a in range(31):
            keys.add((1, 1, a))
            for s in {max(1, units - 8), max(1, units - 4), min(30, units)}:
                keys.add((0, s, a))
        keys = sorted(keys)
        self.params = [(m, s, 31, a, 31) for m, s, a in keys]
        modes = np.array([k[0] for k in keys], dtype=np.int64)
        self.mode0 = modes == 0
        self.grad = np.array([_GRAD_ROW[k][:units] for k in keys], dtype=np.int64)
        self.offset = (modes[:, None] * units + np.arange(units)[None, :]) * D_SPAN - D_MIN
        self.header_bits = np.where(self.mode0, 2 + 6 + 6 + 5 + 5 + 4, 2 + 5 + 5 + 4)

    @classmethod
    def get(cls, units):
        if units not in cls._cache:
            cls._cache[units] = cls(units)
        return cls._cache[units]


def _precision_mask(sf, units):
    mask = np.zeros(UNITS, dtype=np.int64)
    d = np.diff(sf[:units])
    up = d > 1
    down = d < -1
    mask[1:units][up] += np.minimum(d[up] - 1, 5)
    mask[:units - 1][down] += np.minimum(-d[down] - 1, 5)
    return mask


# -- psychoacoustics -----------------------------------------------------------

def _bark(f):
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def _ath_db(f):
    k = np.maximum(f, 20.0) / 1000.0
    return 3.64 * k ** -0.8 - 6.5 * np.exp(-0.6 * (k - 3.3) ** 2) + 1e-3 * k ** 4


class _Psycho:
    """Per-unit masking thresholds, in the MDCT coefficient energy domain."""

    FULL_SCALE_SPL = 110.0      # assumed playback level of a full-scale sine, dB SPL (loud)
    SLOPE_BELOW = 27.0          # masking slope towards lower frequencies, dB/Bark
    SLOPE_ABOVE = 12.0          # masking slope towards higher frequencies, dB/Bark
    NOISE_OFFSET = 5.5          # threshold offset below a noise-like masker, dB
    TONE_OFFSET = 14.5          # offset below a tonal masker at 0 Bark, rising by 1 dB/Bark ...
    TONE_OFFSET_MAX = 25.0      # ... up to this limit, dB
    FLATNESS_TONAL = -25.0      # spectral flatness (dB) regarded as fully tonal

    def __init__(self):
        bin_hz = SAMPLE_RATE / 2.0 / N
        lo = QIDX[:-1] * bin_hz
        hi = QIDX[1:] * bin_hz
        z = _bark((lo + hi) / 2.0)
        dz = z[:, None] - z[None, :]            # maskee - masker
        atten = np.where(dz < 0, self.SLOPE_BELOW * -dz, self.SLOPE_ABOVE * dz)
        self.spread = 10.0 ** (-atten / 10.0)   # [maskee, masker]
        self.z = z
        freqs = np.linspace(lo, hi, 9, axis=1)
        ath = _ath_db(freqs).min(axis=1) - self.FULL_SCALE_SPL
        self.ath = 32768.0 ** 2 * 10.0 ** (ath / 10.0)
        self.bins = [(max(int(QIDX[u]) - 1, 0), min(int(QIDX[u + 1]) + 2, N + 1)) for u in range(UNITS)]

    def thresholds(self, energy, power):
        """energy: (..., UNITS) unit energies; power: (..., N + 1) FFT power."""
        logp = np.log(power + 1e-12)
        tonal = np.empty(energy.shape)
        for u, (a, b) in enumerate(self.bins):
            gm = logp[..., a:b].mean(axis=-1)
            am = np.log(power[..., a:b].mean(axis=-1) + 1e-12)
            sfm_db = 10.0 / np.log(10.0) * (gm - am)
            tonal[..., u] = np.clip(sfm_db / self.FLATNESS_TONAL, 0.0, 1.0)
        offset = (tonal * np.minimum(self.TONE_OFFSET + self.z, self.TONE_OFFSET_MAX)
                  + (1.0 - tonal) * self.NOISE_OFFSET)
        masked = (energy @ self.spread.T) * 10.0 ** (-offset / 10.0)
        return np.maximum(masked, self.ath)


# -- bit writer ----------------------------------------------------------------

class _Bits:
    __slots__ = ("acc", "count")

    def __init__(self):
        self.acc = 0
        self.count = 0

    def put(self, value, bits):
        if bits:
            self.acc = (self.acc << bits) | (int(value) & ((1 << bits) - 1))
            self.count += bits

    def put_many(self, values, lengths):
        acc, count = self.acc, self.count
        for v, n in zip(values.tolist(), lengths.tolist()):
            acc = (acc << n) | v
            count += n
        self.acc, self.count = acc, count

    def to_bytes(self):
        pad = -self.count % 8
        return (self.acc << pad).to_bytes((self.count + pad) // 8, "big")


# -- scale factor coding -------------------------------------------------------

def _sf_clc(sf):
    lo, hi = int(sf.min()), int(sf.max())
    for length in (2, 3, 4):
        if hi - lo < (1 << length):
            return (2 + 2 + 5 + len(sf) * length, 1, ("clc", length, lo))
    return (2 + 2 + len(sf) * 5, 1, ("clc", 5, 0))


def _sf_delta_offset(sf, mode):
    units = len(sf)
    t = sf[None, :] + _SF_WEIGHTS[:, :units]                        # (8, units)
    base = np.clip(t.min(axis=1), 0, 31)
    u = t - base[:, None]
    best = None
    for length in (3, 4, 5, 6):
        ok = (u.min(axis=1) >= 0) & (u.max(axis=1) < (1 << length))
        if not ok.any():
            continue
        deltas = np.diff(u, axis=1) & ((1 << length) - 1)
        bits = _SF_U[length][0][deltas].sum(axis=1) + (2 + 3 + 5 + 2 + length)
        bits = np.where(ok, bits, 1 << 30)
        w = int(np.argmin(bits))
        if best is None or bits[w] < best[0]:
            best = (int(bits[w]), mode, ("delta", w, int(base[w]), length))
    return best


def _sf_distance(sf, baseline, baseline_len, mode):
    units = len(sf)
    count = min(units, baseline_len)
    d = sf[:count] - baseline[:count]
    lo, hi = (int(d.min()), int(d.max())) if count else (0, 0)
    best = None
    for length in (2, 3, 4, 5):
        half = 1 << (length - 1)
        if lo < -half or hi >= half:
            continue
        tab = _SF_S[length][0]
        sym = d & ((1 << length) - 1)
        if count and not tab[sym].all():
            continue
        bits = 2 + 2 + int(tab[sym].sum()) + (units - count) * 5
        if best is None or bits < best[0]:
            best = (bits, mode, ("distance", length, baseline, count))
    return best


def _sf_delta_baseline(sf, baseline, baseline_len, mode):
    units = len(sf)
    count = min(units, baseline_len)
    if count < 1:
        return None
    r = sf[:count] - baseline[:count]
    base = min(max(int(r.min()), -16), 15)
    u = r - base
    if u.min() < 0:
        return None
    best = None
    for length in (1, 2, 3, 4):
        if u.max() >= (1 << length):
            continue
        tab = _SF_U[length][0]
        deltas = np.diff(u) & ((1 << length) - 1)
        bits = 2 + 5 + 2 + length + int(tab[deltas].sum()) + (units - count) * 5
        if best is None or bits < best[0]:
            best = (bits, mode, ("offset", length, base, baseline, count))
    return best


def _choose_sf_coding(sf, channel, first, prev_sf, prev_units, ch0_sf):
    if channel == 0:
        options = [_sf_delta_offset(sf, 0), _sf_clc(sf)]
        if not first:
            options += [_sf_distance(sf, prev_sf, prev_units, 2), _sf_delta_baseline(sf, prev_sf, prev_units, 3)]
    else:
        options = [_sf_delta_offset(sf, 0), _sf_distance(sf, ch0_sf, len(sf), 1),
                   _sf_delta_baseline(sf, ch0_sf, len(sf), 2)]
        if not first:
            options.append(_sf_distance(sf, prev_sf, prev_units, 3))
    return min((o for o in options if o is not None), key=lambda o: o[0])


def _write_sf(out, sf, coding):
    _bits, mode, spec = coding
    out.put(mode, 2)
    kind = spec[0]
    units = len(sf)
    if kind == "clc":
        _, length, base = spec
        out.put(length - 2, 2)
        if length < 5:
            out.put(base, 5)
        for v in sf.tolist():
            out.put(v - base, length)
    elif kind == "delta":
        _, w, base, length = spec
        u = sf + _SF_WEIGHTS[w, :units] - base
        out.put(w, 3)
        out.put(base, 5)
        out.put(length - 3, 2)
        out.put(int(u[0]), length)
        bits_tab, codes_tab = _SF_U[length]
        deltas = np.diff(u) & ((1 << length) - 1)
        out.put_many(codes_tab[deltas], bits_tab[deltas])
    elif kind == "distance":
        _, length, baseline, count = spec
        out.put(length - 2, 2)
        bits_tab, codes_tab = _SF_S[length]
        sym = (sf[:count] - baseline[:count]) & ((1 << length) - 1)
        out.put_many(codes_tab[sym], bits_tab[sym])
        for v in sf[count:].tolist():
            out.put(v, 5)
    else:
        _, length, base, baseline, count = spec
        out.put(base + 16, 5)
        out.put(length - 1, 2)
        u = sf[:count] - baseline[:count] - base
        out.put(int(u[0]), length)
        bits_tab, codes_tab = _SF_U[length]
        deltas = np.diff(u) & ((1 << length) - 1)
        out.put_many(codes_tab[deltas], bits_tab[deltas])
        for v in sf[count:].tolist():
            out.put(v, 5)


# -- per channel-frame analysis ------------------------------------------------

def _fix_minimum(q, xn):
    """Enforce the codebook rules for units 0-11 (see tests/test_atrac9.py)."""
    for p in (1, 2, 3):
        rows = q[p - 1, :16].reshape(8, 2)
        need = 1 << (p - 1)
        bad = np.abs(rows).max(axis=1) < need
        if bad.any():
            xs = xn[:16].reshape(8, 2)
            j = np.abs(xs).argmax(axis=1)
            for u in np.nonzero(bad)[0]:
                rows[u, j[u]] = need if xs[u, j[u]] >= 0 else -need
    rows = q[0, 16:32].reshape(4, 4)
    bad = ~rows.any(axis=1)
    if bad.any():
        xs = xn[16:32].reshape(4, 4)
        j = np.abs(xs).argmax(axis=1)
        for u in np.nonzero(bad)[0]:
            rows[u, j[u]] = 1 if xs[u, j[u]] >= 0 else -1


class _ChannelFrame:
    """Quantisation results for every precision of one channel in one frame."""

    __slots__ = ("sf", "sets", "q", "mask", "coding", "sym", "lut")

    def __init__(self, x, sf, units, weight):
        self.sf = sf
        self.sets = np.array(A.codebook_sets(sf.tolist(), units, False), dtype=np.int64)
        self.mask = _precision_mask(sf, units)
        scale = np.exp2(sf[COEFF_UNIT] - 15.0)
        xn = x / scale
        q = np.rint(xn[None, :] * INV_STEP[1:, None])
        np.clip(q, -MAXQ[1:, None], MAXQ[1:, None], out=q)
        q = q.astype(np.int64)
        _fix_minimum(q, xn)
        err = (xn[None, :] - q * STEP[1:, None]) * scale[None, :]
        noise = np.add.reduceat(err * err, QIDX[:-1], axis=1)[:, :units] * weight[None, :units]
        cost = np.empty((15, units))
        q_ext = np.concatenate([q[:6], np.zeros((6, 1), dtype=np.int64)], axis=1)
        sym = {}
        for p in range(1, 7):
            book = _BOOKS[p + 1]
            s = book.symbols(q_ext[p - 1])
            sym[p] = s
            gb = book.bits[self.sets[book.unit], book.cbi, s]
            cost[p - 1] = np.bincount(book.unit, weights=gb, minlength=UNITS)[:units]
        cost[6:] = QCNT[None, :units] * np.arange(8, 17)[:, None]
        table = np.full((2, units, P_CAP + 1), INF)
        table[0, :, 1:16] = cost.T
        table[1, :, 1:16] = noise.T
        # Lookup tables indexed by (mode, unit, d) for the base precision and for +1.
        lut0 = table[:, :, _P_OF_D].transpose(0, 2, 1, 3).reshape(2, -1)
        lut1 = table[:, :, _P1_OF_D].transpose(0, 2, 1, 3).reshape(2, -1)
        self.lut = (lut0, lut1)
        self.q = q
        self.sym = sym


# -- encoder -------------------------------------------------------------------

class Encoder:
    def __init__(self, channels=2, bitrate=192):
        if channels not in (1, 2):
            raise ValueError("ATRAC9 encoder supports mono or stereo")
        allowed = STEREO_BITRATES if channels == 2 else MONO_BITRATES
        if bitrate not in allowed:
            raise ValueError("bit rate must be one of %s kb/s for %d channel(s)" % (allowed, channels))
        self.channels = channels
        self.bitrate = bitrate
        self.frame_bytes = frame_bytes_for(bitrate)
        self.config = A.Config.build(SAMPLE_RATE_INDEX, 2 if channels == 2 else 0, self.frame_bytes, 2)
        self.superframe_bytes = self.config.superframe_bytes
        per_channel = bitrate // channels
        self.max_band = next((b for r, b in _MAX_BAND_BY_CHANNEL_RATE if per_channel >= r), 12)
        analysis, _ = A.mdct_windows(N)
        k = np.arange(N)[:, None] + 0.5
        n = np.arange(2 * N)[None, :] + 0.5 + N / 2.0
        window = np.concatenate([analysis, analysis[::-1]])[None, :]
        self.basis = (np.cos(np.pi / N * k * n) * window).T * (2.0 / N)
        self.psycho = _Psycho()

    # public ------------------------------------------------------------------

    def encode(self, pcm, progress=None, workers=None):
        """Encode float PCM (channels, samples) in 16-bit scale. Returns the data chunk bytes."""
        pcm = np.atleast_2d(np.asarray(pcm, dtype=np.float64))
        if pcm.shape[0] != self.channels:
            raise ValueError("expected %d channel(s)" % self.channels)
        samples = pcm.shape[1]
        superframes = -(-(samples + N) // (N * FRAMES_PER_SUPERFRAME))
        frames = superframes * FRAMES_PER_SUPERFRAME
        padded = np.zeros((self.channels, (frames + 1) * N))
        padded[:, N:N + samples] = pcm
        blocks = np.lib.stride_tricks.sliding_window_view(padded, 2 * N, axis=1)[:, ::N][:, :frames]
        spectra = blocks @ self.basis                                   # (ch, frames, N)
        power = np.abs(np.fft.rfft(blocks * np.hanning(2 * N), axis=-1)) ** 2
        energy = np.add.reduceat(spectra * spectra, QIDX[:-1], axis=-1)
        masks = self.psycho.thresholds(energy, power)
        del blocks, power
        peaks = np.maximum.reduceat(np.abs(spectra), QIDX[:-1], axis=-1)
        with np.errstate(divide="ignore"):
            sfs = np.where(peaks > 0, np.ceil(np.log2(np.maximum(peaks, 1e-300))) + 15, 0)
        sfs = np.clip(sfs, 0, 31).astype(np.int64)
        bands = self._choose_bands(energy, masks, superframes)

        chunk = 8
        jobs = []
        for start in range(0, superframes, chunk):
            stop = min(start + chunk, superframes)
            rows = slice(start * FRAMES_PER_SUPERFRAME, stop * FRAMES_PER_SUPERFRAME)
            jobs.append((self.channels, self.bitrate, spectra[:, rows], sfs[:, rows], masks[:, rows],
                         bands[start:stop]))
        if workers is None:
            workers = os.cpu_count() or 1
        workers = max(1, min(workers, len(jobs)))
        out = []
        done = 0
        if workers == 1:
            results = map(_encode_job, jobs)
            pool = None
        else:
            pool = ProcessPoolExecutor(max_workers=workers)
            results = pool.map(_encode_job, jobs)
        try:
            for job, data in zip(jobs, results):
                out.append(data)
                done += len(job[5])
                if progress:
                    progress(done, superframes)
        finally:
            if pool is not None:
                pool.shutdown(cancel_futures=True)
        return b"".join(out)

    # internals ---------------------------------------------------------------

    def _choose_bands(self, energy, masks, superframes):
        audible = (energy > masks).reshape(self.channels, superframes, FRAMES_PER_SUPERFRAME, UNITS)
        audible = audible.any(axis=(0, 2))                              # (superframes, UNITS)
        top = np.where(audible.any(axis=1), UNITS - np.argmax(audible[:, ::-1], axis=1), 0)
        need = np.array([next((b for b in range(3, 19) if BAND_UNITS[b] >= t), 18) for t in top])
        smooth = need.copy()
        for d in (1, 2):
            smooth[d:] = np.maximum(smooth[d:], need[:-d])
            smooth[:-d] = np.maximum(smooth[:-d], need[d:])
        return np.clip(smooth, 3, self.max_band)

    def encode_superframes(self, spectra, sfs, masks, bands):
        out = bytearray()
        for s, band in enumerate(bands):
            rows = slice(s * FRAMES_PER_SUPERFRAME, (s + 1) * FRAMES_PER_SUPERFRAME)
            args = (spectra[:, rows], sfs[:, rows], masks[:, rows])
            band = int(band)
            while True:
                try:
                    out += self._superframe(*args, band)
                    break
                except _Overflow:
                    if band <= 3:
                        raise RuntimeError("superframe cannot be encoded within its size")
                    band -= 1      # narrow the coded bandwidth until the superframe fits
        return bytes(out)

    def _superframe(self, spectra, sfs, masks, band):
        units = BAND_UNITS[band]
        cands = _Candidates.get(units)
        stereo = self.channels == 2
        frames = []
        prev_sf = [None] * self.channels
        for f in range(FRAMES_PER_SUPERFRAME):
            first = f == 0
            fixed = 2 + ((9 if stereo else 5) if first else 0) + (2 if stereo else 0) + 1
            chans = []
            for c in range(self.channels):
                sf = sfs[c, f].copy()
                sf[units:] = 0
                cf = _ChannelFrame(spectra[c, f], sf, units, 1.0 / masks[c, f])
                cf.coding = _choose_sf_coding(sf[:units], c, first, prev_sf[c], units,
                                              chans[0].sf[:units] if c else None)
                fixed += cf.coding[0]
                chans.append(cf)
            prev_sf = [cf.sf for cf in chans]
            frames.append((chans,) + self._frame_options(chans, cands, fixed, units))
        choice = self._allocate(frames)
        data = bytearray()
        for f, frame in enumerate(frames):
            data += self._write_frame(frame, choice[f], f == 0, band, units, cands)
        if len(data) > self.superframe_bytes:
            raise RuntimeError("internal error: superframe overflow")
        # Sony's decoder rejects superframes whose unused tail is not filled with 0x01 bytes.
        data += b"\x01" * (self.superframe_bytes - len(data))
        return data

    def _frame_options(self, chans, cands, fixed_bits, units):
        with np.errstate(invalid="ignore"):
            return self._frame_options_inner(chans, cands, fixed_bits, units)

    def _frame_options_inner(self, chans, cands, fixed_bits, units):
        kmax = min(15, units)
        bits = dist = None
        for cf in chans:
            base = np.where(cands.mode0[:, None], cf.sf[None, :units], (cf.sf + cf.mask)[None, :units])
            idx = cands.offset + np.clip(base - cands.grad, D_MIN, D_MAX)
            lut0, lut1 = cf.lut
            c0 = np.take(lut0[0], idx)
            n0 = np.take(lut0[1], idx)
            dc = np.take(lut1[0], idx[:, :kmax]) - c0[:, :kmax]
            dn = np.take(lut1[1], idx[:, :kmax]) - n0[:, :kmax]
            b = np.empty((len(idx), kmax + 1))
            d = np.empty((len(idx), kmax + 1))
            b[:, 0] = c0.sum(axis=1)
            d[:, 0] = n0.sum(axis=1)
            np.cumsum(dc, axis=1, out=b[:, 1:])
            np.cumsum(dn, axis=1, out=d[:, 1:])
            b[:, 1:] += b[:, :1]
            d[:, 1:] += d[:, :1]
            bits = b if bits is None else bits + b
            dist = d if dist is None else dist + d
        bits += fixed_bits + cands.header_bits[:, None]
        nbytes = np.ceil(bits.ravel() / 8.0)
        flat = np.flatnonzero(nbytes <= self.superframe_bytes)
        if not len(flat):
            raise _Overflow()
        b = nbytes[flat].astype(np.int16)
        d = dist.ravel()[flat]
        order = np.argsort(b, kind="stable")
        b, d, flat = b[order], d[order], flat[order]
        starts = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
        mins = np.minimum.reduceat(d, starts)
        sizes = np.diff(np.r_[starts, len(b)])
        hit = np.flatnonzero(d == np.repeat(mins, sizes))
        group = np.searchsorted(starts, hit, side="right") - 1
        first = hit[np.r_[True, group[1:] != group[:-1]]]
        b, d, flat = b[first].astype(np.int64), d[first], flat[first]
        keep = np.r_[True, d[1:] < np.minimum.accumulate(d)[:-1]]
        return b[keep], d[keep], flat[keep], kmax

    def _allocate(self, frames):
        cap = self.superframe_bytes
        dp = np.full(cap + 1, INF)
        dp[0] = 0.0
        back = []
        for _chans, b, d, _flat, _k in frames:
            new = np.full(cap + 1, INF)
            arg = np.full(cap + 1, -1, dtype=np.int64)
            for i in range(len(b)):
                size = int(b[i])
                cand = dp[:cap + 1 - size] + d[i]
                better = cand < new[size:]
                new[size:][better] = cand[better]
                arg[size:][better] = i
            back.append(arg)
            dp = new
        if not np.isfinite(dp).any():
            raise _Overflow()
        pos = int(np.argmin(dp))
        choice = [0] * len(frames)
        for f in range(len(frames) - 1, -1, -1):
            i = int(back[f][pos])
            choice[f] = i
            pos -= int(frames[f][1][i])
        return choice

    def _write_frame(self, frame, index, first, band, units, cands):
        chans, b, _d, flat, kmax = frame
        cand, boundary = divmod(int(flat[index]), kmax + 1)
        mode, start_unit, end_unit, start_value, end_value = cands.params[cand]
        out = _Bits()
        out.put(0 if first else 1, 1)
        out.put(0 if first else 1, 1)
        if first:
            out.put(band - 3, 4)
            if self.channels == 2:
                out.put(band - 3, 4)
            out.put(0, 1)
        out.put(mode, 2)
        if mode:
            out.put(start_unit, 5)
            out.put(start_value, 5)
        else:
            out.put(start_unit, 6)
            out.put(end_unit - 1, 6)
            out.put(start_value, 5)
            out.put(end_value, 5)
        out.put(boundary, 4)
        if self.channels == 2:
            out.put(0, 1)      # primary channel 0
            out.put(0, 1)      # no intensity-stereo signs
        out.put(0, 1)          # no band-extension data
        grad = A.gradient_curve(mode, start_unit, end_unit, start_value, end_value, units)
        for cf in chans:
            coarse, fine = A.precisions(cf.sf.tolist(), grad, mode, boundary, units)
            if any(fine[:units]):
                raise RuntimeError("internal error: fine precision selected")
            _write_sf(out, cf.sf[:units], cf.coding)
            for u in range(units):
                p = coarse[u]
                wl = p + 1
                if wl <= 7:
                    book = _BOOKS[wl]
                    g0, g1 = book.group_start[u], book.group_start[u + 1]
                    sym = cf.sym[p][g0:g1]
                    s = cf.sets[u]
                    lengths = book.bits[s, CBI[u], sym]
                    if not lengths.all():
                        raise RuntimeError("internal error: invalid spectrum symbol")
                    out.put_many(book.codes[s, CBI[u], sym], lengths)
                else:
                    for v in cf.q[p - 1, QIDX[u]:QIDX[u + 1]].tolist():
                        out.put(v, wl)
        data = out.to_bytes()
        if len(data) != int(b[index]):
            raise RuntimeError("internal error: frame size mismatch (%d != %d)" % (len(data), int(b[index])))
        return data


_WORKER_ENCODERS = {}


def _encode_job(job):
    channels, bitrate, spectra, sfs, masks, bands = job
    key = (channels, bitrate)
    enc = _WORKER_ENCODERS.get(key)
    if enc is None:
        enc = _WORKER_ENCODERS[key] = Encoder(channels, bitrate)
    return enc.encode_superframes(spectra, sfs, masks, bands)
