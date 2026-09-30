# PS5 AT9 Converter - Resampling, loudness, true peak, limiting and fades.
# Copyright (C) 2026 BlackBearReloaded
# SPDX-License-Identifier: GPL-3.0-or-later

"""Audio preparation: channel mapping, resampling, loudness, true peak, limiting, fades."""

import math

import numpy as np


# -- channels ------------------------------------------------------------------

def to_stereo(pcm):
    """Map mono or multichannel audio to stereo (ITU-R BS.775 downmix for 5.1)."""
    ch = pcm.shape[0]
    if ch == 2:
        return pcm
    if ch == 1:
        return np.vstack([pcm[0], pcm[0]])
    if ch == 6:          # FL FR FC LFE BL BR
        k = 1.0 / math.sqrt(2.0)
        left = pcm[0] + k * pcm[2] + k * pcm[4]
        right = pcm[1] + k * pcm[2] + k * pcm[5]
        return np.vstack([left, right]) / (1.0 + 2.0 * k)
    return pcm[:2]


# -- resampling ----------------------------------------------------------------

def _kaiser_lowpass_phases(up, down, rate_in, pass_hz, stop_hz, atten_db=100.0):
    """Polyphase table h[phase, tap] and half width for rate_in*up/down conversion."""
    beta = 0.1102 * (atten_db - 8.7)
    width = (stop_hz - pass_hz) / rate_in                  # normalised to input rate
    taps = int(math.ceil((atten_db - 8.0) / (2.285 * 2.0 * math.pi * width)))
    half = max(8, taps // 2 + 1)
    cutoff = (pass_hz + stop_hz) / 2.0 / rate_in           # cycles per input sample
    phases = np.arange(up)[:, None] / up                  # fractional position of the output
    k = np.arange(-half + 1, half + 1)[None, :]
    t = k - phases                                        # input sample offset from output time
    h = 2.0 * cutoff * np.sinc(2.0 * cutoff * t)
    win = np.i0(beta * np.sqrt(np.clip(1.0 - (t / half) ** 2, 0.0, 1.0))) / np.i0(beta)
    h *= win
    h /= h.sum(axis=1, keepdims=True)
    return h, half


def resample(pcm, rate_in, rate_out=48000):
    """High-quality windowed-sinc sample-rate conversion of pcm[channels, samples]."""
    if rate_in == rate_out:
        return pcm
    g = math.gcd(rate_in, rate_out)
    up, down = rate_out // g, rate_in // g
    nyquist = min(rate_in, rate_out) / 2.0
    pass_hz = min(20000.0, 0.9 * nyquist)
    h, half = _kaiser_lowpass_phases(up, down, rate_in, pass_hz, nyquist)
    n_in = pcm.shape[1]
    n_out = int(math.ceil(n_in * up / down))
    pad = np.zeros((pcm.shape[0], n_in + 2 * half + 2))
    pad[:, half:half + n_in] = pcm
    out = np.empty((pcm.shape[0], n_out))
    offsets = np.arange(-half + 1, half + 1)
    chunk = max(1, (1 << 21) // len(offsets))
    for start in range(0, n_out, chunk):
        n = np.arange(start, min(start + chunk, n_out))
        pos = n * down
        base = pos // up
        phase = pos % up
        idx = base[:, None] + offsets[None, :] + half
        coeff = h[phase]
        for c in range(pcm.shape[0]):
            out[c, start:start + len(n)] = np.einsum("ij,ij->i", pad[c][idx], coeff)
    return out


# -- loudness (ITU-R BS.1770-4 / EBU R128) --------------------------------------

_K_SHELF = ([1.53512485958697, -2.69169618940638, 1.19839281085285], [1.0, -1.69065929318241, 0.73248077421585])
_K_HIGHPASS = ([1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621])


def _biquad_impulse(b, a, length):
    out = [0.0] * length
    x1 = x2 = y1 = y2 = 0.0
    for n in range(length):
        x0 = 1.0 if n == 0 else 0.0
        y0 = b[0] * x0 + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
        out[n] = y0
        x2, x1 = x1, x0
        y2, y1 = y1, y0
    return np.array(out)


_K_IMPULSE = None


def _k_weighting(pcm):
    """Apply the 48 kHz K-weighting filter (exact IIR response truncated at -200 dB)."""
    global _K_IMPULSE
    if _K_IMPULSE is None:
        shelf = _biquad_impulse(*_K_SHELF, 16384)
        hp = _biquad_impulse(*_K_HIGHPASS, 16384)
        _K_IMPULSE = np.convolve(shelf, hp)[:16384]
    n = pcm.shape[1]
    size = 1 << int(math.ceil(math.log2(n + len(_K_IMPULSE))))
    spec = np.fft.rfft(_K_IMPULSE, size)
    return np.fft.irfft(np.fft.rfft(pcm, size, axis=1) * spec, size, axis=1)[:, :n]


def integrated_loudness(pcm, rate=48000):
    """Integrated loudness in LUFS of pcm[channels, samples] (full scale = 1.0)."""
    if rate != 48000:
        raise ValueError("loudness is measured at 48 kHz")
    weighted = _k_weighting(pcm)
    block = int(0.4 * rate)
    step = int(0.1 * rate)
    n = weighted.shape[1]
    if n < block:
        power = np.mean(weighted ** 2, axis=1).sum()
        return -0.691 + 10.0 * math.log10(power) if power > 0 else -math.inf
    csum = np.concatenate([np.zeros((weighted.shape[0], 1)), np.cumsum(weighted ** 2, axis=1)], axis=1)
    starts = np.arange(0, n - block + 1, step)
    z = ((csum[:, starts + block] - csum[:, starts]) / block).sum(axis=0)
    with np.errstate(divide="ignore"):
        lk = -0.691 + 10.0 * np.log10(z)
    gated = z[lk > -70.0]
    if not len(gated):
        return -math.inf
    relative = -0.691 + 10.0 * math.log10(gated.mean()) - 10.0
    final = z[(lk > -70.0) & (lk > relative)]
    return -0.691 + 10.0 * math.log10(final.mean())


# -- true peak -----------------------------------------------------------------

def _oversample4(x):
    h, half = _kaiser_lowpass_phases(4, 1, 48000, 18000.0, 24000.0, atten_db=60.0)
    n = len(x)
    pad = np.zeros(n + 2 * half + 2)
    pad[half:half + n] = x
    out = np.empty((n, 4))
    offsets = np.arange(-half + 1, half + 1)
    chunk = max(1, (1 << 20) // len(offsets))
    for start in range(0, n, chunk):
        idx = np.arange(start, min(start + chunk, n))[:, None] + offsets[None, :] + half
        seg = pad[idx]
        out[start:start + idx.shape[0]] = seg @ h.T
    return out


def true_peak_envelope(pcm):
    """Per-sample true peak (max |x| over 4x oversampled points), max over channels."""
    env = None
    for ch in pcm:
        e = np.abs(_oversample4(ch)).max(axis=1)
        env = e if env is None else np.maximum(env, e)
    return env


def true_peak_db(pcm):
    peak = float(true_peak_envelope(pcm).max()) if pcm.size else 0.0
    return 20.0 * math.log10(peak) if peak > 0 else -math.inf


def _sliding_min(x, width):
    """Minimum over the centred window [n - width, n + width]."""
    if width <= 0:
        return x.copy()
    w = 2 * width + 1
    n = len(x)
    pad = np.concatenate([np.full(width, x[0]), x, np.full(width + w, x[-1])])
    blocks = -(-len(pad) // w)
    padded = np.concatenate([pad, np.full(blocks * w - len(pad), x[-1])]).reshape(blocks, w)
    prefix = np.minimum.accumulate(padded, axis=1).ravel()
    suffix = np.minimum.accumulate(padded[:, ::-1], axis=1)[:, ::-1].ravel()
    i = np.arange(n)
    return np.minimum(suffix[i], prefix[i + w - 1])


def _moving_average(x, width):
    """Mean over the centred window [n - width, n + width]."""
    if width <= 0:
        return x.copy()
    pad = np.concatenate([np.full(width, x[0]), x, np.full(width, x[-1])])
    c = np.concatenate([[0.0], np.cumsum(pad)])
    w = 2 * width + 1
    return (c[w:] - c[:-w]) / w


def limit_true_peak(pcm, ceiling_db, rate=48000, attack=0.002, release=0.05):
    """Smooth look-ahead gain reduction so the true peak stays at or below ceiling_db.

    Returns (pcm, limited) where `limited` tells whether any gain reduction was applied.
    """
    env = true_peak_envelope(pcm)
    if float(env.max()) <= 10.0 ** (ceiling_db / 20.0):
        return pcm, False
    target = ceiling_db
    a = max(1, int(attack * rate))
    r = max(1, int(release * rate))
    for _ in range(8):
        need = np.minimum(1.0, 10.0 ** (target / 20.0) / np.maximum(env, 1e-12))
        gain = _moving_average(_sliding_min(need, a), a // 2)
        gain = _moving_average(_sliding_min(gain, r), r // 2)
        pcm = pcm * gain[None, :]
        env = true_peak_envelope(pcm)
        if float(env.max()) <= 10.0 ** (ceiling_db / 20.0):
            break
        target -= 0.05
    return pcm, True


# -- fades ---------------------------------------------------------------------

def apply_fades(pcm, rate, fade_in, fade_out):
    n = pcm.shape[1]
    out = pcm.copy()
    fi = min(n, int(round(fade_in * rate)))
    fo = min(n, int(round(fade_out * rate)))
    if fi > 0:
        out[:, :fi] *= 0.5 - 0.5 * np.cos(np.pi * (np.arange(fi) + 0.5) / fi)
    if fo > 0:
        out[:, n - fo:] *= 0.5 + 0.5 * np.cos(np.pi * (np.arange(fo) + 0.5) / fo)
    return out
