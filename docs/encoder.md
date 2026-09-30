# How the encoder works

ATRAC9 decoders derive most of their decisions from a few transmitted parameters. The encoder in [`ps5at9/atrac9_encoder.py`](../ps5at9/atrac9_encoder.py) searches those parameters and recomputes everything the decoder derives with the decoder's own integer rules. It can therefore predict the exact size of every frame and the exact quantisation noise of every choice.

## Transform

Each channel is cut into 512-sample blocks that advance by 256 samples, so one frame covers 5.3 ms at 48 kHz. Each block is windowed with ATRAC9's analysis window, `(sin(((n + 0.5) / 256 − 0.5)·π) + 1) / 2`, mirrored for the second half. It is then transformed with a 256-coefficient MDCT scaled by 2/N. The decoder's synthesis window is the matching biorthogonal window, so an unquantised spectrum reconstructs the input exactly, delayed by 256 samples. This delay is the encoder delay in the `fact` chunk.

## Scale factors and quantisation

The 256 coefficients form 30 quantisation units of 2 to 16 coefficients. Each unit's scale factor is the smallest value whose range `2^(sf − 15)` covers the unit's peak. The decoder computes each unit's precision (its word length) from the scale factors and the transmitted *gradient* parameters. The gradient parameters are a mode, a start unit, a start value, an end unit, an end value and a boundary; the boundary gives one extra bit to the lowest units.

For every unit and every precision from 1 to 15, the encoder quantises the coefficients and records two numbers:

- the exact Huffman cost, using the codebook set the decoder will pick from the scale factors;
- the quantisation noise, weighted by the masking threshold.

A precision table lookup then gives the bits and weighted noise of any gradient setting without quantising again.

## Psychoacoustic weighting

Masking thresholds are computed per unit and frame. Unit energies are spread across the Bark scale, with slopes of 27 dB/Bark below a masker and 12 dB/Bark above it. The spread energy is lowered by an offset that depends on tonality, measured by the spectral flatness of a Hann-windowed FFT: about 5.5 dB for noise and up to 25 dB for tones. The result is limited below by the absolute threshold of hearing, assuming a loud playback level (full scale at 110 dB SPL; PEAQ scores improve up to that level and then level off). Quantisation noise divided by this threshold (the noise-to-mask ratio) is the distortion measure.

## Bit allocation

For every frame the encoder evaluates about 1,300 gradient settings, each with every boundary from 0 to 15. The settings cover all four gradient modes and start units around the coded bandwidth. For each possible frame size in bytes, it keeps the setting with the lowest total noise-to-mask ratio.

The four frames of a superframe share 512 bytes. A small dynamic programme picks one option per frame to minimise the total noise-to-mask ratio within that budget, so hard frames borrow bytes from easy ones. If a superframe cannot fit even at the lowest precisions, the coded bandwidth is narrowed until it does. This only happens at low bit rates.

## Bitstream

- Each superframe is independent: the first frame sends the band parameters and the next three reuse them. Scale factors use whichever of the eight coding modes is cheapest, including prediction from the other channel or the previous frame of the same superframe.
- Spectra use Huffman codes up to word length 7 and fixed-length codes above it.
- Intensity stereo and band extension are not used. The coded bandwidth follows the audible content per superframe, up to band 16 (18 kHz) at 192 kb/s, with smoothing across neighbouring superframes.
- Unused superframe bytes are filled with `0x01`.

Superframes are encoded in parallel worker processes. The output is identical to a single-process run and identical across platforms.

## Verification

- The built-in decoder in [`ps5at9/atrac9.py`](../ps5at9/atrac9.py) follows LibAtrac9. It reproduces Sony's `at9tool -d` output for Sony-encoded files to within 1e-7.
- Encoder output decodes bit-exactly the same in Sony's decoder as in the built-in decoder, across all bit rates and the edge cases listed in [development.md](development.md).
