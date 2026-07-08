// Real-FFT abstraction: same frontend.c compiles against esp-dsp on device
// and the portable reference implementation on the host (parity tests).
#pragma once

void rfft512_init(void);
// in: 512 windowed samples. out: 257 power-spectrum bins (re^2 + im^2).
void rfft512_power(const float *in, float *out);
