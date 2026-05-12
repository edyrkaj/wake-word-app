// Pure JS log-mel spectrogram matching the params used in
// model_gen/train_model.py:
//   sample_rate=16000, n_fft=512, hop_length=160, n_mels=40,
//   window=hann (periodic), center=True, pad_mode=reflect,
//   mel_scale='htk', power=2.0, norm=None
// Output shape: Float32Array(N_MELS * N_FRAMES) in [mel, frame] order
// (rows = mel bins). The model expects [1, 1, N_MELS, N_FRAMES].

export const SAMPLE_RATE = 16000;
export const N_FFT = 512;
export const HOP_LENGTH = 160;
export const N_MELS = 40;
export const N_FRAMES = 99;
export const WINDOW_SAMPLES = SAMPLE_RATE;

const F_MIN = 0;
const F_MAX = SAMPLE_RATE / 2;
const HALF_FFT = N_FFT / 2;

// Periodic Hann window: w[n] = 0.5 - 0.5 cos(2 pi n / N)
const HANN = new Float32Array(N_FFT);
for (let i = 0; i < N_FFT; i += 1) {
  HANN[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N_FFT);
}

function hzToMel(hz) {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel) {
  return 700 * (10 ** (mel / 2595) - 1);
}

// Build mel filterbank: triangular filters with peaks placed on mel scale.
const FILTERBANK = (() => {
  const melMin = hzToMel(F_MIN);
  const melMax = hzToMel(F_MAX);
  const melPoints = new Float64Array(N_MELS + 2);
  for (let i = 0; i < N_MELS + 2; i += 1) {
    melPoints[i] = melMin + ((melMax - melMin) * i) / (N_MELS + 1);
  }
  const hzPoints = new Float64Array(N_MELS + 2);
  for (let i = 0; i < N_MELS + 2; i += 1) {
    hzPoints[i] = melToHz(melPoints[i]);
  }
  const binHz = new Float64Array(HALF_FFT + 1);
  for (let k = 0; k <= HALF_FFT; k += 1) {
    binHz[k] = (k * SAMPLE_RATE) / N_FFT;
  }
  const fb = [];
  for (let m = 1; m <= N_MELS; m += 1) {
    const left = hzPoints[m - 1];
    const center = hzPoints[m];
    const right = hzPoints[m + 1];
    const filter = new Float32Array(HALF_FFT + 1);
    for (let k = 0; k <= HALF_FFT; k += 1) {
      const f = binHz[k];
      if (f >= left && f <= center) {
        filter[k] = (f - left) / (center - left || 1);
      } else if (f > center && f <= right) {
        filter[k] = (right - f) / (right - center || 1);
      }
    }
    fb.push(filter);
  }
  return fb;
})();

// Iterative Cooley-Tukey radix-2 FFT (in-place).
function fft(re, im) {
  const n = re.length;
  // bit reversal
  let j = 0;
  for (let i = 1; i < n; i += 1) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      let tmp = re[i];
      re[i] = re[j];
      re[j] = tmp;
      tmp = im[i];
      im[i] = im[j];
      im[j] = tmp;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const angle = (-2 * Math.PI) / len;
    const wStepRe = Math.cos(angle);
    const wStepIm = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let wRe = 1;
      let wIm = 0;
      for (let k = 0; k < half; k += 1) {
        const uRe = re[i + k];
        const uIm = im[i + k];
        const tRe = re[i + k + half] * wRe - im[i + k + half] * wIm;
        const tIm = re[i + k + half] * wIm + im[i + k + half] * wRe;
        re[i + k] = uRe + tRe;
        im[i + k] = uIm + tIm;
        re[i + k + half] = uRe - tRe;
        im[i + k + half] = uIm - tIm;
        const newWRe = wRe * wStepRe - wIm * wStepIm;
        const newWIm = wRe * wStepIm + wIm * wStepRe;
        wRe = newWRe;
        wIm = newWIm;
      }
    }
  }
}

// Reflect padding (PyTorch-style): pad=2 on [a,b,c,d,e] -> [c,b,a,b,c,d,e,d,c]
function reflectPad(samples, pad) {
  const out = new Float32Array(samples.length + 2 * pad);
  for (let i = 0; i < pad; i += 1) {
    out[i] = samples[pad - i];
  }
  out.set(samples, pad);
  for (let i = 0; i < pad; i += 1) {
    out[pad + samples.length + i] = samples[samples.length - 2 - i];
  }
  return out;
}

const _re = new Float32Array(N_FFT);
const _im = new Float32Array(N_FFT);
const _power = new Float32Array(HALF_FFT + 1);

export function computeLogMel(samples) {
  // Ensure exactly WINDOW_SAMPLES.
  let buf;
  if (samples.length === WINDOW_SAMPLES) {
    buf = samples;
  } else if (samples.length > WINDOW_SAMPLES) {
    const start = ((samples.length - WINDOW_SAMPLES) / 2) | 0;
    buf = samples.subarray(start, start + WINDOW_SAMPLES);
  } else {
    buf = new Float32Array(WINDOW_SAMPLES);
    buf.set(samples, 0);
  }

  const pad = N_FFT / 2;
  const padded = reflectPad(buf, pad);

  // center=True framing: 1 + N / hop
  const totalFrames = 1 + Math.floor(WINDOW_SAMPLES / HOP_LENGTH);
  const out = new Float32Array(N_MELS * N_FRAMES);

  const usableFrames = Math.min(totalFrames, N_FRAMES);

  for (let t = 0; t < usableFrames; t += 1) {
    const start = t * HOP_LENGTH;
    for (let i = 0; i < N_FFT; i += 1) {
      _re[i] = padded[start + i] * HANN[i];
      _im[i] = 0;
    }
    fft(_re, _im);
    for (let k = 0; k <= HALF_FFT; k += 1) {
      _power[k] = _re[k] * _re[k] + _im[k] * _im[k];
    }
    for (let m = 0; m < N_MELS; m += 1) {
      const filter = FILTERBANK[m];
      let sum = 0;
      for (let k = 0; k < filter.length; k += 1) {
        sum += _power[k] * filter[k];
      }
      out[m * N_FRAMES + t] = Math.log(sum + 1e-6);
    }
  }

  // If usableFrames < N_FRAMES the remaining cells are already 0; in
  // practice totalFrames=101 so we trim and never pad.
  return out;
}
