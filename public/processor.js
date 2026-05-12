// Audio worklet that maintains a rolling 1-second window of mic samples
// at 16 kHz and posts a fresh copy to the main thread every ~250 ms once
// the buffer is full.

class AudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.bufferSize = 16000; // 1 second @ 16 kHz
    this.buffer = new Float32Array(this.bufferSize);
    this.writeIdx = 0;
    this.filled = 0;
    this.samplesSincePost = 0;
    this.postIntervalSamples = 4000; // ~250 ms between posts
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const channel = input[0];
    if (!channel) return true;

    for (let i = 0; i < channel.length; i += 1) {
      this.buffer[this.writeIdx] = channel[i];
      this.writeIdx = (this.writeIdx + 1) % this.bufferSize;
      if (this.filled < this.bufferSize) this.filled += 1;
      this.samplesSincePost += 1;

      if (
        this.filled >= this.bufferSize &&
        this.samplesSincePost >= this.postIntervalSamples
      ) {
        this.samplesSincePost = 0;
        const window = new Float32Array(this.bufferSize);
        for (let j = 0; j < this.bufferSize; j += 1) {
          window[j] = this.buffer[(this.writeIdx + j) % this.bufferSize];
        }
        this.port.postMessage(window, [window.buffer]);
      }
    }
    return true;
  }
}

registerProcessor('audio-processor', AudioProcessor);
