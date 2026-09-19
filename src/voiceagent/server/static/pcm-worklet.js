// AudioWorklet that forwards mono Float32 frames to the main thread.
// Capture happens at 16 kHz because we ask the AudioContext for that rate —
// the browser does the resampling properly, so the server never has to.
class PCMCapture extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input[0] && input[0].length) {
      this.port.postMessage(input[0].slice(0));
    }
    return true;
  }
}
registerProcessor("pcm-capture", PCMCapture);
