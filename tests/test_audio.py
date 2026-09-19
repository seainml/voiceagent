"""Audio primitives: the PCM dialect, resampling, VAD and endpointing."""

from __future__ import annotations

import io
import math
import wave

import numpy as np

from voiceagent.audio.pcm import (
    PCM_MIME,
    AudioBuffer,
    AudioChunk,
    AudioClip,
    concat,
    dbfs,
    decode_wav,
    normalize,
)
from voiceagent.audio.vad import (
    FrameVAD,
    UtteranceSegmenter,
    VADConfig,
    segment_clip,
    trim_silence,
)


def tone(ms: int, freq: float = 440.0, rate: int = 16000, amp: float = 0.5) -> AudioClip:
    n = int(rate * ms / 1000)
    t = np.arange(n, dtype=np.float32) / rate
    return AudioClip.from_samples(amp * np.sin(2 * math.pi * freq * t), rate)


class TestAudioClip:
    def test_roundtrip_through_wav(self):
        clip = tone(200)
        decoded = decode_wav(clip.to_wav_bytes())
        assert decoded.sample_rate == clip.sample_rate
        assert abs(decoded.duration_s - clip.duration_s) < 0.001
        assert np.allclose(decoded.samples, clip.samples, atol=1e-4)

    def test_wav_header_is_a_real_riff_file(self):
        data = tone(50).to_wav_bytes()
        with wave.open(io.BytesIO(data), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000

    def test_downmix_stereo(self):
        rate = 16000
        left = tone(50, 440, rate).samples
        right = tone(50, 440, rate).samples
        interleaved = np.empty(left.size * 2, dtype=np.int16)
        interleaved[0::2] = (left * 32767).astype(np.int16)
        interleaved[1::2] = (right * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(interleaved.tobytes())
        decoded = decode_wav(buf.getvalue())
        assert decoded.channels == 1
        assert decoded.n_samples == left.size

    def test_rejects_multichannel_clips(self):
        import pytest

        with pytest.raises(ValueError):
            AudioClip(b"\x00\x00", 16000, channels=2)

    def test_resample_changes_rate_and_duration_is_preserved(self):
        clip = tone(500, rate=48000)
        out = clip.resample(16000)
        assert out.sample_rate == 16000
        assert abs(out.duration_s - clip.duration_s) < 0.01

    def test_resample_to_same_rate_is_identity(self):
        clip = tone(50)
        assert clip.resample(16000) is clip

    def test_duration_and_rms(self):
        clip = tone(100, amp=0.5)
        assert abs(clip.duration_s - 0.1) < 0.005
        assert 0.3 < clip.rms() < 0.4

    def test_silence_detection(self):
        assert AudioClip.silence(50).is_silent()
        assert not tone(50).is_silent()

    def test_dbfs_of_silence_is_very_low(self):
        assert dbfs(AudioClip.silence(20)) < -100

    def test_normalize_hits_the_target_peak(self):
        quiet = tone(100, amp=0.01)
        assert abs(float(np.max(np.abs(normalize(quiet).samples))) - 0.9) < 0.05

    def test_normalize_leaves_true_silence_alone(self):
        silent = AudioClip.silence(50)
        assert normalize(silent).pcm == silent.pcm

    def test_base64_roundtrip(self):
        clip = tone(20)
        assert AudioClip.from_base64(clip.to_base64()).pcm == clip.pcm

    def test_concat_stitches_and_resamples(self):
        a = tone(100, rate=16000)
        b = tone(100, rate=8000)
        joined = concat([a, b])
        assert joined.sample_rate == 16000
        assert abs(joined.duration_s - 0.2) < 0.02


class TestAudioChunk:
    def test_pcm_chunk_becomes_a_clip(self):
        clip = tone(20)
        chunk = AudioChunk(clip.pcm, PCM_MIME, 16000)
        assert chunk.is_pcm
        assert chunk.as_clip().sample_rate == 16000

    def test_encoded_chunk_refuses_to_be_a_clip(self):
        import pytest

        with pytest.raises(ValueError):
            AudioChunk(b"xx", "audio/mpeg", None).as_clip()


class TestAudioBuffer:
    def test_accumulates_and_takes(self):
        buf = AudioBuffer(16000)
        buf.append(tone(100).pcm)
        buf.append(tone(100).pcm)
        assert abs(buf.duration_s - 0.2) < 0.01
        clip = buf.take()
        assert abs(clip.duration_s - 0.2) < 0.01
        assert buf.n_bytes == 0

    def test_empty_buffer_yields_empty_clip(self):
        assert AudioBuffer().clip().n_samples == 0


class TestFrameVAD:
    def test_detects_speech_in_a_tone(self):
        vad = FrameVAD(VADConfig(threshold=0.01), 16000)
        assert vad.speech_ratio(tone(300, amp=0.4)) > 0.9

    def test_reports_no_speech_for_silence(self):
        vad = FrameVAD(VADConfig(threshold=0.01), 16000)
        assert vad.speech_ratio(AudioClip.silence(300)) < 0.1

    def test_adaptive_floor_does_not_suppress_loud_speech(self):
        vad = FrameVAD(VADConfig(threshold=0.015, adaptive=True), 16000)
        # Feed quiet room tone first, then speech: the floor must not swallow it.
        vad.flags(AudioClip.from_samples(
            np.random.default_rng(0).normal(0, 0.002, 16000).astype(np.float32), 16000))
        assert vad.speech_ratio(tone(300, amp=0.4)) > 0.8


class TestUtteranceSegmenter:
    def test_closes_on_trailing_silence(self):
        seg = UtteranceSegmenter(VADConfig(silence_ms=200, min_utterance_ms=100), 16000)
        decision = seg.push(tone(400).pcm)
        assert not decision.finished
        decision = seg.push(AudioClip.silence(300).pcm)
        assert decision.finished
        assert decision.reason == "silence"
        assert seg.has_speech

    def test_ignores_a_blip_shorter_than_the_minimum(self):
        seg = UtteranceSegmenter(VADConfig(silence_ms=100, min_utterance_ms=400), 16000)
        seg.push(tone(60).pcm)
        decision = seg.push(AudioClip.silence(200).pcm)
        assert not decision.finished

    def test_keeps_a_pre_roll_so_the_first_phoneme_survives(self):
        seg = UtteranceSegmenter(VADConfig(silence_ms=200, min_utterance_ms=60), 16000)
        seg.push(AudioClip.silence(200).pcm)   # nothing buffered yet
        seg.push(tone(300).pcm)
        seg.push(AudioClip.silence(250).pcm)
        clip = seg.take()
        # silence pre-roll (~200 ms) + speech (~300 ms)
        assert clip.duration_s > 0.35

    def test_hard_stops_a_runaway_utterance(self):
        seg = UtteranceSegmenter(
            VADConfig(silence_ms=10_000, min_utterance_ms=50, max_utterance_s=0.5), 16000)
        decision = seg.push(tone(700).pcm)
        assert decision.finished
        assert decision.reason == "max_duration"

    def test_segment_clip_finds_two_utterances(self):
        clip = concat([
            tone(300), AudioClip.silence(600),
            tone(300), AudioClip.silence(600),
        ])
        parts = segment_clip(clip, VADConfig(silence_ms=300, min_utterance_ms=100))
        assert len(parts) == 2


class TestTrimSilence:
    def test_removes_leading_and_trailing_silence(self):
        clip = concat([AudioClip.silence(500), tone(300), AudioClip.silence(500)])
        trimmed = trim_silence(clip, VADConfig(threshold=0.01), pad_ms=50)
        assert trimmed.duration_s < clip.duration_s
        assert 0.2 < trimmed.duration_s < 0.6

    def test_returns_input_when_there_is_no_speech(self):
        silent = AudioClip.silence(300)
        assert trim_silence(silent).pcm == silent.pcm
