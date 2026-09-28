import struct

import numpy as np

from voicecmd.audio import FRAME_SAMPLES, RingBuffer, parse_wav, pcm_to_wav
from voicecmd.tts import split_sentences
from voicecmd.vad import Endpointer


def test_wav_roundtrip():
    pcm = (np.sin(np.arange(1600) / 5) * 8000).astype(np.int16)
    out, rate, ch = parse_wav(pcm_to_wav(pcm, 16000))
    assert rate == 16000 and ch == 1 and np.array_equal(out, pcm)


def test_wav_with_streaming_size_header():
    pcm = np.arange(-500, 500, dtype=np.int16)
    wav = bytearray(pcm_to_wav(pcm, 24000))
    wav[4:8] = struct.pack("<I", 0x7FFFFFFF)
    wav[40:44] = struct.pack("<I", 0x7FFFFFFF)
    out, rate, _ = parse_wav(bytes(wav))
    assert rate == 24000 and np.array_equal(out, pcm)


def test_ring_buffer_tail():
    ring = RingBuffer(ms=400)
    for i in range(10):
        ring.push(np.full(FRAME_SAMPLES, i, dtype=np.int16))
    tail = ring.tail(160)
    assert len(tail) == 2 * FRAME_SAMPLES and tail[0] == 8 and tail[-1] == 9


def test_endpointer_times_out_on_silence():
    ep = Endpointer(listen_timeout_ms=400)
    ep.reset()
    silence = np.zeros(FRAME_SAMPLES, dtype=np.int16)
    statuses = [ep.feed(silence).status for _ in range(8)]
    assert "timeout" in statuses


def test_split_sentences_merges_short_ones():
    parts = split_sentences("Okay. Your pasta timer is set for twelve minutes. Enjoy dinner tonight!")
    assert parts[0].startswith("Okay. Your pasta timer")
    assert parts[-1] == "Enjoy dinner tonight!"
