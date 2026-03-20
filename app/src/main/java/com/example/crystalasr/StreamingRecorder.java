package com.example.crystalasr;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Captures mono PCM16 at 16 kHz and forwards chunks to {@link AsrNative}.
 */
public final class StreamingRecorder {
    private static final int SAMPLE_RATE = 16000;
    private static final int CHANNEL = AudioFormat.CHANNEL_IN_MONO;
    private static final int FORMAT = AudioFormat.ENCODING_PCM_16BIT;

    private final AsrNative asrNative;
    private final long nativeHandle;

    private Thread thread;
    private AudioRecord record;
    private final AtomicBoolean running = new AtomicBoolean(false);

    public StreamingRecorder(AsrNative asrNative, long nativeHandle) {
        this.asrNative = asrNative;
        this.nativeHandle = nativeHandle;
    }

    public void start() {
        if (running.getAndSet(true)) return;

        int minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL, FORMAT);
        int bufferSize = Math.max(minBuf, SAMPLE_RATE / 5 * 2); // ~200ms

        record = new AudioRecord(
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                SAMPLE_RATE,
                CHANNEL,
                FORMAT,
                bufferSize);

        if (record.getState() != AudioRecord.STATE_INITIALIZED) {
            running.set(false);
            if (record != null) {
                record.release();
                record = null;
            }
            throw new IllegalStateException("AudioRecord failed to initialize");
        }

        asrNative.nativeStart(nativeHandle);
        record.startRecording();

        thread = new Thread(this::loop, "crystal-asr-audio");
        thread.start();
    }

    public void stop() {
        running.set(false);
        Thread t = thread;
        thread = null;
        if (t != null) {
            try {
                t.join(2000);
            } catch (InterruptedException ignored) {
                Thread.currentThread().interrupt();
            }
        }
        AudioRecord r = record;
        record = null;
        if (r != null) {
            try {
                r.stop();
            } catch (IllegalStateException ignored) {
            }
            r.release();
        }
        asrNative.nativeStop(nativeHandle);
    }

    private void loop() {
        short[] buf = new short[SAMPLE_RATE / 5]; // 200ms
        while (running.get() && record != null) {
            int n = record.read(buf, 0, buf.length);
            if (n > 0) {
                asrNative.nativeFeedPcm16(nativeHandle, buf, 0, n);
            }
        }
    }
}
