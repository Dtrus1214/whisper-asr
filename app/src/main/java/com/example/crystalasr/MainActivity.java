package com.example.crystalasr;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.atomic.AtomicBoolean;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import com.google.android.material.button.MaterialButton;

public class MainActivity extends AppCompatActivity implements AsrListener {
    private static final int REQ_RECORD_AUDIO = 1001;
    private static final String MODEL_ASSET_PATH = "asr/ggml-tiny.en.bin";
    private static final String MODEL_FILENAME = "ggml-tiny.en.bin";
    private static final String WAV_SIM_ASSET_PATH = "1.wav";
    private static final int SAMPLE_RATE = 16000;

    private final AsrNative asrNative = new AsrNative();
    private final AtomicBoolean simulateRunning = new AtomicBoolean(false);
    private long nativeHandle;
    private StreamingRecorder recorder;
    private Thread simulateThread;

    private MaterialButton btnListen;
    private MaterialButton btnStop;
    private MaterialButton btnSimulate;
    private TextView transcript;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        nativeHandle = asrNative.nativeCreate();
        asrNative.nativeSetListener(nativeHandle, this);

        File modelFile = new File(getFilesDir(), MODEL_FILENAME);
        if (!ensureModelFile(modelFile)) {
            Toast.makeText(this, R.string.asr_model_load_failed, Toast.LENGTH_LONG).show();
        }
        if (!asrNative.nativeLoadModel(nativeHandle, modelFile.getAbsolutePath())) {
            Toast.makeText(this, R.string.asr_model_load_failed, Toast.LENGTH_LONG).show();
        }

        btnListen = findViewById(R.id.btnListen);
        btnStop = findViewById(R.id.btnStop);
        btnSimulate = findViewById(R.id.btnSimulate);
        transcript = findViewById(R.id.transcript);

        btnListen.setOnClickListener(v -> onListenClicked());
        btnStop.setOnClickListener(v -> onStopClicked());
        btnSimulate.setOnClickListener(v -> onSimulateClicked());
    }

    @Override
    protected void onDestroy() {
        if (recorder != null) {
            recorder.stop();
            recorder = null;
        }
        stopSimulation();
        if (nativeHandle != 0) {
            asrNative.nativeDestroy(nativeHandle);
            nativeHandle = 0;
        }
        super.onDestroy();
    }

    private void onListenClicked() {
        if (recorder != null) return;

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                    this,
                    new String[]{Manifest.permission.RECORD_AUDIO},
                    REQ_RECORD_AUDIO
            );
            return;
        }
        startPipeline();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode != REQ_RECORD_AUDIO) return;
        if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startPipeline();
        } else {
            Toast.makeText(this, R.string.mic_permission_rationale, Toast.LENGTH_LONG).show();
        }
    }

    private void startPipeline() {
        if (recorder != null || simulateRunning.get()) return;
        try {
            recorder = new StreamingRecorder(asrNative, nativeHandle);
            recorder.start();
        } catch (IllegalStateException e) {
            Toast.makeText(this, "Could not start microphone: " + e.getMessage(), Toast.LENGTH_LONG).show();
            recorder = null;
            return;
        }
        setUiRunningState(true);
    }

    private void onStopClicked() {
        if (recorder != null) {
            recorder.stop();
            recorder = null;
        }
        stopSimulation();
        asrNative.nativeStop(nativeHandle);
        setUiRunningState(false);
    }

    @Override
    public void onRecognizedText(@NonNull String text) {
        runOnUiThread(() -> {
            CharSequence cur = transcript.getText();
            String next = cur.toString().trim().isEmpty() ? text : cur + "\n" + text;
            transcript.setText(next);
        });
    }

    private boolean ensureModelFile(@NonNull File target) {
        if (target.exists() && target.length() > 0) return true;
        File parent = target.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) return false;
        try (InputStream in = getAssets().open(MODEL_ASSET_PATH);
             OutputStream out = openFileOutput(MODEL_FILENAME, MODE_PRIVATE)) {
            byte[] buffer = new byte[8192];
            int read;
            while ((read = in.read(buffer)) != -1) {
                out.write(buffer, 0, read);
            }
            out.flush();
            return target.exists() && target.length() > 0;
        } catch (IOException ignored) {
            return false;
        }
    }

    private void onSimulateClicked() {
        if (recorder != null || simulateRunning.get()) return;
        simulateRunning.set(true);
        setUiRunningState(true);
        asrNative.nativeStart(nativeHandle);
        simulateThread = new Thread(() -> {
            boolean ok = runWavSimulationFromAssets();
            runOnUiThread(() -> {
                if (!ok) {
                    Toast.makeText(this, R.string.wav_simulation_failed, Toast.LENGTH_LONG).show();
                }
                asrNative.nativeStop(nativeHandle);
                stopSimulation();
                setUiRunningState(false);
            });
        }, "crystal-asr-simulate");
        simulateThread.start();
    }

    private void stopSimulation() {
        simulateRunning.set(false);
        Thread t = simulateThread;
        simulateThread = null;
        if (t != null && t != Thread.currentThread()) {
            t.interrupt();
        }
    }

    private void setUiRunningState(boolean running) {
        btnListen.setEnabled(!running);
        btnSimulate.setEnabled(!running);
        btnStop.setEnabled(running);
    }

    private boolean runWavSimulationFromAssets() {
        try (InputStream in = getAssets().open(WAV_SIM_ASSET_PATH)) {
            byte[] header = new byte[44];
            if (in.read(header) != 44) return false;
            if (!isSupportedWavHeader(header)) return false;

            byte[] pcmBytes = new byte[4096];
            while (simulateRunning.get()) {
                int read = in.read(pcmBytes);
                if (read <= 0) break;
                int evenRead = read & ~1;
                if (evenRead <= 0) continue;
                short[] pcm16 = new short[evenRead / 2];
                ByteBuffer.wrap(pcmBytes, 0, evenRead)
                        .order(ByteOrder.LITTLE_ENDIAN)
                        .asShortBuffer()
                        .get(pcm16);
                asrNative.nativeFeedPcm16(nativeHandle, pcm16, 0, pcm16.length);
            }
            return true;
        } catch (IOException e) {
            return false;
        }
    }

    private boolean isSupportedWavHeader(byte[] h) {
        if (h.length < 44) return false;
        if (!(h[0] == 'R' && h[1] == 'I' && h[2] == 'F' && h[3] == 'F')) return false;
        if (!(h[8] == 'W' && h[9] == 'A' && h[10] == 'V' && h[11] == 'E')) return false;
        if (!(h[12] == 'f' && h[13] == 'm' && h[14] == 't' && h[15] == ' ')) return false;
        int audioFormat = u16le(h, 20);
        int channels = u16le(h, 22);
        int sampleRate = u32le(h, 24);
        int bitsPerSample = u16le(h, 34);
        return audioFormat == 1 && channels == 1 && sampleRate == SAMPLE_RATE && bitsPerSample == 16;
    }

    private int u16le(byte[] b, int off) {
        return (b[off] & 0xff) | ((b[off + 1] & 0xff) << 8);
    }

    private int u32le(byte[] b, int off) {
        return (b[off] & 0xff)
                | ((b[off + 1] & 0xff) << 8)
                | ((b[off + 2] & 0xff) << 16)
                | ((b[off + 3] & 0xff) << 24);
    }
}
