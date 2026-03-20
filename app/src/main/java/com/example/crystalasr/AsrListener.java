package com.example.crystalasr;

/**
 * Called from the audio capture thread when the native layer finishes a voiced segment.
 * Use {@code runOnUiThread} (or a {@code Handler}) if you touch the UI.
 */
public interface AsrListener {
    void onRecognizedText(String text);
}
