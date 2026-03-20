package com.example.crystalasr;

/**
 * JNI bridge to {@code libasr.so}. Replace the C++ implementation with your VAD + ASR stack;
 * keep this Java API stable so the app layer does not change.
 */
public final class AsrNative {
    static {
        System.loadLibrary("asr");
    }

    public native long nativeCreate();

    public native void nativeDestroy(long handle);

    /**
     * Load and initialize the ASR model from a readable file on disk (absolute path).
     * Typical pattern: copy a bundled asset to {@code getFilesDir()} or {@code getCacheDir()}, then pass that path.
     *
     * @return {@code true} if the native engine initialized successfully for that path
     */
    public native boolean nativeLoadModel(long handle, String modelPath);

    /** Release model resources; safe to call while stopped. {@link #nativeDestroy} also unloads. */
    public native void nativeUnloadModel(long handle);

    /** Pass null to clear. Listener must define {@code void onRecognizedText(String)}. */
    public native void nativeSetListener(long handle, AsrListener listener);

    public native void nativeStart(long handle);

    public native void nativeStop(long handle);

    /** Little-endian PCM16 mono, sample rate must match native ({@code 16000} in stub). */
    public native void nativeFeedPcm16(long handle, short[] samples, int offset, int length);
}
