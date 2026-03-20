#include <jni.h>
#include <android/log.h>
#include <whisper.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

#define LOG_TAG "libasr"
#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

namespace {

constexpr int kSampleRate = 16000;
// ~20 ms frames
constexpr int kFrameSamples = 320;
// RMS threshold (tune for mic sensitivity); rough energy gate as VAD placeholder
constexpr float kSpeechRmsThreshold = 800.0f;
// End of utterance after this many ms of "silence" while in speech
constexpr int kTrailingSilenceMs = 400;
// Minimum voiced duration before we consider it a segment (ms)
constexpr int kMinSpeechMs = 200;

inline float rmsFrame(const int16_t* p, int n) {
    if (n <= 0) return 0.f;
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        double v = static_cast<double>(p[i]);
        acc += v * v;
    }
    return static_cast<float>(std::sqrt(acc / static_cast<double>(n)));
}

struct Engine {
    std::mutex mutex;
    bool running = false;
    JavaVM* vm = nullptr;
    jobject callback = nullptr;
    jmethodID onTextMid = nullptr;

    std::string model_path;
    bool model_ready = false;
    whisper_context* whisper_ctx = nullptr;

    std::vector<int16_t> segment;
    bool in_speech = false;
    int trailing_silent_frames = 0;
    int voiced_frames = 0;

    void reset_segment_state() {
        segment.clear();
        in_speech = false;
        trailing_silent_frames = 0;
        voiced_frames = 0;
    }

    // Call only with mutex held. Extend with real session/teardown (interpreter, mmap, etc.).
    void unload_model_locked() {
        if (whisper_ctx) {
            whisper_free(whisper_ctx);
            whisper_ctx = nullptr;
        }
        model_path.clear();
        model_ready = false;
    }

    static void emit_transcription(JNIEnv* env, jobject callback, jmethodID mid, const std::string& text) {
        if (!callback || !mid || text.empty()) return;
        jstring jt = env->NewStringUTF(text.c_str());
        if (!jt) return;
        env->CallVoidMethod(callback, mid, jt);
        env->DeleteLocalRef(jt);
        if (env->ExceptionCheck()) {
            env->ExceptionDescribe();
            env->ExceptionClear();
        }
    }

    // VAD segmentation. Appends completed PCM16 speech segments to `completed`.
    void process_frames(const int16_t* samples, int sample_count, std::vector<std::vector<int16_t>>& completed) {
        int offset = 0;
        while (offset < sample_count) {
            int take = std::min(kFrameSamples, sample_count - offset);
            float rms = rmsFrame(samples + offset, take);
            bool voice = rms >= kSpeechRmsThreshold;

            if (voice) {
                if (!in_speech) {
                    in_speech = true;
                    segment.clear();
                    ALOGI("VAD: speech start (rms=%.1f)", rms);
                }
                trailing_silent_frames = 0;
                voiced_frames++;
                segment.insert(segment.end(), samples + offset, samples + offset + take);
            } else if (in_speech) {
                trailing_silent_frames++;
                segment.insert(segment.end(), samples + offset, samples + offset + take);
                int max_trail = (kTrailingSilenceMs * kSampleRate) / (1000 * kFrameSamples);
                int min_voice = (kMinSpeechMs * kSampleRate) / (1000 * kFrameSamples);
                if (trailing_silent_frames >= max_trail && voiced_frames >= min_voice) {
                    if (model_ready) {
                        ALOGI("VAD: speech end, enqueue segment (%zu samples)", segment.size());
                        completed.push_back(segment);
                    } else {
                        ALOGI("VAD: speech end, skipping inference (no model loaded)");
                    }
                    reset_segment_state();
                }
            }
            offset += take;
        }
    }
};

Engine* from_handle(jlong h) {
    return reinterpret_cast<Engine*>(static_cast<uintptr_t>(h));
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL Java_com_example_crystalasr_AsrNative_nativeCreate(JNIEnv* env, jobject /*thiz*/) {
    auto* e = new Engine();
    env->GetJavaVM(&e->vm);
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(e));
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_example_crystalasr_AsrNative_nativeLoadModel(JNIEnv* env, jobject /*thiz*/, jlong handle,
                                                                                          jstring jpath) {
    auto* e = from_handle(handle);
    if (!e) return JNI_FALSE;
    if (!jpath) return JNI_FALSE;

    const char* utf = env->GetStringUTFChars(jpath, nullptr);
    if (!utf) return JNI_FALSE;
    std::string path(utf);
    env->ReleaseStringUTFChars(jpath, utf);

    if (path.empty()) {
        ALOGI("nativeLoadModel: empty path");
        return JNI_FALSE;
    }

    bool ok = false;
    {
        std::lock_guard<std::mutex> lock(e->mutex);
        e->unload_model_locked();
        whisper_context_params cparams = whisper_context_default_params();
        cparams.use_gpu = false;
        cparams.flash_attn = false;
        whisper_context* ctx = whisper_init_from_file_with_params(path.c_str(), cparams);
        if (ctx) {
            e->whisper_ctx = ctx;
            e->model_path = std::move(path);
            e->model_ready = true;
            ok = true;
            ALOGI("ASR model initialized: %s", e->model_path.c_str());
        } else {
            ALOGI("ASR model load failed (whisper init failed): %s", path.c_str());
        }
    }
    return ok ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL Java_com_example_crystalasr_AsrNative_nativeUnloadModel(JNIEnv* /*env*/, jobject /*thiz*/, jlong handle) {
    auto* e = from_handle(handle);
    if (!e) return;
    std::lock_guard<std::mutex> lock(e->mutex);
    e->unload_model_locked();
    ALOGI("ASR model unloaded");
}

extern "C" JNIEXPORT void JNICALL Java_com_example_crystalasr_AsrNative_nativeDestroy(JNIEnv* env, jobject /*thiz*/, jlong handle) {
    auto* e = from_handle(handle);
    if (!e) return;
    std::lock_guard<std::mutex> lock(e->mutex);
    e->unload_model_locked();
    if (e->callback) {
        env->DeleteGlobalRef(e->callback);
        e->callback = nullptr;
    }
    e->onTextMid = nullptr;
    delete e;
}

extern "C" JNIEXPORT void JNICALL Java_com_example_crystalasr_AsrNative_nativeSetListener(JNIEnv* env, jobject /*thiz*/, jlong handle,
                                                                                          jobject listener) {
    auto* e = from_handle(handle);
    if (!e) return;
    std::lock_guard<std::mutex> lock(e->mutex);
    if (e->callback) {
        env->DeleteGlobalRef(e->callback);
        e->callback = nullptr;
        e->onTextMid = nullptr;
    }
    if (!listener) return;
    e->callback = env->NewGlobalRef(listener);
    jclass cls = env->GetObjectClass(listener);
    e->onTextMid = env->GetMethodID(cls, "onRecognizedText", "(Ljava/lang/String;)V");
    env->DeleteLocalRef(cls);
    if (!e->onTextMid) {
        ALOGI("Listener missing onRecognizedText(String); callback disabled");
        env->DeleteGlobalRef(e->callback);
        e->callback = nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL Java_com_example_crystalasr_AsrNative_nativeStart(JNIEnv* /*env*/, jobject /*thiz*/, jlong handle) {
    auto* e = from_handle(handle);
    if (!e) return;
    std::lock_guard<std::mutex> lock(e->mutex);
    e->running = true;
    e->reset_segment_state();
}

extern "C" JNIEXPORT void JNICALL Java_com_example_crystalasr_AsrNative_nativeStop(JNIEnv* /*env*/, jobject /*thiz*/, jlong handle) {
    auto* e = from_handle(handle);
    if (!e) return;
    std::lock_guard<std::mutex> lock(e->mutex);
    e->running = false;
    e->reset_segment_state();
}

extern "C" JNIEXPORT void JNICALL Java_com_example_crystalasr_AsrNative_nativeFeedPcm16(JNIEnv* env, jobject /*thiz*/, jlong handle,
                                                                                        jshortArray samples, jint offset, jint length) {
    auto* e = from_handle(handle);
    if (!e || length <= 0) return;

    std::vector<std::vector<int16_t>> completed;
    {
        std::lock_guard<std::mutex> lock(e->mutex);
        if (!e->running) return;

        jshort* ptr = env->GetShortArrayElements(samples, nullptr);
        if (!ptr) return;
        const int16_t* base = reinterpret_cast<const int16_t*>(ptr) + offset;
        e->process_frames(base, length, completed);
        env->ReleaseShortArrayElements(samples, ptr, JNI_ABORT);
    }

    jobject cb;
    jmethodID mid;
    whisper_context* ctx = nullptr;
    {
        std::lock_guard<std::mutex> lock(e->mutex);
        cb = e->callback;
        mid = e->onTextMid;
        ctx = e->whisper_ctx;
    }
    if (!ctx || !cb || !mid) return;

    const int n_threads = 2;
    for (const auto& seg16 : completed) {
        if (seg16.empty()) continue;

        std::vector<float> segf;
        segf.reserve(seg16.size());
        for (int16_t s : seg16) {
            segf.push_back(static_cast<float>(s) / 32768.0f);
        }

        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams.n_threads = n_threads;
        wparams.print_progress = false;
        wparams.print_realtime = false;
        wparams.print_timestamps = false;
        wparams.print_special = false;
        wparams.translate = false;
        wparams.no_context = true;
        wparams.single_segment = false;
        wparams.language = "en";

        if (whisper_full(ctx, wparams, segf.data(), static_cast<int>(segf.size())) != 0) {
            ALOGI("whisper_full failed for segment of %zu samples", seg16.size());
            continue;
        }

        int nseg = whisper_full_n_segments(ctx);
        std::string text;
        for (int i = 0; i < nseg; ++i) {
            const char* t = whisper_full_get_segment_text(ctx, i);
            if (t && t[0]) text += t;
        }
        if (!text.empty()) {
            Engine::emit_transcription(env, cb, mid, text);
        }
    }
}
