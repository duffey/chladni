/*
 * Audio Reactive Chladni Plate Visualizer - Win32 C Port
 *
 * Build with:
 *   cl /O2 chladni.c /link opengl32.lib user32.lib gdi32.lib ole32.lib
 *
 * Or with MinGW:
 *   gcc -O2 chladni.c -o chladni.exe -lopengl32 -lgdi32 -lole32 -lm
 */

#define WIN32_LEAN_AND_MEAN
#define COBJMACROS
#define _USE_MATH_DEFINES

#include <windows.h>
#include <gl/gl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <initguid.h>
#include <mmdeviceapi.h>
#include <audioclient.h>

/* Define GUIDs for WASAPI interfaces */
DEFINE_GUID(CLSID_MMDeviceEnumerator, 0xbcde0395, 0xe52f, 0x467c,
            0x8e, 0x3d, 0xc4, 0x57, 0x92, 0x91, 0x69, 0x2e);
DEFINE_GUID(IID_IMMDeviceEnumerator,  0xa95664d2, 0x9614, 0x4f35,
            0xa7, 0x46, 0xde, 0x8d, 0xb6, 0x36, 0x17, 0xe6);
DEFINE_GUID(IID_IAudioClient,         0x1cb9ad4c, 0xdbfa, 0x4c32,
            0xb1, 0x78, 0xc2, 0xf5, 0x68, 0xa7, 0x03, 0xb2);
DEFINE_GUID(IID_IAudioCaptureClient,  0xc8adbd64, 0xe71e, 0x48a0,
            0xa4, 0xde, 0x18, 0x5c, 0x39, 0x5c, 0xd3, 0x17);

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "ole32.lib")

/* ============================================================================
 * OpenGL Extensions
 * ============================================================================ */

#define GL_ARRAY_BUFFER                   0x8892
#define GL_ELEMENT_ARRAY_BUFFER           0x8893
#define GL_STATIC_DRAW                    0x88E4
#define GL_DYNAMIC_DRAW                   0x88E8
#define GL_FRAGMENT_SHADER                0x8B30
#define GL_VERTEX_SHADER                  0x8B31
#define GL_COMPILE_STATUS                 0x8B81
#define GL_LINK_STATUS                    0x8B82
#define GL_INFO_LOG_LENGTH                0x8B84
#define GL_TEXTURE_BUFFER                 0x8C2A
#define GL_R32F                           0x822E
#define GL_TEXTURE0                       0x84C0

typedef char GLchar;
typedef ptrdiff_t GLsizeiptr;
typedef ptrdiff_t GLintptr;

typedef GLuint (APIENTRY *PFNGLCREATESHADERPROC)(GLenum type);
typedef void (APIENTRY *PFNGLSHADERSOURCEPROC)(GLuint shader, GLsizei count, const GLchar *const*string, const GLint *length);
typedef void (APIENTRY *PFNGLCOMPILESHADERPROC)(GLuint shader);
typedef void (APIENTRY *PFNGLGETSHADERIVPROC)(GLuint shader, GLenum pname, GLint *params);
typedef void (APIENTRY *PFNGLGETSHADERINFOLOGPROC)(GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
typedef GLuint (APIENTRY *PFNGLCREATEPROGRAMPROC)(void);
typedef void (APIENTRY *PFNGLATTACHSHADERPROC)(GLuint program, GLuint shader);
typedef void (APIENTRY *PFNGLLINKPROGRAMPROC)(GLuint program);
typedef void (APIENTRY *PFNGLGETPROGRAMIVPROC)(GLuint program, GLenum pname, GLint *params);
typedef void (APIENTRY *PFNGLGETPROGRAMINFOLOGPROC)(GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
typedef void (APIENTRY *PFNGLUSEPROGRAMPROC)(GLuint program);
typedef void (APIENTRY *PFNGLDELETESHADERPROC)(GLuint shader);
typedef GLint (APIENTRY *PFNGLGETUNIFORMLOCATIONPROC)(GLuint program, const GLchar *name);
typedef void (APIENTRY *PFNGLUNIFORM1FPROC)(GLint location, GLfloat v0);
typedef void (APIENTRY *PFNGLUNIFORM1IPROC)(GLint location, GLint v0);
typedef void (APIENTRY *PFNGLUNIFORM2FPROC)(GLint location, GLfloat v0, GLfloat v1);
typedef void (APIENTRY *PFNGLGENVERTEXARRAYSPROC)(GLsizei n, GLuint *arrays);
typedef void (APIENTRY *PFNGLBINDVERTEXARRAYPROC)(GLuint array);
typedef void (APIENTRY *PFNGLGENBUFFERSPROC)(GLsizei n, GLuint *buffers);
typedef void (APIENTRY *PFNGLBINDBUFFERPROC)(GLenum target, GLuint buffer);
typedef void (APIENTRY *PFNGLBUFFERDATAPROC)(GLenum target, GLsizeiptr size, const void *data, GLenum usage);
typedef void (APIENTRY *PFNGLBUFFERSUBDATAPROC)(GLenum target, GLintptr offset, GLsizeiptr size, const void *data);
typedef void (APIENTRY *PFNGLVERTEXATTRIBPOINTERPROC)(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer);
typedef void (APIENTRY *PFNGLENABLEVERTEXATTRIBARRAYPROC)(GLuint index);
typedef void (APIENTRY *PFNGLACTIVETEXTUREPROC)(GLenum texture);
typedef void (APIENTRY *PFNGLTEXBUFFERPROC)(GLenum target, GLenum internalformat, GLuint buffer);
typedef BOOL (APIENTRY *PFNWGLSWAPINTERVALEXTPROC)(int interval);
typedef HGLRC (APIENTRY *PFNWGLCREATECONTEXTATTRIBSARBPROC)(HDC hDC, HGLRC hShareContext, const int *attribList);

#define WGL_CONTEXT_MAJOR_VERSION_ARB     0x2091
#define WGL_CONTEXT_MINOR_VERSION_ARB     0x2092
#define WGL_CONTEXT_PROFILE_MASK_ARB      0x9126
#define WGL_CONTEXT_CORE_PROFILE_BIT_ARB  0x00000001

static PFNGLCREATESHADERPROC glCreateShader;
static PFNGLSHADERSOURCEPROC glShaderSource;
static PFNGLCOMPILESHADERPROC glCompileShader;
static PFNGLGETSHADERIVPROC glGetShaderiv;
static PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog;
static PFNGLCREATEPROGRAMPROC glCreateProgram;
static PFNGLATTACHSHADERPROC glAttachShader;
static PFNGLLINKPROGRAMPROC glLinkProgram;
static PFNGLGETPROGRAMIVPROC glGetProgramiv;
static PFNGLGETPROGRAMINFOLOGPROC glGetProgramInfoLog;
static PFNGLUSEPROGRAMPROC glUseProgram;
static PFNGLDELETESHADERPROC glDeleteShader;
static PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation;
static PFNGLUNIFORM1FPROC glUniform1f;
static PFNGLUNIFORM1IPROC glUniform1i;
static PFNGLUNIFORM2FPROC glUniform2f;
static PFNGLGENVERTEXARRAYSPROC glGenVertexArrays;
static PFNGLBINDVERTEXARRAYPROC glBindVertexArray;
static PFNGLGENBUFFERSPROC glGenBuffers;
static PFNGLBINDBUFFERPROC glBindBuffer;
static PFNGLBUFFERDATAPROC glBufferData;
static PFNGLBUFFERSUBDATAPROC glBufferSubData;
static PFNGLVERTEXATTRIBPOINTERPROC glVertexAttribPointer;
static PFNGLENABLEVERTEXATTRIBARRAYPROC glEnableVertexAttribArray;
static PFNGLACTIVETEXTUREPROC glActiveTexture;
static PFNGLTEXBUFFERPROC glTexBuffer;
static PFNWGLSWAPINTERVALEXTPROC wglSwapIntervalEXT;
static PFNWGLCREATECONTEXTATTRIBSARBPROC wglCreateContextAttribsARB;

static void LoadGLExtensions(void) {
    glCreateShader = (PFNGLCREATESHADERPROC)wglGetProcAddress("glCreateShader");
    glShaderSource = (PFNGLSHADERSOURCEPROC)wglGetProcAddress("glShaderSource");
    glCompileShader = (PFNGLCOMPILESHADERPROC)wglGetProcAddress("glCompileShader");
    glGetShaderiv = (PFNGLGETSHADERIVPROC)wglGetProcAddress("glGetShaderiv");
    glGetShaderInfoLog = (PFNGLGETSHADERINFOLOGPROC)wglGetProcAddress("glGetShaderInfoLog");
    glCreateProgram = (PFNGLCREATEPROGRAMPROC)wglGetProcAddress("glCreateProgram");
    glAttachShader = (PFNGLATTACHSHADERPROC)wglGetProcAddress("glAttachShader");
    glLinkProgram = (PFNGLLINKPROGRAMPROC)wglGetProcAddress("glLinkProgram");
    glGetProgramiv = (PFNGLGETPROGRAMIVPROC)wglGetProcAddress("glGetProgramiv");
    glGetProgramInfoLog = (PFNGLGETPROGRAMINFOLOGPROC)wglGetProcAddress("glGetProgramInfoLog");
    glUseProgram = (PFNGLUSEPROGRAMPROC)wglGetProcAddress("glUseProgram");
    glDeleteShader = (PFNGLDELETESHADERPROC)wglGetProcAddress("glDeleteShader");
    glGetUniformLocation = (PFNGLGETUNIFORMLOCATIONPROC)wglGetProcAddress("glGetUniformLocation");
    glUniform1f = (PFNGLUNIFORM1FPROC)wglGetProcAddress("glUniform1f");
    glUniform1i = (PFNGLUNIFORM1IPROC)wglGetProcAddress("glUniform1i");
    glUniform2f = (PFNGLUNIFORM2FPROC)wglGetProcAddress("glUniform2f");
    glGenVertexArrays = (PFNGLGENVERTEXARRAYSPROC)wglGetProcAddress("glGenVertexArrays");
    glBindVertexArray = (PFNGLBINDVERTEXARRAYPROC)wglGetProcAddress("glBindVertexArray");
    glGenBuffers = (PFNGLGENBUFFERSPROC)wglGetProcAddress("glGenBuffers");
    glBindBuffer = (PFNGLBINDBUFFERPROC)wglGetProcAddress("glBindBuffer");
    glBufferData = (PFNGLBUFFERDATAPROC)wglGetProcAddress("glBufferData");
    glBufferSubData = (PFNGLBUFFERSUBDATAPROC)wglGetProcAddress("glBufferSubData");
    glVertexAttribPointer = (PFNGLVERTEXATTRIBPOINTERPROC)wglGetProcAddress("glVertexAttribPointer");
    glEnableVertexAttribArray = (PFNGLENABLEVERTEXATTRIBARRAYPROC)wglGetProcAddress("glEnableVertexAttribArray");
    glActiveTexture = (PFNGLACTIVETEXTUREPROC)wglGetProcAddress("glActiveTexture");
    glTexBuffer = (PFNGLTEXBUFFERPROC)wglGetProcAddress("glTexBuffer");
    wglSwapIntervalEXT = (PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT");
    wglCreateContextAttribsARB = (PFNWGLCREATECONTEXTATTRIBSARBPROC)wglGetProcAddress("wglCreateContextAttribsARB");
}

/* ============================================================================
 * Constants
 * ============================================================================ */

#define SAMPLE_RATE     44100
#define CHUNK_SIZE      64
#define MAX_FFT_SIZE    32768
#define MAX_BINS        (MAX_FFT_SIZE / 2 + 1)
#define PI              3.14159265358979323846f

/* ============================================================================
 * Shaders
 * ============================================================================ */

static const char *VERTEX_SHADER_SRC =
    "#version 330 core\n"
    "layout(location = 0) in vec2 position;\n"
    "out vec2 fragCoord;\n"
    "uniform vec2 iResolution;\n"
    "void main() {\n"
    "    fragCoord = (position + 1.0) * 0.5 * iResolution;\n"
    "    gl_Position = vec4(position, 0.0, 1.0);\n"
    "}\n";

static const char *FRAGMENT_SHADER_SRC =
    "#version 330 core\n"
    "in vec2 fragCoord;\n"
    "out vec4 outColor;\n"
    "uniform vec2 iResolution;\n"
    "uniform samplerBuffer iSpectrum;\n"
    "uniform float iFundamental;\n"
    "uniform float iFreqPerBin;\n"
    "uniform int iNumBins;\n"
    "uniform float iComplexity;\n"
    "uniform float iMaxFreq;\n"
    "uniform float iDominantHue;\n"
    "uniform int iColorMode;\n"
    "uniform float iThreshold;\n"
    "\n"
    "#define PI 3.14159265\n"
    "\n"
    "vec3 hsv2rgb(vec3 c) {\n"
    "    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);\n"
    "    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);\n"
    "    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);\n"
    "}\n"
    "\n"
    "float chladni(vec2 p, float n, float m, float a) {\n"
    "    float kn = n * PI / a;\n"
    "    float km = m * PI / a;\n"
    "    return cos(kn * p.x) * cos(m * PI * p.y)\n"
    "         - cos(km * p.x) * cos(n * PI * p.y);\n"
    "}\n"
    "\n"
    "vec2 freqToMode(float freq, float a) {\n"
    "    float ratio = freq / iFundamental;\n"
    "    float lambda_1_2 = 1.0 / (a * a) + 4.0;\n"
    "    float target = lambda_1_2 * ratio * ratio;\n"
    "    if (target <= lambda_1_2) return vec2(1.0, 2.0);\n"
    "    float sqrtT = sqrt(target);\n"
    "    float n = max(1.0, floor(sqrtT * a * iComplexity));\n"
    "    float na = n / a;\n"
    "    float m2 = target - na * na;\n"
    "    float m = max(1.0, round(sqrt(max(0.0, m2))));\n"
    "    if (m == n) m = n + 1.0;\n"
    "    return vec2(n, m);\n"
    "}\n"
    "\n"
    "void main() {\n"
    "    float aspect = iResolution.x / iResolution.y;\n"
    "    vec2 plate = (fragCoord - 0.5 * iResolution.xy) / (0.5 * iResolution.y);\n"
    "    float displacement = 0.0;\n"
    "    int maxBin = min(iNumBins, int(iMaxFreq / iFreqPerBin));\n"
    "    for (int i = 1; i < maxBin; i++) {\n"
    "        float amp = texelFetch(iSpectrum, i).r;\n"
    "        float freq = float(i) * iFreqPerBin;\n"
    "        vec2 nm = freqToMode(freq, aspect);\n"
    "        displacement += amp * chladni(plate, nm.x, nm.y, aspect);\n"
    "    }\n"
    "    float d = abs(displacement);\n"
    "    float fw = fwidth(displacement) * 1.5;\n"
    "    float intensity = 1.0 - smoothstep(iThreshold - fw, iThreshold + fw, d);\n"
    "    vec3 color = iColorMode == 0 ? hsv2rgb(vec3(iDominantHue, 0.85, 1.0)) : vec3(1.0);\n"
    "    outColor = vec4(color * intensity, 1.0);\n"
    "}\n";

/* ============================================================================
 * Audio Capture (WASAPI Loopback)
 * ============================================================================ */

typedef struct {
    IAudioClient *pAudioClient;
    IAudioCaptureClient *pCaptureClient;
    WAVEFORMATEX *pwfx;
    HANDLE hThread;
    volatile BOOL running;
    volatile BOOL paused;

    float ringBuffer[MAX_FFT_SIZE * 4];
    volatile LONG writePos;

    int fftSize;
    float window[MAX_FFT_SIZE];
    float spectrum[MAX_BINS];
    float smoothSpectrum[MAX_BINS];
    float runningMax;
    float output[MAX_BINS];

    CRITICAL_SECTION cs;
} AudioCapture;

static AudioCapture g_audio;

/* Simple radix-2 FFT - Cooley-Tukey algorithm */
static void fft_complex(float *real, float *imag, int n) {
    /* Bit-reversal permutation */
    int i, j, k;
    for (i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = real[i], ti = imag[i];
            real[i] = real[j]; imag[i] = imag[j];
            real[j] = tr; imag[j] = ti;
        }
    }

    /* Cooley-Tukey iterative FFT */
    for (int len = 2; len <= n; len <<= 1) {
        float angle = -2.0f * PI / len;
        float wpr = cosf(angle);
        float wpi = sinf(angle);
        for (i = 0; i < n; i += len) {
            float wr = 1.0f, wi = 0.0f;
            for (j = 0; j < len / 2; j++) {
                int u = i + j;
                int v = i + j + len / 2;
                float tr = wr * real[v] - wi * imag[v];
                float ti = wr * imag[v] + wi * real[v];
                real[v] = real[u] - tr;
                imag[v] = imag[u] - ti;
                real[u] += tr;
                imag[u] += ti;
                float wt = wr;
                wr = wr * wpr - wi * wpi;
                wi = wt * wpi + wi * wpr;
            }
        }
    }
}

static void AudioUpdateFFTParams(AudioCapture *a) {
    int nBins = a->fftSize / 2 + 1;
    for (int i = 0; i < a->fftSize; i++) {
        a->window[i] = 0.5f * (1.0f - cosf(2.0f * PI * i / (a->fftSize - 1)));
    }
    memset(a->spectrum, 0, sizeof(a->spectrum));
    memset(a->smoothSpectrum, 0, sizeof(a->smoothSpectrum));
    a->runningMax = 0.01f;
}

static DWORD WINAPI AudioCaptureThread(LPVOID param) {
    AudioCapture *a = (AudioCapture *)param;
    HRESULT hr;

    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);

    __try {
        while (a->running) {
            /* When paused, just sleep and discard audio data */
            if (a->paused) {
                Sleep(50);
                /* Drain buffer to prevent buildup */
                if (a->pCaptureClient) {
                    UINT32 packetLength = 0;
                    hr = IAudioCaptureClient_GetNextPacketSize(a->pCaptureClient, &packetLength);
                    while (SUCCEEDED(hr) && packetLength > 0) {
                        BYTE *pData = NULL;
                        UINT32 numFrames = 0;
                        DWORD flags = 0;
                        hr = IAudioCaptureClient_GetBuffer(a->pCaptureClient, &pData, &numFrames, &flags, NULL, NULL);
                        if (SUCCEEDED(hr)) {
                            IAudioCaptureClient_ReleaseBuffer(a->pCaptureClient, numFrames);
                        }
                        hr = IAudioCaptureClient_GetNextPacketSize(a->pCaptureClient, &packetLength);
                    }
                }
                continue;
            }

            if (!a->pCaptureClient) break;

            /* Polling mode - check for available data */
            UINT32 packetLength = 0;
            hr = IAudioCaptureClient_GetNextPacketSize(a->pCaptureClient, &packetLength);

            if (SUCCEEDED(hr) && packetLength > 0) {
                while (SUCCEEDED(hr) && packetLength > 0 && a->running && !a->paused) {
                    BYTE *pData = NULL;
                    UINT32 numFrames = 0;
                    DWORD flags = 0;

                    hr = IAudioCaptureClient_GetBuffer(a->pCaptureClient, &pData, &numFrames, &flags, NULL, NULL);
                    if (FAILED(hr) || !pData) break;

                    float *samples = (float *)pData;
                    int channels = a->pwfx->nChannels;
                    int bufLen = MAX_FFT_SIZE * 4;

                    EnterCriticalSection(&a->cs);
                    for (UINT32 i = 0; i < numFrames; i++) {
                        float sample = 0.0f;
                        for (int c = 0; c < channels; c++) {
                            sample += samples[i * channels + c];
                        }
                        sample /= channels;

                        int pos = a->writePos % bufLen;
                        a->ringBuffer[pos] = sample;
                        a->writePos = (a->writePos + 1) % bufLen;
                    }
                    LeaveCriticalSection(&a->cs);

                    hr = IAudioCaptureClient_ReleaseBuffer(a->pCaptureClient, numFrames);
                    if (FAILED(hr)) break;
                    hr = IAudioCaptureClient_GetNextPacketSize(a->pCaptureClient, &packetLength);
                }
            } else {
                /* No data available, sleep briefly */
                Sleep(1);
            }
        }
    } __except(EXCEPTION_EXECUTE_HANDLER) {
        printf("\nAudio thread crashed!\n");
        fflush(stdout);
    }

    return 0;
}

static BOOL AudioInit(AudioCapture *a, int fftSize) {
    HRESULT hr;
    IMMDeviceEnumerator *pEnumerator = NULL;
    IMMDevice *pDevice = NULL;

    memset(a, 0, sizeof(*a));
    a->fftSize = fftSize;
    a->runningMax = 0.01f;
    a->paused = FALSE;
    InitializeCriticalSection(&a->cs);
    AudioUpdateFFTParams(a);

    hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    if (FAILED(hr) && hr != S_FALSE && hr != RPC_E_CHANGED_MODE) {
        printf("CoInitializeEx failed: 0x%08lx\n", hr);
        return FALSE;
    }

    hr = CoCreateInstance(&CLSID_MMDeviceEnumerator, NULL, CLSCTX_ALL,
                          &IID_IMMDeviceEnumerator, (void **)&pEnumerator);
    if (FAILED(hr)) {
        printf("CoCreateInstance failed: 0x%08lx\n", hr);
        return FALSE;
    }

    hr = IMMDeviceEnumerator_GetDefaultAudioEndpoint(pEnumerator, eRender, eConsole, &pDevice);
    IMMDeviceEnumerator_Release(pEnumerator);
    if (FAILED(hr)) {
        printf("GetDefaultAudioEndpoint failed: 0x%08lx\n", hr);
        return FALSE;
    }

    hr = IMMDevice_Activate(pDevice, &IID_IAudioClient, CLSCTX_ALL, NULL, (void **)&a->pAudioClient);
    IMMDevice_Release(pDevice);
    if (FAILED(hr)) {
        printf("Activate failed: 0x%08lx\n", hr);
        return FALSE;
    }

    hr = IAudioClient_GetMixFormat(a->pAudioClient, &a->pwfx);
    if (FAILED(hr)) {
        printf("GetMixFormat failed: 0x%08lx\n", hr);
        return FALSE;
    }

    /* Use polling mode instead of event-driven (more stable with focus changes) */
    hr = IAudioClient_Initialize(a->pAudioClient, AUDCLNT_SHAREMODE_SHARED,
                                  AUDCLNT_STREAMFLAGS_LOOPBACK,
                                  10000000, 0, a->pwfx, NULL);
    if (FAILED(hr)) {
        printf("Initialize failed: 0x%08lx\n", hr);
        return FALSE;
    }

    hr = IAudioClient_GetService(a->pAudioClient, &IID_IAudioCaptureClient, (void **)&a->pCaptureClient);
    if (FAILED(hr)) {
        printf("GetService failed: 0x%08lx\n", hr);
        return FALSE;
    }

    a->running = TRUE;
    a->hThread = CreateThread(NULL, 0, AudioCaptureThread, a, 0, NULL);

    hr = IAudioClient_Start(a->pAudioClient);
    if (FAILED(hr)) {
        printf("Start failed: 0x%08lx\n", hr);
        return FALSE;
    }

    printf("Audio initialized: %d Hz, %d channels\n", a->pwfx->nSamplesPerSec, a->pwfx->nChannels);
    return TRUE;
}

static void AudioStop(AudioCapture *a) {
    a->running = FALSE;
    if (a->hThread) {
        WaitForSingleObject(a->hThread, 1000);
        CloseHandle(a->hThread);
    }
    if (a->pAudioClient) IAudioClient_Stop(a->pAudioClient);
    if (a->pCaptureClient) IAudioCaptureClient_Release(a->pCaptureClient);
    if (a->pAudioClient) IAudioClient_Release(a->pAudioClient);
    if (a->pwfx) CoTaskMemFree(a->pwfx);
    DeleteCriticalSection(&a->cs);
}

static void AudioSetFFTSize(AudioCapture *a, int size) {
    if (size < 256) size = 256;
    if (size > MAX_FFT_SIZE) size = MAX_FFT_SIZE;
    if (size != a->fftSize) {
        EnterCriticalSection(&a->cs);
        a->fftSize = size;
        AudioUpdateFFTParams(a);
        LeaveCriticalSection(&a->cs);
    }
}

static float *AudioGetData(AudioCapture *a) {
    static float fftReal[MAX_FFT_SIZE];
    static float fftImag[MAX_FFT_SIZE];

    int fftSize = a->fftSize;
    int nBins = fftSize / 2 + 1;
    int bufLen = MAX_FFT_SIZE * 4;

    EnterCriticalSection(&a->cs);
    int currentPos = a->writePos;
    int start = (currentPos - fftSize + bufLen) % bufLen;

    for (int i = 0; i < fftSize; i++) {
        int idx = (start + i) % bufLen;
        fftReal[i] = a->ringBuffer[idx] * a->window[i];
        fftImag[i] = 0.0f;
    }
    LeaveCriticalSection(&a->cs);

    fft_complex(fftReal, fftImag, fftSize);

    for (int i = 0; i < nBins; i++) {
        a->spectrum[i] = sqrtf(fftReal[i] * fftReal[i] + fftImag[i] * fftImag[i]);
    }

    float currentMax = 0.0f;
    for (int i = 0; i < nBins; i++) {
        if (a->spectrum[i] > currentMax) currentMax = a->spectrum[i];
    }

    if (currentMax > a->runningMax) {
        a->runningMax = currentMax;
    } else {
        a->runningMax = a->runningMax * 0.995f;
        if (a->runningMax < currentMax) a->runningMax = currentMax;
        if (a->runningMax < 0.01f) a->runningMax = 0.01f;
    }

    for (int i = 0; i < nBins; i++) {
        float normalized = a->spectrum[i] / (a->runningMax + 1e-6f);
        if (normalized > 1.5f) normalized = 1.5f;
        if (normalized < 0.0f) normalized = 0.0f;

        if (normalized > a->smoothSpectrum[i]) {
            a->smoothSpectrum[i] = normalized;
        } else {
            a->smoothSpectrum[i] = a->smoothSpectrum[i] * 0.85f + normalized * 0.15f;
        }
    }

    memset(a->output, 0, sizeof(a->output));
    for (int i = 0; i < nBins; i++) {
        a->output[i] = a->smoothSpectrum[i];
    }

    return a->output;
}

/* ============================================================================
 * Visualizer
 * ============================================================================ */

typedef struct {
    int w, h;
    float fundamental;
    float complexity;
    float maxFreq;
    float threshold;
    int colorMode;
    float smoothHue;

    GLuint program;
    GLuint vao;
    GLuint spectrumTbo;
    GLuint spectrumTex;

    GLint locResolution;
    GLint locSpectrum;
    GLint locFundamental;
    GLint locFreqPerBin;
    GLint locNumBins;
    GLint locComplexity;
    GLint locMaxFreq;
    GLint locDominantHue;
    GLint locColorMode;
    GLint locThreshold;
} Visualizer;

static Visualizer g_viz;

static GLuint CompileShader(GLenum type, const char *source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (!status) {
        char log[512];
        glGetShaderInfoLog(shader, 512, NULL, log);
        printf("Shader compile error: %s\n", log);
        return 0;
    }
    return shader;
}

static BOOL VizInit(Visualizer *v, int w, int h) {
    v->w = w;
    v->h = h;
    v->fundamental = 100.0f;
    v->complexity = 0.4f;
    v->maxFreq = 1100.0f;
    v->threshold = 0.1f;
    v->colorMode = 0;
    v->smoothHue = 0.0f;

    GLuint vs = CompileShader(GL_VERTEX_SHADER, VERTEX_SHADER_SRC);
    GLuint fs = CompileShader(GL_FRAGMENT_SHADER, FRAGMENT_SHADER_SRC);
    if (!vs || !fs) return FALSE;

    v->program = glCreateProgram();
    glAttachShader(v->program, vs);
    glAttachShader(v->program, fs);
    glLinkProgram(v->program);

    GLint status;
    glGetProgramiv(v->program, GL_LINK_STATUS, &status);
    if (!status) {
        char log[512];
        glGetProgramInfoLog(v->program, 512, NULL, log);
        printf("Program link error: %s\n", log);
        return FALSE;
    }

    glDeleteShader(vs);
    glDeleteShader(fs);

    float verts[] = { -1, -1, 1, -1, 1, 1, -1, 1 };
    unsigned int inds[] = { 0, 1, 2, 2, 3, 0 };

    glGenVertexArrays(1, &v->vao);
    GLuint vbo, ebo;
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);

    glBindVertexArray(v->vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(inds), inds, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, NULL);
    glEnableVertexAttribArray(0);

    glGenBuffers(1, &v->spectrumTbo);
    glGenTextures(1, &v->spectrumTex);

    float initialData[MAX_BINS] = {0};
    glBindBuffer(GL_TEXTURE_BUFFER, v->spectrumTbo);
    glBufferData(GL_TEXTURE_BUFFER, sizeof(initialData), initialData, GL_DYNAMIC_DRAW);

    glBindTexture(GL_TEXTURE_BUFFER, v->spectrumTex);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, v->spectrumTbo);

    v->locResolution = glGetUniformLocation(v->program, "iResolution");
    v->locSpectrum = glGetUniformLocation(v->program, "iSpectrum");
    v->locFundamental = glGetUniformLocation(v->program, "iFundamental");
    v->locFreqPerBin = glGetUniformLocation(v->program, "iFreqPerBin");
    v->locNumBins = glGetUniformLocation(v->program, "iNumBins");
    v->locComplexity = glGetUniformLocation(v->program, "iComplexity");
    v->locMaxFreq = glGetUniformLocation(v->program, "iMaxFreq");
    v->locDominantHue = glGetUniformLocation(v->program, "iDominantHue");
    v->locColorMode = glGetUniformLocation(v->program, "iColorMode");
    v->locThreshold = glGetUniformLocation(v->program, "iThreshold");

    return TRUE;
}

static void VizRender(Visualizer *v, float *spectrum, int fftSize) {
    int nBins = fftSize / 2 + 1;
    float freqPerBin = (float)SAMPLE_RATE / fftSize;

    /* Compute dominant hue */
    int maxBin = (int)(v->maxFreq / freqPerBin);
    if (maxBin > nBins) maxBin = nBins;

    float totalAmp = 0.0f;
    float weightedFreq = 0.0f;
    for (int i = 1; i < maxBin; i++) {
        float freq = i * freqPerBin;
        float amp = spectrum[i];
        totalAmp += amp;
        weightedFreq += freq * amp;
    }

    float avgFreq = weightedFreq / (totalAmp + 1e-6f);
    float targetHue = 0.8f * avgFreq / v->maxFreq;
    if (targetHue < 0.0f) targetHue = 0.0f;
    if (targetHue > 0.8f) targetHue = 0.8f;
    v->smoothHue = v->smoothHue * 0.85f + targetHue * 0.15f;

    /* Update spectrum buffer */
    glBindBuffer(GL_TEXTURE_BUFFER, v->spectrumTbo);
    glBufferSubData(GL_TEXTURE_BUFFER, 0, MAX_BINS * sizeof(float), spectrum);

    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(v->program);

    glUniform2f(v->locResolution, (float)v->w, (float)v->h);
    glUniform1f(v->locFundamental, v->fundamental);
    glUniform1f(v->locFreqPerBin, freqPerBin);
    glUniform1i(v->locNumBins, nBins);
    glUniform1f(v->locComplexity, v->complexity);
    glUniform1f(v->locMaxFreq, v->maxFreq);
    glUniform1f(v->locDominantHue, v->smoothHue);
    glUniform1i(v->locColorMode, v->colorMode);
    glUniform1f(v->locThreshold, v->threshold);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_BUFFER, v->spectrumTex);
    glUniform1i(v->locSpectrum, 0);

    glBindVertexArray(v->vao);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, NULL);
}

static void PrintStatus(Visualizer *v, int fftSize) {
    int nBins = fftSize / 2 + 1;
    printf("\rF=%.1fHz  C=%.2f  MaxF=%.0fHz  T=%.2f  FFT=%d (%d bins)                    ",
           v->fundamental, v->complexity, v->maxFreq, v->threshold, fftSize, nBins);
    fflush(stdout);
}

/* ============================================================================
 * Win32 Window
 * ============================================================================ */

static HWND g_hwnd = NULL;
static HDC g_hdc;
static HGLRC g_hglrc;
static BOOL g_fullscreen = FALSE;
static RECT g_windowedRect;
static DWORD g_windowedStyle;

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_SIZE:
            if (wParam != SIZE_MINIMIZED && hwnd == g_hwnd && g_hglrc) {
                g_viz.w = LOWORD(lParam);
                g_viz.h = HIWORD(lParam);
            }
            break;
        case WM_DESTROY:
            if (hwnd == g_hwnd) {
                PostQuitMessage(0);
            }
            return 0;
        case WM_CLOSE:
            if (hwnd == g_hwnd) {
                PostQuitMessage(0);
            }
            return 0;
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

static void ToggleFullscreen(void) {
    if (g_fullscreen) {
        /* Go windowed */
        SetWindowLong(g_hwnd, GWL_STYLE, g_windowedStyle);
        SetWindowPos(g_hwnd, NULL, g_windowedRect.left, g_windowedRect.top,
                     g_windowedRect.right - g_windowedRect.left,
                     g_windowedRect.bottom - g_windowedRect.top,
                     SWP_FRAMECHANGED | SWP_NOZORDER);
        g_fullscreen = FALSE;
    } else {
        /* Save windowed state */
        g_windowedStyle = GetWindowLong(g_hwnd, GWL_STYLE);
        GetWindowRect(g_hwnd, &g_windowedRect);

        /* Go borderless fullscreen */
        MONITORINFO mi = { sizeof(mi) };
        GetMonitorInfo(MonitorFromWindow(g_hwnd, MONITOR_DEFAULTTOPRIMARY), &mi);
        SetWindowLong(g_hwnd, GWL_STYLE, WS_POPUP | WS_VISIBLE);
        SetWindowPos(g_hwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top,
                     mi.rcMonitor.right - mi.rcMonitor.left,
                     mi.rcMonitor.bottom - mi.rcMonitor.top,
                     SWP_FRAMECHANGED);
        g_fullscreen = TRUE;
    }
}

static BOOL CreateOpenGLContext(int w, int h) {
    WNDCLASS wc = {0};
    wc.style = CS_OWNDC;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = GetModuleHandle(NULL);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.lpszClassName = "ChladniClass";
    RegisterClass(&wc);

    /* Create temporary window for extension loading */
    HWND tempHwnd = CreateWindow("ChladniClass", "temp", WS_POPUP, 0, 0, 1, 1, NULL, NULL, wc.hInstance, NULL);
    HDC tempDC = GetDC(tempHwnd);

    PIXELFORMATDESCRIPTOR pfd = {
        sizeof(PIXELFORMATDESCRIPTOR), 1,
        PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,
        PFD_TYPE_RGBA, 32,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        24, 8, 0, PFD_MAIN_PLANE, 0, 0, 0, 0
    };

    int pf = ChoosePixelFormat(tempDC, &pfd);
    SetPixelFormat(tempDC, pf, &pfd);
    HGLRC tempRC = wglCreateContext(tempDC);
    wglMakeCurrent(tempDC, tempRC);

    LoadGLExtensions();

    wglMakeCurrent(NULL, NULL);
    wglDeleteContext(tempRC);
    ReleaseDC(tempHwnd, tempDC);
    DestroyWindow(tempHwnd);

    /* Create real window */
    RECT rect = { 0, 0, w, h };
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);

    int screenW = GetSystemMetrics(SM_CXSCREEN);
    int screenH = GetSystemMetrics(SM_CYSCREEN);
    int winW = rect.right - rect.left;
    int winH = rect.bottom - rect.top;

    g_hwnd = CreateWindow("ChladniClass", "Chladni", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                          (screenW - winW) / 2, (screenH - winH) / 2, winW, winH,
                          NULL, NULL, wc.hInstance, NULL);
    if (!g_hwnd) return FALSE;

    g_hdc = GetDC(g_hwnd);
    pf = ChoosePixelFormat(g_hdc, &pfd);
    SetPixelFormat(g_hdc, pf, &pfd);

    /* Create OpenGL 3.3 core context */
    if (wglCreateContextAttribsARB) {
        int attribs[] = {
            WGL_CONTEXT_MAJOR_VERSION_ARB, 3,
            WGL_CONTEXT_MINOR_VERSION_ARB, 3,
            WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
            0
        };
        g_hglrc = wglCreateContextAttribsARB(g_hdc, NULL, attribs);
    } else {
        g_hglrc = wglCreateContext(g_hdc);
    }

    if (!g_hglrc) return FALSE;
    wglMakeCurrent(g_hdc, g_hglrc);

    /* Enable vsync */
    if (wglSwapIntervalEXT) {
        wglSwapIntervalEXT(1);
    }

    return TRUE;
}

static void DestroyOpenGLContext(void) {
    wglMakeCurrent(NULL, NULL);
    if (g_hglrc) wglDeleteContext(g_hglrc);
    if (g_hdc) ReleaseDC(g_hwnd, g_hdc);
    if (g_hwnd) DestroyWindow(g_hwnd);
}

/* ============================================================================
 * Main
 * ============================================================================ */


int main(int argc, char *argv[]) {
    /* Initialize COM on main thread as STA (required for WASAPI callbacks) */
    CoInitializeEx(NULL, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);

    int winW = 1280, winH = 720;

    if (!CreateOpenGLContext(winW, winH)) {
        printf("Failed to create OpenGL context\n");
        return 1;
    }

    if (!VizInit(&g_viz, winW, winH)) {
        printf("Failed to initialize visualizer\n");
        DestroyOpenGLContext();
        return 1;
    }

    if (!AudioInit(&g_audio, 4096)) {
        printf("Failed to initialize audio capture\n");
        DestroyOpenGLContext();
        return 1;
    }

    Sleep(50);

    printf("\nControls:\n");
    printf("  UP/DOWN     = fundamental +/-2%%    PGUP/PGDN = fundamental +/-10%%\n");
    printf("  W/S         = complexity +/-0.01   A/D       = max freq +/-500 Hz\n");
    printf("  Q/E         = threshold +/-0.01    LEFT/RIGHT = halve/double FFT\n");
    printf("  C           = toggle color/B&W     ALT+ENTER  = toggle fullscreen\n");
    printf("  ESC         = quit\n\n");

    PrintStatus(&g_viz, g_audio.fftSize);

    BOOL running = TRUE;
    DWORD lastKeyTime = 0;
    DWORD repeatDelay = 150;
    BOOL prevLeft = FALSE, prevRight = FALSE, prevC = FALSE;
    BOOL prevAltEnter = FALSE;

    while (running) {
        MSG msg;
        while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) {
                running = FALSE;
            }
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        if (!running) break;

        DWORD now = GetTickCount();
        BOOL needUpdate = FALSE;

        /* Only process keyboard input when window is focused */
        BOOL hasFocus = (GetForegroundWindow() == g_hwnd);

        if (hasFocus) {
            /* ESC to quit */
            if (GetAsyncKeyState(VK_ESCAPE) & 0x8000) {
                printf("\nESC pressed - exiting\n");
                fflush(stdout);
                running = FALSE;
                continue;
            }

            /* Alt+Enter for fullscreen */
            BOOL altHeld = (GetAsyncKeyState(VK_MENU) & 0x8000) != 0;
            BOOL enterPressed = (GetAsyncKeyState(VK_RETURN) & 0x8000) != 0;
            BOOL currAltEnter = altHeld && enterPressed;
            if (currAltEnter && !prevAltEnter) {
                ToggleFullscreen();
            }
            prevAltEnter = currAltEnter;

            /* Fundamental frequency */
            if ((GetAsyncKeyState(VK_UP) & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_viz.fundamental *= 1.02f;
                if (g_viz.fundamental > 10000.0f) g_viz.fundamental = 10000.0f;
                needUpdate = TRUE;
                lastKeyTime = now;
            }
            if ((GetAsyncKeyState(VK_DOWN) & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_viz.fundamental /= 1.02f;
                if (g_viz.fundamental < 1.0f) g_viz.fundamental = 1.0f;
                needUpdate = TRUE;
                lastKeyTime = now;
            }
            if ((GetAsyncKeyState(VK_PRIOR) & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_viz.fundamental *= 1.1f;
                if (g_viz.fundamental > 10000.0f) g_viz.fundamental = 10000.0f;
                needUpdate = TRUE;
                lastKeyTime = now;
            }
            if ((GetAsyncKeyState(VK_NEXT) & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_viz.fundamental /= 1.1f;
                if (g_viz.fundamental < 1.0f) g_viz.fundamental = 1.0f;
                needUpdate = TRUE;
                lastKeyTime = now;
            }

            /* Complexity */
            if ((GetAsyncKeyState('W') & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_viz.complexity += 0.01f;
                if (g_viz.complexity > 1.0f) g_viz.complexity = 1.0f;
                needUpdate = TRUE;
                lastKeyTime = now;
            }
            if ((GetAsyncKeyState('S') & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_viz.complexity -= 0.01f;
                if (g_viz.complexity < 0.01f) g_viz.complexity = 0.01f;
                needUpdate = TRUE;
                lastKeyTime = now;
            }

            /* Max frequency */
            if ((GetAsyncKeyState('D') & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_viz.maxFreq += 500.0f;
                if (g_viz.maxFreq > 20000.0f) g_viz.maxFreq = 20000.0f;
                needUpdate = TRUE;
                lastKeyTime = now;
            }
            if ((GetAsyncKeyState('A') & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_viz.maxFreq -= 500.0f;
                if (g_viz.maxFreq < 100.0f) g_viz.maxFreq = 100.0f;
                needUpdate = TRUE;
                lastKeyTime = now;
            }

            /* Threshold */
            if ((GetAsyncKeyState('E') & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_viz.threshold += 0.01f;
                if (g_viz.threshold > 1.0f) g_viz.threshold = 1.0f;
                needUpdate = TRUE;
                lastKeyTime = now;
            }
            if ((GetAsyncKeyState('Q') & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_viz.threshold -= 0.01f;
                if (g_viz.threshold < 0.01f) g_viz.threshold = 0.01f;
                needUpdate = TRUE;
                lastKeyTime = now;
            }

            /* FFT size (edge-triggered) */
            BOOL currRight = (GetAsyncKeyState(VK_RIGHT) & 0x8000) != 0;
            BOOL currLeft = (GetAsyncKeyState(VK_LEFT) & 0x8000) != 0;
            if (currRight && !prevRight) {
                AudioSetFFTSize(&g_audio, g_audio.fftSize * 2);
                needUpdate = TRUE;
            }
            if (currLeft && !prevLeft) {
                AudioSetFFTSize(&g_audio, g_audio.fftSize / 2);
                needUpdate = TRUE;
            }
            prevRight = currRight;
            prevLeft = currLeft;

            /* Color mode toggle (edge-triggered) */
            BOOL currC = (GetAsyncKeyState('C') & 0x8000) != 0;
            if (currC && !prevC) {
                g_viz.colorMode = 1 - g_viz.colorMode;
            }
            prevC = currC;

            if (needUpdate) {
                PrintStatus(&g_viz, g_audio.fftSize);
            }
        } else {
            /* Reset edge-trigger states when not focused */
            prevRight = FALSE;
            prevLeft = FALSE;
            prevC = FALSE;
            prevAltEnter = FALSE;
        }

        /* Update viewport if window was resized */
        static int lastW = 0, lastH = 0;
        if (g_viz.w != lastW || g_viz.h != lastH) {
            if (g_viz.w > 0 && g_viz.h > 0) {
                glViewport(0, 0, g_viz.w, g_viz.h);
                lastW = g_viz.w;
                lastH = g_viz.h;
            }
        }

        /* Skip rendering if minimized */
        if (g_viz.w <= 0 || g_viz.h <= 0) {
            Sleep(16);
            continue;
        }

        /* Render */
        if (g_hdc && wglGetCurrentContext()) {
            float *spectrum = AudioGetData(&g_audio);
            if (spectrum) {
                VizRender(&g_viz, spectrum, g_audio.fftSize);
                SwapBuffers(g_hdc);
            }
        } else {
            Sleep(16);
        }
    }

    printf("\n");
    AudioStop(&g_audio);
    DestroyOpenGLContext();
    CoUninitialize();

    return 0;
}

