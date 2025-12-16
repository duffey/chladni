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
    "\n"
    "uniform vec2 iResolution;\n"
    "uniform samplerBuffer iSpectrum;\n"
    "uniform int iNumBins;\n"
    "uniform float iFreqPerBin;\n"
    "uniform float iMaxFreq;\n"
    "uniform float iBaseFreq;\n"
    "uniform float iModeScale;\n"
    "uniform float iContrast;\n"
    "uniform int iColorMode;\n"
    "uniform float iTime;\n"
    "uniform int iBoundary;\n"
    "uniform int iAspectMode;\n"
    "\n"
    "#define PI 3.14159265\n"
    "#define TAU 6.28318530\n"
    "\n"
    "// ============================================================================\n"
    "// COLORMAPS\n"
    "// ============================================================================\n"
    "\n"
    "vec3 plasma(float t) {\n"
    "    t = clamp(t, 0.0, 1.0);\n"
    "    vec3 c0 = vec3(0.050383, 0.029803, 0.527975);\n"
    "    vec3 c1 = vec3(0.417642, 0.000564, 0.658390);\n"
    "    vec3 c2 = vec3(0.692840, 0.165141, 0.564522);\n"
    "    vec3 c3 = vec3(0.881443, 0.392529, 0.383229);\n"
    "    vec3 c4 = vec3(0.987622, 0.645320, 0.039886);\n"
    "    vec3 c5 = vec3(0.940015, 0.975158, 0.131326);\n"
    "    float s = t * 5.0;\n"
    "    int idx = int(floor(s));\n"
    "    float f = fract(s);\n"
    "    if (idx == 0) return mix(c0, c1, f);\n"
    "    if (idx == 1) return mix(c1, c2, f);\n"
    "    if (idx == 2) return mix(c2, c3, f);\n"
    "    if (idx == 3) return mix(c3, c4, f);\n"
    "    return mix(c4, c5, f);\n"
    "}\n"
    "\n"
    "vec3 magma(float t) {\n"
    "    t = clamp(t, 0.0, 1.0);\n"
    "    vec3 c0 = vec3(0.001462, 0.000466, 0.013866);\n"
    "    vec3 c1 = vec3(0.316654, 0.071862, 0.485380);\n"
    "    vec3 c2 = vec3(0.716387, 0.214982, 0.474720);\n"
    "    vec3 c3 = vec3(0.974417, 0.462840, 0.359756);\n"
    "    vec3 c4 = vec3(0.995131, 0.766837, 0.534094);\n"
    "    vec3 c5 = vec3(0.987053, 0.991438, 0.749504);\n"
    "    float s = t * 5.0;\n"
    "    int idx = int(floor(s));\n"
    "    float f = fract(s);\n"
    "    if (idx == 0) return mix(c0, c1, f);\n"
    "    if (idx == 1) return mix(c1, c2, f);\n"
    "    if (idx == 2) return mix(c2, c3, f);\n"
    "    if (idx == 3) return mix(c3, c4, f);\n"
    "    return mix(c4, c5, f);\n"
    "}\n"
    "\n"
    "vec3 turbo(float t) {\n"
    "    t = clamp(t, 0.0, 1.0);\n"
    "    vec3 c0 = vec3(0.18995, 0.07176, 0.23217);\n"
    "    vec3 c1 = vec3(0.25107, 0.25237, 0.63374);\n"
    "    vec3 c2 = vec3(0.15992, 0.53830, 0.72889);\n"
    "    vec3 c3 = vec3(0.09140, 0.74430, 0.54318);\n"
    "    vec3 c4 = vec3(0.52876, 0.85393, 0.21546);\n"
    "    vec3 c5 = vec3(0.88092, 0.73551, 0.07741);\n"
    "    vec3 c6 = vec3(0.97131, 0.45935, 0.05765);\n"
    "    vec3 c7 = vec3(0.84299, 0.15070, 0.15090);\n"
    "    float s = t * 7.0;\n"
    "    int idx = int(floor(s));\n"
    "    float f = fract(s);\n"
    "    if (idx == 0) return mix(c0, c1, f);\n"
    "    if (idx == 1) return mix(c1, c2, f);\n"
    "    if (idx == 2) return mix(c2, c3, f);\n"
    "    if (idx == 3) return mix(c3, c4, f);\n"
    "    if (idx == 4) return mix(c4, c5, f);\n"
    "    if (idx == 5) return mix(c5, c6, f);\n"
    "    return mix(c6, c7, f);\n"
    "}\n"
    "\n"
    "vec3 viridis(float t) {\n"
    "    t = clamp(t, 0.0, 1.0);\n"
    "    vec3 c0 = vec3(0.267004, 0.004874, 0.329415);\n"
    "    vec3 c1 = vec3(0.282327, 0.140926, 0.457517);\n"
    "    vec3 c2 = vec3(0.253935, 0.265254, 0.529983);\n"
    "    vec3 c3 = vec3(0.206756, 0.371758, 0.553117);\n"
    "    vec3 c4 = vec3(0.143936, 0.522773, 0.556295);\n"
    "    vec3 c5 = vec3(0.119512, 0.607464, 0.540218);\n"
    "    vec3 c6 = vec3(0.166383, 0.690856, 0.496502);\n"
    "    vec3 c7 = vec3(0.319809, 0.770914, 0.411152);\n"
    "    vec3 c8 = vec3(0.525776, 0.833491, 0.288127);\n"
    "    vec3 c9 = vec3(0.762373, 0.876424, 0.137064);\n"
    "    vec3 c10 = vec3(0.993248, 0.906157, 0.143936);\n"
    "    float s = t * 10.0;\n"
    "    int idx = int(floor(s));\n"
    "    float f = fract(s);\n"
    "    if (idx == 0) return mix(c0, c1, f);\n"
    "    if (idx == 1) return mix(c1, c2, f);\n"
    "    if (idx == 2) return mix(c2, c3, f);\n"
    "    if (idx == 3) return mix(c3, c4, f);\n"
    "    if (idx == 4) return mix(c4, c5, f);\n"
    "    if (idx == 5) return mix(c5, c6, f);\n"
    "    if (idx == 6) return mix(c6, c7, f);\n"
    "    if (idx == 7) return mix(c7, c8, f);\n"
    "    if (idx == 8) return mix(c8, c9, f);\n"
    "    return mix(c9, c10, f);\n"
    "}\n"
    "\n"
    "vec3 diverging(float t) {\n"
    "    t = clamp(t, -1.0, 1.0);\n"
    "    vec3 cold = vec3(0.085, 0.180, 0.525);\n"
    "    vec3 cool = vec3(0.350, 0.550, 0.850);\n"
    "    vec3 neutral = vec3(0.970, 0.970, 0.970);\n"
    "    vec3 warm = vec3(0.900, 0.450, 0.350);\n"
    "    vec3 hot = vec3(0.600, 0.050, 0.100);\n"
    "    if (t < -0.5) return mix(cold, cool, (t + 1.0) * 2.0);\n"
    "    if (t < 0.0) return mix(cool, neutral, (t + 0.5) * 2.0);\n"
    "    if (t < 0.5) return mix(neutral, warm, t * 2.0);\n"
    "    return mix(warm, hot, (t - 0.5) * 2.0);\n"
    "}\n"
    "\n"
    "// ============================================================================\n"
    "// PLATE EIGENMODES\n"
    "// ============================================================================\n"
    "\n"
    "float beamCC(float x, float n) {\n"
    "    float k = (n + 0.5) * PI;\n"
    "    float s = sin(k * x);\n"
    "    float edge = 1.0 - exp(-3.0 * min(x + 1.0, 1.0 - x));\n"
    "    return s * edge;\n"
    "}\n"
    "\n"
    "float beamFF(float x, float n) {\n"
    "    return cos(n * PI * x);\n"
    "}\n"
    "\n"
    "float beamCF(float x, float n) {\n"
    "    float k = (n + 0.25) * PI;\n"
    "    float xNorm = (x + 1.0) * 0.5;\n"
    "    return sin(k * xNorm) - sinh(k * xNorm) * exp(-k);\n"
    "}\n"
    "\n"
    "float modeSimplySupported(vec2 p, float n, float m, float aspect) {\n"
    "    float qx = (p.x / aspect + 1.0) * 0.5;\n"
    "    float qy = (p.y + 1.0) * 0.5;\n"
    "    return sin(n * PI * qx) * sin(m * PI * qy);\n"
    "}\n"
    "\n"
    "float modeClamped(vec2 p, float n, float m, float aspect) {\n"
    "    float px = p.x / aspect;\n"
    "    return beamCC(px, n) * beamCC(p.y, m);\n"
    "}\n"
    "\n"
    "float modeFree(vec2 p, float n, float m, float aspect) {\n"
    "    float px = p.x / aspect;\n"
    "    float xMode = (n < 0.5) ? 1.0 : beamFF(px, n);\n"
    "    float yMode = (m < 0.5) ? 1.0 : beamFF(p.y, m);\n"
    "    return xMode * yMode;\n"
    "}\n"
    "\n"
    "float modeSSF(vec2 p, float n, float m, float aspect) {\n"
    "    float px = p.x / aspect;\n"
    "    float qx = (px + 1.0) * 0.5;\n"
    "    float xMode = sin(n * PI * qx);\n"
    "    float yMode = (m < 0.5) ? 1.0 : cos(m * PI * p.y);\n"
    "    return xMode * yMode;\n"
    "}\n"
    "\n"
    "float modeCSS(vec2 p, float n, float m, float aspect) {\n"
    "    float px = p.x / aspect;\n"
    "    float qy = (p.y + 1.0) * 0.5;\n"
    "    return beamCC(px, n) * sin(m * PI * qy);\n"
    "}\n"
    "\n"
    "float modeCF(vec2 p, float n, float m, float aspect) {\n"
    "    float px = p.x / aspect;\n"
    "    float xMode = beamCC(px, n);\n"
    "    float yMode = (m < 0.5) ? 1.0 : beamFF(p.y, m);\n"
    "    return xMode * yMode;\n"
    "}\n"
    "\n"
    "float modeCantilever(vec2 p, float n, float m, float aspect) {\n"
    "    float px = p.x / aspect;\n"
    "    float xMode = beamCF(px, n);\n"
    "    float yMode = (m < 0.5) ? 1.0 : beamFF(p.y, m);\n"
    "    return xMode * yMode;\n"
    "}\n"
    "\n"
    "float modeGuided(vec2 p, float n, float m, float aspect) {\n"
    "    float px = p.x / aspect;\n"
    "    return cos(n * PI * px) * cos(m * PI * p.y);\n"
    "}\n"
    "\n"
    "float modeChladni(vec2 p, float n, float m, float aspect) {\n"
    "    float px = p.x / aspect;\n"
    "    float py = p.y;\n"
    "    float mode_nm = cos(n * PI * px) * cos(m * PI * py);\n"
    "    float mode_mn = cos(m * PI * px) * cos(n * PI * py);\n"
    "    return mode_nm - mode_mn;\n"
    "}\n"
    "\n"
    "float computeSingleMode(vec2 p, float n, float m, float aspect, int boundary) {\n"
    "    vec2 pc = p;\n"
    "    if (boundary == 1 || boundary == 4 || boundary == 5 || boundary == 6) {\n"
    "        pc.x = clamp(p.x, -aspect, aspect);\n"
    "        pc.y = clamp(p.y, -1.0, 1.0);\n"
    "    }\n"
    "    if (boundary == 0) return modeSimplySupported(p, n, m, aspect);\n"
    "    else if (boundary == 1) return modeClamped(pc, n, m, aspect);\n"
    "    else if (boundary == 2) return modeFree(p, n, m, aspect);\n"
    "    else if (boundary == 3) return modeSSF(p, n, m, aspect);\n"
    "    else if (boundary == 4) return modeCSS(pc, n, m, aspect);\n"
    "    else if (boundary == 5) return modeCF(pc, n, m, aspect);\n"
    "    else if (boundary == 6) return modeCantilever(pc, n, m, aspect);\n"
    "    else return modeGuided(p, n, m, aspect);\n"
    "}\n"
    "\n"
    "float computeModeSum(vec2 p, float targetLambda, float aspect, int boundary, float time) {\n"
    "    float sqrtL = sqrt(max(5.0, targetLambda));\n"
    "    float n = max(2.0, floor(sqrtL * 0.9 + 0.5));\n"
    "    float m = max(1.0, floor(sqrtL * 0.5 + 0.5));\n"
    "    if (n <= m) n = m + 1.0;\n"
    "    if (boundary == 0) return modeChladni(p, n, m, aspect);\n"
    "    else return computeSingleMode(p, n, m, aspect, boundary);\n"
    "}\n"
    "\n"
    "// ============================================================================\n"
    "// MAIN\n"
    "// ============================================================================\n"
    "\n"
    "void main() {\n"
    "    vec2 uv = fragCoord / iResolution;\n"
    "    float windowAspect = iResolution.x / iResolution.y;\n"
    "\n"
    "    vec2 p;\n"
    "    bool outOfBounds = false;\n"
    "    float plateAspect = 1.0;\n"
    "\n"
    "    if (iAspectMode == 0) {\n"
    "        plateAspect = windowAspect;\n"
    "        p.x = (uv.x - 0.5) * 2.0 * windowAspect;\n"
    "        p.y = (uv.y - 0.5) * 2.0;\n"
    "    } else if (iAspectMode == 1) {\n"
    "        plateAspect = 1.0;\n"
    "        if (windowAspect > 1.0) {\n"
    "            float plateWidth = 1.0 / windowAspect;\n"
    "            float margin = (1.0 - plateWidth) / 2.0;\n"
    "            if (uv.x < margin || uv.x > 1.0 - margin) {\n"
    "                outOfBounds = true;\n"
    "            } else {\n"
    "                float localX = (uv.x - margin) / plateWidth;\n"
    "                p.x = (localX - 0.5) * 2.0;\n"
    "                p.y = (uv.y - 0.5) * 2.0;\n"
    "            }\n"
    "        } else {\n"
    "            float plateHeight = windowAspect;\n"
    "            float margin = (1.0 - plateHeight) / 2.0;\n"
    "            if (uv.y < margin || uv.y > 1.0 - margin) {\n"
    "                outOfBounds = true;\n"
    "            } else {\n"
    "                float localY = (uv.y - margin) / plateHeight;\n"
    "                p.x = (uv.x - 0.5) * 2.0;\n"
    "                p.y = (localY - 0.5) * 2.0;\n"
    "            }\n"
    "        }\n"
    "    } else {\n"
    "        plateAspect = 1.0;\n"
    "        if (windowAspect > 1.0) {\n"
    "            p.x = (uv.x - 0.5) * 2.0 * windowAspect;\n"
    "            p.y = (uv.y - 0.5) * 2.0;\n"
    "        } else {\n"
    "            p.x = (uv.x - 0.5) * 2.0;\n"
    "            p.y = (uv.y - 0.5) * 2.0 / windowAspect;\n"
    "        }\n"
    "    }\n"
    "\n"
    "    if (outOfBounds) {\n"
    "        outColor = vec4(0.0, 0.0, 0.0, 1.0);\n"
    "        return;\n"
    "    }\n"
    "\n"
    "    float displacement = 0.0;\n"
    "    float totalEnergy = 0.0;\n"
    "    int maxBin = min(iNumBins, int(iMaxFreq / iFreqPerBin));\n"
    "\n"
    "    for (int i = 1; i < 2048; i++) {\n"
    "        if (i >= maxBin) break;\n"
    "        float freq = float(i) * iFreqPerBin;\n"
    "        float amp = texelFetch(iSpectrum, i).r;\n"
    "        if (amp < 0.005) continue;\n"
    "        float ratio = freq / iBaseFreq;\n"
    "        float targetLambda = 5.0 + ratio * iModeScale * 10.0;\n"
    "        float mode = computeModeSum(p, targetLambda, plateAspect, iBoundary, iTime);\n"
    "        displacement += amp * mode;\n"
    "        totalEnergy += amp;\n"
    "    }\n"
    "\n"
    "    if (totalEnergy > 0.1) {\n"
    "        displacement /= sqrt(totalEnergy);\n"
    "    }\n"
    "\n"
    "    float d = displacement * iContrast;\n"
    "    vec3 color;\n"
    "\n"
    "    if (iColorMode == 0) {\n"
    "        float energy = tanh(abs(d));\n"
    "        color = plasma(energy);\n"
    "    } else if (iColorMode == 1) {\n"
    "        float energy = tanh(abs(d));\n"
    "        color = magma(energy);\n"
    "    } else if (iColorMode == 2) {\n"
    "        float energy = tanh(abs(d));\n"
    "        color = turbo(energy);\n"
    "    } else if (iColorMode == 3) {\n"
    "        float energy = tanh(abs(d));\n"
    "        color = viridis(energy);\n"
    "    } else {\n"
    "        float signed_d = tanh(d);\n"
    "        color = diverging(signed_d);\n"
    "    }\n"
    "\n"
    "    outColor = vec4(color, 1.0);\n"
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

/* Default values matching Python version */
#define DEFAULT_BASE_FREQ    40.0f
#define DEFAULT_MODE_SCALE   0.5f
#define DEFAULT_MAX_FREQ     7000.0f
#define DEFAULT_CONTRAST     1.0f
#define DEFAULT_COLOR_MODE   4
#define DEFAULT_BOUNDARY     0
#define DEFAULT_ASPECT_MODE  2
#define DEFAULT_FFT_SIZE     8192

#define NUM_COLOR_MODES      5
#define NUM_BOUNDARIES       8
#define NUM_ASPECT_MODES     3

static const char *COLOR_NAMES[] = {"Plasma", "Magma", "Turbo", "Viridis", "Signed"};
static const char *BOUNDARY_NAMES[] = {
    "Chladni", "Clamped", "Free", "SS-Free",
    "Clamped-SS", "Clamped-Free", "Cantilever", "Guided"
};
static const char *ASPECT_NAMES[] = {"Full", "1:1 Letterbox", "1:1 Crop"};

typedef struct {
    int w, h;
    float baseFreq;
    float modeScale;
    float maxFreq;
    float contrast;
    int colorMode;
    int boundary;
    int aspectMode;
    float time;

    GLuint program;
    GLuint vao;
    GLuint spectrumTbo;
    GLuint spectrumTex;

    GLint locResolution;
    GLint locSpectrum;
    GLint locNumBins;
    GLint locFreqPerBin;
    GLint locMaxFreq;
    GLint locBaseFreq;
    GLint locModeScale;
    GLint locContrast;
    GLint locColorMode;
    GLint locTime;
    GLint locBoundary;
    GLint locAspectMode;
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

static void VizResetDefaults(Visualizer *v) {
    v->baseFreq = DEFAULT_BASE_FREQ;
    v->modeScale = DEFAULT_MODE_SCALE;
    v->maxFreq = DEFAULT_MAX_FREQ;
    v->contrast = DEFAULT_CONTRAST;
    v->colorMode = DEFAULT_COLOR_MODE;
    v->boundary = DEFAULT_BOUNDARY;
    v->aspectMode = DEFAULT_ASPECT_MODE;
}

static BOOL VizInit(Visualizer *v, int w, int h) {
    v->w = w;
    v->h = h;
    v->time = 0.0f;
    VizResetDefaults(v);

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
    v->locNumBins = glGetUniformLocation(v->program, "iNumBins");
    v->locFreqPerBin = glGetUniformLocation(v->program, "iFreqPerBin");
    v->locMaxFreq = glGetUniformLocation(v->program, "iMaxFreq");
    v->locBaseFreq = glGetUniformLocation(v->program, "iBaseFreq");
    v->locModeScale = glGetUniformLocation(v->program, "iModeScale");
    v->locContrast = glGetUniformLocation(v->program, "iContrast");
    v->locColorMode = glGetUniformLocation(v->program, "iColorMode");
    v->locTime = glGetUniformLocation(v->program, "iTime");
    v->locBoundary = glGetUniformLocation(v->program, "iBoundary");
    v->locAspectMode = glGetUniformLocation(v->program, "iAspectMode");

    return TRUE;
}

static void VizRender(Visualizer *v, float *spectrum, int fftSize) {
    int nBins = fftSize / 2 + 1;
    float freqPerBin = (float)SAMPLE_RATE / fftSize;

    /* Update spectrum buffer */
    glBindBuffer(GL_TEXTURE_BUFFER, v->spectrumTbo);
    glBufferSubData(GL_TEXTURE_BUFFER, 0, MAX_BINS * sizeof(float), spectrum);

    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(v->program);

    glUniform2f(v->locResolution, (float)v->w, (float)v->h);
    glUniform1i(v->locNumBins, nBins);
    glUniform1f(v->locFreqPerBin, freqPerBin);
    glUniform1f(v->locMaxFreq, v->maxFreq);
    glUniform1f(v->locBaseFreq, v->baseFreq);
    glUniform1f(v->locModeScale, v->modeScale);
    glUniform1f(v->locContrast, v->contrast);
    glUniform1i(v->locColorMode, v->colorMode);
    glUniform1f(v->locTime, v->time);
    glUniform1i(v->locBoundary, v->boundary);
    glUniform1i(v->locAspectMode, v->aspectMode);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_BUFFER, v->spectrumTex);
    glUniform1i(v->locSpectrum, 0);

    glBindVertexArray(v->vao);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, NULL);

    v->time += 0.016f;
}

static void PrintStatus(Visualizer *v, int fftSize) {
    char status[256];
    snprintf(status, sizeof(status),
             "Base=%.0fHz  Scale=%.2f  MaxF=%.0fHz  Contrast=%.1f  Color=%s  Boundary=%s  Aspect=%s  FFT=%d",
             v->baseFreq, v->modeScale, v->maxFreq, v->contrast,
             COLOR_NAMES[v->colorMode], BOUNDARY_NAMES[v->boundary], ASPECT_NAMES[v->aspectMode], fftSize);
    printf("\r%-140s", status);
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

    if (!AudioInit(&g_audio, DEFAULT_FFT_SIZE)) {
        printf("Failed to initialize audio capture\n");
        DestroyOpenGLContext();
        return 1;
    }

    Sleep(50);

    printf("Wave Plate Visualizer - Steady-state plate vibration from audio FFT\n");
    printf("Controls: UP/DOWN=base freq  W/S=scale  A/D=max freq  Z/X=contrast  LEFT/RIGHT=FFT size  V=color  P=boundary  R=aspect  ALT+ENTER=fullscreen  SPACE=reset  ESC=quit\n\n");

    PrintStatus(&g_viz, g_audio.fftSize);

    BOOL running = TRUE;
    DWORD lastKeyTime = 0;
    DWORD repeatDelay = 100;
    BOOL prevLeft = FALSE, prevRight = FALSE;
    BOOL prevV = FALSE, prevP = FALSE, prevR = FALSE, prevSpace = FALSE;
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

            /* Base frequency */
            if ((GetAsyncKeyState(VK_UP) & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_viz.baseFreq += 5.0f;
                if (g_viz.baseFreq > 500.0f) g_viz.baseFreq = 500.0f;
                needUpdate = TRUE;
                lastKeyTime = now;
            }
            if ((GetAsyncKeyState(VK_DOWN) & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_viz.baseFreq -= 5.0f;
                if (g_viz.baseFreq < 10.0f) g_viz.baseFreq = 10.0f;
                needUpdate = TRUE;
                lastKeyTime = now;
            }

            /* Mode scale */
            if ((GetAsyncKeyState('W') & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_viz.modeScale += 0.05f;
                if (g_viz.modeScale > 2.0f) g_viz.modeScale = 2.0f;
                needUpdate = TRUE;
                lastKeyTime = now;
            }
            if ((GetAsyncKeyState('S') & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_viz.modeScale -= 0.05f;
                if (g_viz.modeScale < 0.1f) g_viz.modeScale = 0.1f;
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
                if (g_viz.maxFreq < 500.0f) g_viz.maxFreq = 500.0f;
                needUpdate = TRUE;
                lastKeyTime = now;
            }

            /* Contrast */
            if ((GetAsyncKeyState('X') & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_viz.contrast += 0.1f;
                if (g_viz.contrast > 5.0f) g_viz.contrast = 5.0f;
                needUpdate = TRUE;
                lastKeyTime = now;
            }
            if ((GetAsyncKeyState('Z') & 0x8000) && (now - lastKeyTime >= repeatDelay)) {
                g_viz.contrast -= 0.1f;
                if (g_viz.contrast < 0.1f) g_viz.contrast = 0.1f;
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
            BOOL currV = (GetAsyncKeyState('V') & 0x8000) != 0;
            if (currV && !prevV) {
                g_viz.colorMode = (g_viz.colorMode + 1) % NUM_COLOR_MODES;
                needUpdate = TRUE;
            }
            prevV = currV;

            /* Boundary mode toggle (edge-triggered) */
            BOOL currP = (GetAsyncKeyState('P') & 0x8000) != 0;
            if (currP && !prevP) {
                g_viz.boundary = (g_viz.boundary + 1) % NUM_BOUNDARIES;
                needUpdate = TRUE;
            }
            prevP = currP;

            /* Aspect mode toggle (edge-triggered) */
            BOOL currR = (GetAsyncKeyState('R') & 0x8000) != 0;
            if (currR && !prevR) {
                g_viz.aspectMode = (g_viz.aspectMode + 1) % NUM_ASPECT_MODES;
                needUpdate = TRUE;
            }
            prevR = currR;

            /* Reset to defaults (edge-triggered) */
            BOOL currSpace = (GetAsyncKeyState(VK_SPACE) & 0x8000) != 0;
            if (currSpace && !prevSpace) {
                VizResetDefaults(&g_viz);
                AudioSetFFTSize(&g_audio, DEFAULT_FFT_SIZE);
            }
            prevSpace = currSpace;
        } else {
            /* Reset edge-trigger states when not focused */
            prevRight = FALSE;
            prevLeft = FALSE;
            prevV = FALSE;
            prevP = FALSE;
            prevR = FALSE;
            prevSpace = FALSE;
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
                PrintStatus(&g_viz, g_audio.fftSize);
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

