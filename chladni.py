"""
Wave Plate Visualizer - Modal Superposition from Audio FFT

Approximates plate vibration as a modal superposition driven by
audio-spectrum amplitudes. Each FFT bin excites modes whose natural
frequency is nearby, weighted by spectral amplitude.

For simply supported rectangular plates, modes are sin(nπx/a)sin(mπy/b).
Other boundary conditions use approximate beam-function products.

For thin Kirchhoff-Love plates: f_nm ∝ (n²/a² + m²/b²).
Actual response depends on damping and forcing overlap with each mode.
"""

import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import soundcard as sc
import threading
import time
import ctypes

SAMPLE_RATE = 44100
CHUNK_SIZE = 64
MAX_FFT_SIZE = 32768
MAX_BINS = MAX_FFT_SIZE // 2 + 1

VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec2 position;
out vec2 fragCoord;
uniform vec2 iResolution;
void main() {
    fragCoord = (position + 1.0) * 0.5 * iResolution;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec2 fragCoord;
out vec4 outColor;

uniform vec2 iResolution;
uniform sampler1D iSpectrum;
uniform int iNumBins;
uniform float iFreqPerBin;
uniform float iMaxFreq;
uniform float iBaseFreq;        // Lowest mode frequency
uniform float iModeScale;       // Controls mode density
uniform float iContrast;
uniform int iColorMode;         // 0=Plasma, 1=Magma, 2=Turbo, 3=Signed
uniform float iTime;
uniform int iBoundary;          // 0-7 various boundary conditions
uniform int iAspectMode;        // 0=Full, 1=1:1 Letterbox, 2=1:1 Crop

#define PI 3.14159265
#define TAU 6.28318530

// ============================================================================
// COLORMAPS
// ============================================================================

// Plasma - perceptually uniform, purple -> pink -> orange -> yellow
vec3 plasma(float t) {
    t = clamp(t, 0.0, 1.0);
    // Attempt at more accurate plasma using polynomial approximation
    vec3 c0 = vec3(0.050383, 0.029803, 0.527975);
    vec3 c1 = vec3(0.417642, 0.000564, 0.658390);
    vec3 c2 = vec3(0.692840, 0.165141, 0.564522);
    vec3 c3 = vec3(0.881443, 0.392529, 0.383229);
    vec3 c4 = vec3(0.987622, 0.645320, 0.039886);
    vec3 c5 = vec3(0.940015, 0.975158, 0.131326);

    float s = t * 5.0;
    int idx = int(floor(s));
    float f = fract(s);

    if (idx == 0) return mix(c0, c1, f);
    if (idx == 1) return mix(c1, c2, f);
    if (idx == 2) return mix(c2, c3, f);
    if (idx == 3) return mix(c3, c4, f);
    return mix(c4, c5, f);
}

// Magma - black -> purple -> red -> yellow
vec3 magma(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.001462, 0.000466, 0.013866);
    vec3 c1 = vec3(0.316654, 0.071862, 0.485380);
    vec3 c2 = vec3(0.716387, 0.214982, 0.474720);
    vec3 c3 = vec3(0.974417, 0.462840, 0.359756);
    vec3 c4 = vec3(0.995131, 0.766837, 0.534094);
    vec3 c5 = vec3(0.987053, 0.991438, 0.749504);

    float s = t * 5.0;
    int idx = int(floor(s));
    float f = fract(s);

    if (idx == 0) return mix(c0, c1, f);
    if (idx == 1) return mix(c1, c2, f);
    if (idx == 2) return mix(c2, c3, f);
    if (idx == 3) return mix(c3, c4, f);
    return mix(c4, c5, f);
}

// Turbo - blue -> cyan -> green -> yellow -> red
vec3 turbo(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.18995, 0.07176, 0.23217);
    vec3 c1 = vec3(0.25107, 0.25237, 0.63374);
    vec3 c2 = vec3(0.15992, 0.53830, 0.72889);
    vec3 c3 = vec3(0.09140, 0.74430, 0.54318);
    vec3 c4 = vec3(0.52876, 0.85393, 0.21546);
    vec3 c5 = vec3(0.88092, 0.73551, 0.07741);
    vec3 c6 = vec3(0.97131, 0.45935, 0.05765);
    vec3 c7 = vec3(0.84299, 0.15070, 0.15090);

    float s = t * 7.0;
    int idx = int(floor(s));
    float f = fract(s);

    if (idx == 0) return mix(c0, c1, f);
    if (idx == 1) return mix(c1, c2, f);
    if (idx == 2) return mix(c2, c3, f);
    if (idx == 3) return mix(c3, c4, f);
    if (idx == 4) return mix(c4, c5, f);
    if (idx == 5) return mix(c5, c6, f);
    return mix(c6, c7, f);
}

// Viridis - dark purple -> blue -> teal -> green -> yellow
vec3 viridis(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.267004, 0.004874, 0.329415);
    vec3 c1 = vec3(0.282327, 0.140926, 0.457517);
    vec3 c2 = vec3(0.253935, 0.265254, 0.529983);
    vec3 c3 = vec3(0.206756, 0.371758, 0.553117);
    vec3 c4 = vec3(0.143936, 0.522773, 0.556295);
    vec3 c5 = vec3(0.119512, 0.607464, 0.540218);
    vec3 c6 = vec3(0.166383, 0.690856, 0.496502);
    vec3 c7 = vec3(0.319809, 0.770914, 0.411152);
    vec3 c8 = vec3(0.525776, 0.833491, 0.288127);
    vec3 c9 = vec3(0.762373, 0.876424, 0.137064);
    vec3 c10 = vec3(0.993248, 0.906157, 0.143936);

    float s = t * 10.0;
    int idx = int(floor(s));
    float f = fract(s);

    if (idx == 0) return mix(c0, c1, f);
    if (idx == 1) return mix(c1, c2, f);
    if (idx == 2) return mix(c2, c3, f);
    if (idx == 3) return mix(c3, c4, f);
    if (idx == 4) return mix(c4, c5, f);
    if (idx == 5) return mix(c5, c6, f);
    if (idx == 6) return mix(c6, c7, f);
    if (idx == 7) return mix(c7, c8, f);
    if (idx == 8) return mix(c8, c9, f);
    return mix(c9, c10, f);
}

// Diverging colormap for signed values (blue-white-red)
vec3 diverging(float t) {
    // t in [-1, 1]
    t = clamp(t, -1.0, 1.0);

    vec3 cold = vec3(0.085, 0.180, 0.525);   // Deep blue
    vec3 cool = vec3(0.350, 0.550, 0.850);   // Light blue
    vec3 neutral = vec3(0.970, 0.970, 0.970); // Near white
    vec3 warm = vec3(0.900, 0.450, 0.350);   // Light red
    vec3 hot = vec3(0.600, 0.050, 0.100);    // Deep red

    if (t < -0.5) return mix(cold, cool, (t + 1.0) * 2.0);
    if (t < 0.0) return mix(cool, neutral, (t + 0.5) * 2.0);
    if (t < 0.5) return mix(neutral, warm, t * 2.0);
    return mix(warm, hot, (t - 0.5) * 2.0);
}

// ============================================================================
// PLATE EIGENMODES - Various boundary conditions for rectangular plates
// ============================================================================

// Beam function approximations for clamped-clamped boundary
// These satisfy: X(0) = X(1) = X'(0) = X'(1) = 0
float beamCC(float x, float n) {
    // Approximation using cos-cosh combination
    // For higher modes, approaches sin((n+0.5)*pi*x)
    float k = (n + 0.5) * PI;
    float s = sin(k * x);
    float c = cos(k * x);
    // Blend with sinh/cosh behavior at edges
    float edge = 1.0 - exp(-3.0 * min(x + 1.0, 1.0 - x));
    return s * edge;
}

// Beam function for free-free boundary
// These satisfy: X''(0) = X''(1) = X'''(0) = X'''(1) = 0
float beamFF(float x, float n) {
    // Free-free modes are symmetric/antisymmetric
    // Use cos for symmetric modes
    return cos(n * PI * x);
}

// Beam function for clamped-free (cantilever)
// Satisfies: X(0) = X'(0) = 0, X''(1) = X'''(1) = 0
float beamCF(float x, float n) {
    // Cantilever modes - fixed at x=-1, free at x=1
    float k = (n + 0.25) * PI;
    float xNorm = (x + 1.0) * 0.5;  // Map [-1,1] to [0,1]
    return sin(k * xNorm) - sinh(k * xNorm) * exp(-k);
}

// 0: Simply Supported (SS-SS-SS-SS)
// All edges pinned: w = 0, M = 0 (moment free)
// Classic Navier solution: sin(nπx/a)sin(mπy/b)
// p is in [-aspect, aspect] x [-1, 1] for physical rectangular plate
float modeSimplySupported(vec2 p, float n, float m, float aspect) {
    // Map to [0, 1] range for each dimension
    float qx = (p.x / aspect + 1.0) * 0.5;  // x in [-aspect, aspect] -> [0, 1]
    float qy = (p.y + 1.0) * 0.5;            // y in [-1, 1] -> [0, 1]
    return sin(n * PI * qx) * sin(m * PI * qy);
}

// 1: Clamped (C-C-C-C)
// All edges clamped: w = 0, dw/dn = 0
// Uses product of clamped-clamped beam functions
float modeClamped(vec2 p, float n, float m, float aspect) {
    // Normalize x from [-aspect, aspect] to [-1, 1]
    float px = p.x / aspect;
    return beamCC(px, n) * beamCC(p.y, m);
}

// 2: Free (F-F-F-F)
// All edges free: M = 0, V = 0 (shear)
// Uses product of free-free beam functions (cos basis)
float modeFree(vec2 p, float n, float m, float aspect) {
    // Normalize x from [-aspect, aspect] to [-1, 1]
    float px = p.x / aspect;
    // Include rigid body modes for n=0 or m=0
    float xMode = (n < 0.5) ? 1.0 : beamFF(px, n);
    float yMode = (m < 0.5) ? 1.0 : beamFF(p.y, m);
    return xMode * yMode;
}

// 3: Simply Supported - Free (SS-F-SS-F)
// x-edges simply supported, y-edges free
float modeSSF(vec2 p, float n, float m, float aspect) {
    // Normalize x from [-aspect, aspect] to [-1, 1]
    float px = p.x / aspect;
    float qx = (px + 1.0) * 0.5;
    float xMode = sin(n * PI * qx);
    float yMode = (m < 0.5) ? 1.0 : cos(m * PI * p.y);
    return xMode * yMode;
}

// 4: Clamped - Simply Supported (C-SS-C-SS)
// x-edges clamped, y-edges simply supported
float modeCSS(vec2 p, float n, float m, float aspect) {
    // Normalize x from [-aspect, aspect] to [-1, 1]
    float px = p.x / aspect;
    float qy = (p.y + 1.0) * 0.5;
    return beamCC(px, n) * sin(m * PI * qy);
}

// 5: Clamped - Free (C-F-C-F)
// x-edges clamped, y-edges free
float modeCF(vec2 p, float n, float m, float aspect) {
    // Normalize x from [-aspect, aspect] to [-1, 1]
    float px = p.x / aspect;
    float xMode = beamCC(px, n);
    float yMode = (m < 0.5) ? 1.0 : beamFF(p.y, m);
    return xMode * yMode;
}

// 6: Cantilever (C-F-F-F)
// One edge clamped (x=-1), all others free
float modeCantilever(vec2 p, float n, float m, float aspect) {
    // Normalize x from [-aspect, aspect] to [-1, 1]
    float px = p.x / aspect;
    float xMode = beamCF(px, n);
    float yMode = (m < 0.5) ? 1.0 : beamFF(p.y, m);
    return xMode * yMode;
}

// 7: Guided (G-G-G-G)
// All edges guided: dw/dn = 0, V = 0
// Uses cos functions (slope zero at edges)
float modeGuided(vec2 p, float n, float m, float aspect) {
    // Normalize x from [-aspect, aspect] to [-1, 1]
    float px = p.x / aspect;
    return cos(n * PI * px) * cos(m * PI * p.y);
}

// Chladni pattern: plate constrained at center, free edges
// Formula: cos(nπx)cos(mπy) - cos(mπx)cos(nπy)
// This automatically includes the degenerate mode with correct sign
float modeChladni(vec2 p, float n, float m, float aspect) {
    // Normalize to [-1, 1] x [-1, 1]
    float px = p.x / aspect;
    float py = p.y;
    float mode_nm = cos(n * PI * px) * cos(m * PI * py);
    float mode_mn = cos(m * PI * px) * cos(n * PI * py);
    return mode_nm - mode_mn;
}

// ============================================================================
// FREQUENCY TO MODE MAPPING
// ============================================================================

// Compute single mode shape based on boundary type
float computeSingleMode(vec2 p, float n, float m, float aspect, int boundary) {
    // Clamp coordinates for modes that use sinh/cosh (they explode outside [-1,1])
    // Modes 1 (Clamped), 4 (CSS), 5 (CF), 6 (Cantilever) use beamCC or beamCF
    vec2 pc = p;
    if (boundary == 1 || boundary == 4 || boundary == 5 || boundary == 6) {
        pc.x = clamp(p.x, -aspect, aspect);
        pc.y = clamp(p.y, -1.0, 1.0);
    }

    if (boundary == 0) {
        return modeSimplySupported(p, n, m, aspect);
    } else if (boundary == 1) {
        return modeClamped(pc, n, m, aspect);
    } else if (boundary == 2) {
        return modeFree(p, n, m, aspect);
    } else if (boundary == 3) {
        return modeSSF(p, n, m, aspect);
    } else if (boundary == 4) {
        return modeCSS(pc, n, m, aspect);
    } else if (boundary == 5) {
        return modeCF(pc, n, m, aspect);
    } else if (boundary == 6) {
        return modeCantilever(pc, n, m, aspect);
    } else {
        return modeGuided(p, n, m, aspect);
    }
}

// For Chladni plate: eigenvalue λ_nm = n² + m² (square plate)
// Map target λ to mode numbers using direct calculation (no loops)
float computeModeSum(vec2 p, float targetLambda, float aspect, int boundary, float time) {
    // Direct calculation of (n, m) from eigenvalue
    // For λ = n² + m², we want n > m and n ≠ m
    // Approximate: n ≈ sqrt(λ * 0.8), m ≈ sqrt(λ * 0.2)
    float sqrtL = sqrt(max(5.0, targetLambda));

    // Ensure asymmetric modes (n ≠ m) for visible Chladni patterns
    float n = max(2.0, floor(sqrtL * 0.9 + 0.5));
    float m = max(1.0, floor(sqrtL * 0.5 + 0.5));

    // Ensure n > m for consistent patterns
    if (n <= m) {
        n = m + 1.0;
    }

    // Use Chladni formula which includes degenerate mode subtraction
    if (boundary == 0) {
        // Chladni: cos(nπx)cos(mπy) - cos(mπx)cos(nπy)
        return modeChladni(p, n, m, aspect);
    } else {
        // Other boundaries: use their specific mode shapes
        return computeSingleMode(p, n, m, aspect, boundary);
    }
}

// ============================================================================
// MAIN
// ============================================================================

void main() {
    vec2 uv = fragCoord / iResolution;
    float windowAspect = iResolution.x / iResolution.y;

    // Handle aspect modes
    // 0 = Full (use window aspect), 1 = 1:1 Letterbox, 2 = 1:1 Crop
    vec2 p;  // Plate coordinates, will be in [-1,1] x [-1,1] for square plate
    bool outOfBounds = false;
    float plateAspect = 1.0;

    if (iAspectMode == 0) {
        // Full: rectangular plate fills window
        plateAspect = windowAspect;
        p.x = (uv.x - 0.5) * 2.0 * windowAspect;
        p.y = (uv.y - 0.5) * 2.0;
    } else if (iAspectMode == 1) {
        // Letterbox: square plate centered with black bars
        plateAspect = 1.0;
        if (windowAspect > 1.0) {
            // Wide window: bars on left/right
            float plateWidth = 1.0 / windowAspect;  // fraction of window width
            float margin = (1.0 - plateWidth) / 2.0;
            if (uv.x < margin || uv.x > 1.0 - margin) {
                outOfBounds = true;
            } else {
                float localX = (uv.x - margin) / plateWidth;  // 0 to 1 within plate
                p.x = (localX - 0.5) * 2.0;
                p.y = (uv.y - 0.5) * 2.0;
            }
        } else {
            // Tall window: bars on top/bottom
            float plateHeight = windowAspect;  // fraction of window height
            float margin = (1.0 - plateHeight) / 2.0;
            if (uv.y < margin || uv.y > 1.0 - margin) {
                outOfBounds = true;
            } else {
                float localY = (uv.y - margin) / plateHeight;  // 0 to 1 within plate
                p.x = (uv.x - 0.5) * 2.0;
                p.y = (localY - 0.5) * 2.0;
            }
        }
    } else {
        // Crop: square plate fills window, edges cropped
        plateAspect = 1.0;
        if (windowAspect > 1.0) {
            // Wide window: full height shown, sides cropped
            p.x = (uv.x - 0.5) * 2.0 * windowAspect;  // extends beyond [-1,1]
            p.y = (uv.y - 0.5) * 2.0;
        } else {
            // Tall window: full width shown, top/bottom cropped
            p.x = (uv.x - 0.5) * 2.0;
            p.y = (uv.y - 0.5) * 2.0 / windowAspect;  // extends beyond [-1,1]
        }
    }

    // Black bars for letterbox mode
    if (outOfBounds) {
        outColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Accumulate displacement from all frequency components
    float displacement = 0.0;
    float totalEnergy = 0.0;

    int maxBin = min(iNumBins, int(iMaxFreq / iFreqPerBin));

    // Sum over all FFT bins
    for (int i = 1; i < 2048; i++) {
        if (i >= maxBin) break;

        float freq = float(i) * iFreqPerBin;
        float u = float(i) / float(iNumBins);
        float amp = texture(iSpectrum, u).r;

        if (amp < 0.005) continue;

        // Map frequency to target eigenvalue
        // Lowest interesting Chladni mode is (1,2) with λ = 5
        // Scale so baseFreq maps to λ = 5
        float ratio = freq / iBaseFreq;
        float targetLambda = 5.0 + ratio * iModeScale * 10.0;

        // Sum all modes with eigenvalue close to target (handles degeneracy)
        float mode = computeModeSum(p, targetLambda, plateAspect, iBoundary, iTime);

        displacement += amp * mode;
        totalEnergy += amp;
    }

    // Normalize by total energy for consistent brightness
    if (totalEnergy > 0.1) {
        displacement /= sqrt(totalEnergy);
    }

    // Apply contrast
    float d = displacement * iContrast;

    // Color mapping
    vec3 color;

    if (iColorMode == 0) {
        // Plasma (default) - maps absolute displacement
        float energy = tanh(abs(d));
        color = plasma(energy);
    } else if (iColorMode == 1) {
        // Magma
        float energy = tanh(abs(d));
        color = magma(energy);
    } else if (iColorMode == 2) {
        // Turbo
        float energy = tanh(abs(d));
        color = turbo(energy);
    } else if (iColorMode == 3) {
        // Viridis
        float energy = tanh(abs(d));
        color = viridis(energy);
    } else {
        // Signed displacement - diverging colormap
        float signed_d = tanh(d);
        color = diverging(signed_d);
    }

    outColor = vec4(color, 1.0);
}
"""


class AudioCapture:
    def __init__(self, fft_size=8192):
        self.running = False
        self.thread = None
        self.fft_size = fft_size
        self._ring_buffer = np.zeros(MAX_FFT_SIZE * 4, dtype=np.float32)
        self._write_pos = 0
        self._running_max = 0.01
        self._output = np.zeros(MAX_BINS, dtype=np.float32)
        self._update_fft_params()

    def _update_fft_params(self):
        self._n_bins = self.fft_size // 2 + 1
        self._window = np.hanning(self.fft_size).astype(np.float32)
        self._spectrum = np.zeros(self._n_bins, dtype=np.float32)
        self._smooth_spectrum = np.zeros(self._n_bins, dtype=np.float32)
        self._running_max = 0.01

    def set_fft_size(self, size, viz=None):
        size = max(256, min(MAX_FFT_SIZE, size))
        if size != self.fft_size:
            self.fft_size = size
            self._update_fft_params()
            if viz:
                viz.print_status()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)

    def _capture_loop(self):
        try:
            try:
                ctypes.windll.kernel32.SetThreadPriority(
                    ctypes.windll.kernel32.GetCurrentThread(), 2)
            except:
                pass

            speaker = sc.default_speaker()
            with sc.get_microphone(id=str(speaker.name), include_loopback=True).recorder(
                samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE) as mic:
                while self.running:
                    data = mic.record(numframes=CHUNK_SIZE)
                    if data.ndim > 1:
                        data = data.mean(axis=1)

                    chunk_len = len(data)
                    buf_len = len(self._ring_buffer)
                    end_pos = self._write_pos + chunk_len

                    if end_pos <= buf_len:
                        self._ring_buffer[self._write_pos:end_pos] = data
                    else:
                        first_part = buf_len - self._write_pos
                        self._ring_buffer[self._write_pos:] = data[:first_part]
                        self._ring_buffer[:chunk_len - first_part] = data[first_part:]

                    self._write_pos = end_pos % buf_len
        except Exception as e:
            print(f"Audio error: {e}")

    def get_spectrum(self):
        fft_size = self.fft_size
        n_bins = fft_size // 2 + 1

        current_pos = self._write_pos
        buf_len = len(self._ring_buffer)
        start = (current_pos - fft_size) % buf_len

        if start + fft_size <= buf_len:
            samples = self._ring_buffer[start:start + fft_size].copy()
        else:
            first_part = buf_len - start
            samples = np.concatenate([self._ring_buffer[start:], self._ring_buffer[:fft_size - first_part]])

        magnitude = np.abs(np.fft.rfft(samples * self._window[:fft_size]))
        self._spectrum[:n_bins] = magnitude

        current_max = np.max(self._spectrum[:n_bins])
        if current_max > self._running_max:
            self._running_max = current_max
        else:
            self._running_max = max(self._running_max * 0.995, current_max, 0.01)

        normalized = np.clip(self._spectrum[:n_bins] / (self._running_max + 1e-6), 0.0, 1.5)

        # Fast attack, slow decay
        mask = normalized > self._smooth_spectrum[:n_bins]
        self._smooth_spectrum[:n_bins][mask] = normalized[mask]
        self._smooth_spectrum[:n_bins][~mask] = self._smooth_spectrum[:n_bins][~mask] * 0.85 + normalized[~mask] * 0.15

        self._output[:] = 0
        self._output[:n_bins] = self._smooth_spectrum[:n_bins]
        return self._output, n_bins


class Visualizer:
    DEFAULT_BASE_FREQ = 40.0
    DEFAULT_MODE_SCALE = 0.5
    DEFAULT_MAX_FREQ = 7000.0
    DEFAULT_CONTRAST = 1.0
    DEFAULT_COLOR_MODE = 4  # Signed
    DEFAULT_BOUNDARY = 0    # Chladni
    DEFAULT_FFT_SIZE = 8192
    DEFAULT_ASPECT_MODE = 2  # Crop

    def __init__(self, w, h):
        self.w, self.h = w, h
        self._fft_size = self.DEFAULT_FFT_SIZE
        self._color_names = ["Plasma", "Magma", "Turbo", "Viridis", "Signed"]
        self._boundary_names = [
            "Chladni",           # 0: Classic Chladni (center-constrained, free edges)
            "Clamped",           # 1: C-C-C-C (all edges fixed)
            "Free",              # 2: F-F-F-F (all edges free)
            "SS-Free",           # 3: SS-F-SS-F (x pinned, y free)
            "Clamped-SS",        # 4: C-SS-C-SS (x clamped, y pinned)
            "Clamped-Free",      # 5: C-F-C-F (x clamped, y free)
            "Cantilever",        # 6: C-F-F-F (one edge clamped)
            "Guided",            # 7: G-G-G-G (all edges guided)
        ]
        self._aspect_names = ["Full", "1:1 Letterbox", "1:1 Crop"]
        self._fps = 0.0
        self._frame_times = []
        self._time = 0.0
        self.reset_defaults()

    def reset_defaults(self):
        self.base_freq = self.DEFAULT_BASE_FREQ
        self.mode_scale = self.DEFAULT_MODE_SCALE
        self.max_freq = self.DEFAULT_MAX_FREQ
        self.contrast = self.DEFAULT_CONTRAST
        self.color_mode = self.DEFAULT_COLOR_MODE
        self.boundary = self.DEFAULT_BOUNDARY
        self.aspect_mode = self.DEFAULT_ASPECT_MODE
        self.print_status()

    def init(self):
        self.program = compileProgram(
            compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
            validate=False
        )

        verts = np.array([-1, -1, 1, -1, 1, 1, -1, 1], dtype=np.float32)
        inds = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

        self.vao = glGenVertexArrays(1)
        vbo, ebo = glGenBuffers(2)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, inds.nbytes, inds, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, None)
        glEnableVertexAttribArray(0)

        # 1D texture for spectrum
        self.spectrum_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_1D, self.spectrum_tex)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexImage1D(GL_TEXTURE_1D, 0, GL_R32F, MAX_BINS, 0, GL_RED, GL_FLOAT, None)

        self.locs = {
            'iResolution': glGetUniformLocation(self.program, "iResolution"),
            'iSpectrum': glGetUniformLocation(self.program, "iSpectrum"),
            'iNumBins': glGetUniformLocation(self.program, "iNumBins"),
            'iFreqPerBin': glGetUniformLocation(self.program, "iFreqPerBin"),
            'iMaxFreq': glGetUniformLocation(self.program, "iMaxFreq"),
            'iBaseFreq': glGetUniformLocation(self.program, "iBaseFreq"),
            'iModeScale': glGetUniformLocation(self.program, "iModeScale"),
            'iContrast': glGetUniformLocation(self.program, "iContrast"),
            'iColorMode': glGetUniformLocation(self.program, "iColorMode"),
            'iTime': glGetUniformLocation(self.program, "iTime"),
            'iBoundary': glGetUniformLocation(self.program, "iBoundary"),
            'iAspectMode': glGetUniformLocation(self.program, "iAspectMode"),
        }

    def set_base_freq(self, value):
        self.base_freq = max(10.0, min(500.0, value))
        self.print_status()

    def set_mode_scale(self, value):
        self.mode_scale = max(0.1, min(2.0, value))
        self.print_status()

    def set_max_freq(self, value):
        self.max_freq = max(500.0, min(20000.0, value))
        self.print_status()

    def set_contrast(self, value):
        self.contrast = max(0.1, min(5.0, value))
        self.print_status()

    def update_fps(self):
        now = time.time()
        self._frame_times.append(now)
        self._frame_times = [t for t in self._frame_times if now - t < 1.0]
        if len(self._frame_times) >= 2:
            self._fps = len(self._frame_times) / (self._frame_times[-1] - self._frame_times[0] + 0.001)

    def print_status(self):
        color_str = self._color_names[self.color_mode]
        boundary_str = self._boundary_names[self.boundary]
        aspect_str = self._aspect_names[self.aspect_mode]
        freq_per_bin = SAMPLE_RATE / self._fft_size
        status = (f"Base={self.base_freq:.0f}Hz  Scale={self.mode_scale:.2f}  "
                  f"MaxF={self.max_freq:.0f}Hz  Contrast={self.contrast:.1f}  "
                  f"Color={color_str}  Boundary={boundary_str}  Aspect={aspect_str}  "
                  f"FFT={self._fft_size}  FPS={self._fps:.0f}")
        print(f"\r{status:<140}", end="", flush=True)

    def render(self, spectrum, n_bins, fft_size):
        freq_per_bin = SAMPLE_RATE / fft_size
        self._fft_size = fft_size

        glBindTexture(GL_TEXTURE_1D, self.spectrum_tex)
        glTexSubImage1D(GL_TEXTURE_1D, 0, 0, n_bins, GL_RED, GL_FLOAT, spectrum[:n_bins])

        glViewport(0, 0, self.w, self.h)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self.program)
        glUniform2f(self.locs['iResolution'], self.w, self.h)
        glUniform1i(self.locs['iNumBins'], n_bins)
        glUniform1f(self.locs['iFreqPerBin'], freq_per_bin)
        glUniform1f(self.locs['iMaxFreq'], self.max_freq)
        glUniform1f(self.locs['iBaseFreq'], self.base_freq)
        glUniform1f(self.locs['iModeScale'], self.mode_scale)
        glUniform1f(self.locs['iContrast'], self.contrast)
        glUniform1i(self.locs['iColorMode'], self.color_mode)
        glUniform1f(self.locs['iTime'], self._time)
        glUniform1i(self.locs['iBoundary'], self.boundary)
        glUniform1i(self.locs['iAspectMode'], self.aspect_mode)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_1D, self.spectrum_tex)
        glUniform1i(self.locs['iSpectrum'], 0)

        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

        self._time += 0.016


def main():
    if not glfw.init():
        return

    monitor = glfw.get_primary_monitor()
    mode = glfw.get_video_mode(monitor)
    screen_w, screen_h = mode.size.width, mode.size.height

    win_w, win_h = 1280, 720
    glfw.window_hint(glfw.DECORATED, glfw.TRUE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(win_w, win_h, "Wave Plate Visualizer", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.set_window_pos(window, (screen_w - win_w) // 2, (screen_h - win_h) // 2)
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    viz = Visualizer(win_w, win_h)
    viz.init()
    audio = AudioCapture()
    audio.start()
    time.sleep(0.05)

    def on_resize(window, width, height):
        if width > 0 and height > 0:
            viz.w, viz.h = width, height
            glViewport(0, 0, width, height)

    glfw.set_framebuffer_size_callback(window, on_resize)

    is_fullscreen = False
    windowed_pos = glfw.get_window_pos(window)
    windowed_size = (win_w, win_h)

    def toggle_fullscreen():
        nonlocal is_fullscreen, windowed_pos, windowed_size
        if is_fullscreen:
            glfw.set_window_monitor(window, None, windowed_pos[0], windowed_pos[1],
                                    windowed_size[0], windowed_size[1], 0)
            glfw.set_window_attrib(window, glfw.DECORATED, glfw.TRUE)
            viz.w, viz.h = windowed_size
            is_fullscreen = False
        else:
            windowed_pos = glfw.get_window_pos(window)
            windowed_size = glfw.get_window_size(window)
            glfw.set_window_attrib(window, glfw.DECORATED, glfw.FALSE)
            glfw.set_window_monitor(window, None, 0, 0, screen_w, screen_h, 0)
            viz.w, viz.h = screen_w, screen_h
            is_fullscreen = True

    print("Wave Plate Visualizer - Steady-state plate vibration from audio FFT")
    print("Controls: UP/DOWN=base freq  W/S=scale  A/D=max freq  Z/X=contrast  LEFT/RIGHT=FFT size  V=color  P=boundary  R=aspect  ALT+ENTER=fullscreen  SPACE=reset  ESC=quit")
    print()
    viz.print_status()

    prev_keys = {}
    key_cooldowns = {}
    repeat_delay = 0.1

    def key_pressed(key):
        curr = glfw.get_key(window, key) == glfw.PRESS
        prev = prev_keys.get(key, False)
        prev_keys[key] = curr
        return curr and not prev

    def key_ready(key):
        now = time.time()
        if glfw.get_key(window, key) != glfw.PRESS:
            return False
        last = key_cooldowns.get(key, 0)
        if now - last >= repeat_delay:
            key_cooldowns[key] = now
            return True
        return False

    while not glfw.window_should_close(window):
        glfw.poll_events()
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break

        alt_held = (glfw.get_key(window, glfw.KEY_LEFT_ALT) == glfw.PRESS or
                    glfw.get_key(window, glfw.KEY_RIGHT_ALT) == glfw.PRESS)
        if alt_held and key_pressed(glfw.KEY_ENTER):
            toggle_fullscreen()

        if key_ready(glfw.KEY_UP):
            viz.set_base_freq(viz.base_freq + 5)
        if key_ready(glfw.KEY_DOWN):
            viz.set_base_freq(viz.base_freq - 5)

        if key_ready(glfw.KEY_W):
            viz.set_mode_scale(viz.mode_scale + 0.05)
        if key_ready(glfw.KEY_S):
            viz.set_mode_scale(viz.mode_scale - 0.05)

        if key_ready(glfw.KEY_D):
            viz.set_max_freq(viz.max_freq + 500)
        if key_ready(glfw.KEY_A):
            viz.set_max_freq(viz.max_freq - 500)

        if key_ready(glfw.KEY_X):
            viz.set_contrast(viz.contrast + 0.1)
        if key_ready(glfw.KEY_Z):
            viz.set_contrast(viz.contrast - 0.1)

        if key_pressed(glfw.KEY_RIGHT):
            audio.set_fft_size(audio.fft_size * 2, viz)
        if key_pressed(glfw.KEY_LEFT):
            audio.set_fft_size(audio.fft_size // 2, viz)

        if key_pressed(glfw.KEY_V):
            viz.color_mode = (viz.color_mode + 1) % len(viz._color_names)
            viz.print_status()

        if key_pressed(glfw.KEY_P):
            viz.boundary = (viz.boundary + 1) % len(viz._boundary_names)
            viz.print_status()

        if key_pressed(glfw.KEY_R):
            viz.aspect_mode = (viz.aspect_mode + 1) % len(viz._aspect_names)
            viz.print_status()

        if key_pressed(glfw.KEY_SPACE):
            viz.reset_defaults()
            audio.set_fft_size(Visualizer.DEFAULT_FFT_SIZE, viz)

        spectrum, n_bins = audio.get_spectrum()
        viz.render(spectrum, n_bins, audio.fft_size)
        glfw.swap_buffers(window)
        viz.update_fps()
        viz.print_status()

    print()
    audio.stop()
    glfw.terminate()


if __name__ == "__main__":
    main()

