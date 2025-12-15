"""Audio Reactive Chladni Plate Visualizer"""

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
MAX_FFT_SIZE = 32768  # Maximum supported FFT size
MAX_BINS = MAX_FFT_SIZE // 2 + 1  # 16385 bins

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
uniform samplerBuffer iSpectrum;  // Spectrum data as texture buffer
uniform float iFundamental;       // Plate fundamental frequency
uniform float iFreqPerBin;        // Hz per FFT bin (SAMPLE_RATE / FFT_SIZE)
uniform int iNumBins;             // Number of active bins
uniform float iComplexity;        // Mode complexity factor (0.1 - 1.0)
uniform float iMaxFreq;           // Maximum frequency to visualize (Hz)
uniform float iDominantHue;       // Dominant hue from audio (0-1)
uniform int iColorMode;           // 0 = color, 1 = black & white
uniform float iThreshold;         // Nodal line thickness threshold

#define PI 3.14159265

// HSV to RGB conversion
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// Chladni eigenmode for rectangular plate (dimensions a x 1)
// Standard formula: cos(n*pi*x/a)*cos(m*pi*y) - cos(m*pi*x/a)*cos(n*pi*y)
float chladni(vec2 p, float n, float m, float a) {
    float kn = n * PI / a;
    float km = m * PI / a;
    return cos(kn * p.x) * cos(m * PI * p.y)
         - cos(km * p.x) * cos(n * PI * p.y);
}

// Find mode (n,m) for a given frequency on rectangular plate
// Direct O(1) calculation - no loops
vec2 freqToMode(float freq, float a) {
    // Eigenvalue: lambda = (n/a)^2 + m^2
    // Frequency ratio squared gives eigenvalue ratio
    float ratio = freq / iFundamental;
    float lambda_1_2 = 1.0 / (a * a) + 4.0;  // eigenvalue of (1,2) mode
    float target = lambda_1_2 * ratio * ratio;

    if (target <= lambda_1_2) return vec2(1.0, 2.0);

    // Direct solve: pick n based on sqrt(target), then solve for m
    float sqrtT = sqrt(target);
    float n = max(1.0, floor(sqrtT * a * iComplexity));

    // Solve: (n/a)^2 + m^2 = target => m = sqrt(target - (n/a)^2)
    float na = n / a;
    float m2 = target - na * na;
    float m = max(1.0, round(sqrt(max(0.0, m2))));

    // Ensure n != m for non-trivial pattern
    if (m == n) m = n + 1.0;

    return vec2(n, m);
}

void main() {
    float aspect = iResolution.x / iResolution.y;
    vec2 plate = (fragCoord - 0.5 * iResolution.xy) / (0.5 * iResolution.y);

    float displacement = 0.0;

    int maxBin = min(iNumBins, int(iMaxFreq / iFreqPerBin));
    for (int i = 1; i < maxBin; i++) {
        float amp = texelFetch(iSpectrum, i).r;
        float freq = float(i) * iFreqPerBin;
        vec2 nm = freqToMode(freq, aspect);
        displacement += amp * chladni(plate, nm.x, nm.y, aspect);
    }

    float d = abs(displacement);
    float fw = fwidth(displacement) * 1.5;
    float intensity = 1.0 - smoothstep(iThreshold - fw, iThreshold + fw, d);

    // Single color from dominant frequency or B&W
    vec3 color = iColorMode == 0 ? hsv2rgb(vec3(iDominantHue, 0.85, 1.0)) : vec3(1.0);
    outColor = vec4(color * intensity, 1.0);
}
"""


class AudioCapture:
    def __init__(self, fft_size=4096):
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
                viz.print_status(self.fft_size, self._n_bins)

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

    def get_data(self):
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

        mask = normalized > self._smooth_spectrum[:n_bins]
        self._smooth_spectrum[:n_bins][mask] = normalized[mask]
        self._smooth_spectrum[:n_bins][~mask] = self._smooth_spectrum[:n_bins][~mask] * 0.85 + normalized[~mask] * 0.15

        self._output[:] = 0
        self._output[:n_bins] = self._smooth_spectrum[:n_bins]
        return self._output


class Visualizer:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.fundamental = 100.0
        self.complexity = 0.4
        self.max_freq = 1100.0
        self._fft_size = 4096
        self._n_bins = 2049
        self.color_mode = 0  # 0 = color, 1 = B&W
        self.threshold = 0.1  # Nodal line thickness

    def init(self):
        self.program = compileProgram(
            compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
            validate=False  # Skip validation, we'll validate after setting uniforms
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

        # Create texture buffer for spectrum data
        self.spectrum_tbo = glGenBuffers(1)
        self.spectrum_tex = glGenTextures(1)

        # Initialize buffer with zeros
        initial_data = np.zeros(MAX_BINS, dtype=np.float32)
        glBindBuffer(GL_TEXTURE_BUFFER, self.spectrum_tbo)
        glBufferData(GL_TEXTURE_BUFFER, initial_data.nbytes, initial_data, GL_DYNAMIC_DRAW)

        # Bind texture to buffer
        glBindTexture(GL_TEXTURE_BUFFER, self.spectrum_tex)
        glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, self.spectrum_tbo)

        self.loc_resolution = glGetUniformLocation(self.program, "iResolution")
        self.loc_spectrum = glGetUniformLocation(self.program, "iSpectrum")
        self.loc_fundamental = glGetUniformLocation(self.program, "iFundamental")
        self.loc_freq_per_bin = glGetUniformLocation(self.program, "iFreqPerBin")
        self.loc_num_bins = glGetUniformLocation(self.program, "iNumBins")
        self.loc_complexity = glGetUniformLocation(self.program, "iComplexity")
        self.loc_max_freq = glGetUniformLocation(self.program, "iMaxFreq")
        self.loc_dominant_hue = glGetUniformLocation(self.program, "iDominantHue")
        self.loc_color_mode = glGetUniformLocation(self.program, "iColorMode")
        self.loc_threshold = glGetUniformLocation(self.program, "iThreshold")
        self._smooth_hue = 0.0

    def set_fundamental(self, value):
        self.fundamental = max(1.0, min(10000.0, value))
        self.print_status()

    def set_complexity(self, value):
        self.complexity = max(0.01, min(1.0, value))
        self.print_status()

    def set_max_freq(self, value):
        self.max_freq = max(100.0, min(20000.0, value))
        self.print_status()

    def set_threshold(self, value):
        self.threshold = max(0.01, min(1.0, value))
        self.print_status()

    def print_status(self, fft_size=None, n_bins=None):
        if fft_size is not None:
            self._fft_size = fft_size
            self._n_bins = n_bins
        status = f"F={self.fundamental:.1f}Hz  C={self.complexity:.2f}  MaxF={self.max_freq:.0f}Hz  T={self.threshold:.2f}  FFT={self._fft_size} ({self._n_bins} bins)"
        print(f"\r{status:<100}", end="", flush=True)

    def render(self, spectrum, fft_size):
        n_bins = fft_size // 2 + 1
        freq_per_bin = SAMPLE_RATE / fft_size

        # Compute dominant hue from weighted average of frequencies
        max_bin = min(n_bins, int(self.max_freq / freq_per_bin))
        freqs = np.arange(1, max_bin) * freq_per_bin
        amps = spectrum[1:max_bin]
        total_amp = np.sum(amps) + 1e-6
        weighted_freq = np.sum(freqs * amps) / total_amp
        # Map to hue: 0 (red) to 0.8 (violet)
        target_hue = 0.8 * weighted_freq / self.max_freq
        target_hue = np.clip(target_hue, 0.0, 0.8)
        # Smooth the hue transition
        self._smooth_hue = self._smooth_hue * 0.85 + target_hue * 0.15

        # Update spectrum texture buffer
        glBindBuffer(GL_TEXTURE_BUFFER, self.spectrum_tbo)
        glBufferSubData(GL_TEXTURE_BUFFER, 0, spectrum.nbytes, spectrum)

        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.program)
        glUniform2f(self.loc_resolution, self.w, self.h)
        glUniform1f(self.loc_fundamental, self.fundamental)
        glUniform1f(self.loc_freq_per_bin, freq_per_bin)
        glUniform1i(self.loc_num_bins, n_bins)
        glUniform1f(self.loc_complexity, self.complexity)
        glUniform1f(self.loc_max_freq, self.max_freq)
        glUniform1f(self.loc_dominant_hue, self._smooth_hue)
        glUniform1i(self.loc_color_mode, self.color_mode)
        glUniform1f(self.loc_threshold, self.threshold)

        # Bind spectrum texture to texture unit 0
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_BUFFER, self.spectrum_tex)
        glUniform1i(self.loc_spectrum, 0)

        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)


def main():
    if not glfw.init():
        return

    monitor = glfw.get_primary_monitor()
    mode = glfw.get_video_mode(monitor)
    screen_w, screen_h = mode.size.width, mode.size.height

    # Start windowed
    win_w, win_h = 1280, 720
    glfw.window_hint(glfw.DECORATED, glfw.TRUE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(win_w, win_h, "Chladni", None, None)
    if not window:
        glfw.terminate()
        return

    # Center the window
    glfw.set_window_pos(window, (screen_w - win_w) // 2, (screen_h - win_h) // 2)
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    viz = Visualizer(win_w, win_h)
    viz.init()
    audio = AudioCapture()
    audio.start()
    time.sleep(0.05)

    # Resize callback to update viewport and shader resolution
    def on_resize(window, width, height):
        if width > 0 and height > 0:
            viz.w, viz.h = width, height
            glViewport(0, 0, width, height)

    glfw.set_framebuffer_size_callback(window, on_resize)

    # Fullscreen toggle state
    is_fullscreen = False
    windowed_pos = glfw.get_window_pos(window)
    windowed_size = (win_w, win_h)
    prev_alt_enter = False

    def toggle_fullscreen():
        nonlocal is_fullscreen, windowed_pos, windowed_size
        if is_fullscreen:
            # Go windowed
            glfw.set_window_monitor(window, None, windowed_pos[0], windowed_pos[1],
                                    windowed_size[0], windowed_size[1], 0)
            glfw.set_window_attrib(window, glfw.DECORATED, glfw.TRUE)
            viz.w, viz.h = windowed_size
            glViewport(0, 0, windowed_size[0], windowed_size[1])
            is_fullscreen = False
        else:
            # Save windowed state
            windowed_pos = glfw.get_window_pos(window)
            windowed_size = glfw.get_window_size(window)
            # Go borderless fullscreen
            glfw.set_window_attrib(window, glfw.DECORATED, glfw.FALSE)
            glfw.set_window_monitor(window, None, 0, 0, screen_w, screen_h, 0)
            viz.w, viz.h = screen_w, screen_h
            glViewport(0, 0, screen_w, screen_h)
            is_fullscreen = True

    print("Controls:")
    print("  UP/DOWN     = fundamental ±2%    PGUP/PGDN = fundamental ±10%")
    print("  W/S         = complexity ±0.01   A/D       = max freq ±500 Hz")
    print("  Q/E         = threshold ±0.01    LEFT/RIGHT = halve/double FFT")
    print("  C           = toggle color/B&W   ALT+ENTER  = toggle fullscreen")
    print("  ESC         = quit")
    print()
    viz.print_status(audio.fft_size, audio._n_bins)

    prev_left = False
    prev_right = False
    prev_alt_enter = False
    prev_c = False

    # Key repeat timing
    key_cooldowns = {}
    repeat_delay = 0.15  # seconds between repeats when holding

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

        # Alt+Enter for fullscreen toggle
        alt_held = (glfw.get_key(window, glfw.KEY_LEFT_ALT) == glfw.PRESS or
                    glfw.get_key(window, glfw.KEY_RIGHT_ALT) == glfw.PRESS)
        enter_pressed = glfw.get_key(window, glfw.KEY_ENTER) == glfw.PRESS
        curr_alt_enter = alt_held and enter_pressed
        if curr_alt_enter and not prev_alt_enter:
            toggle_fullscreen()
        prev_alt_enter = curr_alt_enter

        if key_ready(glfw.KEY_UP):
            viz.set_fundamental(viz.fundamental * 1.02)
        if key_ready(glfw.KEY_DOWN):
            viz.set_fundamental(viz.fundamental / 1.02)
        if key_ready(glfw.KEY_PAGE_UP):
            viz.set_fundamental(viz.fundamental * 1.1)
        if key_ready(glfw.KEY_PAGE_DOWN):
            viz.set_fundamental(viz.fundamental / 1.1)

        if key_ready(glfw.KEY_W):
            viz.set_complexity(viz.complexity + 0.01)
        if key_ready(glfw.KEY_S):
            viz.set_complexity(viz.complexity - 0.01)

        if key_ready(glfw.KEY_D):
            viz.set_max_freq(viz.max_freq + 500)
        if key_ready(glfw.KEY_A):
            viz.set_max_freq(viz.max_freq - 500)

        if key_ready(glfw.KEY_E):
            viz.set_threshold(viz.threshold + 0.01)
        if key_ready(glfw.KEY_Q):
            viz.set_threshold(viz.threshold - 0.01)

        curr_right = glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS
        curr_left = glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS
        if curr_right and not prev_right:
            audio.set_fft_size(audio.fft_size * 2, viz)
        if curr_left and not prev_left:
            audio.set_fft_size(audio.fft_size // 2, viz)
        prev_right = curr_right
        prev_left = curr_left

        # Toggle color mode
        curr_c = glfw.get_key(window, glfw.KEY_C) == glfw.PRESS
        if curr_c and not prev_c:
            viz.color_mode = 1 - viz.color_mode
        prev_c = curr_c

        viz.render(audio.get_data(), audio.fft_size)
        glfw.swap_buffers(window)

    print()  # newline after status line
    audio.stop()
    glfw.terminate()


if __name__ == "__main__":
    main()
