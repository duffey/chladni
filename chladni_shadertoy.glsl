// Wave Plate Visualizer - Chladni Patterns from Audio
// C and Python versions: https://github.com/duffey/chladni
// IMAGE TAB - Main visualization shader
//
// Setup:
//   iChannel0 = SoundCloud/Microphone (audio input)
//   iChannel1 = Buffer A (for persistent state from chladni_shadertoy_bufferA.glsl)
//
// Controls (keyboard - handled by Buffer A):
//   UP/DOWN: Base frequency +/-
//   W/S: Mode scale +/-
//   A/D: Max frequency +/-
//   Z/X: Contrast +/-
//   Q/E: Boundary mode prev/next (Chladni, Clamped, Free, SS-Free, etc.)
//   V: Cycle color mode (Plasma, Magma, Turbo, Viridis, Signed)
//   B: Cycle aspect mode (Fill, Fit, Crop)
//   SPACE: Reset to defaults

#define PI 3.14159265
#define TAU 6.28318530

// ============================================================================
// COLORMAPS
// ============================================================================

vec3 plasma(float t) {
    t = clamp(t, 0.0, 1.0);
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

vec3 diverging(float t) {
    t = clamp(t, -1.0, 1.0);
    vec3 cold = vec3(0.085, 0.180, 0.525);
    vec3 cool = vec3(0.350, 0.550, 0.850);
    vec3 neutral = vec3(0.970, 0.970, 0.970);
    vec3 warm = vec3(0.900, 0.450, 0.350);
    vec3 hot = vec3(0.600, 0.050, 0.100);
    if (t < -0.5) return mix(cold, cool, (t + 1.0) * 2.0);
    if (t < 0.0) return mix(cool, neutral, (t + 0.5) * 2.0);
    if (t < 0.5) return mix(neutral, warm, t * 2.0);
    return mix(warm, hot, (t - 0.5) * 2.0);
}

// ============================================================================
// PLATE EIGENMODES
// ============================================================================

float beamCC(float x, float n) {
    float k = (n + 0.5) * PI;
    float s = sin(k * x);
    float edge = 1.0 - exp(-3.0 * min(x + 1.0, 1.0 - x));
    return s * edge;
}

float beamFF(float x, float n) {
    return cos(n * PI * x);
}

float beamCF(float x, float n) {
    float k = (n + 0.25) * PI;
    float xNorm = (x + 1.0) * 0.5;
    return sin(k * xNorm) - sinh(k * xNorm) * exp(-k);
}

float modeSimplySupported(vec2 p, float n, float m, float aspect) {
    float qx = (p.x / aspect + 1.0) * 0.5;
    float qy = (p.y + 1.0) * 0.5;
    return sin(n * PI * qx) * sin(m * PI * qy);
}

float modeClamped(vec2 p, float n, float m, float aspect) {
    float px = p.x / aspect;
    return beamCC(px, n) * beamCC(p.y, m);
}

float modeFree(vec2 p, float n, float m, float aspect) {
    float px = p.x / aspect;
    float xMode = (n < 0.5) ? 1.0 : beamFF(px, n);
    float yMode = (m < 0.5) ? 1.0 : beamFF(p.y, m);
    return xMode * yMode;
}

float modeSSF(vec2 p, float n, float m, float aspect) {
    float px = p.x / aspect;
    float qx = (px + 1.0) * 0.5;
    float xMode = sin(n * PI * qx);
    float yMode = (m < 0.5) ? 1.0 : cos(m * PI * p.y);
    return xMode * yMode;
}

float modeCSS(vec2 p, float n, float m, float aspect) {
    float px = p.x / aspect;
    float qy = (p.y + 1.0) * 0.5;
    return beamCC(px, n) * sin(m * PI * qy);
}

float modeCF(vec2 p, float n, float m, float aspect) {
    float px = p.x / aspect;
    float xMode = beamCC(px, n);
    float yMode = (m < 0.5) ? 1.0 : beamFF(p.y, m);
    return xMode * yMode;
}

float modeCantilever(vec2 p, float n, float m, float aspect) {
    float px = p.x / aspect;
    float xMode = beamCF(px, n);
    float yMode = (m < 0.5) ? 1.0 : beamFF(p.y, m);
    return xMode * yMode;
}

float modeGuided(vec2 p, float n, float m, float aspect) {
    float px = p.x / aspect;
    return cos(n * PI * px) * cos(m * PI * p.y);
}

float modeChladni(vec2 p, float n, float m, float aspect) {
    float px = p.x / aspect;
    float py = p.y;
    float mode_nm = cos(n * PI * px) * cos(m * PI * py);
    float mode_mn = cos(m * PI * px) * cos(n * PI * py);
    return mode_nm - mode_mn;
}

float computeSingleMode(vec2 p, float n, float m, float aspect, int boundary) {
    vec2 pc = p;
    if (boundary == 1 || boundary == 4 || boundary == 5 || boundary == 6) {
        pc.x = clamp(p.x, -aspect, aspect);
        pc.y = clamp(p.y, -1.0, 1.0);
    }
    if (boundary == 0) return modeSimplySupported(p, n, m, aspect);
    else if (boundary == 1) return modeClamped(pc, n, m, aspect);
    else if (boundary == 2) return modeFree(p, n, m, aspect);
    else if (boundary == 3) return modeSSF(p, n, m, aspect);
    else if (boundary == 4) return modeCSS(pc, n, m, aspect);
    else if (boundary == 5) return modeCF(pc, n, m, aspect);
    else if (boundary == 6) return modeCantilever(pc, n, m, aspect);
    else return modeGuided(p, n, m, aspect);
}

float computeModeSum(vec2 p, float targetLambda, float aspect, int boundary) {
    float sqrtL = sqrt(max(5.0, targetLambda));
    float n = max(2.0, floor(sqrtL * 0.9 + 0.5));
    float m = max(1.0, floor(sqrtL * 0.5 + 0.5));
    if (n <= m) n = m + 1.0;
    if (boundary == 0) return modeChladni(p, n, m, aspect);
    else return computeSingleMode(p, n, m, aspect, boundary);
}

// ============================================================================
// MAIN
// ============================================================================

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    float windowAspect = iResolution.x / iResolution.y;

    // Read state from Buffer A (iChannel1)
    // Pixel (0,0): baseFreq, modeScale, maxFreq, colorMode
    // Pixel (1,0): contrast, aspectMode, boundary, unused
    vec4 state0 = texelFetch(iChannel1, ivec2(0, 0), 0);
    vec4 state1 = texelFetch(iChannel1, ivec2(1, 0), 0);

    float baseFreq = state0.x;
    float modeScale = state0.y;
    float maxFreq = state0.z;
    int colorMode = int(state0.w);
    float contrast = state1.x;
    int aspectMode = int(state1.y);
    int boundary = int(state1.z);

    // Fallback defaults if buffer not initialized
    if (baseFreq < 1.0) {
        baseFreq = 40.0;
        modeScale = 0.5;
        maxFreq = 7000.0;
        contrast = 1.0;
        colorMode = 4;
        aspectMode = 2;
        boundary = 0;
    }

    // Handle aspect modes
    vec2 p;
    bool outOfBounds = false;
    float plateAspect = 1.0;

    if (aspectMode == 0) {
        plateAspect = windowAspect;
        p.x = (uv.x - 0.5) * 2.0 * windowAspect;
        p.y = (uv.y - 0.5) * 2.0;
    } else if (aspectMode == 1) {
        plateAspect = 1.0;
        if (windowAspect > 1.0) {
            float plateWidth = 1.0 / windowAspect;
            float margin = (1.0 - plateWidth) / 2.0;
            if (uv.x < margin || uv.x > 1.0 - margin) {
                outOfBounds = true;
            } else {
                float localX = (uv.x - margin) / plateWidth;
                p.x = (localX - 0.5) * 2.0;
                p.y = (uv.y - 0.5) * 2.0;
            }
        } else {
            float plateHeight = windowAspect;
            float margin = (1.0 - plateHeight) / 2.0;
            if (uv.y < margin || uv.y > 1.0 - margin) {
                outOfBounds = true;
            } else {
                float localY = (uv.y - margin) / plateHeight;
                p.x = (uv.x - 0.5) * 2.0;
                p.y = (localY - 0.5) * 2.0;
            }
        }
    } else {
        plateAspect = 1.0;
        if (windowAspect > 1.0) {
            p.x = (uv.x - 0.5) * 2.0 * windowAspect;
            p.y = (uv.y - 0.5) * 2.0;
        } else {
            p.x = (uv.x - 0.5) * 2.0;
            p.y = (uv.y - 0.5) * 2.0 / windowAspect;
        }
    }

    if (outOfBounds) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Audio parameters
    float freqStep = 11025.0 / 512.0;  // ~21.5 Hz per bin
    int maxBin = min(512, int(maxFreq / freqStep));

    // Accumulate displacement from audio spectrum
    float displacement = 0.0;
    float totalEnergy = 0.0;

    for (int i = 1; i < 512; i++) {
        if (i >= maxBin) break;

        float freq = float(i) * freqStep;
        float u = freq / 11025.0;

        // Read FFT with dB conversion (critical for correct visual density)
        float v = texture(iChannel0, vec2(u, 0.25)).x;
        float amp = pow(10.0, (-100.0 + v * 70.0) / 20.0) * 30.0;

        if (amp < 0.01) continue;

        float ratio = freq / baseFreq;
        float targetLambda = 5.0 + ratio * modeScale * 10.0;

        float mode = computeModeSum(p, targetLambda, plateAspect, boundary);

        displacement += amp * mode;
        totalEnergy += amp;
    }

    if (totalEnergy > 0.1) {
        displacement /= sqrt(totalEnergy);
    }

    float d = displacement * contrast;
    vec3 color;

    if (colorMode == 0) {
        float energy = tanh(abs(d));
        color = plasma(energy);
    } else if (colorMode == 1) {
        float energy = tanh(abs(d));
        color = magma(energy);
    } else if (colorMode == 2) {
        float energy = tanh(abs(d));
        color = turbo(energy);
    } else if (colorMode == 3) {
        float energy = tanh(abs(d));
        color = viridis(energy);
    } else {
        float signed_d = tanh(d);
        color = diverging(signed_d);
    }

    fragColor = vec4(color, 1.0);
}
