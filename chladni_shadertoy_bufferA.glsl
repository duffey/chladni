// Wave Plate Visualizer - Chladni Patterns from Audio
// BUFFER A: State storage for keyboard controls
//
// Setup:
//   iChannel0 = Buffer A (itself, for persistence)
//   iChannel1 = Keyboard
//
// Pixel (0,0) stores: baseFreq, modeScale, maxFreq, colorMode
// Pixel (1,0) stores: contrast, aspectMode, boundary, unused
//
// Controls:
//   UP/DOWN: Base frequency +/-
//   W/S: Mode scale +/-
//   A/D: Max frequency +/-
//   Z/X: Contrast +/-
//   Q/E: Boundary mode prev/next
//   V: Cycle color mode (Plasma, Magma, Turbo, Viridis, Signed)
//   B: Cycle aspect mode (Fill, Fit, Crop)
//   SPACE: Reset to defaults

#define keyClick(ascii) (texelFetch(iChannel1, ivec2(ascii, 1), 0).x > 0.)
#define keyDown(ascii)  (texelFetch(iChannel1, ivec2(ascii, 0), 0).x > 0.)

// Key codes
#define KEY_V     86
#define KEY_B     66
#define KEY_W     87
#define KEY_S     83
#define KEY_A     65
#define KEY_D     68
#define KEY_Q     81
#define KEY_E     69
#define KEY_Z     90
#define KEY_X     88
#define KEY_UP    38
#define KEY_DOWN  40
#define KEY_SPACE 32

// Defaults matching Python version
#define DEFAULT_BASE_FREQ 40.0
#define DEFAULT_MODE_SCALE 0.5
#define DEFAULT_MAX_FREQ 7000.0
#define DEFAULT_THRESHOLD 0.01
#define DEFAULT_CONTRAST 1.0
#define DEFAULT_COLOR_MODE 4.0   // 4=Signed
#define DEFAULT_ASPECT_MODE 2.0  // 0=Fill, 1=Fit, 2=Crop
#define DEFAULT_BOUNDARY 0.0     // 0=Chladni

void mainImage(out vec4 O, in vec2 F) {
    ivec2 p = ivec2(F);

    // Only process state pixels
    if (p.y != 0 || p.x > 1) {
        O = vec4(0);
        return;
    }

    // Load previous state
    vec4 state0 = texelFetch(iChannel0, ivec2(0, 0), 0);
    vec4 state1 = texelFetch(iChannel0, ivec2(1, 0), 0);

    // Initialize on first frame or SPACE reset
    if (iFrame == 0 || keyClick(KEY_SPACE)) {
        state0 = vec4(DEFAULT_BASE_FREQ, DEFAULT_MODE_SCALE, DEFAULT_MAX_FREQ, DEFAULT_COLOR_MODE);
        state1 = vec4(DEFAULT_CONTRAST, DEFAULT_ASPECT_MODE, DEFAULT_BOUNDARY, 0.0);
    }

    float baseFreq = state0.x;
    float modeScale = state0.y;
    float maxFreq = state0.z;
    float colorMode = state0.w;
    float contrast = state1.x;
    float aspectMode = state1.y;
    float boundary = state1.z;

    // Handle key inputs
    // V cycles color mode (0=Plasma, 1=Magma, 2=Turbo, 3=Viridis, 4=Signed)
    if (keyClick(KEY_V)) {
        colorMode = mod(colorMode + 1.0, 5.0);
    }
    // B cycles aspect mode (Fill -> Fit -> Crop)
    if (keyClick(KEY_B)) {
        aspectMode = mod(aspectMode + 1.0, 3.0);
    }

    // Continuous adjustments while held (matching Python controls)
    if (keyDown(KEY_UP))   baseFreq += 1.0;      // Base frequency
    if (keyDown(KEY_DOWN)) baseFreq -= 1.0;
    if (keyDown(KEY_W))    modeScale += 0.01;    // Mode scale
    if (keyDown(KEY_S))    modeScale -= 0.01;
    if (keyDown(KEY_D))    maxFreq += 100.0;     // Max frequency
    if (keyDown(KEY_A))    maxFreq -= 100.0;
    if (keyDown(KEY_X))    contrast += 0.05;     // Contrast
    if (keyDown(KEY_Z))    contrast -= 0.05;

    // Q/E cycle boundary mode
    if (keyClick(KEY_Q))   boundary = mod(boundary - 1.0 + 8.0, 8.0);
    if (keyClick(KEY_E))   boundary = mod(boundary + 1.0, 8.0);

    // Clamp parameters
    baseFreq = clamp(baseFreq, 10.0, 500.0);
    modeScale = clamp(modeScale, 0.1, 2.0);
    maxFreq = clamp(maxFreq, 500.0, 20000.0);
    contrast = clamp(contrast, 0.1, 5.0);

    // Output based on pixel
    if (p.x == 0) {
        O = vec4(baseFreq, modeScale, maxFreq, colorMode);
    } else {
        O = vec4(contrast, aspectMode, boundary, 0.0);
    }
}
