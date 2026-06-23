# OCR Reasoning Robustness Benchmark — Perturbation Catalog

> Total: **26 perturbation types**, each with **5 severity levels**, applied to 100 sample images.

---

## Overview

| Category | Count | Perturbation Types |
|----------|-------|--------------------|
| Noise | 3 | Gaussian Noise, Shot Noise, Impulse Noise |
| Blur | 5 | Defocus Blur, Glass Blur, Motion Blur, Motion Blur (Angle), Zoom Blur |
| Weather | 3 | Snow, Frost, Fog |
| Digital | 5 | JPEG Compression, Brightness, Contrast, Elastic Transform, Pixelate |
| Geometric | 3 | Rotation, Perspective, Slight Rotation |
| Screen/Capture Artifacts | 3 | Screen Recapture, Moiré, Glare |
| OCR-Specific | 4 | Watermark, Font Degradation, Paper Fold/Crease, Color Channel Shift |

---

## 1. Noise

### 1.1 Gaussian Noise

| Item | Detail |
|------|--------|
| Script | `gaussian_noise_perturbation.py` |
| Description | Adds zero-mean Gaussian noise with varying standard deviation |
| Levels | sigma = [8, 12, 18, 26, 38] |
| Output prefix | `sample_100_gaussian_{sigma}` |
| Reference | ImageNet-C |

### 1.2 Shot Noise

| Item | Detail |
|------|--------|
| Script | `shot_noise_perturbation.py` |
| Description | Adds Poisson (shot) noise to simulate photographic sensor noise |
| Levels | lambda = [60, 25, 12, 5, 3] (lower = more noise) |
| Output prefix | `sample_100_shot_{lambda}` |
| Reference | ImageNet-C |

### 1.3 Impulse Noise

| Item | Detail |
|------|--------|
| Script | `impulse_noise_perturbation.py` |
| Description | Adds salt-and-pepper noise to random pixels |
| Levels | proportion = [1%, 2%, 3%, 5%, 7%] |
| Output prefix | `sample_100_impulse_{n}pct` |
| Reference | ImageNet-C |

---

## 2. Blur

### 2.1 Defocus Blur

| Item | Detail |
|------|--------|
| Script | `defocus_blur_perturbation.py` |
| Description | Applies disk-shaped convolution kernel to simulate out-of-focus camera |
| Levels | radius = [3, 4, 6, 8, 10] pixels |
| Output prefix | `sample_100_defocus_{radius}` |
| Reference | ImageNet-C |

### 2.2 Glass Blur

| Item | Detail |
|------|--------|
| Script | `glass_blur_perturbation.py` |
| Description | Random pixel displacement to simulate frosted glass effect |
| Levels | displacement = [1, 2, 3, 4, 5] pixels |
| Output prefix | `sample_100_glass_{level}` |
| Reference | ImageNet-C |

### 2.3 Motion Blur

| Item | Detail |
|------|--------|
| Script | `motion_blur_perturbation.py` |
| Description | Horizontal motion blur with varying kernel sizes |
| Levels | kernel = [3, 7, 11, 15, 21] pixels |
| Output prefix | `sample_100_motion_blur_{kernel}` |
| Reference | ImageNet-C |

### 2.4 Motion Blur (Angle)

| Item | Detail |
|------|--------|
| Script | `motion_blur_angle_perturbation.py` |
| Description | Motion blur at different directional angles (fixed kernel size 15) |
| Levels | angle = [0, 45, 90, 135, 180] degrees |
| Output prefix | `sample_100_motion_blur_angle_{angle}` |
| Reference | ImageNet-C variant |

### 2.5 Zoom Blur

| Item | Detail |
|------|--------|
| Script | `zoom_blur_perturbation.py` |
| Description | Radial zoom blur simulating camera zoom during exposure |
| Levels | zoom factor = [1.05, 1.10, 1.15, 1.20, 1.25] |
| Output prefix | `sample_100_zoom_{n}pct` |
| Reference | ImageNet-C |

---

## 3. Weather

### 3.1 Snow

| Item | Detail |
|------|--------|
| Script | `snow_perturbation.py` |
| Description | Adds falling snow streaks with varying density |
| Levels | intensity = [1, 2, 3, 4, 5] |
| Output prefix | `sample_100_snow_{level}` |
| Reference | ImageNet-C |

### 3.2 Frost

| Item | Detail |
|------|--------|
| Script | `frost_perturbation.py` |
| Description | Adds frost/ice crystal overlay with multi-octave noise pattern |
| Levels | intensity = [1, 2, 3, 4, 5] |
| Output prefix | `sample_100_frost_{level}` |
| Reference | ImageNet-C |

### 3.3 Fog

| Item | Detail |
|------|--------|
| Script | `fog_perturbation.py` |
| Description | Adds fog effect via white blending and contrast reduction |
| Levels | intensity = [1, 2, 3, 4, 5] |
| Output prefix | `sample_100_fog_{level}` |
| Reference | ImageNet-C |

---

## 4. Digital

### 4.1 JPEG Compression

| Item | Detail |
|------|--------|
| Script | `compress_images.py` |
| Description | JPEG compression with varying quality levels |
| Levels | quality = [30, 50, 70, 90, 100] |
| Output prefix | `sample_100_compress_{quality}` |
| Reference | ImageNet-C |

### 4.2 Brightness / Lighting

| Item | Detail |
|------|--------|
| Script | `lighting_perturbation.py` |
| Description | Adjusts image brightness by a multiplicative factor |
| Levels | factor = [0.3, 0.7, 1.0, 1.3, 1.7] |
| Output prefix | `sample_100_brightness_{factor}` |
| Reference | ImageNet-C |

### 4.3 Contrast

| Item | Detail |
|------|--------|
| Script | `contrast_perturbation.py` |
| Description | Adjusts image contrast by a multiplicative factor |
| Levels | factor = [0.3, 0.7, 1.0, 1.3, 1.7] |
| Output prefix | `sample_100_contrast_{factor}` |
| Reference | ImageNet-C |

### 4.4 Elastic Transform

| Item | Detail |
|------|--------|
| Script | `elastic_perturbation.py` |
| Description | Smooth random displacement field deformation |
| Levels | alpha = [50, 100, 150, 200, 250] |
| Output prefix | `sample_100_elastic_{alpha}` |
| Reference | ImageNet-C |

### 4.5 Pixelate

| Item | Detail |
|------|--------|
| Script | `pixelate_perturbation.py` |
| Description | Downscale then upscale with nearest neighbor to create blocky pixels |
| Levels | block size = [4, 8, 12, 16, 20] pixels |
| Output prefix | `sample_100_pixelate_{size}` |
| Reference | ImageNet-C |

---

## 5. Geometric

### 5.1 Rotation

| Item | Detail |
|------|--------|
| Script | `rotation_perturbation.py` |
| Description | Rotates image by specified angles with white border padding |
| Levels | angle = [0, 45, 90, 135, 180] degrees |
| Output prefix | `sample_100_rotation_{angle}` |
| Reference | Custom |

### 5.2 Perspective

| Item | Detail |
|------|--------|
| Script | `perspective.py` |
| Description | Perspective transformation to simulate camera angle distortion |
| Levels | intensity = [1, 2, 3, 4, 5] |
| Output prefix | `sample_100_perspective_{level}` |
| Reference | Custom |

### 5.3 Slight Rotation

| Item | Detail |
|------|--------|
| Script | `slight_rotation_perturbation.py` |
| Description | Small-angle rotation keeping original dimensions, white-filled corners. Simulates handheld camera tilt or scanner skew. |
| Levels | angle = [1, 2, 4, 7, 10] degrees |
| Output prefix | `sample_100_slight_rotation_{angle}` |
| Reference | Custom |

---

### 6.1 Screen Recapture

| Item | Detail |
|------|--------|
| Script | `screen_recapture_perturbation.py` |
| Description | Combines moiré pattern + screen glare to simulate photographing a screen |
| Levels | (freq, moiré, glare) = [(8, 0.03, 0.05), (12, 0.06, 0.10), (18, 0.10, 0.15), (25, 0.15, 0.22), (35, 0.22, 0.30)] |
| Output prefix | `sample_100_screen_recapture_{level}` |
| Reference | Custom |

### 6.2 Moiré

| Item | Detail |
|------|--------|
| Script | `moire_perturbation.py` |
| Description | Adds moiré interference pattern to simulate screen pixel grid artifacts |
| Levels | (freq, intensity) = [(8, 0.03), (12, 0.12), (18, 0.20), (25, 0.30), (35, 0.40)] |
| Output prefix | `sample_100_moire_{level}` |
| Reference | Custom |

### 6.3 Glare

| Item | Detail |
|------|--------|
| Script | `glare_perturbation.py` |
| Description | Adds radial gradient glare to simulate screen reflection |
| Levels | intensity = [0.05, 0.10, 0.15, 0.22, 0.30] |
| Output prefix | `sample_100_glare_{level}` |
| Reference | Custom |

---

## 7. OCR-Specific (New)

### 7.1 Watermark

| Item | Detail |
|------|--------|
| Script | `watermark_perturbation.py` |
| Description | Overlays tiled semi-transparent watermark text at -30° angle. Word randomly selected per image from pool ["SAMPLE", "DRAFT", "COPY", "PREVIEW", "CONFIDENTIAL"], fixed across levels. |
| Levels | (opacity, font_size, spacing) = [(0.08, 28, 160), (0.15, 36, 120), (0.25, 44, 90), (0.38, 52, 60), (0.55, 64, 40)] |
| Output prefix | `sample_100_watermark_{level}` |
| Reference | Custom — simulates web image watermarks |

### 7.2 Font Degradation

| Item | Detail |
|------|--------|
| Script | `font_degradation_perturbation.py` |
| Description | Combines resolution degradation (downscale + nearest-neighbor upscale) with morphological erosion/dilation to simulate low-quality scanning, faded ink, or ink bleeding |
| Levels | (scale, erode_k, dilate_k) = [(0.70, 0, 0), (0.50, 2, 0), (0.35, 2, 2), (0.25, 3, 2), (0.15, 3, 3)] |
| Output prefix | `sample_100_font_degrade_{level}` |
| Reference | Custom — simulates document print/scan degradation |

### 7.3 Paper Fold / Crease

| Item | Detail |
|------|--------|
| Script | `paper_fold_perturbation.py` |
| Description | Adds visible fold crease lines with highlight-shadow brightness profile and geometric warping near the fold, simulating scanned folded documents |
| Levels | (folds, intensity, warp, width) = [(1, 0.08, 0, 20), (1, 0.15, 1.0, 28), (2, 0.20, 2.0, 35), (3, 0.28, 3.0, 42), (4, 0.35, 4.5, 50)] |
| Output prefix | `sample_100_paper_fold_{level}` |
| Reference | Custom — simulates physical document damage |

### 7.4 Color Channel Shift

| Item | Detail |
|------|--------|
| Script | `color_shift_perturbation.py` |
| Description | Shifts R/G/B channels by different pixel offsets to simulate chromatic aberration or scanner color misregistration. G channel stays fixed; R and B shift in opposite diagonal directions. |
| Levels | (R_offset, G_offset, B_offset) = [((0,1),(0,0),(0,-1)), ((1,2),(0,0),(-1,-2)), ((2,4),(0,0),(-2,-4)), ((3,6),(0,0),(-3,-6)), ((5,10),(0,0),(-5,-10))] |
| Output prefix | `sample_100_color_shift_{level}` |
| Reference | Custom — simulates optical chromatic aberration |

---

## Directory Structure

```
pertubation/
├── sample_100_gaussian_{8,12,18,26,38}/
├── sample_100_shot_{60,25,12,5,3}/
├── sample_100_impulse_{1,2,3,5,7}pct/
├── sample_100_defocus_{3,4,6,8,10}/
├── sample_100_glass_{1..5}/
├── sample_100_motion_blur_{3,7,11,15,21}/
├── sample_100_motion_blur_angle_{0,45,90,135,180}/
├── sample_100_zoom_{5,10,15,20,25}pct/
├── sample_100_snow_{1..5}/
├── sample_100_frost_{1..5}/
├── sample_100_fog_{1..5}/
├── sample_100_compress_{30,50,70,90,100}/
├── sample_100_brightness_{0.3,0.7,1.0,1.3,1.7}/
├── sample_100_contrast_{0.3,0.7,1.0,1.3,1.7}/
├── sample_100_elastic_{50,100,150,200,250}/
├── sample_100_pixelate_{4,8,12,16,20}/
├── sample_100_rotation_{0,45,90,135,180}/
├── sample_100_perspective_{1..5}/
├── sample_100_slight_rotation_{1,2,4,7,10}/
├── sample_100_screen_recapture_{1..5}/
├── sample_100_moire_{1..5}/
├── sample_100_glare_{1..5}/
├── sample_100_watermark_{1..5}/
├── sample_100_font_degrade_{1..5}/
├── sample_100_paper_fold_{1..5}/
└── sample_100_color_shift_{1..5}/
```

## eval_full.py Registration

```python
PERTURBATIONS = [
    # --- ImageNet-C: Noise ---
    ("gaussian_noise",     "sample_100_gaussian_",          [8, 12, 18, 26, 38],          ""),
    ("shot_noise",         "sample_100_shot_",              [60, 25, 12, 5, 3],           ""),
    ("impulse_noise",      "sample_100_impulse_",           [1, 2, 3, 5, 7],              "pct"),
    # --- ImageNet-C: Blur ---
    ("defocus_blur",       "sample_100_defocus_",           [3, 4, 6, 8, 10],             ""),
    ("glass_blur",         "sample_100_glass_",             [1, 2, 3, 4, 5],              ""),
    ("motion_blur",        "sample_100_motion_blur_",       [3, 7, 11, 15, 21],           ""),
    ("motion_blur_angle",  "sample_100_motion_blur_angle_", [0, 45, 90, 135, 180],        ""),
    ("zoom_blur",          "sample_100_zoom_",              [5, 10, 15, 20, 25],          "pct"),
    # --- ImageNet-C: Weather ---
    ("snow",               "sample_100_snow_",              [1, 2, 3, 4, 5],              ""),
    ("frost",              "sample_100_frost_",             [1, 2, 3, 4, 5],              ""),
    ("fog",                "sample_100_fog_",               [1, 2, 3, 4, 5],              ""),
    # --- ImageNet-C: Digital ---
    ("compression",        "sample_100_compress_",          [30, 50, 70, 90, 100],        ""),
    ("brightness",         "sample_100_brightness_",        [0.3, 0.7, 1.0, 1.3, 1.7],   ""),
    ("contrast",           "sample_100_contrast_",          [0.3, 0.7, 1.0, 1.3, 1.7],   ""),
    ("elastic",            "sample_100_elastic_",           [50, 100, 150, 200, 250],     ""),
    ("pixelate",           "sample_100_pixelate_",          [4, 8, 12, 16, 20],           ""),
    # --- Geometric ---
    ("rotation",           "sample_100_rotation_",          [0, 45, 90, 135, 180],        ""),
    ("perspective",        "sample_100_perspective_",       [1, 2, 3, 4, 5],              ""),
    ("slight_rotation",    "sample_100_slight_rotation_",   [1, 2, 4, 7, 10],             ""),
    # --- Screen/Capture ---
    ("screen_recapture",   "sample_100_screen_recapture_",  [1, 2, 3, 4, 5],             ""),
    ("moire",              "sample_100_moire_",             [1, 2, 3, 4, 5],              ""),
    ("glare",              "sample_100_glare_",             [1, 2, 3, 4, 5],              ""),
    # --- OCR-Specific ---
    ("watermark",          "sample_100_watermark_",         [1, 2, 3, 4, 5],              ""),
    ("font_degrade",       "sample_100_font_degrade_",      [1, 2, 3, 4, 5],             ""),
    ("paper_fold",         "sample_100_paper_fold_",        [1, 2, 3, 4, 5],              ""),
    ("color_shift",        "sample_100_color_shift_",       [1, 2, 3, 4, 5],              ""),
]
```
