#save as fit_rgb_leds_with_piecewise_bg.py and run with: python3 fit_rgb_leds_with_piecewise_bg.py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# --- load image and extract intensity profile (average across columns) ---
img_name = 'purp_spec1_cropped.png'
img_path = 'photos/' + img_name  # change if needed
img = Image.open(img_path).convert('RGB')
arr = np.array(img)
gray = arr.mean(axis=2)       # simple average of R,G,B channels -> grayscale
I = gray.mean(axis=1)        # average intensity per row (vertical axis)

# --- pixel -> wavelength mapping (assumed) ---
# adjust these endpoints if you have a better calibration
lam_start = 400.0
lam_end = 700.0
rows = len(I)
lam = np.linspace(lam_start, lam_end, rows)

# --- LED Gaussian parameters (typical approximate values) ---
lam_R00, lam_G00, lam_B00 = 660.0, 500.0, 450.0   # center wavelengths (nm)
l_rad = 40
lam_R0l, lam_G0l, lam_B0l = lam_R00 - l_rad, lam_G00 - l_rad, lam_B00 - l_rad
lam_R0u, lam_G0u, lam_B0u = lam_R00 + l_rad, lam_G00 + l_rad, lam_B00 + l_rad
fwhm_R00, fwhm_G00, fwhm_B00 = 16.0, 30.0, 20.0     # typical FWHMs (nm)
fwhm_R0l, fwhm_G0l, fwhm_B0l = 0, 0, 0 
fwhm_R0u, fwhm_G0u, fwhm_B0u = 100, 100, 100

# --- background model: constant between [left_edge, right_edge],
#     Gaussian-decay outside (sd = sd_bg) ---
bg_left_edge0 = 425.0
bg_right_edge0 = 650.0
bg_left_edgel = 400
bg_left_edgeu = 500
bg_right_edgel = 600
bg_right_edgeu = 675
sd_bg0 = 25.0
sd_bgl = 0
sd_bgu = 100

def background_model(bg_amp, bg_left_edge, bg_right_edge, sd_bg, lam_array):
    bg = np.empty_like(lam_array)
    inside = (lam_array >= bg_left_edge) & (lam_array <= bg_right_edge)
    bg[inside] = bg_amp
    left = lam_array < bg_left_edge
    bg[left] = bg_amp * np.exp(- (lam_array[left] - bg_left_edge)**2 / (2 * sd_bg**2))
    right = lam_array > bg_right_edge
    bg[right] = bg_amp * np.exp(- (lam_array[right] - bg_right_edge)**2 / (2 * sd_bg**2))
    return bg

def components(params):
    lam_R0, lam_G0, lam_B0, fwhm_R, fwhm_G, fwhm_B, cR, cG, cB, bg_amp, bg_left_edge, bg_right_edge, sd_bg = params
    sigma_R = fwhm_R / (2 * np.sqrt(2 * np.log(2)))
    sigma_G = fwhm_G / (2 * np.sqrt(2 * np.log(2)))
    sigma_B = fwhm_B / (2 * np.sqrt(2 * np.log(2)))
    SR = cR * np.exp(- (lam - lam_R0)**2 / (2 * sigma_R**2))
    SG = cG * np.exp(- (lam - lam_G0)**2 / (2 * sigma_G**2))
    SB = cB * np.exp(- (lam - lam_B0)**2 / (2 * sigma_B**2))
    BG = background_model(bg_amp, bg_left_edge, bg_right_edge, sd_bg, lam)
    return SR, SG, SB, BG

def model(params):
    SR, SG, SB, BG = components(params)
    return SR + SG + SB + BG

def residuals(params):
    return model(params) - I

# --- initial guess and bounds ---
cR0, cG0, cB0, bg_amp0 = 10., 8., 1., 1.
p0 = [lam_R00, lam_G00, lam_B00, fwhm_R00, fwhm_G00, fwhm_B00, cR0, cG0, cB0, bg_amp0, bg_left_edge0, bg_right_edge0, sd_bg0]   # initial guesses for cR, cG, cB, bg_amp
lb = [lam_R0l, lam_G0l, lam_B0l, fwhm_R0l, fwhm_G0l, fwhm_B0l, 0.0, 0.0, 0.0, 0.0, bg_left_edgel, bg_right_edgel, sd_bgl]
ub = [lam_R0u, lam_G0u, lam_B0u, fwhm_R0u, fwhm_G0u, fwhm_B0u, np.inf, np.inf, np.inf, np.inf, bg_left_edgeu, bg_right_edgeu, sd_bgu]

# --- run least-squares fit ---
res = least_squares(residuals, p0, bounds=(lb, ub))
lam_R0, lam_G0, lam_B0, fwhm_R0, fwhm_G0, fwhm_B0, cR, cG, cB, bg_amp, bg_left_edge, bg_right_edge, sd_bg = res.x
SR, SG, SB, BG = components(res.x)

# --- print results ---
coeffs = np.array([cR, cG, cB])
norm = coeffs / coeffs.sum() if coeffs.sum() > 0 else coeffs
print("Fitted coefficients (cR, cG, cB, bg_amp):")
print("  cR = {:.6g}, cG = {:.6g}, cB = {:.6g}, bg_amp = {:.6g}".format(cR, cG, cB, bg_amp))
print("Normalized R:G:B ratio (fractions):", norm)
print("Normalized R:G:B ratio (proportions): {:.3f} : {:.3f} : {:.3f}".format(*(norm * (1.0 / norm.min()) if norm.min()>0 else norm)))

# --- plot measured spectrum, fit, components, and background ---
plt.figure(figsize=(10,6))
plt.plot(lam, I, label='Measured intensity', linewidth=1, color='orange')
plt.plot(lam, model(res.x), label='Fit (sum)', linewidth=1.2, color='purple')
plt.plot(lam, SR, label=f'Red (λ={lam_R0:.1f} +/- {fwhm_R0:.1f} nm)', color='red')
plt.plot(lam, SG, label=f'Green (λ={lam_G0:.1f} +/- {fwhm_G0:.1f} nm)', color='green')
plt.plot(lam, SB, label=f'Blue (λ={lam_B0:.1f} +/- {fwhm_B0:.1f} nm)', color='blue')
plt.plot(lam, BG, label=f'Background model (bg_l={bg_left_edge:.1f}, bg_r={bg_right_edge:.1f}, sd_bg={sd_bg:.1f})', linestyle='--', color='brown')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (arb units)')
plt.title('R/G/B LED fit with piecewise Gaussian-decaying background (wavelengths are kinda madeup)')
plt.legend()
plt.tight_layout()

plt.savefig("plots/" + img_name)
plt.show()

