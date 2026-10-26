"""
Occurrence Rate Estimator for Giant Planets and Brown Dwarfs

This Streamlit application calculates and visualizes the frequency of giant planets and 
brown dwarfs around different stellar types based on a companion population model.
The model uses log-normal distributions for orbital separations and power-law distributions
for mass ratios to estimate companion frequencies.

Author: Wiliam Meynardie and Yiting Li
Date: October 2026
Paper Reference: Meyer-Li-Meynardie et al. 2026, Demographics of Planetary and Brown Dwarf Companions
"""

import numpy as np
import streamlit as st
from scipy import integrate, stats
import matplotlib.pyplot as plt
from astropy import constants as con

ln10 = np.log(10)


# ===============================================================
# GLOBAL CSS + TITLE BLOCK
# ===============================================================

st.markdown("""
<style>
.title-container {
    background: linear-gradient(to right, #1E88E5, #5E35B1);
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 25px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.main-title {
    color: white;
    font-size: 36px;
    font-weight: 800;
    text-align: center;
}
.subtitle {
    color: rgba(255,255,255,0.9);
    text-align: center;
    font-size: 18px;
    font-style: italic;
}
.section-header {
    color: #1E88E5;
    font-size: 28px;
    font-weight: bold;
    margin: 30px 0 20px 0;
    padding-bottom: 10px;
    border-bottom: 2px solid #1E88E5;
}
.tool-header {
    color: white;
    font-size: 28px;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(to right, #1E88E5, #5E35B1);
    padding: 12px;
    border-radius: 5px;
    margin-top: 25px;
}
</style>

<div class='title-container'>
    <div class='main-title'>Occurrence Rate Estimator</div>
    <div class='subtitle'>for Planets and Brown Dwarfs</div>
    <div class='subtitle'>Companion population model from Meyer-Li-Meynardie et al. (2026)</div>
</div>
""", unsafe_allow_html=True)


# ===============================================================
# INTRO TEXT
# ===============================================================

st.write("""Welcome to the online tool based on Meyer-Li-Meynardie et al. (new draft pending) meant to provide estimates of the expectation values of the mean number of gas giant planets per star and the mean number of brown dwarfs per star generated from our model. The model assumes that the companion mass ratio distribution of gas giants and brown dwarf companions does not vary with orbital separation. However, it explicitly treats brown dwarf companions as an extension of stellar mass companions drawn from the same orbital separations as a function of host star mass.

In the paper we fit the orbital distribution of gas giants and find that a log-normal function provides a good fit (two parameters). We also fit for power-law exponents for the companion mass ratio distribution for the gas giant population, and a stellar mass "efficiency" term (two parameters). We assume that the brown dwarf mass ratio distribution is flat in q. Finally, we fit for the normalization of both populations (two parameters).

Note: The data are fitted in the natural ln but we present the results in log-10 here for consistency with stellar binary orbital distributions.
""")


# ===============================================================
# MATHEMATICAL MODEL
# ===============================================================

st.markdown("<div class='section-header'>Mathematical Model</div>", unsafe_allow_html=True)

st.write("Total companion frequency:")
st.latex(r"""
N_{\text{TOTAL}} = 
\int \phi_{pl}(x)\,\psi_{pl}(q)\,dq\,dx 
+ 
\int \phi_{bd}(x)\,\psi_{bd}(q)\,dq\,dx
""")

st.write("Mass-ratio power laws:")
st.latex(r"\psi_{pl}(q) = M^{\gamma}q^{-\alpha}")
st.latex(r"\psi_{bd}(q) = 1")

st.write("Orbital log-normal distributions:")
st.latex(r"\phi_{pl}(a) = \frac{A_{pl}\,e^{-(x-\mu_{pl})^2/(2\sigma_{pl}^2)}}{a\sqrt{2\pi}\sigma_{pl}\ln 10}")
st.latex(r"\phi_{bd}(a) = \frac{A_{bd}\,e^{-(x-\mu_{bd})^2/(2\sigma_{bd}^2)}}{a\sqrt{2\pi}\sigma_{bd}\ln 10}")

st.write("These distributions depend on stellar type (M / FGK / A).")


# ===============================================================
# TABLE OF BD ORBITAL DISTRIBUTIONS
# ===============================================================

st.write("**Table 1: Companion Frequency (CF) & Log-Normal Separation Distribution vs. Host Type**")

st.table({
    'Spectral Type': ['M', 'FGK', 'A'],
    'CF': ['0.236', '0.61', '0.219'],
    'μ (log10 AU)': ['1.43', '1.70', '2.72'],
    'σ (log10 AU)': ['1.21', '1.68', '0.79']
})

st.markdown("---")
st.markdown("<div class='tool-header'>Frequency Calculation Tool</div>", unsafe_allow_html=True)


# ===============================================================
# PARAMETERS
# ===============================================================

@st.cache_data
def load_posterior_samples(path="samples.txt"):
    data = np.loadtxt(path, skiprows=1)
    cols = ["ln_A_p", "ln_sigma_p", "mu_p", "alpha", "gamma", "A_bd", "log_w"]
    return {name: data[:, i] for i, name in enumerate(cols)}

posterior = load_posterior_samples()
post_log_w = posterior["log_w"]
post_weights = np.exp(post_log_w - post_log_w.max())
post_weights /= post_weights.sum()

st.markdown("<div class='section-header'>Parameters</div>", unsafe_allow_html=True)

# ln_A_bd_default = -1.637
# ln_A_pl_default = -5.844
# alpha_bd_default = -0.764
# alpha_gp_default = 1.476
# mu_pl_default = 1.159
# sigma_pl_default = np.exp(0.425)

mu_bd_M = 1.43
s_bd_M = 1.21
mu_bd_FGK = 1.70
s_bd_FGK = 1.68
mu_bd_A = 2.72
s_bd_A = 0.79

# --- Slider <-> number-input syncing helpers ---------------------------
# Widgets below live outside any st.form, so these on_change callbacks can
# keep a range slider and its two "exact value" number boxes mirrored in
# real time. The heavy computation further down still only (re)runs after
# the Submit button is pressed, via the plain st.button() gate below.

def _sync_range_from_slider(slider_key, min_key, max_key):
    lo, hi = st.session_state[slider_key]
    st.session_state[min_key] = lo
    st.session_state[max_key] = hi

def _sync_slider_from_range(slider_key, min_key, max_key, lo_bound, hi_bound):
    mn = min(st.session_state[min_key], st.session_state[max_key])
    mx = max(st.session_state[min_key], st.session_state[max_key])
    mn = max(lo_bound, min(hi_bound, mn))
    mx = max(lo_bound, min(hi_bound, mx))
    st.session_state[slider_key] = (mn, mx)
    st.session_state[min_key] = mn
    st.session_state[max_key] = mx

# Orbital separation uses a plain slider in log10-space (converted to AU via
# 10**x) rather than st.select_slider: that widget has a bug in this Streamlit
# version where, if it's only ever set via a key/session_state (no `value=`
# every single rerun), its range value silently collapses to a lone float on
# ANY unrelated widget's rerun, crashing the paired on_change callback.
A_LO, A_HI = 0.1, 3000.0

def _sync_a_nums_from_slider():
    lo, hi = st.session_state.a_log_slider
    st.session_state.a_min_num = round(10**lo, 3)
    st.session_state.a_max_num = round(10**hi, 3)

def _sync_a_slider_from_nums():
    mn = min(st.session_state.a_min_num, st.session_state.a_max_num)
    mx = max(st.session_state.a_min_num, st.session_state.a_max_num)
    mn = max(A_LO, min(A_HI, mn))
    mx = max(A_LO, min(A_HI, mx))
    st.session_state.a_log_slider = (np.log10(mn), np.log10(mx))
    st.session_state.a_min_num = mn
    st.session_state.a_max_num = mx

if "ms_slider" not in st.session_state:
    st.session_state.ms_slider = (0.1, 5.0)
    st.session_state.ms_min_num = 0.1
    st.session_state.ms_max_num = 5.0
if "jup_slider" not in st.session_state:
    st.session_state.jup_slider = (0.1, 80.0)
    st.session_state.jup_min_num = 0.1
    st.session_state.jup_max_num = 80.0
if "a_log_slider" not in st.session_state:
    st.session_state.a_log_slider = (np.log10(A_LO), np.log10(A_HI))
    st.session_state.a_min_num = A_LO
    st.session_state.a_max_num = A_HI

st.markdown("<div class='section-header'>Stellar Mass Range (M<sub>☉</sub>)</div>", unsafe_allow_html=True)

st.slider("", 0.1, 5.0, key="ms_slider",
          on_change=_sync_range_from_slider, args=("ms_slider", "ms_min_num", "ms_max_num"))
c1, c2 = st.columns(2)
with c1:
    st.number_input("Min", 0.1, 5.0, key="ms_min_num", step=0.01,
                     on_change=_sync_slider_from_range, args=("ms_slider", "ms_min_num", "ms_max_num", 0.1, 5.0))
with c2:
    st.number_input("Max", 0.1, 5.0, key="ms_max_num", step=0.01,
                     on_change=_sync_slider_from_range, args=("ms_slider", "ms_min_num", "ms_max_num", 0.1, 5.0))

Ms_min, Ms_max = st.session_state.ms_slider
host_mass = (Ms_max + Ms_min) / 2

# ===============================================================
# SECTION 4 — COMPANION MASS RANGE
# ===============================================================

st.markdown("<div class='section-header'>Companion Mass Range (M<sub>J</sub>)</div>", unsafe_allow_html=True)

st.slider("", 0.1, 80.0, key="jup_slider",
          on_change=_sync_range_from_slider, args=("jup_slider", "jup_min_num", "jup_max_num"))
c1, c2 = st.columns(2)
with c1:
    st.number_input("Min", 0.1, 80.0, key="jup_min_num", step=0.1,
                     on_change=_sync_slider_from_range, args=("jup_slider", "jup_min_num", "jup_max_num", 0.1, 80.0))
with c2:
    st.number_input("Max", 0.1, 80.0, key="jup_max_num", step=0.1,
                     on_change=_sync_slider_from_range, args=("jup_slider", "jup_min_num", "jup_max_num", 0.1, 80.0))

Jup_min, Jup_max = st.session_state.jup_slider

# ===============================================================
# SECTION 5 — ORBITAL SEPARATION (only once)
# ===============================================================

st.markdown("<div class='section-header'>Orbital Separation Range (au)</div>", unsafe_allow_html=True)

st.slider("", np.log10(A_LO), np.log10(A_HI), key="a_log_slider",
          on_change=_sync_a_nums_from_slider, format="10^%.2f au")
c1, c2 = st.columns(2)
with c1:
    st.number_input("Min (au)", A_LO, A_HI, key="a_min_num", step=0.1,
                     on_change=_sync_a_slider_from_nums)
with c2:
    st.number_input("Max (au)", A_LO, A_HI, key="a_max_num", step=0.1,
                     on_change=_sync_a_slider_from_nums)

a_log_lo, a_log_hi = st.session_state.a_log_slider
amin_calc, amax_calc = 10**a_log_lo, 10**a_log_hi

submitted = st.button("Submit")

if not submitted:
    st.stop()

q_Jupiter = con.M_jup.value / (host_mass * con.M_sun.value )

if amin_calc >= amax_calc:
    st.error("Minimum separation must be less than maximum separation.")
    st.stop()


# ===============================================================
# MLE VALUES
# ===============================================================

border_m_fgk = 0.59
border_fgk_a = 1.61

alpha_bd = 0
A_bd = 0.47109464

# Same overlap-weighted blend Section 7 uses, so the plot and the printed
# numbers represent the same mix of spectral types across [Ms_min, Ms_max].
if Ms_max > Ms_min:
    f_bd_m_plot = (min(Ms_max, border_m_fgk) - min(Ms_min, border_m_fgk)) / (Ms_max - Ms_min)
    f_bd_a_plot = (max(Ms_max, border_fgk_a) - max(Ms_min, border_fgk_a)) / (Ms_max - Ms_min)
else:
    f_bd_m_plot = 1.0 if host_mass <= border_m_fgk else 0.0
    f_bd_a_plot = 1.0 if host_mass >= border_fgk_a else 0.0
f_bd_fgk_plot = 1 - f_bd_m_plot - f_bd_a_plot

alpha_gp = 1.52467272
A_pl = np.exp(-5.95239706)
mu_pl = 1.2797595
sigma_pl = np.exp(0.73345162)
gamma_pl = 0.92249493


# ===============================================================
# EXACT SURFACE DENSITY FUNCTIONS
# ===============================================================

def _lognorm_den(a, mu, sigma):
    return A_bd/ln10 * np.exp(-(np.log10(a)-mu)**2/(2*sigma**2)) / (a*np.sqrt(2*np.pi)*sigma)

def surface_den_bd_exact(a):
    return (f_bd_m_plot*_lognorm_den(a, mu_bd_M, s_bd_M)
            + f_bd_fgk_plot*_lognorm_den(a, mu_bd_FGK, s_bd_FGK)
            + f_bd_a_plot*_lognorm_den(a, mu_bd_A, s_bd_A))

def surface_den_pl_exact(a):
    return A_pl/ln10 * np.exp(-(np.log10(a)-mu_pl)**2/(2*sigma_pl**2)) / (a*np.sqrt(2*np.pi)*sigma_pl)


# Needed wrappers (fix NameError)
def surface_den_bd(a): return surface_den_bd_exact(a)
def surface_den_pl(a): return surface_den_pl_exact(a)


# Precompute integrals
orb_bd = integrate.quad(surface_den_bd_exact, amin_calc, amax_calc)[0]
orb_pl = integrate.quad(surface_den_pl_exact, amin_calc, amax_calc)[0]


# Mass fctns
def mass_fctn_bd(q): return q**(-alpha_bd)
def mass_fctn_pl(q): return q**(-alpha_gp)

def dN_bd(q): return mass_fctn_bd(q)*orb_bd
def dN_pl(q): return mass_fctn_pl(q)*orb_pl*(host_mass**gamma_pl)


# ===============================================================
# SECTION 6 — PLOTTING
# ===============================================================

fig, ax = plt.subplots(figsize=(10, 8))

q_pl_min = 0.03 * q_Jupiter
q_pl_max = 0.1
q_bd_min = 3 * q_Jupiter
q_bd_max = 0.67

mass_ratio_values_pl = np.logspace(np.log10(q_pl_min), np.log10(q_pl_max), 1000)
mass_ratio_values_bd = np.logspace(np.log10(q_bd_min), np.log10(q_bd_max), 1000)
mass_ratio_values_total = np.logspace(np.log10(q_pl_min), np.log10(q_bd_max), 1000)

pl_freq = [dN_pl(q)*q*ln10 if q_pl_min <= q <= q_pl_max else 0 for q in mass_ratio_values_pl]
bd_freq = [dN_bd(q)*q*ln10 if q_bd_min <= q <= q_bd_max else 0 for q in mass_ratio_values_bd]
total_freq = [
    (dN_pl(q)*q*ln10 if q_pl_min <= q <= q_pl_max else 0) +
    (dN_bd(q)*q*ln10 if q_bd_min <= q <= q_bd_max else 0)
    for q in mass_ratio_values_total
]

ax.plot(np.log10(mass_ratio_values_pl), pl_freq, 'r', linewidth=3, label='Giant Planet Model')
ax.plot(np.log10(mass_ratio_values_bd), bd_freq, 'b', linewidth=3, label='Brown Dwarf Model')
ax.plot(np.log10(mass_ratio_values_total), total_freq, color='orange', linewidth=2, label='Total Frequency')

ax.set_xlabel("log(q)", fontsize=20)
ax.set_ylabel("dN / dlog(q)", fontsize=20)
ax.legend(fontsize=14)
ax.set_title("Companion Frequency Distribution", fontsize=22)

st.pyplot(fig)


# ===============================================================
# SECTION 7 — FREQUENCY CALCULATIONS (POSTERIOR DRAWS)
# ===============================================================

n_draws = 10000
np.random.seed(24601)
draw_idx = np.random.choice(len(post_log_w), size=n_draws, replace=True, p=post_weights)

M_J = con.M_jup.value/con.M_sun.value  # M_Jup / M_sun

# Fixed per-spectral-type BD orbital log-normal params (natural-log lognorm shape/scale,
# converted from the log10(AU) mu/sigma values above), matching simdata's s_m/mu_m etc.
s_bd_m_ln, scale_bd_m = ln10*s_bd_M, 10**mu_bd_M
s_bd_fgk_ln, scale_bd_fgk = ln10*s_bd_FGK, 10**mu_bd_FGK
s_bd_a_ln, scale_bd_a = ln10*s_bd_A, 10**mu_bd_A

def cum_lnfunc(x, amp, s, scale):
    return amp * stats.lognorm.cdf(x, s, 0.0, scale)

def simdata_point(A_pl_i, sigma_pl_i, mu_pl_i, alpha_gp_i, gamma_i, A_bd_i,
                   sm_avg, sm_min, sm_max, q_min, q_max, a_min, a_max, opac_lim=3):
    # --- Giant planets: log-normal orbital CDF x gamma-weighted power-law mass ratio ---
    scale_pl = np.exp(mu_pl_i)
    a_pl = cum_lnfunc(a_max, A_pl_i, sigma_pl_i, scale_pl) - cum_lnfunc(a_min, A_pl_i, sigma_pl_i, scale_pl)

    q_max_pl = min(q_max, 0.1)
    if alpha_gp_i != 1:
        q_pl = (sm_avg**gamma_i) * (q_max_pl**(1-alpha_gp_i) - q_min**(1-alpha_gp_i)) / (1-alpha_gp_i)
    else:
        q_pl = (sm_avg**gamma_i) * np.log(q_max_pl/q_min)
    q_pl = max(q_pl, 0)

    fy_planet = a_pl * q_pl

    # --- Brown dwarfs: computed per spectral type, combined by host-mass-range overlap ---
    a_bd_m = cum_lnfunc(a_max, A_bd_i, s_bd_m_ln, scale_bd_m) - cum_lnfunc(a_min, A_bd_i, s_bd_m_ln, scale_bd_m)
    a_bd_fgk = cum_lnfunc(a_max, A_bd_i, s_bd_fgk_ln, scale_bd_fgk) - cum_lnfunc(a_min, A_bd_i, s_bd_fgk_ln, scale_bd_fgk)
    a_bd_a = cum_lnfunc(a_max, A_bd_i, s_bd_a_ln, scale_bd_a) - cum_lnfunc(a_min, A_bd_i, s_bd_a_ln, scale_bd_a)

    q_min_bd = max(q_min, (opac_lim*M_J)/sm_avg)
    q_bd = max(q_max - q_min_bd, 0)  # flat distribution in q (alpha_bd fixed at 0)

    if sm_max > sm_min:
        f_bd_m = (min(sm_max, border_m_fgk) - min(sm_min, border_m_fgk)) / (sm_max - sm_min)
        f_bd_a = (max(sm_max, border_fgk_a) - max(sm_min, border_fgk_a)) / (sm_max - sm_min)
    else:
        f_bd_m = 1.0 if sm_avg <= border_m_fgk else 0.0
        f_bd_a = 1.0 if sm_avg >= border_fgk_a else 0.0
    f_bd_fgk = 1 - f_bd_m - f_bd_a

    fy_bd = q_bd * (f_bd_m*a_bd_m + f_bd_fgk*a_bd_fgk + f_bd_a*a_bd_a)

    return fy_planet, fy_bd

q_min_total = (Jup_min*M_J)/host_mass
q_max_total = (Jup_max*M_J)/host_mass

mean_num_pl_samples = np.empty(n_draws)
mean_num_bd_samples = np.empty(n_draws)

for i, idx in enumerate(draw_idx):
    A_pl_i = np.exp(posterior["ln_A_p"][idx])
    sigma_pl_i = np.exp(posterior["ln_sigma_p"][idx])
    mu_pl_i = posterior["mu_p"][idx]
    alpha_gp_i = posterior["alpha"][idx]
    gamma_i = posterior["gamma"][idx]
    A_bd_i = posterior["A_bd"][idx]

    mean_num_pl_samples[i], mean_num_bd_samples[i] = simdata_point(
        A_pl_i, sigma_pl_i, mu_pl_i, alpha_gp_i, gamma_i, A_bd_i,
        host_mass, Ms_min, Ms_max, q_min_total, q_max_total, amin_calc, amax_calc
    )

pl_lo, pl_median, pl_hi = np.percentile(mean_num_pl_samples, [16, 50, 84])
bd_lo, bd_median, bd_hi = np.percentile(mean_num_bd_samples, [16, 50, 84])

pl_median_str = f"{pl_median:.4f}"
pl_hi_str = f"+{pl_hi - pl_median:.4f}"
pl_lo_str = f"-{pl_median - pl_lo:.4f}"

bd_median_str = f"{bd_median:.4f}"
bd_hi_str = f"+{bd_hi - bd_median:.4f}"
bd_lo_str = f"-{bd_median - bd_lo:.4f}"

st.write("Number of Planets Per Star (Median and 68% confidence interval):")
st.latex(pl_median_str + r"^{" + pl_hi_str + r"}_{" + pl_lo_str + r"}")

st.write("Number of Brown Dwarfs Per Star (Median and 68% confidence interval):")
st.latex(bd_median_str + r"^{" + bd_hi_str + r"}_{" + bd_lo_str + r"}")

# st.write("*Note: These values represent the expected number of companions per star within the specified mass ratio and orbital separation ranges, marginalized over 1000 posterior draws from `samples.txt` (resampled by their `log_w` weights).*")
