import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import healpy as hp
from classy import Class
import qrcode
from io import BytesIO
from matplotlib.patches import FancyArrowPatch

# Set up the Streamlit page
st.set_page_config(page_title="Cosmolunch Appetizer", layout="wide")
st.title("Cosmolunch Appetizer")
st.markdown("""
    Welcome to the **Cosmolunch Appetizer**, the parameter inference is like cooking. This app is prepared especially for cosmo lunch 🌌  
    Here, you'll be the chef of the cosmos, mixing ingredients like dark matter, baryons, and dark energy to create the perfect recipe for the universe.  
    """)
st.markdown("---")
left_col, right_col = st.columns(2)
st.markdown("---")
st.markdown("""
### 🍽️ What’s on the Menu?
- **Basics**: Start with the Hubble parameter and density evolution. Zeldovich Pancake 🥞 is not implemented as technical reason. 
- **CMB**: This is highly smooth source like source of ideal caccio e pepe [🇮🇹](https://arxiv.org/html/2501.00536v1), however we want to find imperfections!.
- **Galaxy Clustering**: Observing the black peppers 🌶️ in the cosmic soup, find a pair of pepper 🫑 and BAO.
- **Cosmic Shear**: Garnish with convergence maps and correlation functions, banana 🍌 is not implemented as technical reason.
- **Supernovae**: There would be no alchohol 🥃 without stellar evolution ⭐️ and supernova 💥, since big bang produces only few metals.""")
# Sidebar for cosmological parameters
st.sidebar.header("Cosmological Parameters")
st.sidebar.markdown("Fit weird curves by hand")
h = st.sidebar.slider(r"$h$ Hubble Parameter", 0.5, 0.9, 0.6736, step=0.001)
Omega_m = st.sidebar.slider(r"$\Omega_m$ Matter Density", 0.1, 0.5, 0.315, step=0.001)
Omega_b = st.sidebar.slider(r"$\Omega_b$ Baryon Density", 0.01, 0.1, 0.0493, step=0.0001)
Omega_r = st.sidebar.slider(r"$\Omega_r$ Radiation Density", 0.0, 0.0002, 9.2e-5, step=1e-6, format="%.6f")
sigma8 = st.sidebar.slider(r"$\sigma_8$ Amplitude of Matter Fluctuations", 0.5, 1.2, 0.811, step=0.001)
n_s = st.sidebar.slider(r"$n_s$ Spectral Index", 0.9, 1.1, 0.9649, step=0.001)
halofit = st.sidebar.checkbox("Use Halofit for Non-linear Power Spectrum", value=True)

with left_col:
    
    st.markdown("""
    ### 🧑‍🍳 The Recipe of the Universe
    - **Baseline Recipes**: A flat LambdaCDM cosmology (but feel free to add a pinch of massive neutrinos 🥑 or a dash of dynamical dark energy 🍺 if you're allergic to cosmological constant).  
    - Many plots are inspired by the [Modern Cosmology](https://www.amazon.co.uk/Modern-Cosmology-Scott-Dodelson/dp/0128159480).  
    - **Tools**:  
      - [CLASS](http://class-code.net/): The Boltzmann code that simmers the evolution of cosmological perturbations.  
      - [Matplotlib](https://matplotlib.org/stable/): For plating your cosmic dishes with beautiful visualizations.  
      - [Healpy](https://healpy.readthedocs.io/en/latest/index.html): To project your creations onto the celestial sphere.  
      - [GLASS](https://glass.readthedocs.io/stable/index.html): For crafting lognormal cosmic shear and convergence maps.  
      - [QR Code Generator](https://pypi.org/project/qrcode/): For generating QR codes to access the app easily.  
    """)
    st.markdown("""
    ### 👨‍🍳 Developed by [Rintaro Kanaki](https://github.com/Rintaro0406/Spatial_Selection_Photo_z)
    This app is lovingly prepared with [Streamlit](https://streamlit.io/) and served fresh for your cosmological appetite.  
    Buon Appetite, and let’s find the perfect recipe for the universe! 🌟
    """)

with right_col:
    st.markdown(f"""
    

    ### 🍅 Ingredients

    To cook up the universe, you'll need the following ingredients, carefully measured to taste:

    - **Hubble Parameter (h)**: The expansion rate of the universe. This is like baking soda in shortbread 🍞, highly controversial between [different recipes](https://arxiv.org/abs/2203.06142).
      Current setting: **{h:.2f}** (adjustable between 0.5 and 0.9).  
    - **Matter Density (Ωm)**: The amount of matter in the universe. This is your main ingredient, but we have no idea what it is, since no [spherical idiots](https://en.wikipedia.org/wiki/Fritz_Zwicky) have been found yet [🇧🇬](https://en.wikipedia.org/wiki/Fritz_Zwicky). Don't worry this is healthy [🥬](https://www.sciencedirect.com/science/article/pii/S0370269312009884)
      Current setting: **{Omega_m:.2f}** (adjustable between 0.1 and 0.5).  
    - **Baryon Density (Ωb)**: This is the amount of matter which we can see in the universe, but just for seasoning 🧂.
      Current setting: **{Omega_b:.2f}** (adjustable between 0.01 and 0.1).  
    - **Radiation Density (Ωr)**: The amount of radiation in the universe, like a sprinkle of persely 🌿, but not to add too much, or it will burn your cosmic cake.
      Current setting: **{Omega_r:.5f}** (adjustable between 0.0 and 0.0001).
    - **σ8 (Amplitude of Matter Fluctuations)**: The "fluffiness" of your cosmic cake, important to make for banana 🍌, also [highly controversial](https://arxiv.org/abs/1606.05338) to make banana cake.
      Current setting: **{sigma8:.2f}** (adjustable between 0.5 and 1.2).  
    - **Spectral Index (ns)**: The legendary parameter which determines the tilt of the primordial power spectrum. This is like the secret ingredient in your grandpa [👨🏾‍🌾](https://link.springer.com/article/10.1140/epjc/s10052-013-2486-7) recipe 🍜.
      Current setting: **{n_s:.2f}** (adjustable between 0.9 and 1.1).  
    - **[Halofit](https://arxiv.org/abs/1208.2701)**: A special non-linear power spectrum enhancer. Add this for extra complexity in your cosmic dish like a pinch of maggi [🧋](https://www.amazon.co.uk/s?srs=16256131031).
      Current setting: **{halofit}** (toggle on or off).""")



# Generate QR Code for the app
st.sidebar.header("QR Code")
app_url = "http://localhost:8501"  # Future I will deploy this app on the server
qr = qrcode.QRCode()
qr.add_data(app_url)
qr.make(fit=True)
img = qr.make_image(fill="black", back_color="white")
buffer = BytesIO()
img.save(buffer, format="PNG")
st.sidebar.image(buffer.getvalue(), caption="Scan to Open App", use_column_width=True)

# Initialize CLASS
cosmo = Class()
common_settings = {
    'h': h,
    'omega_b': Omega_b * h**2,
    'omega_cdm': Omega_m * h**2 - Omega_b * h**2,
    'n_s': n_s,
    'sigma8': sigma8,
    'output': 'mPk,tCl,pCl,lCl',
    'P_k_max_1/Mpc': 10.0,
    'l_max_scalars': 2500,
    'tau_reio': 0.054,
    'non linear': 'halofit' if halofit else 'linear', 
}

if halofit:
    common_settings['non linear'] = 'halofit'
cosmo.set(common_settings)
cosmo.compute()

# Tabs for different visualizations
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Basics", "CMB", "Galaxy Clustering", "Cosmic Shear", "Supernovae"])

# basics
with tab1:
    st.header("Basics")
    st.markdown("""
    ### Basics of Cosmology
    In this section, we explore the fundamental quantities of cosmology, including the Hubble parameter, distance measures, and density parameter evolution. These plots are inspired by **Chapter 1** and **Chapter 2** of [Modern Cosmology](https://www.amazon.co.uk/Modern-Cosmology-Scott-Dodelson/dp/0128159480) by Scott Dodelson and Fabian Schmidt. 
    Using the [CLASS](http://class-code.net/) Boltzmann code, we calculate and visualize these quantities to understand the evolution of the universe.
    """)
    st.header("Hubble Parameter")
    st.markdown("""
### This plot corresponds to figure 2.8 on [Modern Cosmology](https://www.amazon.co.uk/Modern-Cosmology-Scott-Dodelson/dp/0128159480)

This plot shows the evolution of $H(z)/(1+z)$ with redshift $z$, comparing theoretical predictions ([CLASS](http://class-code.net/0)) with observational data:

- **[Riess et al. 2019](https://arxiv.org/abs/1903.07603)**: Local $H_0$.
- **[BOSS DR12](https://arxiv.org/abs/1607.03155)**: BAO at intermediate $z$.
- **[DR14 Quasars](https://arxiv.org/abs/1910.10395)**: High-$z$ quasars.
- **[DR14 Ly-alpha](https://arxiv.org/abs/1910.10395)**: Lyman-alpha forest at very high $z$.

The blue curve is the theoretical model using [CLASS](http://class-code.net/), while points with error bars show observations. The log scale highlights the wide range of \( H(z)/(1+z) \).
""")
    # Button to toggle observational data
    show_obs_data = st.checkbox("Show Observational Data", value=False)

    # Data: [z, H(z), error], values from Dodelson
    Riess_2019 = np.array([0.0, 74.03, 1.42])  # Riess et al. 2019
    BOSS_DR12 = np.array([
        [0.38, 81.5, 1.9],
        [0.51, 90.5, 1.9],
        [0.61, 97.3, 2.1]
    ])  # BOSS DR12
    DR14_quasars = np.array([1.52, 162, 12])  # DR14 quasars
    DR14_Ly_alpha = np.array([2.34, 222, 7])  # DR14 Ly-alpha

    # Redshift range
    z = np.linspace(0, 2.5, 200)
    Hz = np.array([cosmo.Hubble(z_i) for z_i in z])  # in Mpc^-1
    c_kms = 299_792.458  # km/s
    Hz_kmsMpc = Hz * c_kms

    # Plot H(z)/(1+z)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(z, Hz_kmsMpc / (1 + z), color='blue', lw=2, label='Calculated using CLASS')

    if show_obs_data:
        ax.errorbar(Riess_2019[0], Riess_2019[1] / (1 + Riess_2019[0]), yerr=Riess_2019[2] / (1 + Riess_2019[0]), fmt='o', color='red', label='Riess et al. 2019')
        ax.errorbar(BOSS_DR12[:, 0], BOSS_DR12[:, 1] / (1 + BOSS_DR12[:, 0]), yerr=BOSS_DR12[:, 2] / (1 + BOSS_DR12[:, 0]), fmt='o', color='black', label='BOSS DR12')
        ax.errorbar(DR14_quasars[0], DR14_quasars[1] / (1 + DR14_quasars[0]), yerr=DR14_quasars[2] / (1 + DR14_quasars[0]), fmt='o', color='green', label='DR14 quasars')
        ax.errorbar(DR14_Ly_alpha[0], DR14_Ly_alpha[1] / (1 + DR14_Ly_alpha[0]), yerr=DR14_Ly_alpha[2] / (1 + DR14_Ly_alpha[0]), fmt='o', color='orange', label='DR14 Ly-alpha')

    ax.legend(fontsize=12, loc='upper right')
    ax.set_xlabel('Redshift $z$', fontsize=16)
    ax.set_ylabel(r'$H(z)/(1+z)$ [km s$^{-1}$ Mpc$^{-1}$]', fontsize=16)
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=13, length=7, width=1.5, direction='in', top=True, right=True)
    ax.tick_params(axis='both', which='minor', labelsize=11, length=4, width=1, direction='in', top=True, right=True)
    st.pyplot(fig)
    st.header("Distance Measures")
    st.markdown(r"""
### This plot corresponds to figure 2.3 in [Modern Cosmology](https://www.amazon.co.uk/Modern-Cosmology-Scott-Dodelson/dp/0128159480)

This plot shows the comoving distance $\chi(z)$, angular diameter distance $d_A(z)$, and luminosity distance $d_L(z)$ as functions of redshift $z$.

- The comoving distance $\chi(z)$ is the distance between two points in the universe at a given redshift, given by the following integral:
  $$\chi(z) = \int_0^z \frac{c}{H(z')} \, dz'$$
  where $c$ is the speed of light and $H(z)$ is the Hubble parameter at redshift $z$.

- The angular diameter distance $d_A(z)$ is the distance to an object whose angular size is measured in the sky:
  $$d_A(z) = \frac{\chi(z)}{1+z}$$

- The luminosity distance $d_L(z)$ is the distance to an object whose brightness is measured in the sky:
  $$d_L(z) = \chi(z)(1+z)$$
""")
    # Redshift range (reuse z)
    chi = np.array([cosmo.angular_distance(z_i) * (1 + z_i) for z_i in z])  # comoving distance, Mpc
    d_A = chi / (1 + z)  # angular diameter distance, Mpc
    d_L = chi * (1 + z)  # luminosity distance, Mpc

    # Plot distance measures
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(z, chi, label=r'$\chi(z)$ (Comoving Distance)', color='blue', lw=2)
    ax.plot(z, d_A, label=r'$d_A(z)$ (Angular Diameter Distance)', color='green', lw=2, linestyle='--')
    ax.plot(z, d_L, label=r'$d_L(z)$ (Luminosity Distance)', color='red', lw=2, linestyle='-.')
    ax.set_xlabel('Redshift $z$', fontsize=15)
    ax.set_ylabel(r'Distance [Mpc]', fontsize=15)
    ax.set_yscale('log')
    ax.legend(fontsize=13, loc='upper left')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    st.pyplot(fig)


    st.header("Density Parameter Evolution")
    st.markdown(r"""
### This plot corresponds to figure 1.3 in [Modern Cosmology](https://www.amazon.co.uk/Modern-Cosmology-Scott-Dodelson/dp/0128159480)
This plot shows the evolution of the density parameters $\Omega_r(a)$, $\Omega_m(a)$, and $\Omega_\Lambda(a)$ as functions of the scale factor $a$.
- The density parameters are defined as:
$$\Omega_i(a) = \frac{\rho_i(a)}{\rho_{\rm crit}(a)}$$
where $\rho_i(a)$ is the density of component $i$ at scale factor $a$, and $\rho_{\rm crit}(a)$ is the critical density at scale factor $a$.
- The critical density is given by:
$$\rho_{\rm crit}(a) = \frac{3H^2(a)}{8\pi G}$$
where $H(a)$ is the Hubble parameter at scale factor $a$, and $G$ is the gravitational constant.
- The density parameters evolve as:
$$\Omega_r(a) = \frac{\Omega_{r,0}}{a^4}$$
$$\Omega_m(a) = \frac{\Omega_{m,0}}{a^3}$$
$$\Omega_\Lambda(a) = \Omega_{\Lambda,0}$$
where $\Omega_{i,0}$ is the density parameter of component $i$ at the present time ($a=1$).
- The equality scale factors $a_{\rm eq}$ and $a_{\Lambda}$ are defined as:
$$a_{\rm eq} = \frac{\Omega_{r,0}}{\Omega_{m,0}}$$
$$a_{\Lambda} = \left(\frac{\Omega_{m,0}}{\Omega_{\Lambda,0}}\right)^{1/3}$$
where $a_{\rm eq}$ is the scale factor at matter-radiation equality, and $a_{\Lambda}$ is the scale factor at $\Lambda$-matter equality.
""")
    # Physical densities relative to rho_crit,0
    omega_lambda = 1.0 - Omega_m - Omega_r  # Assuming flat universe
    a_plot = np.logspace(-6, 0, 1000)  # Scale factor range
    rho_r = Omega_r * a_plot**(-4)  # Radiation ∝ a⁻⁴
    rho_m = Omega_m * a_plot**(-3)  # Matter ∝ a⁻³
    rho_L = omega_lambda * a_plot**(0)  # Dark energy ∝ a⁰

    # Compute equality scale-factors
    a_eq = Omega_r / Omega_m
    a_lambda = (Omega_m / omega_lambda)**(1/3)

    shows_equality = st.checkbox("Show Equality Lines (Density Evolution)", value=False, key="density_equality")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(a_plot, rho_r, label=r'$\Omega_r(a)$ (Radiation)', color='orange', lw=2)
    ax.plot(a_plot, rho_m, label=r'$\Omega_m(a)$ (Matter)', color='blue', lw=2)
    ax.plot(a_plot, rho_L, label=r'$\Omega_\Lambda(a)$ (Dark Energy)', color='green', lw=2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Scale factor $a$', fontsize=15)
    ax.set_ylabel(r'$\rho_s(t)$', fontsize=15)
    if shows_equality:
        ax.axvline(a_eq, color='red', linestyle='--', label=r'$a_{\rm eq}$ (Matter-Radiation Equality)')
        ax.axvline(a_lambda, color='purple', linestyle='--', label=r'$a_{\Lambda}$ ($\Lambda$-Matter Equality)')
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    st.pyplot(fig)
   
    st.header("Scale Factor Evolution")
    st.markdown(r"""
### This plot corresponds to figure 1.2 in [Modern Cosmology](https://www.amazon.co.uk/Modern-Cosmology-Scott-Dodelson/dp/0128159480)
This plot shows the evolution of the scale factor $a(t)$ as a function of cosmic time $t$.
- The scale factor $a(t)$ describes how distances in the universe change over time. It is normalized to 1 at the present time ($a=1$).
- The scale factor evolves according to the Friedmann equations, which describe the expansion of the universe.
- The scale factor is related to the Hubble parameter $H(t)$ by:
$$H(t) = \frac{1}{a(t)} \frac{da(t)}{dt}$$
- The Hubble parameter $H(t)$ is the rate of expansion of the universe at time $t$.
- The scale factor $a(t)$ is related to the redshift $z$ by:
$$a(t) = \frac{1}{1+z}$$
- The cosmic time $t$ is the time elapsed since the Big Bang and is related to the scale factor by:
$$t(a) = \int_0^a \frac{da'}{a' H(a')}$$
- Before the equality of matter and radiation, the scale factor evolves as:
$$a(t) \propto t^{2/3}$$
- After the equality of matter and radiation, the scale factor evolves as:
$$a(t) \propto t^{1/2}$$
- After the equality of matter and dark energy, the scale factor evolves as:
$$a(t) \propto e^{H t}$$""")
    # Constants
    H0 = 100 * h  # Hubble constant in km/s/Mpc
    H0_si = H0 * 1e3 / (3.086e22)  # Hubble constant in s^-1 (Mpc to m conversion)

    # Compute E(a) = H(a)/H0
    E = np.sqrt(Omega_r / a_plot**4 + Omega_m / a_plot**3 + omega_lambda)

    # Numeric integration for t(a)
    # dt/da = 1 / [a H(a)] ⇒ t(a) = ∫ da / [a H0 E(a)]
    da = np.diff(a_plot)
    integrand = 1.0 / (a_plot * H0_si * E)
    t = np.concatenate(([0], np.cumsum(0.5 * (integrand[:-1] + integrand[1:]) * da)))

    # Convert to years
    seconds_per_year = 3600 * 24 * 365
    t_years = t / seconds_per_year


    # Use a unique key for the checkbox to avoid conflicts with the previous one
    shows_equality_2 = st.checkbox("Show Equality Lines (Scale Factor Evolution)", value=False, key="scale_factor_equality")
    
    # Plot a(t) vs t
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(t_years, a_plot, color='C0', lw=2, label='Scale Factor Evolution')
    if shows_equality_2:
        ax.axhline(a_eq, color='red', linestyle='--', label=r'Matter-Radiation Equality')
        ax.axhline(a_lambda, color='purple', linestyle='--', label=r'$\Lambda$-Matter Equality')
    ax.set_xlabel('Cosmic time $t$ [yr]', fontsize=14)
    ax.set_ylabel('Scale factor $a(t)$', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(which='both', linestyle='--', alpha=0.5)
    st.pyplot(fig)

    st.header("Transfer Function")
    st.header("Growth Factor")

# cmb
with tab2:
    st.header("CMB Angular Power Spectrum with Contributions")
    # Get C_l^TT from CLASS (in [μK^2])
    lmax = 2500
    cl = cosmo.raw_cl(lmax)
    ells = cl['ell'][2:]  # drop ell=0,1
    cl_tt = cl['tt'][2:]

    # Load Planck 2018 TT data
    show_obs_data = st.checkbox("Show Planck 2018 TT Data", value=False)
    planck_data = np.loadtxt('/Users/r.kanaki/code/lunch_seminar/Data/COM_PowerSpect_CMB-TT-full_R3.01.txt')
    ell_data = planck_data[:, 0]
    cl_data = planck_data[:, 1]
    cl_err_plus = planck_data[:, 2]
    cl_err_minus = planck_data[:, 3]

    # Downsample Planck data
    # Downsample CLASS prediction
    step = 25
    ell_sampled = ell_data[::step]
    cl_sampled = cl_data[::step]
    cl_err_plus_sampled = cl_err_plus[::step]
    cl_err_minus_sampled = cl_err_minus[::step]

    # Convert CLASS prediction to [μK^2]
    T0 = 2.7255 * 1e6  # [K^2] to [μK^2]
    cl_tt = cl_tt * T0**2  # [μK^2]

    # Contribution toggles
    show_contributions = st.checkbox("Show Contributions", value=False)
    M = Class()
    # Compute contributions
    if show_contributions:
      M.empty()
      M.set(common_settings)
      M.set({'temperature contributions': 'tsw'})
      M.compute()
      cl_TSW = M.raw_cl(lmax)
      M.empty()
      M.set(common_settings)
      M.set({'temperature contributions': 'eisw'})
      M.compute()
      cl_eISW = M.raw_cl(lmax)
      M.empty()
      M.set(common_settings)
      M.set({'temperature contributions': 'lisw'})
      M.compute()
      cl_lISW = M.raw_cl(lmax)
      M.empty()
      M.set(common_settings)
      M.set({'temperature contributions': 'dop'})
      M.compute()
      cl_Doppler = M.raw_cl(lmax)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ells, cl_tt * ells * (ells + 1) / (2 * np.pi), lw=1.8, color='blue', label='CLASS prediction')

    if show_contributions:
      ax.plot(ells, cl_TSW['tt'][2:] * ells * (ells + 1) * T0**2 / (2 * np.pi), lw=1.8, color='black', label='TSW')
      ax.plot(ells, cl_eISW['tt'][2:] * ells * (ells + 1) * T0**2 / (2 * np.pi), lw=1.8, color='green', label='eISW')
      ax.plot(ells, cl_lISW['tt'][2:] * ells * (ells + 1) * T0**2 / (2 * np.pi), lw=1.8, color='orange', label='lISW')
      ax.plot(ells, cl_Doppler['tt'][2:] * ells * (ells + 1) * T0**2 / (2 * np.pi), lw=1.8, color='purple', label='Doppler')

    if show_obs_data:
      ax.errorbar(ell_sampled, cl_sampled, yerr=[cl_err_plus_sampled, cl_err_minus_sampled], fmt='o', markersize=4, capsize=2, color='red', label='Planck 2018 TT')

    ax.set_xlabel(r'Multipole $\ell$', fontsize=16)
    ax.set_ylabel(r'$\ell(\ell+1)C_\ell^{TT}/2\pi\ [\mu K^2]$', fontsize=16)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=16, loc='upper right', frameon=False)
    ax.minorticks_on()
    plt.tight_layout()
    st.pyplot(fig)

    st.header("CMB Temperature Anisotropies")
    # Slider for Healpix resolution
    nside = st.select_slider("Healpix Resolution (nside)", options=[32, 64, 128, 256, 512, 1024], value=1024)
    # Generate a simulated CMB map using the CLASS C_l^TT
    cmb_map = hp.synfast(cl_tt, nside=nside, lmax=lmax, new=True, verbose=False)

    # Button to toggle mask application
    apply_mask = st.checkbox("Apply Planck 2018 UT78 Mask", value=False)

    if apply_mask:
      # === Load Planck 2018 UT78 Mask ===
      mask_path = "/Users/r.kanaki/code/lunch_seminar/Data/COM_Mask_CMB-common-Mask-int_2048_R3.00.fits"
      mask_2048 = hp.read_map(mask_path, verbose=False)

      # Downgrade mask to match the map nside
      mask = hp.ud_grade(mask_2048, nside_out=nside)
      mask = np.where(mask > 0.9, 1, 0)  # Binarize mask

      # === Apply mask ===
      cmb_map_masked = cmb_map * mask
      # Set masked pixels to hp.UNSEEN so they appear as background in the plot
      cmb_map_masked[mask == 0] = hp.UNSEEN

      # Plot the masked CMB map
      fig = plt.figure(figsize=(8, 6))
      hp.mollview(cmb_map_masked, title='CMB Map with Planck 2018 UT78 Mask', unit='μK', cmap='jet', fig=fig.number)
      hp.graticule()
    else:
      # Plot the unmasked CMB map
      fig = plt.figure(figsize=(8, 6))
      hp.mollview(cmb_map, title='Simulated CMB map from CLASS $C_\ell^{TT}$', unit='μK', cmap='jet', fig=fig.number)
      hp.graticule()

    st.pyplot(fig)

    st.header("CMB Polarization Power Spectrum")
    # Slider for scalar-to-tensor ratio r
    r = st.slider("Scalar-to-Tensor Ratio (r)", 0.0, 1.0, 0.1, step=0.01)

    # Tensor and Scalar Power Spectrum Comparison
    l_max_scalars = 2500
    l_max_tensors = 2500

    # Scalar modes only
    M_s = Class()
    M_s.set(common_settings)
    M_s.set({'modes': 's', 'lensing': 'yes', 'l_max_scalars': l_max_scalars})
    M_s.compute()

    # Scalar + Tensor modes
    M_t = Class()
    M_t.set(common_settings)
    M_t.set({'modes': 's,t', 'lensing': 'yes', 'r': r, 'n_t': 0,
         'l_max_scalars': l_max_scalars, 'l_max_tensors': l_max_tensors})
    M_t.compute()

    # Extract power spectra
    clt = M_t.raw_cl(l_max_scalars)
    cls = M_s.raw_cl(l_max_scalars)
    cl_lensed = M_t.lensed_cl(l_max_scalars)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    ell = cls['ell']
    ellt = clt['ell']
    factor = 1.e10 * ell * (ell + 1.) / (2. * np.pi)
    factort = 1.e10 * ellt * (ellt + 1.) / (2. * np.pi)

    ax.loglog(ell, factor * cls['tt'], 'r-', label=r'$\mathrm{TT(s)}$')
    ax.loglog(ellt, factort * clt['tt'], 'r:', label=r'$\mathrm{TT(t)}$')
    ax.loglog(ell, factor * cls['ee'], 'b-', label=r'$\mathrm{EE(s)}$')
    ax.loglog(ellt, factort * clt['ee'], 'b:', label=r'$\mathrm{EE(t)}$')
    ax.loglog(ellt, factort * clt['bb'], 'g:', label=r'$\mathrm{BB(t)}$')
    ax.loglog(ell, factor * (cl_lensed['bb'] - clt['bb']), 'g-', label=r'$\mathrm{BB(lensing)}$')

    ax.set_xlim([2, l_max_scalars])
    ax.set_ylim([1.e-8, 10])
    ax.set_xlabel(r"$\ell$", fontsize=16)
    ax.set_ylabel(r"$\ell (\ell+1) C_\ell^{XY} / 2 \pi \,\,\, [\times 10^{10}]$", fontsize=16)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(loc='right', bbox_to_anchor=(1.4, 0.5), fontsize=12)
    st.pyplot(fig)
    st.header("Polarization Maps")
    # Generate simulated CMB polarization maps (E and B modes) from the lensed power spectra
    # Use cl_lensed['ee'] and cl_lensed['bb'] for E and B modes, respectively

    # Prepare the input power spectra for synfast: [TT, EE, BB, TE]
    # TT is not needed for pure polarization, but synfast expects a 4-list
    cl_synfast = [cl_lensed['tt'], cl_lensed['ee'], cl_lensed['bb'], cl_lensed['te']]

    # Generate Q and U maps (Stokes parameters) using healpy.synfast
    # Set pol=True to get polarization maps
    cmb_maps = hp.synfast(cl_synfast, nside=nside, lmax=lmax, new=True, pol=True, verbose=False)
    cmb_T, cmb_Q, cmb_U = cmb_maps  # T, Q, U maps

    # Optionally, decompose Q/U into E/B maps using healpy
    alm_EB = hp.map2alm([cmb_T, cmb_Q, cmb_U], pol=True, lmax=lmax)
    cmb_E = hp.alm2map(alm_EB[1], nside=nside, lmax=lmax)
    cmb_B = hp.alm2map(alm_EB[2], nside=nside, lmax=lmax)

    nside_plot = 32  # lower resolution for vector field visualization
    npix = hp.nside2npix(nside_plot)
    theta, phi = hp.pix2ang(nside_plot, np.arange(npix))

    # Downsample Q/U/E/B maps for plotting
    Q_plot = hp.ud_grade(cmb_Q, nside_plot)
    U_plot = hp.ud_grade(cmb_U, nside_plot)
    E_plot = hp.ud_grade(cmb_E, nside_plot)
    B_plot = hp.ud_grade(cmb_B, nside_plot)

    # Convert spherical to Mollweide projection coordinates
    lon = np.rad2deg(phi) - 180  # [-180, 180]
    lat = 90 - np.rad2deg(theta)  # [-90, 90]

    # Compute polarization angles and amplitudes
    pol_angle = 0.5 * np.arctan2(U_plot, Q_plot)
    pol_amp = np.sqrt(Q_plot**2 + U_plot**2)

    # Normalize arrows for visibility
    arrow_scale = 0.04 * pol_amp / pol_amp.max()

    fig, axs = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)

    # Plot settings
    arrow_skip = 20  # plot every Nth arrow for clarity
    x = lon[::arrow_skip]
    y = lat[::arrow_skip]
    u = arrow_scale[::arrow_skip] * np.cos(pol_angle[::arrow_skip])
    v = arrow_scale[::arrow_skip] * np.sin(pol_angle[::arrow_skip])

    # E-mode
    im0 = axs[0].scatter(lon, lat, c=E_plot, cmap='RdBu_r', s=10, lw=0, alpha=0.85)
    axs[0].quiver(x, y, u, v, color='k', alpha=0.7, width=0.003, scale=0.4)
    axs[0].set_title('E-mode', fontsize=16)
    axs[0].set_xlabel('RA [deg]')
    axs[0].set_ylabel('Dec [deg]')
    axs[0].set_xlim([-180, 180])
    axs[0].set_ylim([-90, 90])
    fig.colorbar(im0, ax=axs[0], orientation='horizontal', pad=0.1, label='μK')

    # B-mode
    im1 = axs[1].scatter(lon, lat, c=B_plot, cmap='RdBu_r', s=10, lw=0, alpha=0.85)
    axs[1].quiver(x, y, u, v, color='k', alpha=0.7, width=0.003, scale=0.4)
    axs[1].set_title('B-mode', fontsize=16)
    axs[1].set_xlabel('RA [deg]')
    axs[1].set_xlim([-180, 180])
    axs[1].set_ylim([-90, 90])
    fig.colorbar(im1, ax=axs[1], orientation='horizontal', pad=0.1, label='μK')

    plt.tight_layout()
    st.pyplot(fig)

# galaxy clustering
with tab3:
    st.header("Matter Power Spectrum")
    k = np.logspace(-4, 1, 1000)
    pk = [cosmo.pk(ki, 0.0) for ki in k]
    fig, ax = plt.subplots()
    ax.loglog(k, pk, label="z=0")
    ax.set_xlabel(r"$k \, [h/\mathrm{Mpc}]$")
    ax.set_ylabel(r"$P(k) \, [\mathrm{Mpc}^3/h^3]$")
    ax.grid()
    ax.legend()
    st.pyplot(fig)
    st.header("BAO Highlight")
    st.header("Galaxy Clustering Correlation Function")
    st.header("Redshift Space Distortions")

# cosmic shear
with tab4:
    st.header("Convergence Power Spectrum")
    st.header("Convergence Map")
    
    # Slider for Healpix resolution
    nside = st.slider("Healpix Resolution (nside)", 32, 512, 128, step=32)
    
    # Generate a random convergence map for demonstration
    npix = hp.nside2npix(nside)
    convergence_map = np.random.normal(size=npix)
    
    # Plot the convergence map using Mollweide projection
    fig = plt.figure(figsize=(8, 6))
    hp.mollview(convergence_map, title="Convergence Map", fig=fig.number, unit="Convergence")
    st.pyplot(fig)
    st.header("Cosmic Shear Correlation Functions")
    st.header("Cosmic Shear Maps")

#supernovae
with tab5:
    st.header("Supernovae Type Ia")
    st.header("Distance Modulus from Supernovae")
    st.header("Supernovae Light Curves")
    st.header("Supernovae Redshift Distribution")

st.markdown("---")
st.markdown("Developed for lunch seminar by Rintaro Kanaki. © 2025")