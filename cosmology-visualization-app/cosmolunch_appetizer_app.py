import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import healpy as hp
from classy import Class
import qrcode
from io import BytesIO
from matplotlib.patches import FancyArrowPatch
from scipy import interpolate
from scipy import integrate
import camb
import glass
from cosmology import Cosmology
import matplotlib.ticker as ticker
import camb
from cosmology import Cosmology
import glass
import glass.ext.camb

# Set up the Streamlit page
st.set_page_config(page_title="Mundus Ex Machina", layout="wide")
st.title("Mundus Ex Machina: A Cosmology Visualization Application")
st.markdown("""
Welcome to the **Mundus Ex Machina** — an interactive educational tool for exploring cosmological models and the large-scale structure of the Universe.

*Mundus* is a Latin word meaning “the world” or “the ordered universe.” As the name suggests, the Universe is not random, it has structure. Cosmology is the science that studies this structure in order to understand the fundamental laws of physics.
Unlike the ancient Greeks and Romans, we now have computers that allow us to perform complex calculations and simulations. Modern cosmological theories must be consistent with observational data across a wide range of times and scales.
In this application, we present four examples of astronomical phenomena, spanning from the early Universe to the present day, and from small to large scales.
This application is designed for public outreach and teaching, with clear explanations, interactive visualizations, and reproducible simulations based on publicly available data and open-source codes.
""")
st.markdown("---")
left_col, right_col = st.columns(2)
st.markdown("---")
# Sidebar for cosmological parameters
st.sidebar.header("Cosmological Parameters")
st.sidebar.markdown("Fit weird curves by hand")
h = st.sidebar.slider(r"$h$ Hubble Parameter", 0.5, 0.9, 0.6736, step=0.001)
Omega_m = st.sidebar.slider(
    r"$\Omega_m$ Matter Density", 0.07, 1.0, 0.315, step=0.0001)
Omega_b = st.sidebar.slider(
    r"$\Omega_b$ Baryon Density", 0.01, 0.07, 0.049, step=0.0001)
Omega_r = st.sidebar.slider(
    r"$\Omega_r$ Radiation Density", 0.0, 0.0002, 9.2e-5, step=1e-6, format="%.6f")
sigma8 = st.sidebar.slider(
    r"$\sigma_8$ Amplitude of Matter Fluctuations", 0.5, 1.2, 0.811, step=0.001)
n_s = st.sidebar.slider(r"$n_s$ Spectral Index", 0.9, 1.1, 0.9649, step=0.001)

with left_col:
    st.markdown("""
    ### Overview
    - **Basics**: Learn the fundamental properties of the Universe, such as how fast it is expanding, how distances are measured in cosmology, and how different components (matter, radiation, dark energy) evolve over time.  
    💡*What you learn:* How we describe and measure the expanding Universe.
    - **CMB (Cosmic Microwave Background)**: Explore the oldest light in the Universe(a snapshot from when the Universe was only about 380,000 years old). We show theoretical predictions and simulated maps of tiny temperature fluctuations.  
    💡*What you learn:* What the early Universe looked like and how structure began.
    - **Galaxy Clustering**: Investigate how galaxies are distributed across space, not randomly, but in a cosmic web of clusters, filaments, and voids.  
    💡*What you learn:* How large-scale structure forms and how matter is distributed in the Universe.
    - **Weak Lensing**: See how gravity bends light from distant galaxies, slightly distorting their shapes. We simulate these effects to map invisible matter like dark matter.  
    💡*What you learn:* How we “see” dark matter and trace the structure of the Universe through gravity.
    """)
    st.markdown("""
    ### About this App
    - **Model**:  
    We adopt the **flat $\mathrm{\Lambda}$CDM model**, the current standard model of cosmology.  
    In this framework, the Universe is:
    - **Flat** (its geometry follows Euclidean geometry on large scales),  
    - Dominated by **dark energy (Λ)**, which drives the accelerated expansion,  
    - And composed of **cold dark matter (CDM)**, which forms the backbone of cosmic structure.  
    This model successfully explains a wide range of observations, including the Cosmic Microwave Background, galaxy clustering, and the expansion history of the Universe.  
    Advanced extensions (such as massive neutrinos or evolving dark energy) can be explored in future versions.

    ### References: 
    Many figures and methods are inspired by standard cosmology textbooks and literature (e.g. *Modern Cosmology* by Scott Dodelson).
    - **Software**:
        - [CLASS](http://class-code.net/): computes theoretical predictions such as power spectra.  
        - [GLASS](https://glass.readthedocs.io/stable/index.html): generates large-scale structure simulations.
        - [healpy](https://healpy.readthedocs.io/en/latest/): handles spherical maps of the sky.  
        - [Matplotlib](https://matplotlib.org/): used for visualization.  
    - **Developed by [Rintaro Kanaki](https://github.com/Rintaro0406)**. This app is built with [Streamlit](https://streamlit.io/).  
    """)

with right_col:
    st.markdown("""
    ### Cosmological Parameters

    The flat **$\mathrm{\Lambda}$CDM** model describes the Universe using a small set of key parameters.  
    By adjusting them, you can explore how the Universe evolves and how structures form.
    - **Hubble Parameter (h)**  
    Sets how fast the Universe is expanding today. A larger value means galaxies move away from each other more quickly.  
    👉 *Think:* How fast is the Universe stretching?  
    - **Matter Density ($\Omega_m$)**  
    Total amount of matter (both dark matter and normal matter). More matter means stronger gravity, which slows expansion and enhances structure formation.  
    👉 *Think:* How much visible and unvisible matter  pulling things together?  
    - **Baryon Density ($\Omega_b$)**  
    The amount of “ordinary” matter (atoms: stars, gas, you!). This affects features like acoustic oscillations and galaxy formation.  
    👉 *Think:* How much visible matter is there?  
    - **Radiation Density ($\Omega_r$)**  
    Energy in light particles (photons and neutrinos), important in the early Universe. Today it is very small, but it dominated the Universe shortly after the Big Bang.  
    👉 *Think:* How important was radiation in the Universe?  
    - **$\sigma_8$ (Fluctuation Amplitude)**  
    Measures how “clumpy” matter is on scales of 8 Mpc/h. Higher values mean stronger clustering and more pronounced cosmic structure.  
    👉 *Think:* How lumpy is the Universe?  
    - **Spectral Index ($n_s$)**  
    Describes how fluctuations depend on scale (large vs small structures). If $n_s = 1$, fluctuations are scale-invariant; deviations indicate early-Universe physics.  
    👉 *Think:* Are small and large structures equally important?  


    ### Non-science background

    For a more accessible introduction, I recommend [*Big Bang*](https://www.amazon.co.uk/Big-Bang-Important-Scientific-Discovery/dp/0007152523/ref=tmm_pap_swatch_0) by [Simon Singh](https://de.wikipedia.org/wiki/Simon_Singh).

    👉 It presents a brief and engaging history of cosmology, from ancient Greece to modern science without mathematics.
    I also read this book when I was a child, and it gives my interest in physics and cosmology.
    """)


# Generate QR Code for the app
st.sidebar.header("QR Code")
app_url = "http://localhost:8501"  # Future I will deploy this app on the server
qr = qrcode.QRCode()
qr.add_data(app_url)
qr.make(fit=True)
img = qr.make_image(fill="black", back_color="white")
buffer = BytesIO()
img.save(buffer, format="PNG")
st.sidebar.image(buffer.getvalue(), caption="Scan to Open App",
                 use_column_width=True)

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
}


cosmo.set(common_settings)
cosmo.compute()

# Tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs(
    ["Basics", "CMB", "Galaxy Clustering", "Weak Lensing"])

# basics
with tab1:
    st.header("Basics of Cosmology")
    st.markdown("""
    Cosmology is the study of how the Universe expands, evolves, and forms structure over time.  
    In this section, we explore the **fundamental quantities** that describe our Universe.

    👉 **What you will learn in this section:**
    - How fast the Universe is expanding (Hubble parameter)  
    - How we measure distances across cosmic scales  
    - How different components (radiation, matter, dark energy) evolve over time  
    - How the Universe transitioned from the early hot Big Bang to today.

    👉 **Why this matters:**  

    All modern cosmology — from galaxy formation to dark energy — is built on these basic relations.  
    Understanding them is like learning the “rules of the game” of the Universe.

    👉 Use the controls left to see how changing **cosmological parameters** affects the evolution of the Universe.
    """)
    use_basics = st.checkbox("Calculate Basics", value=False)
    st.markdown("---")
    if use_basics:
        st.header("Hubble Parameter")
        st.markdown("""
        ### The Expanding Universe

        In 1929, [Edwin Hubble](https://en.wikipedia.org/wiki/Edwin_Hubble) discovered that distant galaxies are moving away from us and the farther they are, the faster they recede.  

        👉 This was [the first evidence](https://ui.adsabs.harvard.edu/abs/1926ApJ....64..321H/abstract) that the **Universe is expanding**.

        This expansion is described by the **Hubble parameter** $H(z)$, which tells us how fast the Universe is expanding at different times.

        - Today: expansion rate = $H_0$  
        - In the past: expansion was different depending on the contents of the Universe  

        👉 *Key idea:*  
        We are not at the center, **space itself is stretching**, carrying galaxies apart.

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
        ax.plot(z, Hz_kmsMpc / (1 + z), color='blue',
                lw=2, label='Calculated using CLASS')

        if show_obs_data:
            ax.errorbar(Riess_2019[0], Riess_2019[1] / (1 + Riess_2019[0]), yerr=Riess_2019[2] / (
                1 + Riess_2019[0]), fmt='o', color='red', label='Riess et al. 2019')
            ax.errorbar(BOSS_DR12[:, 0], BOSS_DR12[:, 1] / (1 + BOSS_DR12[:, 0]), yerr=BOSS_DR12[:,
                        2] / (1 + BOSS_DR12[:, 0]), fmt='s', color='black', label='BOSS DR12')
            ax.errorbar(DR14_quasars[0], DR14_quasars[1] / (1 + DR14_quasars[0]), yerr=DR14_quasars[2] / (
                1 + DR14_quasars[0]), fmt='^', color='green', label='DR14 quasars')
            ax.errorbar(DR14_Ly_alpha[0], DR14_Ly_alpha[1] / (1 + DR14_Ly_alpha[0]), yerr=DR14_Ly_alpha[2] / (
                1 + DR14_Ly_alpha[0]), fmt='D', color='orange', label='DR14 Ly-alpha')

        ax.legend(fontsize=12, loc='upper right')
        ax.set_xlabel('Redshift $z$', fontsize=16)
        ax.set_ylabel(r'$H(z)/(1+z)$ [km s$^{-1}$ Mpc$^{-1}$]', fontsize=16)
        ax.set_yscale('log')
        ax.grid(True, which='both', linestyle='--', alpha=0.4)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', labelsize=13,
                       length=7, width=1.5, direction='in', top=True, right=True)
        ax.tick_params(axis='both', which='minor', labelsize=11,
                       length=4, width=1, direction='in', top=True, right=True)
        st.pyplot(fig)
        st.markdown("""
        ### Understanding the Plot

        This plot shows how the expansion rate of the Universe changes over time.

        - The horizontal axis (**redshift $z$**) tells us how far back in time we are looking  
        - The vertical axis shows the expansion rate $H(z)/(1+z)$  

        ### What you see:

        - **Blue curve**: theoretical prediction from the ΛCDM model (computed with CLASS)  
        - **Points with error bars**: real observations of the Universe  

        """)
        if show_obs_data:
            st.markdown("""
            ### Observational data (explained simply):

            - **[Riess et al. 2019](https://arxiv.org/abs/1903.07603)**  
            → Measures the expansion rate **today** using nearby galaxies and supernovae(*Local Universe*).

            - **[BOSS DR12](https://arxiv.org/abs/1607.03155)** 
            → Uses the clustering of galaxies to measure distances and expansion(*Intermediate distances*).

            - **[DR14 Quasars](https://arxiv.org/abs/1910.10395)**  
            → Uses very bright, distant objects (quasars) to probe the Universe(*Farther back in time*).

            - **[DR14 Ly-alpha](https://arxiv.org/abs/1910.10395)**  
            → Uses absorption of light by intergalactic hydrogen(*Very early Universe*).

            """)
        st.markdown("""
        👉 **Experiment Idea:**  
        - Is the agreement between observational data shows and your model describes the Universe very well? How is your choice of parameters?
        - What would happen if the Universe had **no dark matter** ($\Omega_m = 0$),  
        - Or if it were made entirely of matter with **no dark energy** ($\Omega_m = 1.0$)?

        This plot corresponds to Figure 2.8 in **[Modern Cosmology by Scott Dodelson & Fabian Schmidt](https://www.amazon.co.uk/Modern-Cosmology-Scott-Dodelson/dp/0128159480)**.

        ---
        """)
        st.header("Distance Measures")
        st.markdown("""
        ### How do we measure distance in the Universe?

        In everyday life, distance is simple — you use a ruler.  
        But in cosmology, things are more complicated:

        👉 The Universe is expanding, and we observe distant objects through light.

        So depending on **how we measure**, we get different definitions of distance.

        """)
        # Redshift range (reuse z)
        chi = np.array([cosmo.angular_distance(z_i) * (1 + z_i)
                        for z_i in z])  # comoving distance, Mpc
        d_A = chi / (1 + z)  # angular diameter distance, Mpc
        d_L = chi * (1 + z)  # luminosity distance, Mpc

        # Plot distance measures
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(z, chi, label=r'$\chi(z)$ (Comoving Distance)', color='blue', lw=2)
        ax.plot(z, d_A, label=r'$d_A(z)$ (Angular Diameter Distance)',
                color='green', lw=2, linestyle='--')
        ax.plot(z, d_L, label=r'$d_L(z)$ (Luminosity Distance)',
                color='red', lw=2, linestyle='-.')
        ax.set_xlabel('Redshift $z$', fontsize=15)
        ax.set_ylabel(r'Distance [Mpc]', fontsize=15)
        ax.set_yscale('log')
        ax.legend(fontsize=13, loc='upper left')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        st.pyplot(fig)
        st.markdown("""
         ### What you see in the plot:

        - **$\chi(z)$ (Comoving Distance)**  
        → The “true” distance if we freeze the expansion of the Universe  
        👉 *Think:* distance in a static snapshot of the Universe  

        - **$d_A(z)$ (Angular Diameter Distance)**  
        → Used when measuring the **apparent size** of objects  
        👉 *Think:* how big something looks in the sky  

        - **$d_L(z)$ (Luminosity Distance)**  
        → Used when measuring **brightness** (e.g. supernovae)  
        👉 *Think:* how bright something appears  

        ### Key insight:

        👉 Because the Universe expands, these distances are **not the same**.

        - Objects at higher redshift are seen further back in time  
        - Light is stretched and diluted as it travels  

        This is why cosmology needs different distance measures.
    
        This plot corresponds to Figure 2.3 in ***[Modern Cosmology by Scott Dodelson & Fabian Schmidt](https://www.amazon.co.uk/Modern-Cosmology-Scott-Dodelson/dp/0128159480)***.

        ----
        """)
        st.markdown(r"""
        ### 👉 Exercise (Distance Measures)

        Can you derive the relation between different cosmological distances?

        1. Starting from the definition of **comoving distance**:
        $$
        \chi(z) = \int_0^z \frac{c}{H(z')} \, dz'
        $$

        👉 Show that:

        - the **angular diameter distance** is:
        $$
        d_A(z) = \frac{\chi(z)}{1+z}
        $$

        - the **luminosity distance** is:
        $$
        d_L(z) = \chi(z)(1+z)
        $$

        ---

        👉 *Hint:*  
        Think about how expansion affects:
        - **apparent size** (angles)  
        - **brightness** (photon energy + arrival rate)
        """)
        show_answer_dist = st.checkbox("Show Answer (Distance Measures)")

        if show_answer_dist:
            st.markdown(r"""
            ### ✅ Answer

            #### 1. Angular Diameter Distance

            Consider an object of physical size $D$.

            Its observed angular size is:
            $$
            \theta = \frac{D}{d_A}
            $$

            However, due to expansion, the physical size at emission is smaller by a factor $(1+z)$:

            $$
            D_{\rm emission} = \frac{D_{\rm today}}{1+z}
            $$

            Using comoving distance $\chi$:
            $$
            \theta = \frac{D_{\rm emission}}{\chi}
            $$

            So:
            $$
            d_A = \frac{\chi}{1+z}
            $$

            ---

            #### 2. Luminosity Distance

            Brightness depends on energy flux:

            $$
            F = \frac{L}{4\pi d_L^2}
            $$

            In an expanding Universe, two effects reduce brightness:

            1. **Photon energy redshift** → factor $(1+z)$  
            2. **Arrival rate of photons** → another factor $(1+z)$  

            👉 Total effect: $(1+z)^2$

            So:
            $$
            F = \frac{L}{4\pi \chi^2 (1+z)^2}
            $$

            Comparing with definition of $d_L$:
            $$
            d_L = \chi (1+z)
            $$

            ---

            ### 👉 Final relation

            $$
            d_L = (1+z)^2 d_A
            $$

            ---
            """)
        st.header("Density Parameter Evolution")
        st.markdown("""
        ### How does the content of the Universe change over time?

        The Universe is made of different components:
        - **Radiation** (light, neutrinos)
        - **Matter** (dark matter + normal matter)
        - **Dark Energy** (drives acceleration)

        👉 These components evolve differently as the Universe expands.

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

        shows_equality = st.checkbox(
            "Show Equality Lines (Density Evolution)", value=False, key="density_equality")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(a_plot, rho_r, label=r'$\Omega_r(a)$ (Radiation)',
                color='orange', lw=2, linestyle='-.')
        ax.plot(a_plot, rho_m, label=r'$\Omega_m(a)$ (Matter)', color='green', lw=2, linestyle='-')
        ax.plot(a_plot, rho_L, label=r'$\Omega_\Lambda(a)$ (Dark Energy)',
                color='purple', lw=2, linestyle='--')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Scale factor $a$', fontsize=15)
        ax.set_ylabel(r'$\rho_s(t)$', fontsize=15)
        if shows_equality:
            ax.axvline(a_eq, color='red', linestyle=':',
                       label=r'$a_{\rm eq}$ (Matter-Radiation Equality)')
            ax.axvline(a_lambda, color='blue', linestyle='--',
                       label=r'$a_{\Lambda}$ ($\Lambda$-Matter Equality)')
        ax.legend(fontsize=12)
        ax.grid(True, which='both', linestyle=':', alpha=0.5)
        st.pyplot(fig)
        st.markdown("""
        ### What you see in the plot:

        - **Radiation ($\Omega_r$)**  
        → Dominates the **very early Universe**  
        👉 decreases very quickly as the Universe expands  

        - **Matter ($\Omega_m$)**  
        → Dominates at **intermediate times**  
        👉 responsible for forming galaxies and structure  

        - **Dark Energy ($\Omega_\Lambda$)**  
        → Dominates **today**  
        👉 drives accelerated expansion  

        ### Key insight:

        👉 The Universe goes through **three phases**:

        1. Radiation-dominated  
        2. Matter-dominated  
        3. Dark-energy-dominated  

        👉 Exercise: 
        - How is mathematical behaviour of each components depending on scale factor $a$?
        - At what scale factor $a$ does **matter equal radiation**? (Hint: set $\Omega_r(a) = \Omega_m(a)$)
        """)
        show_answer = st.checkbox("Show Answer")

        if show_answer:
            st.markdown(r"""
            ### ✅ Answer

            #### 1. Mathematical behaviour of each component

            Each component evolves differently with the scale factor $a$:

            - **Radiation**:
            $$
            \rho_r(a) \propto a^{{-4}}
            $$
            👉 decreases very quickly (expansion + redshift of light)

            - **Matter**:
            $$
            \rho_m(a) \propto a^{{-3}}
            $$
            👉 decreases due to expansion (volume increases)

            - **Dark Energy**:
            $$
            \rho_\Lambda(a) = \text{{constant}}
            $$
            👉 does not dilute as the Universe expands

            ---

            #### 2. Matter–Radiation Equality

            We solve:
            $$
            \Omega_r(a) = \Omega_m(a)
            $$

            Using:
            $$
            \Omega_r(a) = \frac{{\Omega_{{r,0}}}}{{a^4}}, \quad
            \Omega_m(a) = \frac{{\Omega_{{m,0}}}}{{a^3}}
            $$

            Setting them equal:
            $$
            \frac{{\Omega_{{r,0}}}}{{a^4}} = \frac{{\Omega_{{m,0}}}}{{a^3}}
            $$

            Solving for $a$:
            $$
            a_{{\rm eq}} = \frac{{\Omega_{{r,0}}}}{{\Omega_{{m,0}}}}
            $$

            👉 For your current parameters:
            $$
            a_{{\rm eq}} \approx {a_eq:.2e}
            $$
            """)
        st.markdown("""
        This plot corresponds to figure 1.3 in [Modern Cosmology](https://www.amazon.co.uk/Modern-Cosmology-Scott-Dodelson/dp/0128159480)

        ---
        """)
        st.header("Scale Factor Evolution")
        st.markdown("""
        ### How does the Universe expand over time?

        The **scale factor** $a(t)$ tells us how the size of the Universe changes with time.

        - $a(t) = 1$ today  
        - $a(t) < 1$ in the past  
        - $a(t) \sim 0$ near the Big Bang  

        👉 *Think:* the scale factor is a “cosmic ruler” that stretches as the Universe expands.
        """)
        # Constants
        H0 = 100 * h  # Hubble constant in km/s/Mpc
        # Hubble constant in s^-1 (Mpc to m conversion)
        H0_si = H0 * 1e3 / (3.086e22)

        # Compute E(a) = H(a)/H0
        E = np.sqrt(Omega_r / a_plot**4 + Omega_m / a_plot**3 + omega_lambda)

        # Numeric integration for t(a)
        # dt/da = 1 / [a H(a)] ⇒ t(a) = ∫ da / [a H0 E(a)]
        da = np.diff(a_plot)
        integrand = 1.0 / (a_plot * H0_si * E)
        t = np.concatenate(
            ([0], np.cumsum(0.5 * (integrand[:-1] + integrand[1:]) * da)))

        # Convert to years
        seconds_per_year = 3600 * 24 * 365
        t_years = t / seconds_per_year

        # Use a unique key for the checkbox to avoid conflicts with the previous one
        shows_equality_2 = st.checkbox(
            "Show Equality Lines (Scale Factor Evolution)", value=False, key="scale_factor_equality")

        # Plot a(t) vs t
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog(t_years, a_plot, color='purple', lw=2,
                  label='Scale Factor Evolution')
        if shows_equality_2:
            ax.axhline(a_eq, color='red', linestyle=':',
                       label=r'Matter-Radiation Equality')
            ax.axhline(a_lambda, color='blue', linestyle=':',
                       label=r'$\Lambda$-Matter Equality')
        ax.set_xlabel('Cosmic time $t$ [yr]', fontsize=14)
        ax.set_ylabel('Scale factor $a(t)$', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(which='both', linestyle='--', alpha=0.5)
        st.pyplot(fig)
        st.markdown("""
        ### What you will learn from this plot:

        - How fast the Universe expands at different times  
        - How different components (radiation, matter, dark energy) control expansion  
        - Why the expansion history changes over time  

        ### What you see in the plot:

        - The curve shows how the Universe grows from very small ($a \ll 1$) to today ($a=1$)  
        - The slope changes depending on which component dominates  

        👉 The expansion is **not uniform** — it changes over cosmic time.

        ### Key insight:

        The expansion history has **three phases**:

        1. **Radiation-dominated era** → very early Universe  
        2. **Matter-dominated era** → structure formation  
        3. **Dark-energy-dominated era** → accelerated expansion today  

        ---
        """)
        st.markdown("""
        ### 👉 Exercise

        How does the scale factor behave in different epochs?

        - What is the time dependence $a(t)$ when:
        - the Universe is dominated by **radiation**?
        - the Universe is dominated by **matter**?
        - the Universe is dominated by **dark energy**?

        ### Important relations
        """)
        st.markdown(r"""
        - Expansion rate:
        $$
        H(t) = \frac{1}{a(t)} \frac{da(t)}{dt}
        $$

        - Redshift relation:
        $$
        a(t) = \frac{1}{1+z}
        $$

        - Cosmic time:
        $$
        t(a) = \int_0^a \frac{da'}{a' H(a')}
        $$

        👉 Hint: Use the Friedmann equation and how densities scale with $a$.
        """)
        show_answer_sf = st.checkbox("Show Answer (Scale Factor Evolution)")
        if show_answer_sf:
            st.markdown(r"""
            ### ✅ Answer

            The evolution of $a(t)$ depends on which component dominates the energy density.

            ---

            #### 1. Radiation-dominated Universe

            We know:
            $$
            \rho_r \propto a^{-4}
            $$

            From the Friedmann equation:
            $$
            H^2 \propto \rho
            $$

            So:
            $$
            H \propto a^{-2}
            $$

            Using:
            $$
            H = \frac{1}{a}\frac{da}{dt}
            $$

            we get:
            $$
            \frac{da}{dt} \propto a^{-1}
            $$

            Integrating:
            $$
            a(t) \propto t^{1/2}
            $$

            ---

            #### 2. Matter-dominated Universe

            $$
            \rho_m \propto a^{-3}
            $$

            So:
            $$
            H \propto a^{-3/2}
            $$

            Then:
            $$
            \frac{da}{dt} \propto a^{-1/2}
            $$

            Integrating:
            $$
            a(t) \propto t^{2/3}
            $$

            ---

            #### 3. Dark-energy-dominated Universe

            $$
            \rho_\Lambda = \text{constant}
            $$

            So:
            $$
            H = \text{constant}
            $$

            Then:
            $$
            \frac{da}{dt} = H a
            $$

            Solution:
            $$
            a(t) \propto e^{Ht}
            $$

            ---

            ### 👉 Final summary

            - Radiation era:  $a(t) \propto t^{1/2}$  
            - Matter era:   $a(t) \propto t^{2/3}$  
            - Dark energy era: $a(t) \propto e^{Ht}$  

            ---
            """)

        st.markdown("""
        ### 📚 Further Reading & References

        The calculations in this section are based on the **ΛCDM model**, the current standard model of cosmology.  
        We compute theoretical predictions using the [CLASS Boltzmann solver](http://class-code.net/), a widely used tool in modern cosmology.

        ---

        ### 📖 Learn more

        - *[Modern Cosmology by Scott Dodelson & Fabian Schmidt](https://www.amazon.co.uk/Modern-Cosmology-Scott-Dodelson/dp/0128159480)* — Scott Dodelson & Fabian Schmidt  

            👉 A comprehensive introduction to the theory behind these plots  

        - *[Physical Foundations of Cosmology](https://www.amazon.co.uk/Physical-Foundations-Cosmology-Viatcheslav-Mukhanov/dp/0521563984)* — Viatcheslav Mukhanov 

            👉 If you are masochist or confident with mathematics, it is excellent book. Sadly this book does not cover brilliant anekdote. 

        ### 🌐 Data & Observations

        - [Riess et al. (2019)](https://arxiv.org/abs/1903.07603) — Local expansion rate ($H_0$)  
        - [BOSS DR12](https://arxiv.org/abs/1607.03155) — Galaxy clustering and BAO  
        - [eBOSS DR14](https://arxiv.org/abs/1910.10395) — Quasars and Lyman-$\\alpha$ forest  
        """)
# cmb
with tab2:
    st.header("Cosmic Microwave Background")
    st.markdown("""
    ### The oldest light in the Universe

    The **Cosmic Microwave Background (CMB)** is the oldest light we can observe.  
    It was emitted about **380,000 years after the Big Bang**, when the Universe cooled enough for electrons and protons to combine into neutral atoms.

    👉 Before this time, the Universe was a hot plasma, and light could not travel freely.  
    👉 After this transition, light decoupled and has been traveling ever since.

    ---
    """)
    left_col_CMB, right_col_CMB = st.columns(2)
    with left_col_CMB:
        st.markdown("""
        ### Why is the CMB evidence for the Big Bang?

        The Big Bang model predicts that the early Universe was:

        - **Hot**

        - **Dense**

        - Filled with radiation and particles in thermal equilibrium  

        As the Universe expanded, it cooled down. Eventually:

        - Electrons and protons combined into neutral atoms  

        - Light was able to travel freely for the first time  

        👉 This moment is called **recombination**.

        If this picture is correct, we should still observe this radiation today, but:

        - Stretched by cosmic expansion  

        - Cooled from thousands of Kelvin  

        - Shifted into the **microwave** range  

        👉 This is exactly what we observe as the CMB (~2.7 K).

        """)
    with right_col_CMB:
        st.markdown("""
        ### What you will learn in this section:

        - What the early Universe looked like  
        - Why the CMB is almost uniform, but not perfectly  
        - How tiny fluctuations grew into galaxies and large-scale structure  
        - How we extract cosmological information from the CMB  

        ### What does the CMB tell us?

        Although the CMB is extremely uniform (~2.7 K), it contains tiny fluctuations:

        👉 Temperature variations at the level of **one part in 100,000**

        These small anisotropies encode:

        - The content of the Universe (matter, dark matter, radiation)  
        - The geometry of space  
        - The physics of the early Universe  
        - Evidence for cosmic inflation  
        """)
    st.markdown("---")
    # Get C_l^TT from CLASS (in [μK^2])
    use_CMB = st.checkbox("Calculate CMB Power Spectrum", value=False)
    if use_CMB:
        st.header("CMB Angular Power Spectrum")
        st.markdown("""
        ### How do we read the CMB power spectrum?

        The CMB sky is not perfectly uniform — it has tiny temperature fluctuations.  
        The **angular power spectrum** tells us how these fluctuations are distributed across different angular scales.

        👉 Instead of looking at a map, we summarize the information statistically.

        ---
        """)
        lmax = 2500
        cl = cosmo.raw_cl(lmax)
        ells = cl['ell'][2:]  # drop ell=0,1
        cl_tt = cl['tt'][2:]

        # Load Planck 2018 TT data
        show_obs_data = st.checkbox("Show Planck 2018 TT Data", value=False)
        planck_data = np.loadtxt(
            '/Users/r.kanaki/code/lunch_seminar/Data/COM_PowerSpect_CMB-TT-full_R3.01.txt')
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
        ax.plot(ells, cl_tt * ells * (ells + 1) / (2 * np.pi),
                lw=1.8, color='blue', label='CLASS prediction')

        if show_contributions:
            ax.plot(ells, cl_TSW['tt'][2:] * ells * (ells + 1) * T0 **
                    2 / (2 * np.pi), lw=1.8, color='black', label='TSW')
            ax.plot(ells, cl_eISW['tt'][2:] * ells * (ells + 1) * T0 **
                    2 / (2 * np.pi), lw=1.8, color='green', label='eISW')
            ax.plot(ells, cl_lISW['tt'][2:] * ells * (ells + 1) * T0 **
                    2 / (2 * np.pi), lw=1.8, color='orange', label='lISW')
            ax.plot(ells, cl_Doppler['tt'][2:] * ells * (ells + 1) * T0 **
                    2 / (2 * np.pi), lw=1.8, color='purple', label='Doppler')

        if show_obs_data:
            ax.errorbar(ell_sampled, cl_sampled, yerr=[
                        cl_err_plus_sampled, cl_err_minus_sampled], fmt='o', markersize=4, capsize=2, color='red', label='Planck 2018 TT')

        ax.set_xlabel(r'Multipole $\ell$', fontsize=16)
        ax.set_ylabel(r'$\ell(\ell+1)C_{\ell}^{TT}/2\pi\ [\mu K^2]$', fontsize=16)
        ax.grid(True, which='both', ls='--', alpha=0.4)
        ax.tick_params(labelsize=14)
        ax.legend(fontsize=16, loc='upper right', frameon=False)
        ax.minorticks_on()
        plt.tight_layout()
        st.pyplot(fig)
        left_column_CMB_2, right_column_CMB_2 = st.columns(2)
        with left_column_CMB_2:
            st.markdown("""
            ### What you see in the plot:

            - The horizontal axis (**multipole $\ell$**) corresponds to angular scale:
            - Small $\ell$ → large angles (large structures)  
            - Large $\ell$ → small angles (fine details)  

            - The vertical axis shows the **strength of temperature fluctuations**

            ### Key physical idea: acoustic oscillations

            In the early Universe:

            - Photons and baryons formed a tightly coupled fluid  
            - Gravity tried to pull matter inward  
            - Radiation pressure pushed it outward  

            👉 This created **sound waves (acoustic oscillations)** in the plasma.

            """)
        with right_column_CMB_2:
            st.markdown("""
            ### Why do we see peaks?

            At recombination, these oscillations were “frozen”:

            - Some regions were at **maximum compression** → peaks  
            - Others at **maximum rarefaction** → troughs  

            👉 These appear as a series of **peaks in the power spectrum**

            ### What each part tells us:

            - **Peak positions** → geometry of the Universe  
            - **Peak heights** → baryon density and dark matter  
            - **Large scales (low $\ell$)** → initial conditions (inflation)  
            - **Small scales (high $\ell$)** → diffusion and damping  
            
            """)
        if show_contributions:
            st.markdown("""
            ---

            ### Contributions (optional)

            If you turn this on, you can see how different physical effects combine to create the CMB signal.

            👉 *Think of it like different ingredients in a recipe for the Universe.*

            - **SW (Sachs–Wolfe)**  
            → Light loses or gains energy as it climbs out of gravity wells  
            👉 *Gravity affects the color (temperature) of the light*

            - **eISW (Early ISW)**  
            → Changes in gravity in the early Universe affect the light  
            👉 *The Universe is still evolving rapidly*

            - **lISW (Late ISW)**  
            → Dark energy changes gravity at late times  
            👉 *A signature of the accelerated expansion today*

            - **Doppler**  
            → Motion of matter creates shifts in the light  
            👉 *Like the Doppler effect of sound (e.g. a passing siren)*

            👉 All these effects together produce the pattern you see in the CMB power spectrum.

            """)
        if show_obs_data:
            st.markdown("""

            ###Observational Data

            - **[Planck 2018 TT Data](https://pla.esac.esa.int/pla/#home)**: Observational data from the Planck 2018 mission, which provides measurements of the CMB temperature anisotropies.

            """)
        st.markdown("""
        ### Theory vs observation

        - **Blue curve**: theoretical prediction (ΛCDM using CLASS)  
        - **Points**: measurements from the Planck satellite  

        👉 The remarkable agreement shows that the ΛCDM model describes the Universe extremely well.
        👉 Why this angular powerspectrum is sufficient for describing CMB? What assumption or observational fact is behind?
        """)
        st.markdown("""

        ---

        👉 **Experiment Idea:**

        1. Adjust the cosmological parameters to fit the CMB power spectrum.  
        2. Then go back to the *Basics* section and examine the $H(z)$ vs. $z$ curve.

        👉 You may notice a mismatch between the expansion rate inferred from the **early Universe (CMB)** and the **local Universe**.

        This is known as the **Hubble tension** — one of the biggest open problems in modern cosmology.

        - Are the observations incomplete or biased?  
        - Are there unknown systematic errors?  
        - Or do we need new physics beyond the standard $\Lambda$CDM model?

        👉 This question is still actively researched today.

        This plot corresponds to Figure 1.10 in [*Modern Cosmology* by Scott Dodelson & Fabian Schmidt](https://www.amazon.co.uk/Modern-Cosmology-Scott-Dodelson/dp/0128159480).

        ---
        """)

        st.header("CMB Temperature Anisotropies Map")
        st.markdown("""
        ### What does the CMB look like on the sky?

        This map shows the **tiny temperature variations** in the Cosmic Microwave Background across the entire sky.

        👉 *Think of this as a heat map of the early Universe.*

        ---
        """)
        # Slider for Healpix resolution
        nside = st.select_slider("Healpix Resolution (nside)", options=[
            32, 64, 128, 256, 512, 1024], value=1024)
        # Generate a simulated CMB map using the CLASS C_l^TT
        cmb_map = hp.synfast(cl_tt, nside=nside, lmax=lmax,
                             new=True, verbose=False)

        # Button to toggle mask application
        st.markdown("""
                    ‼️ The mask is only applied to the CMB map, not the power spectrum.
        """)
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
            hp.mollview(cmb_map_masked, title='CMB Map with Planck 2018 UT78 Mask',
                        unit='μK', cmap='jet', fig=fig.number)
            hp.graticule()
        else:
            # Plot the unmasked CMB map
            fig = plt.figure(figsize=(8, 6))
            hp.mollview(
                cmb_map, title='Simulated CMB map from CLASS $C_{\ell}^{TT}$', unit='μK', cmap='jet', fig=fig.number)
            hp.graticule()

        st.pyplot(fig)
        st.markdown("""

        ### Controls:

        - **Resolution ($n_{\text{side}}$)**  
        → Adjust how detailed the map is  
        👉 Higher values show finer structures  

        - **Mask (optional)**  
        → Removes regions contaminated by foreground signals (e.g. our Galaxy)  
        👉 Helps isolate the true cosmological signal  

        - **[Planck 2018 UT78 Mask](https://pla.esac.esa.int/pla/#home)**  
        → A standard mask used to clean the sky from foreground contamination  

        ---
        """)
        left_column_CMB_3, right_column_CMB_3 = st.columns(2)
        with left_column_CMB_3:
            st.markdown("""
            ### What you will learn from this map:

            - How the CMB looks across the whole sky  
            - How small the fluctuations really are  
            - How structure in today’s Universe began from tiny differences  

            ### What you see in the map:

            - Colors represent temperature differences:
            - **Red** → slightly hotter regions  
            - **Blue** → slightly colder regions  

            👉 These differences are extremely small:  
            only about **±0.0001 K** around the average temperature (2.7 K)
            """)
        with right_column_CMB_3:
            st.markdown("""
            ### Key idea:

            👉 These tiny fluctuations are the **seeds of all structure** in the Universe  
            (galaxies, clusters, and large-scale structure)


            ### Simulation vs reality

            - This map is **simulated** using the theoretical power spectrum from [CLASS](http://class-code.net/)  
            - Real observations (e.g. Planck) look remarkably similar  

            👉 This agreement is one of the strongest successes of modern cosmology
            👉 How do we compare this(Hint previous plot.)
            """)

        st.markdown("---")
        st.header("CMB Polarization Power Spectrum")
        st.markdown("""
        ### What is CMB polarization?

        The CMB is not only temperature — it is also **polarized light**.

        👉 Polarization tells us about how light was scattered in the early Universe.

        """)
        # Slider for scalar-to-tensor ratio r
        r = st.slider("Scalar-to-Tensor Ratio (r)", 0.0, 1.0, 0.1, step=0.01)

        # Tensor and Scalar Power Spectrum Comparison
        l_max_scalars = 2500
        l_max_tensors = 2500

        # Scalar modes only
        M_s = Class()
        M_s.set(common_settings)
        M_s.set({'modes': 's', 'lensing': 'yes',
                'l_max_scalars': l_max_scalars})
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
        ax.loglog(ell, factor * (cl_lensed['bb'] - clt['bb']),
                  'g-', label=r'$\mathrm{BB(lensing)}$')

        ax.set_xlim([2, l_max_scalars])
        ax.set_ylim([1.e-8, 10])
        ax.set_xlabel(r"$\ell$", fontsize=16)
        ax.set_ylabel(
            r"$\ell (\ell+1) C_{\ell}^{XY} / 2 \pi \,\,\, [\times 10^{10}]$", fontsize=16)
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend(loc='right', bbox_to_anchor=(1.4, 0.5), fontsize=12)
        st.pyplot(fig)
        left_column_CMB_4, right_column_CMB_4 = st.columns(2)
        with left_column_CMB_4:
            st.markdown("""
            ### What you will learn from this plot:

            - What **E-mode** and **B-mode** polarization are  
            - How different physical processes create them  
            - Why B-modes are linked to the very early Universe  
            - How we search for signals from **cosmic inflation**  

            ### What you see in the plot:

            - Different curves show how polarization depends on angular scale  
            - You can adjust the parameter $r$ to explore new physics  

            ### Two types of polarization:

            - **E-mode (gradient pattern)**  
            → Produced by ordinary density fluctuations  
            👉 Already well measured and understood  

            - **B-mode (curl pattern)**  
            → Much weaker and harder to detect  
            👉 Can be produced by two effects:
            - Gravitational lensing (distortion by matter)
            - **Primordial gravitational waves**

            """)
        with right_column_CMB_4:
            st.markdown("""
            ### Why B-modes are so important

            👉 Primordial B-modes would be a signal from the **very early Universe**, possibly from a period called **inflation**.

            Inflation is a theory that says:

            - The Universe expanded extremely rapidly just after the Big Bang  
            - Tiny quantum fluctuations were stretched to cosmic scales  

            👉 This process would generate **gravitational waves**, which leave a signature in the CMB polarization.

            ### The key parameter: $r$

            - $r$ measures the strength of these gravitational waves  
            - Larger $r$ → stronger primordial B-mode signal  

            👉 Try changing $r$ and see how the B-mode signal changes.

            """)
        st.markdown("""
        ### Open question in cosmology

        👉 Detecting primordial B-modes would be **direct evidence for inflation**.

        - So far, we have **not confirmed this signal yet**  
        - Experiments are ongoing (e.g. [Simons Observatory](https://en.wikipedia.org/wiki/Simons_Observatory), [CMB-S4](https://cmb-s4.org/), [LiteBIRD](https://en.wikipedia.org/wiki/LiteBIRD))

        👉 This is one of the biggest open questions in modern cosmology.


        👉 *Big picture:*  
        This plot connects the CMB to the **earliest moments of the Universe — far beyond what we can see directly.**
        """)
        st.markdown("""---""")
        st.header(r"Simulated CMB Polarization Maps")
        st.markdown(r"""
        This section visualizes the simulated CMB polarization maps (E and B modes) using the lensed power spectra $ C_{\ell}^{EE} $ and \( C_{\ell}^{BB} \) computed by CLASS. Generating these maps can be slow; enable the option below to compute them on demand.

        - **E-mode Polarization**: Generated by scalar perturbations, primarily from density fluctuations.
        - **B-mode Polarization**: Generated by tensor perturbations (primordial gravitational waves) and lensing effects.
        - **Polarization Vectors**: Represent the direction and amplitude of polarization. You can toggle the display of arrows for clarity.
        """)

        compute_pol_maps = st.checkbox(
            "Generate polarization maps (heavy) — compute on demand", value=False)
        if compute_pol_maps:
            st.info("Computing polarization maps — this may take some time.")
            st.header("CMB Polarization Maps")
            st.markdown("""
            ### What does polarization look like on the sky?

            So far, we looked at polarization statistically (power spectra).  
            Now we visualize it directly as maps.

            👉 *This lets you see the patterns of polarization across the sky.*

            ---
            """)
            # Prepare the input power spectra for synfast: [TT, EE, BB, TE]
            cl_synfast = [cl_lensed['tt'], cl_lensed['ee'],
                          cl_lensed['bb'], cl_lensed['te']]

            # Generate Q and U maps (Stokes parameters) using healpy.synfast (pol=True)
            cmb_maps = hp.synfast(cl_synfast, nside=nside,
                                  lmax=lmax, new=True, pol=True, verbose=False)
            cmb_T, cmb_Q, cmb_U = cmb_maps

            # Optionally, decompose Q/U into E/B maps using healpy
            alm_EB = hp.map2alm([cmb_T, cmb_Q, cmb_U], pol=True, lmax=lmax)
            cmb_E = hp.alm2map(alm_EB[1], nside=nside, lmax=lmax)
            cmb_B = hp.alm2map(alm_EB[2], nside=nside, lmax=lmax)

            nside_plot = 32  # lower resolution for vector field visualization
            theta, phi = hp.pix2ang(
                nside_plot, np.arange(hp.nside2npix(nside_plot)))

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
            arrow_scale = 0.04 * pol_amp / \
                (pol_amp.max() if pol_amp.max() != 0 else 1.0)

            # Toggle for displaying arrows
            show_arrows = st.checkbox("Show Polarization Arrows", value=True)

            fig, axs = plt.subplots(1, 2, figsize=(
                16, 7), sharex=True, sharey=True)

            # Plot settings
            arrow_skip = 20  # plot every Nth arrow for clarity
            x = lon[::arrow_skip]
            y = lat[::arrow_skip]
            u = arrow_scale[::arrow_skip] * np.cos(pol_angle[::arrow_skip])
            v = arrow_scale[::arrow_skip] * np.sin(pol_angle[::arrow_skip])

            # E-mode
            im0 = axs[0].scatter(lon, lat, c=E_plot,
                                 cmap='RdBu_r', s=10, lw=0, alpha=0.85)
            if show_arrows:
                axs[0].quiver(x, y, u, v, color='k', alpha=0.7,
                              width=0.003, scale=0.4)
            axs[0].set_title('E-mode', fontsize=16)
            axs[0].set_xlabel('RA [deg]')
            axs[0].set_ylabel('Dec [deg]')
            axs[0].set_xlim([-180, 180])
            axs[0].set_ylim([-90, 90])
            fig.colorbar(im0, ax=axs[0],
                         orientation='horizontal', pad=0.1, label='μK')

            # B-mode
            im1 = axs[1].scatter(lon, lat, c=B_plot,
                                 cmap='RdBu_r', s=10, lw=0, alpha=0.85)
            if show_arrows:
                axs[1].quiver(x, y, u, v, color='k', alpha=0.7,
                              width=0.003, scale=0.4)
            axs[1].set_title('B-mode', fontsize=16)
            axs[1].set_xlabel('RA [deg]')
            axs[1].set_xlim([-180, 180])
            axs[1].set_ylim([-90, 90])
            fig.colorbar(im1, ax=axs[1],
                         orientation='horizontal', pad=0.1, label='μK')

            plt.tight_layout()
            st.pyplot(fig)
            left_column_CMB_5, right_column_CMB_5 = st.columns(2)
            with left_column_CMB_5:
                st.markdown("""
                ### What you will learn from these maps:

                - How polarization varies across the sky  
                - The difference between **E-mode** and **B-mode** patterns  
                - How tiny signals encode information about the early Universe  

                ### What you see in the maps:

                - Colors show the **strength of the signal**  
                - Arrows show the **direction of polarization**  

                👉 Each arrow represents how light is oriented at that point in the sky
                """)
            with right_column_CMB_5:
                st.markdown("""

                ### E-mode vs B-mode (visual intuition)

                - **E-mode (left)**  
                → Smooth, radial or circular patterns  
                👉 *Like ripples on water*  

                - **B-mode (right)**  
                → Twisting, swirling patterns  
                👉 *Like a vortex or whirlpool*  

                ### Simulation note:

                - These maps are **simulated** using the power spectra you selected  
                - Real observations are much noisier and harder to measure  

                """)
        st.markdown("---")
        st.markdown("""
        ### 📚 Further Reading & References

        The calculations in this section are based on the **ΛCDM model**, the current standard model of cosmology.  
        We compute theoretical predictions using the [CLASS Boltzmann solver](http://class-code.net/), a widely used tool in modern cosmology.

        ---

        ### 📖 Learn more

        - *[Modern Cosmology by Scott Dodelson & Fabian Schmidt](https://www.amazon.co.uk/Modern-Cosmology-Scott-Dodelson/dp/0128159480)* — Scott Dodelson & Fabian Schmidt  

            👉 A comprehensive introduction to the theory behind these plots  

        - *[Physical Foundations of Cosmology](https://www.amazon.co.uk/Physical-Foundations-Cosmology-Viatcheslav-Mukhanov/dp/0521563984)* — Viatcheslav Mukhanov 

            👉 Inflation background.

        - *[CMB-slow, or How to Estimate Cosmological Parameters by Hand](https://arxiv.org/abs/astro-ph/0303072)* — Viatcheslav Mukhanov 

            👉 If you want to do everything by hand, but you need quite strong approximations.
        
        - *[waynehu's site](https://background.uchicago.edu/)* - Weyne Hu

            👉 Great Introduction about CMB 

        - *[宇宙マイクロ波背景放射 新天文学ライブラリー](https://www.amazon.co.jp/-/en/%E5%B0%8F%E6%9D%BE-%E8%8B%B1%E4%B8%80%E9%83%8E-ebook/dp/B07XP454BV/ref=sr_1_2?dib=eyJ2IjoiMSJ9.VQAl0XmSt32Ap6l-GsL7DZhg3YNDYdxOYK0uNV0UZ5wBLIxJnAxYy0iVsVMX0GdP.ns9TAcT8Ns1OPQsQp-Aqrpku-ne71CD4vbLGGq36eqk&dib_tag=se&qid=1776732668&s=books&sr=1-2)* - Eichiro Komatsu

            👉 This book covers almost everything regarding CMB(primordial gravitational wave or detailed statistical procedure of analysis of CMB), however so far no english translation.

        - *[Cosmology intertwined: A review of the particle physics, astrophysics, and cosmology associated with the cosmological tensions and anomalies](https://ui.adsabs.harvard.edu/abs/2022JHEAp..34...49A/abstract)
            
            👉 Review including discussion about Hubble tension

        ### 🌐 Data & Observations

        - [Planck 2018](https://pla.esac.esa.int/pla/#home) — Data about Planck
        """)


# galaxy clustering
with tab3:
    st.header("Galaxy Clustering")
    st.markdown(r"""
    ### Matter Power Spectrum
    This plot corresponds to Figure 8.14 in [Modern Cosmology](https://www.amazon.co.uk/Modern-Cosmology-Scott-Dodelson/dp/0128159480).
    It shows the matter power spectrum $ P(k) $ as a function of wavenumber $k$, with options to include:
    - **Halofit**: A semi-analytical model for the non-linear evolution of the matter power spectrum.

    #### Halofit:
    - **[Halofit](https://arxiv.org/abs/1208.2701)**: A model for the non-linear matter power spectrum that incorporates the effects of halo formation and evolution.
    - It is based on the idea that the matter power spectrum can be approximated by a sum of contributions from different halo masses.
    - It provides a more accurate prediction of $ P(k) $ at small scales compared to linear theory.
    """)
    use_gal = st.checkbox("Calculate Matter Power Spectrum", value=False)
    if use_gal:
        # Slider for redshift
        redshift = st.slider("Redshift (z)", 0.0, 5.0, 0.0, step=0.1)

        # Toggle for Halofit
        use_halofit = st.checkbox("Enable Halofit", value=True)

        # Define k range and compute linear matter power spectrum
        k = np.logspace(-4, 1, 1000)

        def get_pk_array(k_array, z, engine=cosmo):
            out = np.empty(len(k_array), dtype=float)
            for i, kv in enumerate(k_array):
                try:
                    out[i] = engine.pk(float(kv), float(z))
                except Exception:
                    try:
                        out[i] = engine.pk(float(kv), 0.0)
                    except Exception:
                        out[i] = np.nan
            return out

        pk_linear = get_pk_array(k, redshift, engine=cosmo)

        # Compute non-linear (Halofit) matter power spectrum if enabled
        if use_halofit:
            halofit = Class()
            halofit.set(common_settings)
            halofit.set({'non linear': 'halofit'})
            halofit.compute()
            pk_halofit = get_pk_array(k, redshift, engine=halofit)

        # Plot the matter power spectrum
        fig, ax = plt.subplots()
        ax.loglog(k, pk_linear, label="Linear (z={:.1f})".format(
            redshift), color='blue')
        if use_halofit:
            ax.loglog(k, pk_halofit, label="Halofit (z={:.1f})".format(
                redshift), linestyle='--', color='red')
        ax.set_xlabel(r"$k \, [h/\mathrm{Mpc}]$")
        ax.set_ylabel(r"$P(k) \, [\mathrm{Mpc}^3/h^3]$")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend()
        st.pyplot(fig)
        st.header("Halo Mass Function")
        st.markdown("""
        Compute the halo mass function using the Sheth–Tormen fit and the linear power spectrum from CLASS.
        The redshift is controlled by the slider above.
        """)
        with st.spinner("Computing halo mass function..."):
            # prepare k and linear power from CLASS at selected redshift
            k_hmf = np.logspace(-4, 1, 800)
            pk_lin_hmf = get_pk_array(k_hmf, redshift, engine=cosmo)
            pk_interp = interpolate.interp1d(np.log(k_hmf), np.log(
                pk_lin_hmf), kind='cubic', fill_value='extrapolate')

            def P_of_k(kv):
                return np.exp(pk_interp(np.log(kv)))

            # background densities
            rho_crit = 2.775e11 * h**2  # Msun / Mpc^3
            rho_crit_h = rho_crit / h**3
            rho_m = Omega_m * rho_crit_h  # Msun / (Mpc/h)^3

            def W_tophat(x):
                x_arr = np.asarray(x)
                with np.errstate(divide='ignore', invalid='ignore'):
                    w = 3.0 * (np.sin(x_arr) - x_arr *
                               np.cos(x_arr)) / x_arr**3
                # Safely handle x->0 for both scalars and arrays
                return np.where(np.isclose(x_arr, 0.0), 1.0, w)

            # mass grid
            M_vals = np.logspace(13, 16, 60)  # Msun/h
            R_vals = (3.0 * M_vals / (4.0 * np.pi * rho_m))**(1.0/3.0)
            # Vectorized sigma^2 computation: integrate over k for each R
            kk = k_hmf
            Pk_vals = P_of_k(kk)
            KR = np.outer(kk, R_vals)  # shape (nk, nR)
            W = W_tophat(KR)
            integrand = (kk[:, None]**2) * Pk_vals[:, None] * W**2
            sigma2 = np.trapz(integrand, kk, axis=0) / (2.0 * np.pi**2)
            sigma0 = np.sqrt(sigma2)
            lns = np.log(sigma0)
            lnM = np.log(M_vals)
            dlns_dlnM = np.gradient(lns, lnM)

            # Sheth-Tormen
            A = 0.3222
            a_st = 0.707
            p = 0.3
            delta_c = 1.686

            def sheth_tormen_dn_dM(M, sigma, dlns_dlnM_local):
                nu = delta_c / sigma
                f_nu = A * np.sqrt(2.0 * a_st / np.pi) * nu * np.exp(-0.5 *
                                                                     a_st * nu**2) * (1.0 + (1.0 / (a_st * nu**2))**p)
                dlns_dM = dlns_dlnM_local / M
                dn_dM = (rho_m / M) * f_nu * np.abs(-dlns_dM)
                return dn_dM

            # approximate linear growth (use CLASS growth if available)
            try:
                D_z = cosmo.scale_independent_growth_factor_f(redshift)
            except Exception:
                # fallback to simple fitting formula
                def growth_factor(z):
                    Om0 = Omega_m
                    Ol0 = 1.0 - Om0
                    Ez2 = Om0 * (1 + z)**3 + Ol0
                    Omz = Om0 * (1 + z)**3 / Ez2
                    Olz = Ol0 / Ez2
                    g = 2.5 * Omz / (Omz**(4.0/7.0) - Olz +
                                     (1.0 + Omz/2.0) * (1.0 + Olz/70.0))
                    g0 = 2.5 * Om0 / (Om0**(4.0/7.0) - Ol0 +
                                      (1.0 + Om0/2.0) * (1.0 + Ol0/70.0))
                    return (g / (1 + z)) / g0
                D_z = growth_factor(redshift)

            sigma_z = sigma0 * D_z
            dn_dM = sheth_tormen_dn_dM(M_vals, sigma_z, dlns_dlnM)
            dn_dlog10M = np.log(10) * M_vals * dn_dM

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.loglog(M_vals, dn_dlog10M, color='C2', lw=2)
        ax.set_xlabel('Mass [Msun/h]')
        ax.set_ylabel('dn/dlog10(M) [ (h^3 / Mpc^3) ]')
        ax.grid(which='both', ls='--', alpha=0.4)
        st.pyplot(fig)

        st.header("Cluster Counts in Mass-Redshift Bins")
        st.markdown(
            "Compute expected cluster counts in mass-redshift bins for the full sky.")
        with st.spinner("Computing cluster counts..."):
            z_edges = np.linspace(0.0, 1.0, 11)
            z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])
            M_edges = np.logspace(13, 16, 30)
            M_mid = 0.5 * (M_edges[:-1] + M_edges[1:])
            dM = np.diff(M_edges)

            c_kms = 299792.458

            def comoving_distance(zv):
                return (1.0 + zv) * cosmo.angular_distance(zv)

            def H_of_z(zv):
                return cosmo.Hubble(zv)

            counts = np.zeros((len(z_mid), len(M_mid)))
            for iz in range(len(z_mid)):
                z1, z2 = z_edges[iz], z_edges[iz+1]
                z_samples = np.linspace(z1, z2, 6)
                dz = z_samples[1] - z_samples[0]
                weight_z = np.ones_like(z_samples)
                weight_z[0] = 0.5
                weight_z[-1] = 0.5
                for jz, zz in enumerate(z_samples):
                    # growth
                    try:
                        D_zz = cosmo.scale_independent_growth_factor_f(zz)
                    except Exception:
                        D_zz = D_z * (1.0 / (1.0 + zz))  # rough fallback
                    sigma_z_samples = sigma0 * D_zz
                    sig_interp = interpolate.interp1d(
                        np.log(M_vals), sigma_z_samples, kind='cubic', fill_value='extrapolate')
                    sigma_mid = sig_interp(np.log(M_mid))
                    slope_interp = interpolate.interp1d(
                        np.log(M_vals), dlns_dlnM, kind='cubic', fill_value='extrapolate')
                    slope_mid = slope_interp(np.log(M_mid))
                    dn_dM_mid = sheth_tormen_dn_dM(M_mid, sigma_mid, slope_mid)
                    Dc = comoving_distance(zz)
                    Hz = H_of_z(zz)
                    dV_dz = 4.0 * np.pi * Dc**2 * (c_kms / Hz)
                    counts[iz] += dn_dM_mid * dM * dV_dz * weight_z[jz] * dz

        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.pcolormesh(np.log10(M_edges), z_edges, counts,
                           shading='auto', cmap='viridis')
        ax.set_xlabel('log10(M [Msun/h])')
        ax.set_ylabel('Redshift')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Counts per bin (full sky)')
        st.pyplot(fig)


# cosmic shear
with tab4:
    st.header("Weak Lensing")
    st.markdown("""
    This panel computes a log-normal weak-lensing convergence map using GLASS+CAMB.
    The computation is heavy — enable and click **Run GLASS simulation** to execute.
    """)

    use_glass = st.checkbox(
        "Enable GLASS weak-lensing simulation (heavy)", value=False)
    if use_glass:
        st.markdown("**Simulation settings**")
        col1, col2, col3 = st.columns(3)
        with col1:
            nside = st.selectbox("HEALPix nside", options=[
                                 64, 128, 256], index=0)
            lmax = st.number_input("lmax", value=3 * nside, step=1)
        with col2:
            zmin = st.number_input("zmin", value=0.0, format="%.2f")
            zmax = st.number_input("zmax", value=3.0, format="%.2f")
            dx = st.number_input("dx [Mpc] (shell spacing)", value=200.0)
        with col3:
            ncorr = st.number_input("ncorr", value=3, step=1)
            n_arcmin2 = st.number_input(
                "n_gal / arcmin^2", value=0.03, format="%.3f")
            nbins = st.number_input("tomographic nbins", value=10, step=1)

        run_button = st.button("Run GLASS simulation")
        if run_button:
            with st.spinner("Running GLASS pipeline (this may take a while)..."):
                try:
                    # build CAMB params from sidebar cosmology
                    pars = camb.set_params(
                        H0=100.0 * h,
                        omch2=(Omega_m - Omega_b) * h**2,
                        ombh2=Omega_b * h**2,
                        NonLinear=camb.model.NonLinear_both,
                    )

                    # cosmology helper used by glass
                    cosmo_glass = Cosmology.from_camb(pars)

                    # distance grid and windows
                    zb = glass.distance_grid(cosmo_glass, zmin, zmax, dx=dx)
                    shells = glass.linear_windows(zb)

                    # angular power spectrum from CAMB
                    cls = glass.ext.camb.matter_cls(pars, lmax, shells)
                    cls = glass.discretized_cls(
                        cls, nside=nside, lmax=lmax, ncorr=ncorr)

                    # generate log-normal fields and matter
                    fields = glass.lognormal_fields(shells)
                    gls = glass.solve_gaussian_spectra(fields, cls)
                    matter = glass.generate(fields, gls, nside, ncorr=ncorr)

                    # build convergence
                    convergence = glass.MultiPlaneConvergence(cosmo_glass)
                    for i, delta_i in enumerate(matter):
                        convergence.add_window(delta_i, shells[i])

                    kappa_map = convergence.kappa
                    gamm1_map, gamm2_map = glass.shear_from_convergence(
                        kappa_map)

                    # display maps
                    fig = plt.figure(figsize=(10, 6))
                    hp.mollview(kappa_map,
                                title="Convergence (kappa)",
                                fig=fig,
                                sub=(1, 3, 1))

                    hp.mollview(gamm1_map,
                                title=r"Shear $\gamma_1$",
                                fig=fig,
                                sub=(1, 3, 2))

                    hp.mollview(gamm2_map,
                                title=r"Shear $\gamma_2$",
                                fig=fig,
                                sub=(1, 3, 3))
                    st.pyplot(fig)

                    # small power spectrum check
                    with st.expander("Show convergence power spectrum"):
                        sim_cls = hp.anafast(kappa_map, lmax=lmax)
                        ell = np.arange(len(sim_cls))

                        # try to compute expected kappa C_l from CAMB using a Smail dN/dz
                        theory_kappa = None
                        try:
                            z_theory = np.arange(zmin, zmax, 0.01)
                            dndz_theory = glass.smail_nz(z_theory, z_mode=0.9,
                                                         alpha=2.0, beta=1.5)
                            dndz_theory *= n_arcmin2

                            pars.min_l = 1
                            pars.set_for_lmax(lmax)
                            pars.SourceWindows = [
                                camb.sources.SplinedSourceWindow(
                                    z=z_theory, W=dndz_theory, source_type='lensing')
                            ]
                            theory_cls = camb.get_results(
                                pars).get_source_cls_dict(lmax=lmax, raw_cl=True)
                            theory_kappa = theory_cls.get('W1xW1', None)
                        except Exception:
                            theory_kappa = None

                        fig2, ax = plt.subplots(figsize=(6, 4))
                        ax.plot(ell, sim_cls, '-k',
                                label='Simulated kappa C_l')
                        if theory_kappa is not None:
                            # include HEALPix pixel window
                            pw = hp.pixwin(nside, lmax=lmax)
                            ax.plot(ell, theory_kappa * pw**2, '-r',
                                    lw=1.5, label='Expected kappa C_l (CAMB)')

                        ax.set_yscale('log')
                        ax.set_xscale('log')
                        ax.set_xlabel(r'$\ell$')
                        ax.set_ylabel(r'$C_{\ell}$')
                        ax.legend()
                        st.pyplot(fig2)

                    # Tomographic bins and redshift distributions
                    with st.expander("Show tomographic bins and redshift distributions"):
                        try:
                            z = np.arange(zmin, zmax, 0.01)
                            # simple Smail et al. model parameters (exposed minimally)
                            z_mode = 0.9
                            alpha = 2.0
                            beta = 1.5
                            sigma_z0 = 0.03

                            dndz = glass.smail_nz(z, z_mode=z_mode,
                                                  alpha=alpha, beta=beta)
                            dndz = dndz * n_arcmin2

                            nbins_int = int(nbins)
                            zbins = glass.equal_dens_zbins(
                                z, dndz, nbins=nbins_int)
                            tomo_nz = glass.tomo_nz_gausserr(
                                z, dndz, sigma_z0, zbins)

                            fig3, ax3 = plt.subplots(figsize=(8, 4))
                            for i in range(nbins_int):
                                ax3.plot(z, (tomo_nz[i] / n_arcmin2) * nbins_int,
                                         alpha=0.6, label=f"Input Bin {i}")
                            ax3.plot(z, dndz / n_arcmin2 * nbins_int,
                                     ls='--', c='k', label='Input Total Distribution')
                            ax3.set_xlabel('Redshift z')
                            ax3.set_ylabel('Normalized dN/dz')
                            ax3.legend(ncol=2, fontsize=9)
                            ax3.grid(True, ls=':')
                            st.pyplot(fig3)
                        except Exception as e:
                            st.error(
                                f"Failed to compute tomographic bins: {e}")

                    # Shear correlation function (xi+)
                    with st.expander("Show shear correlation function (xi+)"):
                        try:
                            sim_cls = hp.anafast(kappa_map, lmax=lmax)
                            ell = np.arange(len(sim_cls))
                            # theta grid in degrees
                            theta_deg = np.logspace(-2, 2, 50)
                            theta_rad = np.deg2rad(theta_deg)

                            from scipy.special import eval_legendre

                            xi_plus = np.zeros_like(theta_rad)
                            # sum over multipoles: xi+(theta)=sum (2l+1)/(4pi) C_l P_l(cos theta)
                            for l in range(len(ell)):
                                Cl = sim_cls[l]
                                Pl = eval_legendre(l, np.cos(theta_rad))
                                xi_plus += (2 * l + 1) / \
                                    (4.0 * np.pi) * Cl * Pl

                            fig4, ax4 = plt.subplots(figsize=(7, 4))
                            ax4.semilogx(theta_deg, xi_plus, '-k')
                            ax4.set_xlabel('Angular separation [deg]')
                            ax4.set_ylabel(r'$\xi_{+}(\theta)$')
                            ax4.grid(True, ls=':')
                            st.pyplot(fig4)
                        except Exception as e:
                            st.error(
                                f"Failed to compute correlation function: {e}")

                except Exception as e:
                    st.error(f"GLASS simulation failed: {e}")
st.markdown("---")
st.markdown("Developed for lunch seminar by Rintaro Kanaki. © 2025")
