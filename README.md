# Lunch Seminar: Spatial Modeling of Photometric Redshifts

This repository contains Jupyter Notebooks and a Streamlit app for visualizing cosmological quantities and statistical methodologies.

## Contents
- Basic cosmological quantity computation, CMB, and matter power spectra using [CLASS](http://class-code.net/).
- HEALPix visualization of simulated CMB anisotropy maps using [Healpy](https://healpy.readthedocs.io/en/latest/index.html).
- The app is developed using [Streamlit](https://streamlit.io/).
- Plots for galaxy clustering and cosmic shear will be added soon... ❗️


## How to Run

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/lunch_seminar.git
    cd lunch_seminar
    ```

2. **Create and activate a conda environment**:
    ```bash
    conda create -n lunch_seminar_env python=3.9 -y
    conda activate lunch_seminar_env
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Install `classy`**:
    Install `classy`(Boltzmann Solver CLASS](http://class-code.net/)) separately to ensure all prerequisites are met:
    ```bash
    pip install classy
    ```

5. **Install CLASS**:
    Clone the CLASS repository and compile it:
    ```bash
    git clone https://github.com/lesgourg/class_public.git
    cd class_public
    make
    cd ..
    ```

7. **Run the Streamlit app**:
    ```bash
    cd cosmology-visualization-app
    streamlit run cosmolunch_appetizer_app.py
    ```

8. **Access the app**:
   Open your browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

### Running CLASS
To run CLASS with a specific configuration, use the following command:
```bash
./class_public/class explanatory.ini
```
Replace `explanatory.ini` with your desired configuration file.