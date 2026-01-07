# ===============================================================
# 🌱 Streamlit GUI: Soil Water Retention Curve Fitting with Progress Bar
# Author: Shahab A. Shojaeezadeh
# Soil Science Section, University of Kassel
# https://www.uni-kassel.de/fb11agrar/en/fachgebiete-einrichtungen/bodenkunde/home.html
# GitHub: https://github.com/Bluerrror
# ===============================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad

# ===============================================================
# Metrics
# ===============================================================
def RMSE(obs, sim): 
    return np.sqrt(np.mean((np.array(sim) - np.array(obs)) ** 2))

def NSE(obs, sim): 
    return 1 - np.sum((obs - np.array(sim))**2) / np.sum((obs - np.mean(obs))**2)

def KGE(obs, sim):
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

def compute_aic(n, rss, k):
    return n * np.log(rss / n) + 2 * k

def compute_bic(n, rss, k):
    return n * np.log(rss / n) + k * np.log(n)

# ===============================================================
# Progress wrapper with Streamlit
# ===============================================================
# ===============================================================
# Persistent Progress Wrapper (does not vanish)
# ===============================================================
class FuncWithStreamlitProgress:
    def __init__(self, func, h_vals, theta_obs, maxfev=5000, desc="curve_fit"):
        self.func = func
        self.h_vals = h_vals
        self.theta_obs = theta_obs
        self.ncalls = 0
        self.maxfev = maxfev
        self.desc = desc

        # Persistent elements (don't vanish)
        if "progress_bar" not in st.session_state:
            st.session_state["progress_bar"] = st.progress(0)
        if "status_text" not in st.session_state:
            st.session_state["status_text"] = st.empty()

        # Reset progress for each new fitting
        st.session_state["progress_bar"].progress(0)
        st.session_state["status_text"].text(f"{self.desc} | Starting fitting...")

    def __call__(self, h, *params):
        self.ncalls += 1
        pred = self.func(h, *params)
        rmse_val = RMSE(self.theta_obs, pred)
        frac = min(self.ncalls / self.maxfev, 1.0)
        st.session_state["progress_bar"].progress(frac)
        st.session_state["status_text"].text(
            f"{self.desc} | Eval {self.ncalls}/{self.maxfev} | RMSE={rmse_val:.6f}"
        )
        return pred

    def close(self):
        # Keep progress bar visible after completion
        st.session_state["progress_bar"].progress(1.0)
        st.session_state["status_text"].text(f"{self.desc} | Completed ✅")

# ===============================================================
# Define Models
# ===============================================================
def vG_curve(h, alpha, theta_r, n, theta_s):
    m = 1 - 1.0 / n
    Se = (1.0 / (1.0 + (alpha * np.abs(h)) ** n)) ** m
    return Se * (theta_s - theta_r) + theta_r

def Sc_vG(h, alpha, n):
    m = 1 - 1.0 / n
    return (1.0 / (1.0 + (alpha * np.abs(h)) ** n)) ** m

def Sstar_nC_vec(h_vals, alpha, n, h_eps=1e-6):
    factor = 1.0 / np.log(10.0)
    result = np.zeros_like(h_vals)
    for i, hi in enumerate(h_vals):
        integrand = lambda hp: (Sc_vG(hp, alpha, n) - 1.0) / hp
        val, _ = quad(integrand, h_eps, max(hi, h_eps), epsabs=1e-8, epsrel=1e-6)
        result[i] = factor * val
    return result

def Snc_vec(h_vals, alpha, n, logh0):
    h0 = 10 ** logh0
    s_star = Sstar_nC_vec(h_vals, alpha, n)
    s_star0 = Sstar_nC_vec(np.array([h0]), alpha, n)[0]
    return 1.0 - s_star / s_star0

def Brunswick_curve(h_vals, alpha, theta_r, n, theta_s, f_nc, logh0):
    theta_cs = (1 - f_nc) * theta_s
    theta_ncs = f_nc * theta_s
    Sc_val = Sc_vG(h_vals, alpha, n)
    Snc_val = Snc_vec(h_vals, alpha, n, logh0)
    return np.clip(theta_cs * Sc_val + theta_ncs * Snc_val, 0, 1)

def Sc_PDI_exact(h_vals, alpha, n, h0):
    m = 1 - 1.0 / n
    h_abs = np.abs(h_vals)
    Gamma_h = (1.0 / (1.0 + (alpha * h_abs) ** n)) ** m
    Gamma_h0 = (1.0 / (1.0 + (alpha * h0) ** n)) ** m
    return (Gamma_h - Gamma_h0) / (1.0 - Gamma_h0)

def compute_ha(alpha, n, h0, zeta=0.75):
    m = 1 - 1.0 / n
    Gamma0 = (1.0 / (1.0 + (alpha * h0) ** n)) ** m
    gamma = zeta * (1 - Gamma0) + Gamma0
    inner = max(gamma ** (-1.0 / m) - 1.0, 0.0)
    return (inner ** (1.0 / n)) / alpha if inner > 0 else 1e-12

def Snc_PDI_exact(h_vals, alpha, n, b, logh0):
    h0 = 10 ** logh0
    h_abs = np.maximum(np.abs(h_vals), 1e-20)
    ha = compute_ha(alpha, n, h0)
    denom = np.log(h0 / ha)
    denom = denom if np.abs(denom) > 1e-20 else 1e-20
    exponent = 1.0 / (np.log(10.0) * b)
    power_term = (ha / h_abs) ** exponent
    term_inside_log = 1.0 + power_term
    numer = np.log(h0 / h_abs) - (np.log(10.0) * b) * np.log(term_inside_log)
    return numer / denom

def PDI_curve(h_vals, alpha, theta_r, n, theta_s, logh0):
    h0 = 10 ** logh0
    Sc_val = Sc_PDI_exact(h_vals, alpha, n, h0)
    Snc_val = Snc_PDI_exact(h_vals, alpha, n, b=0.2, logh0=logh0)
    theta = (theta_s - theta_r) * Sc_val + theta_r * Snc_val
    return np.clip(theta, 0, 1)

# ===============================================================
# Streamlit App
# ===============================================================
st.set_page_config(layout="wide", page_title="Soil Water Retention Curve Fitting")
st.title("🌱 Soil Water Retention Curve Fitting GUI")

st.markdown("""
**Author:** Shahab A. Shojaeezadeh  
**Soil Science Section, University of Kassel**  
[Website](https://www.uni-kassel.de/fb11agrar/en/fachgebiete-einrichtungen/bodenkunde/home.html) | [GitHub](https://github.com/Bluerrror)
""")

# -------------------------
# Sidebar Controls
# -------------------------
st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='utf-8', encoding_errors='ignore')
    st.write("Data preview:", df.head())

    # User selects columns
    sample_col = st.sidebar.selectbox("Select Sample ID column", df.columns)
    pF_col = st.sidebar.selectbox("Select pF column", df.columns)
    theta_col = st.sidebar.selectbox("Select theta column", df.columns)
    is_log = st.sidebar.checkbox("Is pF logarithmic?", value=True)

    # Select sample
    sample_ids = df[sample_col].unique()
    selected_sample = st.sidebar.selectbox("Select Sample ID", sample_ids)
    dc = df[df[sample_col] == selected_sample]
    
    theta_v = dc[theta_col].values
    h = dc[pF_col].values
    if is_log:
        h = 10 ** h
    h_plot = np.logspace(-2, 7, 200)

    # Model selection
    model_options = st.sidebar.multiselect("Select models to fit", ["van Genuchten", "Brunswick", "PDI"], default=["van Genuchten"])
    results = {}

    # -------------------------
    # Fit Models
    # -------------------------
    if "van Genuchten" in model_options:
        st.subheader("van Genuchten (vG)")
        vG_progress = FuncWithStreamlitProgress(vG_curve, h, theta_v, maxfev=5000, desc="vG fitting")
        p0_vG = [0.01, 0.05, 1.5, 0.4]
        bounds_vG = ([1e-5, 0.0, 1.01, 0.0], [1.0, 0.4, 15.0, 1.0])
        popt_vG, _ = curve_fit(vG_progress, h, theta_v, p0=p0_vG, bounds=bounds_vG, maxfev=5000)
        vG_progress.close()
        theta_vG_pred = vG_curve(h, *popt_vG)
        results["van Genuchten"] = (popt_vG, theta_vG_pred, ['alpha','theta_r','n','theta_s'])

    if "Brunswick" in model_options:
        st.subheader("Brunswick")
        Brunswick_progress = FuncWithStreamlitProgress(Brunswick_curve, h, theta_v, maxfev=10000, desc="Brunswick fitting")
        p0_Brunswick = [0.01, 0.05, 1.5, 0.4, 0.1, 6.79]
        bounds_Brunswick = ([1e-5,0.0,1.01,0.1,0.0,4.0],[1.0,0.4,15.0,1.0,0.5,6.81])
        popt_Brunswick, _ = curve_fit(Brunswick_progress, h, theta_v, p0=p0_Brunswick, bounds=bounds_Brunswick, maxfev=10000)
        Brunswick_progress.close()
        theta_Brunswick_pred = Brunswick_curve(h, *popt_Brunswick)
        results["Brunswick"] = (popt_Brunswick, theta_Brunswick_pred, ['alpha','theta_r','n','theta_s','f_nc','logh0'])

    if "PDI" in model_options:
        st.subheader("PDI")
        PDI_progress = FuncWithStreamlitProgress(PDI_curve, h, theta_v, maxfev=10000, desc="PDI fitting")
        p0_PDI = [0.01, 0.05, 1.5, 0.4, 6.79]
        bounds_PDI = ([1e-5,0.0,1.01,0,4.0],[1.0,0.4,15.0,0.8,6.81])
        popt_PDI, _ = curve_fit(PDI_progress, h, theta_v, p0=p0_PDI, bounds=bounds_PDI, maxfev=10000)
        PDI_progress.close()
        theta_PDI_pred = PDI_curve(h, *popt_PDI)
        results["PDI"] = (popt_PDI, theta_PDI_pred, ['alpha','theta_r','n','theta_s','logh0'])

    # -------------------------
    # Display Results Table (better visual)
    # -------------------------
    st.subheader("Fitted Parameters and Metrics")

    # Collect all parameter names
    all_param_names = set()
    for _, (_, _, param_names) in results.items():
        all_param_names.update(param_names)
    all_param_names = sorted(list(all_param_names))

    # Prepare table rows
    table_rows = []
    for model_name, (params, theta_pred, param_names) in results.items():
        row = {"Model": model_name}
        for pname in all_param_names:
            if pname in param_names:
                idx = param_names.index(pname)
                row[pname] = round(params[idx], 6)
            else:
                row[pname] = "-"
        # Add metrics
        n_obs = len(theta_v)
        k_params = len(params)
        rss = np.sum((theta_v - theta_pred)**2)
        row["RMSE"] = round(RMSE(theta_v, theta_pred), 6)
        row["NSE"] = round(NSE(theta_v, theta_pred), 4)
        row["KGE"] = round(KGE(theta_v, theta_pred), 4)
        row["AIC"] = round(compute_aic(n_obs, rss, k_params), 2)
        row["BIC"] = round(compute_bic(n_obs, rss, k_params), 2)
        table_rows.append(row)

    results_df = pd.DataFrame(table_rows)
    st.dataframe(results_df.style.format(precision=6))

    # -------------------------
    # Plot
    # -------------------------
    st.subheader("Fitted Curves")
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(np.log10(h), theta_v, c='k', s=70, label='Measured data')
    if "van Genuchten" in results:
        ax.plot(np.log10(h_plot), vG_curve(h_plot, *results["van Genuchten"][0]), 'r--', lw=2.5, label='van Genuchten')
    if "Brunswick" in results:
        ax.plot(np.log10(h_plot), Brunswick_curve(h_plot, *results["Brunswick"][0]), 'b-', lw=2.5, label='Brunswick')
    if "PDI" in results:
        ax.plot(np.log10(h_plot), PDI_curve(h_plot, *results["PDI"][0]), 'g-.', lw=2.5, label='PDI')
    ax.set_xlabel(r'$\log_{10}(h)$ [cm]', fontsize=14)
    ax.set_ylabel(r'$\theta~(cm^3/cm^3)$', fontsize=14)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend()
    st.pyplot(fig)

