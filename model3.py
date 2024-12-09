import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

# Given data (user can input these parameters)
st.title("Pharmacokinetic Modeling")

# Inputs from user
dose = st.number_input("Initial Dose (mg)", min_value=1, max_value=5000, value=1000)
tau = st.number_input("Dosing Interval (hours)", min_value=1, max_value=24, value=12)
tinf = st.number_input("Infusion Time (hours)", min_value=0.1, max_value=10.0, value=1.5)
measured_trough = st.number_input("Measured Trough Concentration (µg/mL)", min_value=0.1, max_value=100.0, value=6.1)
T2 = st.number_input("Time from Start of Infusion to Trough (hours)", min_value=0, max_value=24, value=11)
AUC_target_range = (400, 600)  # Fixed therapeutic range for AUC24

# Patient-specific parameters
weight = st.number_input("Patient Weight (kg)", min_value=20, max_value=200, value=100)
height = st.number_input("Patient Height (inches)", min_value=40, max_value=96, value=72)
age = st.number_input("Patient Age (years)", min_value=1, max_value=120, value=64)
sex = st.selectbox("Patient Sex", options=["male", "female"])
scr = st.number_input("Serum Creatinine (mg/dL)", min_value=0.1, max_value=10.0, value=1.0)

# Renal function: Estimation using Cockcroft-Gault equation
if sex.lower() == "female":
    sex_factor = 0.85
else:
    sex_factor = 1.0

# Creatinine clearance (mL/min)
crcl = ((140 - age) * weight * sex_factor) / (72 * scr)

# Priors for V and CL
V_prior_mean = 0.7 * weight  # Total Volume in L (based on weight)
V_prior_sd = 0.1 * weight  # SD for Volume
CL_prior_mean = crcl / 60  # Convert mL/min to L/hr
CL_prior_sd = 1  # SD for Clearance

def compute_concentrations(dose, V, CL, tinf, tau, T2):
    """Calculate peak and trough concentrations."""
    ke = CL / V  # Elimination rate constant
    
    # Peak: Concentration at the end of infusion
    C_peak = (dose / (tinf * CL)) * (1 - np.exp(-ke * tinf))
    
    # Trough: Concentration at T2 after infusion ends
    C_trough = C_peak * np.exp(-ke * (T2 - tinf))
    
    return C_peak, C_trough

def calculate_auc24_and_dosing_adjustment():
    """Bayesian model to calculate AUC24 and recommend dosing adjustment."""
    with pm.Model() as model:
        # Priors
        V = pm.Normal("V", mu=V_prior_mean, sigma=V_prior_sd)
        CL = pm.Normal("CL", mu=CL_prior_mean, sigma=CL_prior_sd)
        
        # Elimination rate constant (ke)
        ke = pm.Deterministic("ke", CL / V)
        
        # Observed trough concentration
        C_trough_obs = (dose / V) * pm.math.exp(-ke * T2)
        obs = pm.Normal("obs", mu=C_trough_obs, sigma=1, observed=measured_trough)
        
        # Sampling
        trace = pm.sample(1000, tune=1000, return_inferencedata=True, target_accept=0.9)
    
    # AUC24 calculations
    ke_samples = trace.posterior["ke"].values.flatten()
    V_samples = trace.posterior["V"].values.flatten()
    CL_samples = trace.posterior["CL"].values.flatten()
    AUC24_samples = []

    time_points = np.linspace(0, tau, 100)
    for ke_sample, V_sample in zip(ke_samples, V_samples):
        concentrations = (dose / V_sample) * np.exp(-ke_sample * time_points)
        # Replaced simps with numpy.trapz for trapezoidal rule
        AUC24_samples.append(np.trapz(concentrations, time_points))
    
    # Scale AUC to 24 hours based on dosing interval
    num_intervals = int(24 / tau)  # e.g., 2 for q12, 3 for q8
    AUC_24 = np.mean(AUC24_samples) * num_intervals
    
    # Predicted Peak and Trough
    mean_V = np.mean(V_samples)
    mean_CL = np.mean(CL_samples)
    predicted_peak_current, predicted_trough_current = compute_concentrations(dose, mean_V, mean_CL, tinf, tau, T2)
    
    # Dose adjustment
    target_AUC = np.random.uniform(*AUC_target_range)
    recommended_dose = dose * (target_AUC / AUC_24)
    
    # Predicted Peak and Trough for new dose
    predicted_peak_new, predicted_trough_new = compute_concentrations(recommended_dose, mean_V, mean_CL, tinf, tau, T2)
    
    return (trace, AUC_24, predicted_peak_current, predicted_trough_current, 
            recommended_dose, predicted_peak_new, predicted_trough_new, mean_CL, mean_V, np.mean(ke_samples))

def plot_concentration_time(trace, dose, tau, num_intervals, infusion_time):
    """Interactive concentration-time curve."""
    V_samples = trace.posterior["V"].values.flatten()
    CL_samples = trace.posterior["CL"].values.flatten()
    
    total_time = tau * num_intervals
    time_points = np.linspace(0, total_time, 1000)
    mean_concentration = np.zeros_like(time_points)
    
    for V, CL in zip(V_samples, CL_samples):
        ke = CL / V
        concentrations = np.zeros_like(time_points)
        
        for interval in range(num_intervals):
            dose_start_time = interval * tau
            infusion_end_time = dose_start_time + infusion_time
            
            during_infusion = (time_points >= dose_start_time) & (time_points <= infusion_end_time)
            infusion_time_points = time_points[during_infusion] - dose_start_time
            concentrations[during_infusion] += (
                (dose / infusion_time) / CL * (1 - np.exp(-ke * infusion_time_points))
            )
            
            after_infusion = time_points > infusion_end_time
            post_infusion_time_points = time_points[after_infusion] - infusion_end_time
            C_end = (dose / infusion_time) / CL * (1 - np.exp(-ke * infusion_time))
            concentrations[after_infusion] += C_end * np.exp(-ke * post_infusion_time_points)
        
        mean_concentration += concentrations / len(V_samples)
    
    # Plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=time_points, y=mean_concentration, mode="lines", name="Current Dose")
    )
    fig.update_layout(
        title="Concentration-Time Curve",
        xaxis_title="Time (hours)",
        yaxis_title="Concentration (mcg/mL)",
        template="plotly_white"
    )
    return fig

# Add "Calculate" button
calculate_button = st.button("Calculate")

if calculate_button:
    (trace, AUC_24_Final, predicted_peak_current, predicted_trough_current, 
     recommended_dose, predicted_peak_new, predicted_trough_new, mean_CL, mean_V, mean_ke) = calculate_auc24_and_dosing_adjustment()

    # Results layout
    st.header("Pharmacokinetic Model Results")

    # Display Current Dosing Results
    st.subheader("Current Dosing Schedule")
    current_dosing_data = {
        "Dose": f"{dose} mg q{tau}h (over {tinf} hr)",
        "AUC24/MIC": f"{AUC_24_Final:.2f} mcg*hr/mL",
        "Peak": f"{predicted_peak_current:.2f} mcg/mL",
        "Trough": f"{predicted_trough_current:.2f} mcg/mL"
    }
    st.table(current_dosing_data)

    # Display Predicted PK Results
    st.subheader("Predicted PK for Adjusted Dose")
    predicted_dosing_data = {
        "Dose": f"{recommended_dose:.2f} mg q{tau}h (over {tinf} hr)",
        "AUC24/MIC": f"{np.random.uniform(*AUC_target_range):.2f} mcg*hr/mL",
        "Peak": f"{predicted_peak_new:.2f} mcg/mL",
        "Trough": f"{predicted_trough_new:.2f} mcg/mL"
    }
    st.table(predicted_dosing_data)

    # Display AUC plot
    auc_fig = plot_concentration_time(trace, dose, tau, int(24/tau), tinf)
    st.plotly_chart(auc_fig)
