import streamlit as st
from estimation import ABTEST 
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import norm
import streamlit as st
import plotly.graph_objects as go




#st.title('A/B Test Sample Size Calculator')
page = st.sidebar.selectbox('',['A/B Test Sample Size Calculator', 'Vizualization','Minimum Detectable Effect Calculator'])

# Inputs
if page == 'A/B Test Sample Size Calculator':

    baseline_conversion = st.slider('Baseline Conversion Rate (%)', 0.0, 100.0, 10.0) / 100
    minimum_effect = st.slider('Minimum Detectable Effect (%)', 0.1, 100.0, 20.0) / 100
    test_type = st.radio("Hypothesis", ('One-sided Test', 'Two-sided Test'))
    alpha = st.slider('Significance Level (α)', 0.01, 0.10, 0.05)
    beta = st.slider('Statistical Power (1 - β)', 0.65, 0.95, 0.80)
    daily_visitors = st.number_input('Daily Visitors', min_value=100, value=1000)
    #ab_split = st.number_input('Test vs. Control', 0.1, 1.0, 0.5, step=0.01)
    ab_split=1
    # Calculate the sample size per group

    calculator = ABTEST(2,minimum_effect,alpha,1 - beta,baseline_conversion)
    sample_size = calculator.get_sample_size (test_type)
    duration = calculator.calculate_duration(daily_visitors,test_type)
    #st.write(f"Required sample size per group: {sample_size}")
    #st.write(f"Duration in days (assuming equal traffic to both versions): {duration}")
    num_variants = st.number_input('Number of Variants', min_value=2, max_value=10, value=2, step=1)

    # Create an instance of the ABTEST CLASS
    #Number of variations is taken into account in the duration of the test calculation
    # Displaying results with progress bars

    

    st.metric(label="Sample Size per variation", value=f"{sample_size}")
    test_duration=np.ceil(num_variants*sample_size /daily_visitors)

    # Création du graphique de jauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = test_duration,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Test Duration in Days"},
        gauge = {
            'axis': {'range': [None, 30], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkslategray"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 10], 'color': 'lightgray'},
                {'range': [10, 20], 'color': 'gray'},
                {'range': [20, 30], 'color': 'darkgray'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': test_duration}
        }
    ))
    fig.update_layout(autosize=False, width=500, height=300)  # Adjusting size of the gauge

    st.plotly_chart(fig, use_container_width=True)
elif page=='Vizualization' :
    def interactive_plot(mu_MDE, sigma, alpha, power):
        x_min = -10
        x_max = max(-x_min, mu_MDE + 5 * sigma)
        x = np.linspace(x_min, x_max, 1000)
        mu_H0 = 0  # Mean of the null hypothesis distribution
        
        # Create distributions
        H0_distribution = norm.pdf(x, mu_H0, sigma)
        MDE_distribution = norm.pdf(x, mu_MDE, sigma)
        
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.plot(x, H0_distribution, color='blue', label='H0 Sample Distribution (Assumes no effect)')
        ax.plot(x, MDE_distribution, color='green', label='Alternative Hypothesis Distribution')
        
        ax.axvline(mu_H0, color='blue', linestyle='dashed', linewidth=1)
        ax.axvline(mu_MDE, color='green', linestyle='dashed', linewidth=1)
        
        # Calculating the critical z-values for alpha and the inverse of power (beta)
        z_alpha = norm.ppf(1 - alpha, mu_H0, sigma)
        beta = 1 - power
        z_beta = norm.ppf(beta, mu_MDE, sigma)
        
        ax.fill_between(x, H0_distribution, where=(x > z_alpha), color='blue', alpha=0.05, label='Type I Error (α)')
        ax.fill_between(x, MDE_distribution, where=(x < z_beta), color='green', alpha=0.05, label='Type II Error β')
        
        ax.set_title('Properly Powering Your (One-sided) AB Test')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True)
        
        st.pyplot(fig)
    mu_MDE = st.slider("Mean Difference (mu_MDE)", 0.0, 10.0, 2.0, 0.01)
    sigma = st.slider("Standard Deviation (sigma)", 0.1, 3.0, 1.0, 0.1)
    alpha = st.slider("Significance Level (alpha)", 0.01, 0.2, 0.05, 0.01)
    power = st.slider("Power", 0.5, 0.99, 0.8, 0.01)


    # Calling the plot function with current parameters
    interactive_plot(mu_MDE, sigma, alpha, power)
else : 
    

    st.title('Minimum Detectable Effect Calculator')

    with st.form("my_form"):
        n = st.number_input('Number of Visitors', min_value=0, value=15000, step=1000)
        baseline_cr = st.number_input('Conversion Rate (%)', min_value=0.0, value=3.0, step=0.1)
        submitted = st.form_submit_button("Calculate MDE")
        calculator = ABTEST(2,0,0.05,0.2,baseline_cr,n)

        if submitted:

            mde =calculator.calculate_mde()
            new_cr = baseline_cr * (1 + mde / 100)
            st.success(f"Minimal Detectable Effect: {mde:.2f}% (relative)")
            st.success(f"An uplift from {baseline_cr}% to {new_cr:.2f}% will be detectable")
