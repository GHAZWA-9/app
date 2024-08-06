import streamlit as st
from estimation import ABTEST
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import streamlit as st
import plotly.graph_objects as go


# st.title('A/B Test Sample Size Calculator')


page = st.sidebar.selectbox(
    "",
    [
        "A/B Test Sample Size Calculator",
        "Minimum Detectable Effect Calculator",
    ],
)

# Inputs
if page == "A/B Test Sample Size Calculator":

    baseline_conversion = (
        st.slider("Baseline Conversion Rate (%)", 0.0, 100.0, 10.0) / 100
    )
    minimum_effect = st.slider("Minimum Detectable Effect (%)", 0.1, 100.0, 20.0) / 100
    test_type = st.radio("Hypothesis", ("One-sided Test", "Two-sided Test"))
    alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05)
    beta = st.slider("Statistical Power (1 - β)", 0.65, 0.95, 0.80)
    daily_visitors = st.number_input("Daily Visitors", min_value=100, value=1000)
    
    # ab_split = st.number_input('Test vs. Control', 0.1, 1.0, 0.5, step=0.01)
    ab_split = 1
    # Calculate the sample size per group

    calculator = ABTEST(2, minimum_effect, alpha, 1 - beta, baseline_conversion)
    sample_size = calculator.get_sample_size(test_type)
    duration = calculator.calculate_duration(daily_visitors, test_type)
    # st.write(f"Required sample size per group: {sample_size}")
    # st.write(f"Duration in days (assuming equal traffic to both versions): {duration}")
    num_variants = st.number_input(
        "Number of Variants", min_value=2, max_value=10, value=2, step=1
    )
    allocations = {}

# Step 3: Generate input fields dynamically based on the number of variations
    for i in range(1, num_variants + 1):
        allocations[f'Variation {i}'] = st.slider(f'Allocation for Variation {i} (%)', 0, 100, 100 // num_variants)

# Check if allocations sum to 100%
    total_allocation = sum(allocations.values())

# Display the allocations
    st.write("Allocations:")
    for variation, percentage in allocations.items():
        st.write(f"{variation}: {percentage}%")

# Step 4: Validate the total allocation
    if total_allocation == 100:
        st.success("Total allocation correctly sums up to 100%.")
    else:
        st.error(f"Total allocation is off by {100 - total_allocation}%. Please adjust to sum up to 100%.")

    # Create an instance of the ABTEST CLASS
    # Number of variations is taken into account in the duration of the test calculation
    # Displaying results with progress bars

    st.metric(label="Sample Size per variation", value=f"{sample_size}")
    test_duration = np.ceil(num_variants * sample_size / daily_visitors)

    # Création du graphique de jauge
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=test_duration,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Test Duration in Days"},
            gauge={
                "axis": {"range": [None, 30], "tickwidth": 1, "tickcolor": "darkblue"},
                "bar": {"color": "darkslategray"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 10], "color": "lightgray"},
                    {"range": [10, 20], "color": "gray"},
                    {"range": [20, 30], "color": "darkgray"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": test_duration,
                },
            },
        )
    )
    fig.update_layout(
        autosize=False, width=500, height=300
    )  # Adjusting size of the gauge

    st.plotly_chart(fig, use_container_width=True)

else:
    

    st.title("Minimum Detectable Effect Calculator")

    with st.form("my_form"):
        n = st.number_input("Number of Visitors", min_value=0, value=15000, step=1000)
        baseline_cr = st.number_input(
            "Conversion Rate (%)", min_value=0.0, value=3.0, step=0.1
        )
        submitted = st.form_submit_button("Calculate MDE")
        calculator = ABTEST(2, 0, 0.05, 0.2, baseline_cr, n)

        if submitted:

            mde = calculator.calculate_mde()
            new_cr = baseline_cr * (1 + mde / 100)
            st.success(f"Minimal Detectable Effect: {mde:.2f}% (relative)")
            st.success(
                f"An uplift from {baseline_cr}% to {new_cr:.2f}% will be detectable"
            )

