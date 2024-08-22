import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from estimation import ABTEST
from scipy.stats import norm
import pandas as pd

# st.title('A/B Test Sample Size Calculator')


page = st.sidebar.selectbox(
    "",
    [
        "A/B Test Sample Size Calculator",
        "Visualization",
        "Minimum Detectable Effect Calculator",
    ],
)

# Inputs


if page == "A/B Test Sample Size Calculator":

    baseline_conversion = (
        st.number_input("Baseline Conversion Rate (%)", min_value=1) / 100
    )
    minimum_effect = (
        st.number_input("Minimum Detectable Effect (%)", min_value=1) / 100
    )
    test_type = st.radio("Hypothesis", ("One-sided Test", "Two-sided Test"))
    alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05)
    beta = st.slider("Statistical Power (1 - β)", 0.65, 0.95, 0.80)
    daily_visitors = st.number_input("Daily Visitors", min_value=100, value=1000)

    # ab_split = st.number_input('Test vs. Control', 0.1, 1.0, 0.5, step=0.01)
    # Calculate the sample size per group
    # calculator = ABTEST(2, minimum_effect, alpha, 1 - beta, baseline_conversion)
    # sample_size = calculator.get_sample_size(test_type)
    # duration = calculator.calculate_duration(daily_visitors, test_type)
    # st.write(f"Required sample size per group: {sample_size}")
    # st.write(f"Duration in days (assuming equal traffic to both versions): {duration}")

    # Step 1: Get the number of variations
    num_variants = st.number_input(
        "Number of Variants", min_value=2, max_value=10, value=2, step=1)

    # Step 2: Create a dictionary to store the percentage allocations
    allocations = {}

    # Using columns to layout the sliders neatly
    cols = st.columns(num_variants)

    # Step 3: Generate input fields dynamically within columns
    for i, col in enumerate(cols):
        with col:
            if i < num_variants and i == 0:
                allocations[f"Control"] = st.slider(
                    f"Control  Allocation (%)", 0, 100, 100 // num_variants
                )
            elif i < num_variants:
                allocations[f"Variant {i}"] = st.slider(
                    f"Variant {i} Allocation (%)", 0, 100, 100 // num_variants
                )
    variant_allocations = {
        key: val for key, val in allocations.items() if key != "Control"
    }
    if variant_allocations:  #
        ratio_test = min(variant_allocations.values()) / 100
        ratio_control = allocations["Control"] / 100
        calculator = ABTEST(
            2,
            minimum_effect,
            alpha,
            beta,
            baseline_conversion,
            1000,
            ratio_control,
            ratio_test,
        )
        sample_size = calculator.get_sample_size(test_type)

    # Display the allocations using an expander
    with st.expander("See Allocation Details"):
        for variation, percentage in allocations.items():
            st.write(f"{variation}: {percentage}%")
            st.progress(percentage)

        # Create an instance of the ABTEST CLASS
        # Number of variations is taken into account in the duration of the test calculation
        # Displaying results with progress bars

    st.metric(label="Total Sample Size ", value=f"{int(sample_size)}")
    #test_duration = np.round((sample_size / daily_visitors),1)
    duration = calculator.calculate_duration(daily_visitors, test_type)
    st.write(f"Duration in days (assuming equal traffic to both versions): {duration})")
    # Création du graphique de jauge
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=duration,
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
                    "value": duration,
                },
            },
        )
    )
    fig.update_layout(
        autosize=False, width=500, height=300
    )  # Adjusting size 
    

    st.plotly_chart(fig, use_container_width=True)


elif page == "Visualization":

    baseline_conversion = (
        st.number_input("Baseline Conversion Rate (%)", min_value=1) / 100
    )
    minimum_effect = (
        st.number_input("Minimum Detectable Effect (%)", min_value=1) / 100
    )
    test_type = st.radio("Hypothesis", ("One-sided Test", "Two-sided Test"))
    alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05)
    beta = st.slider("Statistical Power (1 - β)", 0.65, 0.95, 0.80)
    sigma = baseline_conversion * (1 - baseline_conversion)
    st.write("## Interactive AB Test Power Analysis")
    calculator = ABTEST(2, minimum_effect, alpha, beta, baseline_conversion)
    if st.button("Generate Plot"):
        fig = calculator.generate_plot(test_type)
        st.pyplot(fig)

else:

    st.title("Minimum Detectable Effect Calculator")

    with st.form("my_form"):



        weekly_traffic = st.number_input(
            "weekly_traffic",
            min_value=100,
            value=1000,
            step=100,
        )
        weekly_conversions = st.number_input(
            "Conversions hebdomadaires", min_value=1, value=50, step=1
        )
        num_variants = st.number_input("Number of Variants", min_value=2, value=2, step=1)
        baseline_cr = round(weekly_conversions / weekly_traffic, 2)
        alpha = 0.05
        beta = 0.2
        submit_button = st.form_submit_button(label="Calculate")

    if submit_button:
        results = []

        for weeks in range(1, 6):
            test = ABTEST(nv=num_variants, mde=0, alpha=0.05, beta=0.2, ctr1=baseline_cr, r1=0.5, r2=0.5, traffic=(weeks * weekly_traffic) ) 
            mde = test.calculate_mde()
            test = ABTEST(nv=num_variants, mde=mde, alpha=0.05, beta=0.2, ctr1=baseline_cr, r1=0.5, r2=0.5, traffic=weeks * weekly_traffic)
            visitors = test.get_sample_size()
            #visitors = weekly_traffic*weeks//num_variants
            results.append(
                {
                "Number of weeks": weeks,
                "Min. Det.Effect (MDE) %": f"{mde * 100:.2f}",
                "Visitors per variant": int(visitors),
                }
            )

    results_df = pd.DataFrame(results)
    results_df = results_df.set_index("Number of weeks")
    st.table(results_df)
  

   