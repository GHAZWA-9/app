import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from estimation import ABTEST
from scipy.stats import norm

st.title(" A/B Test Calculator")

with st.form("test_data"):
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
    st.table(results_df)
