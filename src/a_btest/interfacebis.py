import numpy as np
import plotly.graph_objects as go
import plots
from fasthtml.common import *
from src.estimation import ABTEST

app, rt = fast_app()


"""@rt("/")
def index():
    selected_page = Select(
        "Choose a page",
        options=[
            "A/B Test Sample Size Calculator",
            "Visualization",
            "Minimum Detectable Effect Calculator",
        ],
    )

    if selected_page == "A/B Test Sample Size Calculator":
        return ab_test_sample_size()
    elif selected_page == "Visualization":
        return visualization()
    else:
        return minimum_detectable_effect()

    return Div(selected_page)"""


@rt("/ab-test-size")
def ab_test_sample_size():
    form = Form(
        Input(
            type="number",
            name="baseline_conversion",
            placeholder="Baseline Conversion Rate (%)",
            min=0,
        ),
        Input(
            type="number",
            name="minimum_effect",
            placeholder="Minimum Detectable Effect (%)",
            min=0,
        ),
        src
        / a_btest
        / interfacebis.py("test_type", options=["One-sided Test", "Two-sided Test"]),
        Input(type="range", name="alpha", min=0.01, max=0.1, step=0.01, value=0.05),
        Input(type="range", name="beta", min=0.65, max=0.95, step=0.01, value=0.8),
        Input(
            type="number",
            name="daily_visitors",
            placeholder="Daily Visitors",
            min=100,
            value=1000,
        ),
        Input(type="submit", value="Calculate"),
    )
    if form.is_submitted():
        # Calculation based on inputs
        result = calculate_sample_size(form.data)
        return Div(
            H1("Results"),
            P(f"Sample Size: {result['sample_size']}"),
            P(f"Duration: {result['duration']} days"),
        )
    return form


def create_plot(data):
    fig = go.Figure(data)
    # Convert Plotly figure to HTML or URL to embed
    plot_html = fig.to_html()
    return Html(plot_html)


serve()
