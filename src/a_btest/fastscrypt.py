import streamlit as st
from streamlit_fast_html import fast_html

# Custom HTML content
html_content = """
<div style='background: linear-gradient(135deg, #9575CD 0%, #EDE7F6 100%); 
            min-height: 100vh; 
            padding: 50px; 
            display: flex; 
            flex-direction: column; 
            justify-content: center; 
            align-items: center;'>
    <h1 style='color: #FFFFFF; font-size: 48px; text-align: center;'>Welcome to Kameleoon AI Experiments</h1>
    <p style='color: #FFFFFF; font-size: 24px;'>Explore the potential of AI with our platform.</p>
    <button style='font-size: 20px; padding: 10px 20px; border: none; 
                   background-color: #5F4B8B; color: white; border-radius: 5px;
                   cursor: pointer; margin-top: 20px;'>
        Learn More
    </button>
</div>
"""

# Render the custom HTML using fast_html
fast_html(html_content, width=None, scrolling=True)
