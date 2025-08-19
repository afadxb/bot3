import os
import streamlit as st
import requests

BACKEND = os.getenv("BACKEND_URL", "http://backend:8000")

st.title("Barchart Swing Bot")

if st.button("Check Health"):
    r = requests.get(f"{BACKEND}/health")
    st.write(r.json())
