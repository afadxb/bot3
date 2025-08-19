import os
import requests
import streamlit as st

BACKEND = os.getenv("BACKEND_URL", "http://backend:8000")
API_TOKEN = os.getenv("API_TOKEN", "changeme")

st.title("Barchart Swing Bot")

if st.button("Check Health"):
    r = requests.get(f"{BACKEND}/health")
    st.write(r.json())

st.header("Scheduler")
for job in ["ingest", "verify", "enrich", "analyze", "publish"]:
    if st.button(f"Run {job}"):
        r = requests.post(
            f"{BACKEND}/scheduler/{job}/run",
            headers={"X-API-TOKEN": API_TOKEN},
        )
        st.write(r.json())

st.header("Settings")
if st.button("Toggle Kill-switch"):
    r = requests.post(
        f"{BACKEND}/settings/kill_switch",
        params={"value": "on"},
        headers={"X-API-TOKEN": API_TOKEN},
    )
    st.write(r.json())

if st.button("Toggle Risk Envelope"):
    r = requests.post(
        f"{BACKEND}/settings/risk_envelope",
        params={"value": "off"},
        headers={"X-API-TOKEN": API_TOKEN},
    )
    st.write(r.json())

if st.button("Pushover Test"):
    r = requests.post(
        f"{BACKEND}/pushover/test",
        headers={"X-API-TOKEN": API_TOKEN},
    )
    st.write(r.json())
