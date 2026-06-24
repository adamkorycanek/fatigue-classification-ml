import streamlit as st
import os
import joblib
import numpy as np

# Load model
model = joblib.load("models/final_model.pkl")

REFERENCE_FILE = "data/reference_values.json"


# STATUS

reference_exists = os.path.exists(REFERENCE_FILE)

if reference_exists:
    st.success("Referenční měření je uloženo.")
else:
    st.warning("Nejprve proveďte referenční měření.")


# RESET

if st.button("Reset reference", disabled=not reference_exists):
    os.remove(REFERENCE_FILE)
    st.warning("Reference smazána.")
    st.rerun()


# REFERENCE MEASUREMENT (SIMULATED / IMPORTED DATA)

st.header("1. Referenční měření")

if st.button("Spustit referenční měření"):

    st.warning("Simulace měření...")

    # ⚠️ NA GITHUBU:
    # místo MATLAB -> načti nebo simuluj data
    ref_values = {
        "norm_X_5kHz": 1.0,
        "norm_X_100kHz": 1.0,
        "norm_θ2_100kHz": 1.0,
        "norm_R_5kHz/R_100kHz": 1.0
    }

    import json
    with open(REFERENCE_FILE, "w") as f:
        json.dump(ref_values, f)

    st.success("Reference uložena.")
    st.rerun()


# CURRENT MEASUREMENT

st.header("2. Aktuální měření")

if st.button("Vyhodnotit únavu"):

    if not reference_exists:
        st.error("Nejprve reference.")
    else:

        import json

        with open(REFERENCE_FILE, "r") as f:
            ref_values = json.load(f)

        # ⚠️ simulace aktuálního měření
        cur_values = {
            "norm_X_5kHz": 0.8,
            "norm_X_100kHz": 0.7,
            "norm_θ2_100kHz": 0.6,
            "norm_R_5kHz/R_100kHz": 0.9
        }

        # feature vector
        input_data = np.array([[
            cur_values["norm_X_5kHz"],
            cur_values["norm_X_100kHz"],
            cur_values["norm_θ2_100kHz"],
            cur_values["norm_R_5kHz/R_100kHz"],
        ]])

        prediction = model.predict(input_data)[0]

        st.subheader("Výsledek")

        if prediction == "Nízká":
            st.success("🟢 Nízká únava")

        elif prediction == "Střední":
            st.warning("🟡 Střední únava")

        elif prediction == "Vysoká":
            st.error("🔴 Vysoká únava")


# INFO
st.info("Aplikace běží na ML modelu bez závislosti na MATLAB.")
