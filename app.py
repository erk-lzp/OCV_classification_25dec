# =============================================================================
# app.py
# Streamlit app: identify the cathode chemistry (LCO / LFP / NMC) of a cell 
# from its OCV discharge curve, using the models trained on synthetic data.
#
# The user uploads a CSV, tells the app which column is voltage and which is
# capacity (or time + current), picks the units, and the app converts, extracts
# the features, runs the models, and shows the predicted chemistry with a
# confidence percentage. A PDF report of the result can be downloaded.
# =============================================================================

import os
import io
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import streamlit as st
from datetime import datetime

# trapezoidal integration: NumPy 2.0+ uses np.trapezoid, older uses np.trapz
_trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# where the trained model files live (relative to this app)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


# =============================================================================
# FEATURE EXTRACTION  (must match the training script exactly)
# =============================================================================
def compute_all_features(voltage, capacity):
    # work in normalized units so the absolute voltage window and cell size
    # do not matter - both axes are scaled to 0..1
    v = np.asarray(voltage, dtype=float)
    q = np.asarray(capacity, dtype=float)

    # make sure capacity goes from full to empty (increasing)
    if q[0] > q[-1]:
        v = v[::-1]; q = q[::-1]

    q_norm = (q - q.min()) / (q.max() - q.min() + 1e-12)
    v_norm = (v - v.min()) / (v.max() - v.min() + 1e-12)

    dvdq = np.gradient(v_norm, q_norm)
    abs_slope = np.abs(dvdq)

    def v_at(frac):
        return float(np.interp(frac, q_norm, v_norm))

    knee_idx = int(np.argmax(abs_slope))
    smooth = np.convolve(abs_slope, np.ones(5) / 5, mode="same")
    peaks = 0
    for i in range(1, len(smooth) - 1):
        if smooth[i] > smooth[i - 1] and smooth[i] > smooth[i + 1] and smooth[i] > 0.2:
            peaks += 1

    return {
        "plateau_fraction": float(np.mean(abs_slope < 0.3)),
        "v_start": v_at(0.05), "v_at_25": v_at(0.25), "v_at_50": v_at(0.50),
        "v_at_75": v_at(0.75), "v_end": v_at(0.95),
        "mean_abs_slope": float(np.mean(abs_slope)),
        "max_abs_slope": float(abs_slope[knee_idx]),
        "knee_position": float(knee_idx / (len(abs_slope) - 1)),
        "area_under_curve": float(_trap(v_norm, q_norm)),
        "dvdq_peak_count": float(peaks),
    }


def extract_features(voltage, capacity, feature_names):
    allf = compute_all_features(voltage, capacity)
    return np.array([allf[name] for name in feature_names])


# =============================================================================
# UNIT CONVERSION HELPERS
# =============================================================================
def voltage_to_volts(values, unit):
    # convert the voltage column to volts
    if unit == "mV":
        return values / 1000.0
    return values  # already V


def capacity_to_Ah(values, unit):
    # convert the capacity column to amp-hours (Ah)
    if unit == "mAh":
        return values / 1000.0
    return values  # already Ah


def time_to_hours(values, unit):
    # convert a time column to hours
    if unit == "seconds":
        return values / 3600.0
    if unit == "minutes":
        return values / 60.0
    return values  # already hours


def current_to_A(value, unit):
    # convert a single current value to amps
    if unit == "mA":
        return value / 1000.0
    return value  # already A


# =============================================================================
# LOAD MODELS  (cached so it only reads from disk once)
# =============================================================================
@st.cache_resource
def load_models():
    models = {}
    for path in sorted(glob.glob(os.path.join(MODEL_DIR, "*.joblib"))):
        bundle = joblib.load(path)
        pretty = os.path.basename(path).replace(".joblib", "").replace("_", " ").title()
        models[pretty] = bundle
    return models


# =============================================================================
# PLOTTING
# =============================================================================
def make_curve_figure(q_Ah, v_V):
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=150)
    ax.plot(q_Ah, v_V, lw=1.8, color="#2b6cb0")
    ax.set_xlabel("Capacity [Ah]")
    ax.set_ylabel("Voltage [V]")
    ax.set_title("Uploaded OCV curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def make_probability_figure(classes, proba_pct, predicted):
    fig, ax = plt.subplots(figsize=(6.5, 3.6), dpi=150)
    colors = ["#2ca02c" if c == predicted else "#b0b7c3" for c in classes]
    bars = ax.bar(classes, proba_pct, color=colors)
    for b, p in zip(bars, proba_pct):
        ax.annotate(f"{p:.1f}%", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Confidence [%]")
    ax.set_ylim(0, 108)
    ax.set_title("Chemistry probability")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# =============================================================================
# PDF REPORT
# =============================================================================
def build_pdf(predicted, confidence, model_name, classes, proba_pct,
              curve_fig, prob_fig, feature_dict):
    # build a one-page PDF summary using matplotlib's PdfPages
    from matplotlib.backends.backend_pdf import PdfPages
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # page 1: title, result, and the two figures
        fig = plt.figure(figsize=(8.27, 11.69), dpi=150)  # A4 portrait
        fig.suptitle("Cathode Chemistry Identification Report",
                     fontsize=16, fontweight="bold", y=0.97)

        # header text block
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        header = (f"Generated: {stamp}\n"
                  f"Model used: {model_name}\n\n"
                  f"Predicted chemistry: {predicted}\n"
                  f"Confidence: {confidence:.1f}%")
        fig.text(0.08, 0.86, header, fontsize=12, va="top", family="monospace")

        # probabilities as text
        prob_lines = "  ".join(f"{c}: {p:.1f}%" for c, p in zip(classes, proba_pct))
        fig.text(0.08, 0.78, "All classes -> " + prob_lines, fontsize=10, va="top")

        # paste the curve figure
        ax1 = fig.add_axes([0.08, 0.45, 0.84, 0.28])
        _paste_fig(ax1, curve_fig)
        # paste the probability figure
        ax2 = fig.add_axes([0.08, 0.13, 0.84, 0.26])
        _paste_fig(ax2, prob_fig)

        pdf.savefig(fig); plt.close(fig)

        # page 2: the extracted features table
        fig2 = plt.figure(figsize=(8.27, 11.69), dpi=150)
        fig2.suptitle("Extracted curve features", fontsize=14, fontweight="bold", y=0.96)
        rows = [[k, f"{v:.4f}"] for k, v in feature_dict.items()]
        ax = fig2.add_axes([0.1, 0.1, 0.8, 0.8]); ax.axis("off")
        table = ax.table(cellText=rows, colLabels=["feature", "value"],
                         loc="upper center", cellLoc="left")
        table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.6)
        pdf.savefig(fig2); plt.close(fig2)

    buf.seek(0)
    return buf


def _paste_fig(target_ax, source_fig):
    # render a matplotlib figure to an image and drop it into a target axes
    source_fig.canvas.draw()
    img = np.frombuffer(source_fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = source_fig.canvas.get_width_height()
    img = img.reshape(h, w, 4)
    target_ax.imshow(img); target_ax.axis("off")


# =============================================================================
# STREAMLIT APP
# =============================================================================
st.set_page_config(page_title="Cathode Chemistry ID", page_icon="🔋", layout="wide")

st.title("Cathode Chemistry Identification from OCV Curves")
st.write("Upload a low-rate discharge curve. The app reads the voltage profile, "
         "extracts its shape, and predicts whether the cell is LCO, LFP, or NMC "
         "with a confidence score.")

models = load_models()
if not models:
    st.error("No trained models found. Put the .joblib model files in a 'models' "
             "folder next to this app, then reload.")
    st.stop()

# ---- STEP 1: upload ---------------------------------------------------------
st.header("1. Upload your data")
uploaded = st.file_uploader("CSV file with a voltage column and either a "
                            "capacity column or a time column", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded)
st.write("Preview of your file:")
st.dataframe(df.head(), use_container_width=True)
cols = list(df.columns)

# ---- STEP 2: map columns and units -----------------------------------------
st.header("2. Tell the app what each column is")

left, right = st.columns(2)

with left:
    st.subheader("Voltage")
    v_col = st.selectbox("Which column is voltage?", cols, key="vcol")
    v_unit = st.selectbox("Voltage unit", ["V", "mV"], key="vunit")

with right:
    st.subheader("Capacity source")
    # the user can either give capacity directly, or give time + current
    cap_mode = st.radio("How is capacity available?",
                        ["I have a capacity column",
                         "I only have time (compute from current)"],
                        key="capmode")

    if cap_mode == "I have a capacity column":
        q_col = st.selectbox("Which column is capacity?",
                             [c for c in cols if c != v_col], key="qcol")
        q_unit = st.selectbox("Capacity unit", ["Ah", "mAh"], key="qunit")
    else:
        t_col = st.selectbox("Which column is time?",
                             [c for c in cols if c != v_col], key="tcol")
        t_unit = st.selectbox("Time unit", ["seconds", "minutes", "hours"], key="tunit")
        cur_val = st.number_input("Discharge current (constant)", min_value=0.0,
                                  value=1.0, step=0.1, key="curval")
        cur_unit = st.selectbox("Current unit", ["A", "mA"], key="curunit")

# ---- STEP 3: model choice ---------------------------------------------------
st.header("3. Choose the model")
model_name = st.selectbox("Model", list(models.keys()))
bundle = models[model_name]
model = bundle["model"]; classes = list(bundle["classes"])
feature_names = list(bundle["features"])

# ---- RUN --------------------------------------------------------------------
run = st.button("Identify chemistry", type="primary")

if run:
    # build the voltage array in volts
    try:
        v_raw = df[v_col].values.astype(float)
    except Exception:
        st.error(f"Column '{v_col}' does not look numeric. Pick the voltage column.")
        st.stop()
    v_V = voltage_to_volts(v_raw, v_unit)

    # build the capacity array in Ah, either directly or from time * current
    if cap_mode == "I have a capacity column":
        q_raw = df[q_col].values.astype(float)
        q_Ah = capacity_to_Ah(q_raw, q_unit)
    else:
        t_raw = df[t_col].values.astype(float)
        t_h = time_to_hours(t_raw, t_unit)          # time in hours
        cur_A = current_to_A(cur_val, cur_unit)     # current in amps
        # capacity (Ah) = current (A) * time (h); time starts at zero
        t_h = t_h - t_h.min()
        q_Ah = cur_A * t_h

    # guard against too-short curves
    if len(v_V) < 10:
        st.error("The curve has too few points to analyze. Need at least ~10 rows.")
        st.stop()

    # extract features and predict
    feat_dict_full = compute_all_features(v_V, q_Ah)
    x = np.array([feat_dict_full[name] for name in feature_names]).reshape(1, -1)

    proba = model.predict_proba(x)[0]
    pred_idx = int(np.argmax(proba))
    predicted = classes[pred_idx]
    confidence = float(proba[pred_idx] * 100.0)
    proba_pct = [float(proba[classes.index(c)] * 100.0) for c in classes]

    # ---- results ----
    st.header("Result")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.metric("Predicted chemistry", predicted, f"{confidence:.1f}% confidence")
        # a short plain-language note on how sure the model is
        if confidence >= 80:
            st.success("The model is confident in this prediction.")
        elif confidence >= 55:
            st.warning("Moderate confidence. Check the curve quality and units.")
        else:
            st.error("Low confidence. The curve may be noisy, cropped, or the "
                     "units may be set wrong.")
    with c2:
        st.write("Confidence for each chemistry:")
        st.dataframe(pd.DataFrame({"chemistry": classes,
                                   "confidence_%": [round(p, 1) for p in proba_pct]}),
                     use_container_width=True, hide_index=True)

    # ---- figures ----
    curve_fig = make_curve_figure(q_Ah, v_V)
    prob_fig = make_probability_figure(classes, proba_pct, predicted)
    g1, g2 = st.columns(2)
    with g1:
        st.pyplot(curve_fig)
    with g2:
        st.pyplot(prob_fig)

    # ---- extracted features (transparency for the user) ----
    with st.expander("See the features the model used"):
        st.dataframe(pd.DataFrame({"feature": feature_names,
                                   "value": [round(feat_dict_full[f], 4) for f in feature_names]}),
                     use_container_width=True, hide_index=True)

    # ---- PDF download ----
    pdf_buf = build_pdf(predicted, confidence, model_name, classes, proba_pct,
                        curve_fig, prob_fig,
                        {f: feat_dict_full[f] for f in feature_names})
    st.download_button("Download PDF report", data=pdf_buf,
                       file_name=f"chemistry_report_{predicted}.pdf",
                       mime="application/pdf")

    st.caption("Note: predictions are based on curve shape learned from "
               "synthetic PyBaMM data. Treat the result as a secondary check, "
               "not a definitive assay.")
