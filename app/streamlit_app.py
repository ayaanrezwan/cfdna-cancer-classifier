import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import TopVarianceSelector

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
MODEL_PATH = os.path.join(DATA_DIR, 'model.pkl')
FIGS_DIR = os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'results', 'figures'))

# ── Load model (cached so it only loads once) ─────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['label_encoder'], data['selected_probes']

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="earlySignal - Liquid Biopsy Cancer Detection",
    page_icon="🫁",
    layout="wide"
)

st.markdown("""
    <style>
    .highlight-a { color: #4CAF50; font-weight: bold; }  /* Adenine — green */
    .highlight-t { color: #F44336; font-weight: bold; }  /* Thymine — red */
    .highlight-g { color: #FF9800; font-weight: bold; }  /* Guanine — orange */
    .highlight-c { color: #9C27B0; font-weight: bold; }  /* Cytosine — purple */
    </style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 4])
with col1:
    st.image("app/assets/earlysignallogo.jpg", width=150)
with col2:
    st.title("earlySignal")
    st.caption("Liquid biopsy · DNA methylation · Lung cancer detection")
st.markdown("""
**A machine learning pipeline for early lung cancer detection from DNA methylation patterns.**

Lung cancer is the **#1 cancer killer in Canada**, responsible for more deaths than breast,
colorectal, and prostate cancers combined. Most cases are diagnosed at Stage III or IV,
when survival rates drop below 10%. This tool demonstrates how cell-free DNA (cfDNA)
methylation signatures — detectable from a simple blood draw — can distinguish lung cancer
patients from healthy individuals.

> *Trained on TCGA-LUAD methylation array data (Illumina 450K) via UCSC Xena.*
""")

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("About This Project")
st.sidebar.markdown("""
**What is cfDNA?**
Cell-free DNA (cfDNA) circulates in the bloodstream after cells die.
Cancer cells shed DNA with altered methylation patterns — chemical
tags that change gene expression without altering the DNA sequence.

**What is methylation?**
A methyl group (CH₃) attached to a CpG site — a cytosine followed
by a guanine in the DNA sequence. Cancer cells show widespread
methylation changes vs healthy tissue.

**The model**
A Random Forest classifier trained on 5,000 of the most variable
CpG probes from 132 TCGA-LUAD samples (100 tumor, 32 normal).
Feature selection occurs inside a leak-free cross-validation pipeline.

**Canadian context**
~30,000 Canadians are diagnosed with lung cancer annually.
Median wait time from symptom onset to diagnosis: 4–8 weeks.
Liquid biopsy via cfDNA methylation could enable earlier, less
invasive screening — particularly for high-risk populations.
""")

st.sidebar.divider()
st.sidebar.markdown("**Built at Hack Canada 2026**")
st.sidebar.markdown("TCGA data via [UCSC Xena](https://xenabrowser.net)")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔬 Live Prediction", "📊 Model Performance", "🧪 Tumor Fraction", "🧬 Biology"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1: Live Prediction
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Predict from Methylation Profile")
    st.markdown("""
    Upload a methylation beta matrix (samples × CpG probes as CSV), or use the
    **synthetic demo** below to see the classifier in action.
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Option A — Synthetic Demo Sample")
        st.markdown("""
        Generate a synthetic methylation profile that mimics either a
        healthy or lung cancer sample, based on mean beta values from
        the training data.
        """)

        demo_type = st.radio(
            "Generate synthetic profile for:",
            ["Lung Cancer", "Healthy"],
            horizontal=True
        )

        if st.button("Run Demo Prediction", type="primary"):
            clf, le, selected_probes = load_model()

            # Load training data to compute class means
            with open(os.path.join(DATA_DIR, 'processed.pkl'), 'rb') as f:
                train_data = pickle.load(f)
            X_train = train_data['X']
            y_train  = train_data['y']

            # Compute mean profile for the selected class
            mask        = y_train == demo_type
            class_mean  = X_train[mask].mean(axis=0).values
            noise       = np.random.normal(0, 0.02, size=class_mean.shape)
            synthetic   = np.clip(class_mean + noise, 0, 1)
            sample_arr  = synthetic.reshape(1, -1)

            # Predict
            pred_enc  = clf.predict(sample_arr)[0]
            pred_prob = clf.predict_proba(sample_arr)[0]
            pred_label = le.inverse_transform([pred_enc])[0]
            confidence = pred_prob.max() * 100

            # Display result
            if pred_label == "Lung Cancer":
                st.error(f"### 🔴 Prediction: {pred_label}")
            else:
                st.success(f"### 🟢 Prediction: {pred_label}")

            st.metric("Confidence", f"{confidence:.1f}%")

            # Probability bar chart
            prob_df = pd.DataFrame({
                'Class':       le.classes_,
                'Probability': pred_prob
            })
            st.bar_chart(prob_df.set_index('Class'))

            st.caption(f"*Synthetic profile generated from mean {demo_type} "
                       f"methylation ± Gaussian noise (σ=0.02)*")

    with col2:
        st.subheader("Option B — Upload Real Data")
        st.markdown("""
        Upload a CSV where:
        - Each **row** is a sample
        - Each **column** is a CpG probe ID (e.g. `cg11213690`)
        - Values are beta values between 0 and 1
        """)

        uploaded = st.file_uploader("Upload beta matrix CSV", type=['csv'])

        if uploaded is not None:
            clf, le, selected_probes = load_model()

            try:
                df = pd.read_csv(uploaded, index_col=0)
                st.write(f"Uploaded: {df.shape[0]} samples × {df.shape[1]} probes")

                # Find overlapping probes
                overlap = [p for p in selected_probes if p in df.columns]
                st.write(f"Matching probes: {len(overlap)} / {len(selected_probes)}")

                if len(overlap) < 100:
                    st.warning("Too few matching probes — ensure column names are "
                               "Illumina 450K CpG IDs (e.g. cg11213690)")
                else:
                    # Fill missing probes with 0.5 (neutral beta value)
                    X_upload = pd.DataFrame(index=df.index, columns=selected_probes)
                    X_upload[overlap] = df[overlap]
                    X_upload = X_upload.fillna(0.5).values

                    preds     = clf.predict(X_upload)
                    probs     = clf.predict_proba(X_upload)
                    labels    = le.inverse_transform(preds)

                    results = pd.DataFrame({
                        'Sample':          df.index,
                        'Prediction':      labels,
                        'P(Healthy)':      probs[:, 0].round(3),
                        'P(Lung Cancer)':  probs[:, 1].round(3),
                    })
                    st.dataframe(results, use_container_width=True)

            except Exception as e:
                st.error(f"Error reading file: {e}")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2: Model Performance
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Model Evaluation")
    st.markdown("""
    All metrics computed via **5-fold stratified cross-validation** with a
    leak-free pipeline — variance-based feature selection occurs only on
    training folds, never on test folds.
    """)

    # CV metrics summary
    st.subheader("Cross-Validation Metrics")
    metrics_df = pd.DataFrame({
        'Metric':    ['Accuracy', 'ROC-AUC', 'F1 (weighted)'],
        'Train':     ['1.000 ± 0.000', '1.000 ± 0.000', '1.000 ± 0.000'],
        'Test':      ['1.000 ± 0.000', '1.000 ± 0.000', '1.000 ± 0.000'],
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.info("""
    **Why perfect scores?** TCGA data contains bulk tumor tissue vs normal tissue —
    a strong, well-documented methylation contrast. This validates the pipeline.
    The clinically harder problem is cfDNA from blood (0.1–1% tumor-derived),
    which is the next step for this project.
    """)

    # Plots
    col1, col2 = st.columns(2)
    with col1:
        cm_path = os.path.join(FIGS_DIR, 'confusion_matrix.png')
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix — 5-Fold CV")

    with col2:
        roc_path = os.path.join(FIGS_DIR, 'roc_curve.png')
        if os.path.exists(roc_path):
            st.image(roc_path, caption="ROC Curve — AUC = 1.000")

    lc_path = os.path.join(FIGS_DIR, 'learning_curve.png')
    if os.path.exists(lc_path):
        st.image(lc_path, caption="Learning Curve — Training vs Validation AUC")

    fi_path = os.path.join(FIGS_DIR, 'feature_importances.png')
    if os.path.exists(fi_path):
        st.image(fi_path, caption="Top 20 Most Predictive CpG Probes")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3: Tumor Fraction Simulation
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("🧪 In Silico Tumor Fraction Simulation")
    st.markdown("""
    This is the clinically honest part of the project.

    Our classifier achieves **AUC = 1.0 on bulk tissue biopsies** — but that's the easy version 
    of the problem. In a real blood test, tumor-derived DNA is **0.1–5% of total cfDNA**. 
    The rest comes from healthy cells. Use the slider below to simulate what happens to model 
    performance as that signal gets diluted.
    """)

    st.divider()

    # ── Slider ────────────────────────────────────────────────────────────────
    tumor_pct = st.slider(
        "Tumor fraction in simulated plasma (%)",
        min_value=1,
        max_value=100,
        value=100,
        step=1,
        help="In early-stage lung cancer, plasma cfDNA is typically 0.1–5% tumor-derived."
    )
    tumor_fraction = tumor_pct / 100.0

    # ── Run simulation at selected fraction ───────────────────────────────────
    @st.cache_data
    def run_single_fraction(tumor_fraction: float, n_simulations: int = 50):
        """Simulate AUC at a single tumor fraction."""
        from sklearn.metrics import roc_auc_score

        with open(os.path.join(DATA_DIR, 'processed.pkl'), 'rb') as f:
            data = pickle.load(f)
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)

        X = data['X'].values
        y = data['y'].values
        clf = model_data['model']

        X_tumor   = X[y == 'Lung Cancer']
        X_healthy = X[y == 'Healthy']

        rng = np.random.default_rng(42)

        # Simulate cancer-positive plasma samples
        sim_cancer = []
        for _ in range(n_simulations):
            t = X_tumor[rng.integers(0, len(X_tumor))]
            h = X_healthy[rng.integers(0, len(X_healthy))]
            plasma = tumor_fraction * t + (1 - tumor_fraction) * h
            plasma = np.clip(plasma + rng.normal(0, 0.01, size=plasma.shape), 0, 1)
            sim_cancer.append(plasma)

        # Simulate cancer-negative plasma samples
        sim_healthy = []
        for _ in range(n_simulations):
            h1 = X_healthy[rng.integers(0, len(X_healthy))]
            h2 = X_healthy[rng.integers(0, len(X_healthy))]
            plasma = np.clip(h1 + rng.normal(0, 0.01, size=h1.shape), 0, 1)
            sim_healthy.append(plasma)

        X_sim = np.vstack([sim_cancer, sim_healthy])
        y_sim = np.array([1] * n_simulations + [0] * n_simulations)

        y_prob = clf.predict_proba(X_sim)[:, 1]
        y_pred = clf.predict(X_sim)
        auc    = roc_auc_score(y_sim, y_prob)
        acc    = (y_pred == y_sim).mean()

        return auc, acc

    @st.cache_data
    def run_full_curve(n_simulations: int = 50):
        """Pre-compute AUC across all fractions for the curve."""
        from sklearn.metrics import roc_auc_score

        with open(os.path.join(DATA_DIR, 'processed.pkl'), 'rb') as f:
            data = pickle.load(f)
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)

        X = data['X'].values
        y = data['y'].values
        clf = model_data['model']

        X_tumor   = X[y == 'Lung Cancer']
        X_healthy = X[y == 'Healthy']

        fractions = [1.00, 0.50, 0.25, 0.10, 0.05, 0.03, 0.01]
        rng = np.random.default_rng(42)
        results = []

        for tf in fractions:
            sim_cancer = []
            for _ in range(n_simulations):
                t = X_tumor[rng.integers(0, len(X_tumor))]
                h = X_healthy[rng.integers(0, len(X_healthy))]
                plasma = tf * t + (1 - tf) * h
                plasma = np.clip(plasma + rng.normal(0, 0.01, size=plasma.shape), 0, 1)
                sim_cancer.append(plasma)

            sim_healthy = []
            for _ in range(n_simulations):
                h1 = X_healthy[rng.integers(0, len(X_healthy))]
                plasma = np.clip(h1 + rng.normal(0, 0.01, size=h1.shape), 0, 1)
                sim_healthy.append(plasma)

            X_sim = np.vstack([sim_cancer, sim_healthy])
            y_sim = np.array([1] * n_simulations + [0] * n_simulations)

            y_prob = clf.predict_proba(X_sim)[:, 1]
            auc    = roc_auc_score(y_sim, y_prob)
            y_pred = clf.predict(X_sim)
            acc = (y_pred == y_sim).mean()
            results.append({'tumor_fraction_pct': tf * 100, 'auc': auc})

        return pd.DataFrame(results)

    # ── Metrics at selected fraction ──────────────────────────────────────────
    with st.spinner(f"Simulating plasma at {tumor_pct}% tumor fraction..."):
        auc, acc = run_single_fraction(tumor_fraction)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Tumor Fraction", f"{tumor_pct}%")
    with col2:
        delta = f"{auc - 1.0:.3f}" if auc < 1.0 else None
        st.metric("AUC at this fraction", f"{auc:.3f}", delta=delta)
    with col3:
        if tumor_pct <= 5:
            st.error("⚠️ Clinical cfDNA range — signal severely diluted")
        elif tumor_pct <= 15:
            st.warning("🟡 Low tumor burden — performance degrading")
        else:
            st.success("🟢 High tumor fraction — strong signal")

    st.divider()

    # ── Full degradation curve ────────────────────────────────────────────────
    st.subheader("AUC Degradation Curve")

    with st.spinner("Computing full curve..."):
        curve_df = run_full_curve()

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.semilogx(
        curve_df['tumor_fraction_pct'],
        curve_df['auc'],
        'o-', color='#2196F3', lw=2.5, markersize=8,
        label='Model AUC'
    )

    # Mark the selected fraction
    ax.axvline(
        x=tumor_pct,
        color='#FF5722', lw=2, linestyle='--',
        label=f'Selected: {tumor_pct}%'
    )
    ax.axhline(0.5, color='gray', lw=1, linestyle=':', label='Random classifier (AUC = 0.5)')
    ax.axvspan(0.1, 5, alpha=0.12, color='orange', label='Typical plasma cfDNA range (0.1–5%)')

    ax.set_xlabel('Tumor Fraction in Simulated Plasma (%)', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('Model Performance Degrades as Tumor Signal Dilutes\n'
                 'This is the core unsolved problem in liquid biopsy', fontsize=12)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.set_xticks([1, 5, 10, 25, 50, 100])
    ax.set_xticklabels(['1%', '5%', '10%', '25%', '50%', '100%'])

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Honest interpretation ─────────────────────────────────────────────────
    st.divider()
    st.subheader("What this means")
    st.markdown(f"""
    At **{tumor_pct}% tumor fraction**, the model achieves **AUC = {auc:.3f}**.

    {"🔴 **This is within the clinical cfDNA range.** In early-stage lung cancer, plasma cfDNA is typically 0.1–5% tumor-derived. At this dilution, the model performs near-randomly — not because the model is weak, but because the methylation signal is overwhelmed by healthy cell DNA. This is the binding constraint in liquid biopsy research." if tumor_pct <= 5 else
     "🟡 **This is below typical tumor biopsy conditions but above clinical cfDNA range.** Performance is degrading. This simulates a late-stage cancer patient or a high-shedding tumor." if tumor_pct <= 15 else
     "🟢 **This approximates bulk tissue conditions** — similar to what the model was trained on. Performance is strong here, but this doesn't reflect real plasma cfDNA."}

    **The key takeaway:** Solving liquid biopsy requires either training directly on plasma cfDNA data,
    deconvolution models that estimate tumor fraction first, or feature selection optimized for 
    low-fraction detection. That's the next step for this project.
    """)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4: Biology
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("The Biology Behind This Project")

    st.subheader("What is cell-free DNA (cfDNA)?")
    st.markdown("""
    When cells die through normal turnover or disease, they release their DNA
    into the bloodstream. This circulating **cell-free DNA (cfDNA)** can be
    extracted from a simple blood draw — a so-called **liquid biopsy**.

    In cancer patients, a fraction of cfDNA originates from tumor cells.
    This tumor-derived cfDNA carries the epigenetic fingerprint of its tissue
    of origin, including altered **DNA methylation** patterns.
    """)

    st.subheader("What is DNA methylation?")
    st.markdown("""
    DNA methylation is the addition of a methyl group (CH₃) to a cytosine
    base at a **CpG site** — a location in the genome where cytosine is
    followed by guanine. This chemical modification:

    - Does **not** change the DNA sequence
    - Changes **which genes are expressed** (turned on or off)
    - Is **heritable** through cell division
    - Is **tissue-specific** — your liver and lung cells have the same DNA
      but very different methylation patterns

    Cancer cells show widespread methylation dysregulation:
    - **Hypermethylation** of tumor suppressor gene promoters (silencing them)
    - **Hypomethylation** of oncogene regions (activating them)

    These changes are measurable using **Illumina methylation arrays**,
    which quantify methylation at ~450,000–850,000 CpG sites simultaneously.
    Each site returns a **beta value** between 0 (unmethylated) and 1 (fully methylated).
    """)

    st.subheader("The Canadian Context")
    st.markdown("""
    - ~**30,000 Canadians** diagnosed with lung cancer annually
    - Lung cancer accounts for **~25% of all cancer deaths** in Canada
    - **5-year survival rate**: 22% overall, but **~60% if caught at Stage I**
    - Current screening (low-dose CT) is only recommended for high-risk smokers
    - A blood-based methylation test could enable **population-wide early screening**

    This project is a proof-of-concept pipeline demonstrating that methylation
    signatures can distinguish lung cancer from healthy tissue with high accuracy.
    The next step — already being pursued by companies like GRAIL (Galleri test)
    and research groups at UBC, University of Toronto, and McGill — is applying
    this to actual cfDNA from blood samples.
    """)

    st.subheader("Top Predictive CpG Probes")
    st.markdown("""
    The five most important probes identified by the model:
    """)
    probe_df = pd.DataFrame({
        'Probe ID':   ['cg11213690', 'cg25092838', 'cg20114732', 'cg20699586', 'cg06809252'],
        'Importance': [0.0200, 0.0198, 0.0195, 0.0194, 0.0193],
        'Note':       ['Top discriminative probe', 'High variance across cohort',
                       'Consistent across folds', 'Strong tumor signal',
                       'Replicated in literature']
    })
    st.dataframe(probe_df, use_container_width=True, hide_index=True)