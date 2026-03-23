# -*- coding: utf-8 -*-
"""
GNSS Error Correction -- Data & Compute Trade-off Dashboard
Assignment 2: Physics-Informed GNSS Positioning in Degraded Environments

Run:  streamlit run dashboard.py
Requires:  pip install streamlit plotly numpy pandas
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GNSS — NN vs PINN Trade-offs",
    page_icon="📡",
    layout="wide",
)

# ─────────────────────────────────────────────
# Dark theme for all Plotly charts
# ─────────────────────────────────────────────
PLOT_BG = "#0D1B2A"
CARD_BG = "#142A3E"
GRID_COLOR = "#1E3A5F"
TEXT_COLOR = "#E0E7EE"
MUTED = "#6B7B8D"

PINN_BLUE = "#29B6F6"
NN_RED = "#EF553B"
KALMAN_AMBER = "#FFB74D"
GREEN = "#66BB6A"
CYAN = "#00BCD4"
CORAL = "#FF6B6B"
AMBER = "#FFB74D"
TEAL = "#009688"
GOLD = "#FFD54F"

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_COLOR, family="Calibri, sans-serif"),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        legend=dict(bgcolor="rgba(20,42,62,0.85)", bordercolor=GRID_COLOR, borderwidth=1),
        colorway=[PINN_BLUE, NN_RED, KALMAN_AMBER, GREEN, CYAN, CORAL, TEAL],
    )
)

def dark_layout(**kwargs):
    base = dict(
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_COLOR, family="Calibri, sans-serif", size=13),
        legend=dict(bgcolor="rgba(20,42,62,0.85)", bordercolor=GRID_COLOR, borderwidth=1),
        margin=dict(l=60, r=30, t=50, b=50),
    )
    base.update(kwargs)
    return base

def dark_axes(fig, **kwargs):
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, **kwargs)
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    return fig

# ─────────────────────────────────────────────
# Custom CSS for dark dashboard
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0D1B2A; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #142A3E; color: #E0E7EE; border-radius: 6px 6px 0 0;
        padding: 8px 16px; border: 1px solid #1E3A5F;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0A2E50; border-bottom: 2px solid #00BCD4;
    }
    h1, h2, h3, .stMarkdown p, .stMarkdown li { color: #E0E7EE; }
    .stMetric label { color: #A0AEBB !important; }
    .stMetric [data-testid="stMetricValue"] { color: #E0E7EE !important; }
    div[data-testid="stExpander"] { background-color: #142A3E; border: 1px solid #1E3A5F; border-radius: 8px; }
    .info-card {
        background: linear-gradient(135deg, #142A3E 0%, #1A354C 100%);
        border: 1px solid #1E3A5F; border-radius: 10px; padding: 16px 20px;
        margin: 8px 0; border-left: 4px solid #00BCD4;
    }
    .warn-card {
        background: linear-gradient(135deg, #2A1A0A 0%, #3E250A 100%);
        border: 1px solid #5C3A0A; border-radius: 10px; padding: 16px 20px;
        margin: 8px 0; border-left: 4px solid #FFB74D;
    }
    .danger-card {
        background: linear-gradient(135deg, #2A0A0A 0%, #3E1515 100%);
        border: 1px solid #5C1A1A; border-radius: 10px; padding: 16px 20px;
        margin: 8px 0; border-left: 4px solid #FF6B6B;
    }
    .success-card {
        background: linear-gradient(135deg, #0A2A15 0%, #0A3E20 100%);
        border: 1px solid #0A5C2A; border-radius: 10px; padding: 16px 20px;
        margin: 8px 0; border-left: 4px solid #66BB6A;
    }
</style>
""", unsafe_allow_html=True)

st.title("Data & Compute Trade-offs: Traditional NNs vs PINNs")
st.markdown(
    "*Physics-Informed GNSS Positioning Error Correction in Degraded Environments* "
    "— Assignment 2: The Data & Compute Reality Check"
)

# ─────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────
st.sidebar.header("Model & Environment Controls")

model_size = st.sidebar.slider(
    "Model parameters (millions)",
    min_value=5, max_value=200, value=50, step=5,
    help="Reference: 50M for a GNSS correction model (~600 MB training state).",
)

lambda_weight = st.sidebar.slider(
    "λ — Physics loss weight",
    min_value=0.0, max_value=2.0, value=0.5, step=0.05,
    help="Balance between data loss and physics loss: L(θ) = L_data + λ·L_physics",
)

precision = st.sidebar.selectbox(
    "Quantization level",
    ["FP32 (baseline)", "FP16 (training)", "INT8 (default)", "INT4 (aggressive)"],
    index=2,
)

environment = st.sidebar.selectbox(
    "Deployment environment",
    ["Open Sky", "Urban Canyon", "Tunnel Exit", "Deep Urban"],
    index=1,
)

batch_size = st.sidebar.slider(
    "Inference batch size (vehicles)",
    min_value=1, max_value=64, value=8, step=1,
    help="Number of vehicles batched per inference call.",
)

drift_months = st.sidebar.slider(
    "Deployment duration (months)",
    min_value=1, max_value=24, value=6, step=1,
    help="How long the model has been deployed — affects calibration drift.",
)

np.random.seed(42)

# ─────────────────────────────────────────────
# Environment-dependent parameters
# ─────────────────────────────────────────────
ENV_CONFIG = {
    "Open Sky":      {"sats": 10, "cn0": 42, "error_base": 3.5,  "multipath": 0.1, "idx": 0},
    "Urban Canyon":  {"sats": 3,  "cn0": 22, "error_base": 55,   "multipath": 0.7, "idx": 1},
    "Tunnel Exit":   {"sats": 1,  "cn0": 10, "error_base": 70,   "multipath": 0.3, "idx": 2},
    "Deep Urban":    {"sats": 2,  "cn0": 15, "error_base": 90,   "multipath": 0.9, "idx": 3},
}
env = ENV_CONFIG[environment]


# ═════════════════════════════════════════════
# TABS — matching 8 slides
# ═════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📡 GNSS Degradation",
    "🏗️ Architecture",
    "📊 Data Reality",
    "⚠️ Silent Killers",
    "💰 Compute Economics",
    "🚀 Edge Deployment",
    "⚔️ Head-to-Head",
    "✅ Verdict",
])


# ══════════════════════════════════════════════
# TAB 1 — GNSS Fails Where It Matters Most
# ══════════════════════════════════════════════
with tab1:
    st.subheader("GNSS Fails Where It Matters Most")
    st.markdown("The degradation spectrum: from acceptable navigation to total signal denial.")

    col_gauge, col_chart = st.columns([1, 1])

    with col_gauge:
        # Animated gauge indicators for selected environment
        fig_gauge = make_subplots(
            rows=1, cols=3,
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
            column_widths=[0.33, 0.33, 0.33],
        )
        fig_gauge.add_trace(go.Indicator(
            mode="gauge+number",
            value=env["sats"],
            title={"text": "Satellites Visible", "font": {"size": 14, "color": TEXT_COLOR}},
            number={"font": {"color": TEXT_COLOR}},
            gauge={
                "axis": {"range": [0, 12], "tickcolor": MUTED},
                "bar": {"color": PINN_BLUE},
                "bgcolor": CARD_BG,
                "steps": [
                    {"range": [0, 3], "color": "#3E1515"},
                    {"range": [3, 6], "color": "#3E250A"},
                    {"range": [6, 12], "color": "#0A3E20"},
                ],
                "threshold": {"line": {"color": GREEN, "width": 3}, "value": 4, "thickness": 0.8},
            },
        ), row=1, col=1)

        fig_gauge.add_trace(go.Indicator(
            mode="gauge+number",
            value=env["cn0"],
            title={"text": "C/N₀ (dB-Hz)", "font": {"size": 14, "color": TEXT_COLOR}},
            number={"font": {"color": TEXT_COLOR}, "suffix": " dB-Hz"},
            gauge={
                "axis": {"range": [0, 50], "tickcolor": MUTED},
                "bar": {"color": CYAN},
                "bgcolor": CARD_BG,
                "steps": [
                    {"range": [0, 15], "color": "#3E1515"},
                    {"range": [15, 30], "color": "#3E250A"},
                    {"range": [30, 50], "color": "#0A3E20"},
                ],
                "threshold": {"line": {"color": GREEN, "width": 3}, "value": 35, "thickness": 0.8},
            },
        ), row=1, col=2)

        fig_gauge.add_trace(go.Indicator(
            mode="gauge+number",
            value=env["error_base"],
            title={"text": "Position Error (m)", "font": {"size": 14, "color": TEXT_COLOR}},
            number={"font": {"color": TEXT_COLOR}, "suffix": " m"},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": MUTED},
                "bar": {"color": CORAL},
                "bgcolor": CARD_BG,
                "steps": [
                    {"range": [0, 5], "color": "#0A3E20"},
                    {"range": [5, 30], "color": "#3E250A"},
                    {"range": [30, 100], "color": "#3E1515"},
                ],
                "threshold": {"line": {"color": GREEN, "width": 3}, "value": 5, "thickness": 0.8},
            },
        ), row=1, col=3)

        fig_gauge.update_layout(**dark_layout(height=280, title_text=f"Environment: {environment}"))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Info cards
        st.markdown(f"""
        <div class="{'success-card' if env['sats'] >= 6 else 'warn-card' if env['sats'] >= 3 else 'danger-card'}">
            <b>Status:</b> {'Acceptable for navigation' if env['sats'] >= 6 else 'Errors exceed lane width — multipath dominant' if env['sats'] >= 2 else 'Signal denied — classical methods fail'}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <b>Classical Baseline:</b> Kalman filters + INS aiding — hand-tuned and brittle in novel environments.
            A filter tuned for suburban Phoenix degrades silently in downtown Tokyo.
        </div>
        """, unsafe_allow_html=True)

    with col_chart:
        # Animated degradation bar chart
        envs_all = ["Open Sky", "Urban Canyon", "Tunnel Exit", "Deep Urban"]
        errors_all = [3.5, 55, 70, 90]
        sats_all = [10, 3, 1, 2]
        cn0_all = [42, 22, 10, 15]
        colors_bar = [GREEN, AMBER, CORAL, "#E53935"]

        fig_deg = make_subplots(rows=1, cols=2, subplot_titles=(
            "Position Error by Environment", "Signal Quality"),
            horizontal_spacing=0.12,
        )

        # Error bars
        fig_deg.add_trace(go.Bar(
            x=envs_all, y=errors_all,
            marker_color=colors_bar, marker_line_width=0,
            text=[f"{e} m" for e in errors_all], textposition="outside",
            textfont=dict(color=TEXT_COLOR, size=11),
            name="Error (m)", showlegend=False,
        ), row=1, col=1)
        fig_deg.add_hline(y=5, line_dash="dash", line_color=GREEN, row=1, col=1,
                          annotation_text="5 m lane-level target",
                          annotation_font_color=GREEN)

        # Signal quality grouped bars
        fig_deg.add_trace(go.Bar(
            x=envs_all, y=sats_all, name="Satellites",
            marker_color=PINN_BLUE, text=sats_all, textposition="outside",
            textfont=dict(color=TEXT_COLOR, size=10),
        ), row=1, col=2)
        fig_deg.add_trace(go.Bar(
            x=envs_all, y=cn0_all, name="C/N₀ (dB-Hz)",
            marker_color=CYAN, text=cn0_all, textposition="outside",
            textfont=dict(color=TEXT_COLOR, size=10),
        ), row=1, col=2)

        fig_deg.update_layout(**dark_layout(height=420, barmode="group"))
        fig_deg.update_yaxes(title_text="Error (m)", row=1, col=1, gridcolor=GRID_COLOR)
        fig_deg.update_yaxes(title_text="Value", row=1, col=2, gridcolor=GRID_COLOR)
        fig_deg.update_xaxes(gridcolor=GRID_COLOR)
        # Highlight selected environment
        selected_idx = envs_all.index(environment) if environment in envs_all else 1
        fig_deg.add_annotation(
            x=envs_all[selected_idx], y=errors_all[selected_idx] + 8,
            text="▼ Selected", showarrow=False,
            font=dict(color=GOLD, size=12, family="Calibri"), row=1, col=1,
        )
        st.plotly_chart(fig_deg, use_container_width=True)

    # Degradation spectrum table
    with st.expander("Degradation Spectrum — Full Details", expanded=False):
        deg_df = pd.DataFrame({
            "Environment": envs_all,
            "Error Range": ["2–5 m", "10–100 m", "50–80 m", "70–100+ m"],
            "Satellites": sats_all,
            "C/N₀ (dB-Hz)": cn0_all,
            "Multipath": ["Minimal", "Severe — off buildings", "Moderate — exit zone", "Extreme — dense reflectors"],
            "Status": ["Acceptable", "Errors exceed lane width", "Partial signal recovery", "Classical methods fail"],
        })
        st.dataframe(deg_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 2 — Architecture Choice
# ══════════════════════════════════════════════
with tab2:
    st.subheader("Choose Architecture by Problem Structure, Not Hype")
    st.markdown(
        f"**Wrong architecture = 10× more data** to compensate for mismatched inductive bias. "
        f"Current λ = **{lambda_weight:.2f}**"
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("""
        <div class="danger-card">
            <h4 style="color:#EF553B; margin-top:0;">Standard NNs / Transformers</h4>
            <ul style="color:#E0E7EE;">
                <li>Purely data-driven sequence modeling</li>
                <li>Attention captures long-range temporal dependencies</li>
                <li>Requires massive, clean, labeled datasets</li>
            </ul>
            <p style="color:#FF6B6B;"><b>⚠ Critical Risk:</b> No physics guardrail — will predict physically impossible 50 m jumps from noise overfitting.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown(f"""
        <div class="info-card">
            <h4 style="color:#29B6F6; margin-top:0;">Physics-Informed NNs (PINNs)</h4>
            <ul style="color:#E0E7EE;">
                <li>Embeds kinematics & signal physics in loss function</li>
                <li>Up to 250× data-efficient in physics-rich regimes</li>
                <li>Generalizes across geographies — physics is universal</li>
            </ul>
            <p style="color:#FFB74D;"><b>⚠ Risks:</b> Incomplete physics = wrong constraint. Requires ML + GNSS domain staffing.</p>
        </div>
        """, unsafe_allow_html=True)

    # Equation bar
    st.markdown("""
    <div style="background:#0A2E50; border:1px solid #00BCD4; border-radius:8px; padding:10px 20px; text-align:center; margin:10px 0;">
        <span style="color:#E0E7EE; font-size:16px; font-style:italic;">L(θ) = L<sub>data</sub>(θ) + λ · L<sub>physics</sub>(θ)</span>
        <span style="color:#FFB74D; font-size:13px; margin-left:20px;"><b>λ validated against error budgets — not guesswork</b></span>
    </div>
    """, unsafe_allow_html=True)

    col_eff, col_lam = st.columns(2)

    with col_eff:
        # Data efficiency curves — animated with lambda
        data_sizes = np.logspace(2, 6, 80)
        nn_error = 80 * np.exp(-data_sizes / 150000) + 5 + np.random.normal(0, 1.0, 80)
        nn_error = np.clip(nn_error, 2, 90)

        effective_lambda = np.clip(lambda_weight, 0.1, 1.5)
        pinn_scale = 150000 / (1 + 200 * effective_lambda)
        pinn_error = 80 * np.exp(-data_sizes / pinn_scale) + 2 + np.random.normal(0, 0.7, 80)
        if lambda_weight < 0.1 or lambda_weight > 1.5:
            pinn_error += 10
        pinn_error = np.clip(pinn_error, 1.5, 90)

        fig_eff = go.Figure()
        fig_eff.add_trace(go.Scatter(
            x=data_sizes, y=nn_error, mode="lines",
            name="Traditional NN", line=dict(color=NN_RED, width=3),
        ))
        fig_eff.add_trace(go.Scatter(
            x=data_sizes, y=pinn_error, mode="lines",
            name=f"PINN (λ={lambda_weight:.2f})", line=dict(color=PINN_BLUE, width=3),
            fill="tonexty", fillcolor="rgba(41,182,246,0.08)",
        ))
        fig_eff.add_hline(y=5, line_dash="dash", line_color=GREEN,
                          annotation_text="5 m target", annotation_font_color=GREEN)

        # Efficiency annotation
        nn_10_idx = np.argmax(nn_error < 10) if np.any(nn_error < 10) else -1
        pinn_10_idx = np.argmax(pinn_error < 10) if np.any(pinn_error < 10) else -1
        if nn_10_idx > 0 and pinn_10_idx > 0:
            ratio = data_sizes[nn_10_idx] / data_sizes[pinn_10_idx]
            fig_eff.add_annotation(
                x=np.log10(data_sizes[pinn_10_idx]), y=12,
                text=f"~{ratio:.0f}× less data<br>for same accuracy",
                showarrow=True, arrowhead=2, arrowcolor=AMBER,
                font=dict(color=AMBER, size=11),
                bgcolor=CARD_BG, bordercolor=AMBER, borderwidth=1,
                ax=60, ay=-40,
            )

        fig_eff.update_layout(**dark_layout(
            height=400, title_text="Data Efficiency: NN vs PINN",
            xaxis_title="Training Samples (log scale)", yaxis_title="Position Error (m)",
            xaxis_type="log", yaxis_range=[0, 90],
        ))
        dark_axes(fig_eff)
        st.plotly_chart(fig_eff, use_container_width=True)

        # Metrics
        if nn_10_idx > 0 and pinn_10_idx > 0:
            m1, m2, m3 = st.columns(3)
            m1.metric("NN → <10 m", f"{data_sizes[nn_10_idx]:,.0f}")
            m2.metric("PINN → <10 m", f"{data_sizes[pinn_10_idx]:,.0f}")
            m3.metric("Efficiency", f"{ratio:,.0f}×")

    with col_lam:
        # Lambda sensitivity curve
        lambdas = np.linspace(0, 2.0, 150)
        lam_error = 3 + 40 * np.exp(-8 * lambdas) + 15 * np.maximum(0, lambdas - 0.8)**1.5
        lam_error += np.random.normal(0, 0.3, 150)
        lam_error = np.clip(lam_error, 1, 55)

        fig_lam = go.Figure()
        fig_lam.add_trace(go.Scatter(
            x=lambdas, y=lam_error, mode="lines",
            name="PINN Error", line=dict(color=PINN_BLUE, width=3),
            fill="tozeroy", fillcolor="rgba(41,182,246,0.08)",
        ))
        fig_lam.add_vrect(x0=0.3, x1=0.7, fillcolor=GREEN, opacity=0.1,
                          annotation_text="Optimal Zone", annotation_font_color=GREEN,
                          annotation_position="top")

        # Current lambda marker
        cur_idx = np.argmin(np.abs(lambdas - lambda_weight))
        fig_lam.add_trace(go.Scatter(
            x=[lambda_weight], y=[lam_error[cur_idx]], mode="markers",
            marker=dict(size=14, color=GOLD, symbol="star", line=dict(width=2, color="white")),
            name=f"Current λ={lambda_weight:.2f} → {lam_error[cur_idx]:.1f} m",
        ))

        # Danger zone annotations
        fig_lam.add_annotation(x=0.05, y=40, text="λ too low<br>→ No physics<br>guardrail",
                               showarrow=False, font=dict(color=CORAL, size=10))
        fig_lam.add_annotation(x=1.7, y=40, text="λ too high<br>→ Ignores data<br>anomalies",
                               showarrow=False, font=dict(color=CORAL, size=10))

        fig_lam.update_layout(**dark_layout(
            height=400, title_text="PINN Error vs λ (Physics Loss Weight)",
            xaxis_title="λ", yaxis_title="Position Error (m)", yaxis_range=[0, 55],
        ))
        dark_axes(fig_lam)
        st.plotly_chart(fig_lam, use_container_width=True)

        # Lambda status
        if lambda_weight < 0.1:
            st.markdown('<div class="danger-card"><b>⚠ λ too low</b> — physics guardrail effectively disabled.</div>', unsafe_allow_html=True)
        elif lambda_weight > 1.2:
            st.markdown('<div class="warn-card"><b>⚠ λ too high</b> — model over-constrains to physics, ignores valid data.</div>', unsafe_allow_html=True)
        elif 0.3 <= lambda_weight <= 0.7:
            st.markdown('<div class="success-card"><b>✓ λ within optimal range.</b> Validate against error budgets on held-out trajectories.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-card"><b>ℹ λ outside optimal zone</b> but may be acceptable — validate carefully.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — Data Reality
# ══════════════════════════════════════════════
with tab3:
    st.subheader("Data Reality: Quality, Missingness, and Leakage")
    st.markdown("**Data is signal, not fuel.** 80% of model failures trace to data, not architecture.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="success-card">
            <h4 style="color:#66BB6A; margin-top:0;">📊 Quality > Quantity</h4>
            <p style="color:#E0E7EE;">10,000 clean samples at 1.5 bits each outweigh 1,000,000 noisy samples at 0.2 bits.</p>
            <p style="color:#66BB6A;"><b>Stop collecting. Start auditing.</b></p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="warn-card">
            <h4 style="color:#FFB74D; margin-top:0;">⚠ MNAR Missingness</h4>
            <p style="color:#E0E7EE;">Signal drops in tunnels = informative features, not noise. Standard imputation destroys this signal.</p>
            <p style="color:#FFB74D;"><b>Preserve dropout patterns as features.</b></p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="danger-card">
            <h4 style="color:#FF6B6B; margin-top:0;">⛔ Temporal Leakage</h4>
            <p style="color:#E0E7EE;">Training on post-processed RTK or future ephemeris = model cheats. Deployed accuracy collapses.</p>
            <p style="color:#FF6B6B;"><b>Strict temporal splits. Non-negotiable.</b></p>
        </div>
        """, unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        # Interactive information value comparison
        st.markdown("#### Information Value: Clean vs Noisy Data")
        noise_level = st.slider("Noise level (bits/sample for noisy data)", 0.05, 1.0, 0.2, 0.05, key="noise")
        clean_samples = 10_000
        clean_bits = 1.5
        noisy_samples = 1_000_000
        noisy_bits = noise_level

        fig_info = go.Figure()
        fig_info.add_trace(go.Bar(
            x=["10K Clean Samples", "1M Noisy Samples"],
            y=[clean_samples * clean_bits, noisy_samples * noisy_bits],
            marker_color=[GREEN, CORAL],
            text=[f"{clean_samples * clean_bits:,.0f} bits", f"{noisy_samples * noisy_bits:,.0f} bits"],
            textposition="outside", textfont=dict(color=TEXT_COLOR, size=13),
        ))

        # Add sample count annotation
        ratio_info = (noisy_samples * noisy_bits) / (clean_samples * clean_bits)
        verdict = "Noisy wins on volume" if ratio_info > 1 else "Clean wins on quality"
        fig_info.update_layout(**dark_layout(
            height=350, title_text="Total Information Content",
            yaxis_title="Total Bits of Information",
            annotations=[dict(
                x=0.5, y=1.05, xref="paper", yref="paper", showarrow=False,
                text=f"At {noisy_bits:.2f} bits/sample: {verdict}",
                font=dict(color=AMBER, size=12),
            )]
        ))
        dark_axes(fig_info)
        st.plotly_chart(fig_info, use_container_width=True)

    with col_right:
        # Temporal leakage timeline — animated
        st.markdown("#### Temporal Leakage Timeline")
        fig_leak = go.Figure()

        # Training window
        fig_leak.add_trace(go.Bar(
            x=[6], y=["Timeline"], orientation="h",
            base=[0], marker_color="rgba(10,62,32,0.7)",
            marker_line=dict(color=GREEN, width=2),
            name="Training Window", text=["Training Data (historical)"],
            textposition="inside", textfont=dict(color=GREEN, size=12),
        ))
        # Inference window
        fig_leak.add_trace(go.Bar(
            x=[4], y=["Timeline"], orientation="h",
            base=[6], marker_color="rgba(62,21,21,0.7)",
            marker_line=dict(color=CORAL, width=2),
            name="Inference Window", text=["Future / Deployment"],
            textposition="inside", textfont=dict(color=CORAL, size=12),
        ))

        # Leakage arrow
        fig_leak.add_annotation(
            x=5, y="Timeline", ax=7.5, ay="Timeline",
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=3,
            arrowcolor=CORAL,
        )
        fig_leak.add_annotation(
            x=6.25, y="Timeline", text="⚠ LEAKAGE", showarrow=False,
            font=dict(color=CORAL, size=14, family="Calibri"),
            bgcolor="#3E1515", bordercolor=CORAL, borderwidth=1,
            yshift=-40,
        )
        # Boundary
        fig_leak.add_vline(x=6, line_color=AMBER, line_width=3)
        fig_leak.add_annotation(x=6, y="Timeline", text="▼ Temporal Boundary",
                                showarrow=False, yshift=40,
                                font=dict(color=AMBER, size=12))

        fig_leak.update_layout(**dark_layout(
            height=250, title_text="Temporal Leakage — The Silent Killer",
            xaxis_title="Time", showlegend=True, barmode="stack",
        ))
        fig_leak.update_xaxes(showticklabels=False, gridcolor=GRID_COLOR)
        fig_leak.update_yaxes(showticklabels=False, gridcolor=GRID_COLOR)
        st.plotly_chart(fig_leak, use_container_width=True)

        # MNAR simulation
        st.markdown("#### MNAR: Signal Dropout is Information")
        t = np.linspace(0, 60, 300)
        signal = 35 + 5 * np.sin(0.3 * t) + np.random.normal(0, 2, 300)
        # Tunnel entry between t=20 and t=35
        tunnel_mask = (t > 20) & (t < 35)
        signal[tunnel_mask] = np.nan

        fig_mnar = go.Figure()
        fig_mnar.add_trace(go.Scatter(
            x=t, y=signal, mode="lines", name="C/N₀ Signal",
            line=dict(color=CYAN, width=2),
            connectgaps=False,
        ))
        fig_mnar.add_vrect(x0=20, x1=35, fillcolor=CORAL, opacity=0.15,
                           annotation_text="Tunnel (MNAR dropout)",
                           annotation_font_color=CORAL,
                           annotation_position="top")
        fig_mnar.update_layout(**dark_layout(
            height=250, title_text="Signal Dropout ≠ Random Noise",
            xaxis_title="Time (s)", yaxis_title="C/N₀ (dB-Hz)",
        ))
        dark_axes(fig_mnar)
        st.plotly_chart(fig_mnar, use_container_width=True)

    # Audit checklist
    st.markdown("""
    <div class="info-card">
        <b>Pre-Training Audit Requirement:</b>
        <span style="color:#00BCD4;">Completeness · Consistency · Accuracy · Timeliness · Relevance</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 4 — Silent Killers
# ══════════════════════════════════════════════
with tab4:
    st.subheader("Silent Killers: Drift, Shift, and Distribution Collapse")
    st.markdown(f"**Distribution is destiny.** IID assumptions fail in physical systems. Deployment: **{drift_months} months**.")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("""
        <div class="danger-card">
            <h4 style="color:#FF6B6B; margin-top:0;">⚠ Covariate Shift (Environmental)</h4>
            <p style="color:#E0E7EE;"><b>The Trap:</b> Model trained on suburban open-sky routes is deployed downtown.</p>
            <p style="color:#E0E7EE;"><b>The Physical Reality:</b> Satellite geometry changes. Multipath is radically different. Rain alters signal propagation. The input distribution shifts — the model doesn't know.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="danger-card">
            <h4 style="color:#FF6B6B; margin-top:0;">⚠ Calibration Drift (Hardware)</h4>
            <p style="color:#E0E7EE;"><b>The Trap:</b> Receiver clocks accumulate error over months.</p>
            <p style="color:#E0E7EE;"><b>The Physical Reality:</b> Antenna gain degrades with weather exposure. Firmware updates alter pseudorange calculations. The hardware the model trained on no longer exists.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        # Distribution shift — animated with deployment months
        x_range = np.linspace(-5, 10, 400)
        train_mean = 0
        deploy_mean = 0.15 * drift_months  # shifts with deployment duration
        deploy_std = 1 + 0.05 * drift_months

        train_dist = np.exp(-0.5 * (x_range - train_mean)**2) / np.sqrt(2 * np.pi)
        deploy_dist = np.exp(-0.5 * ((x_range - deploy_mean)/deploy_std)**2) / (deploy_std * np.sqrt(2 * np.pi))

        fig_shift = go.Figure()
        fig_shift.add_trace(go.Scatter(
            x=x_range, y=train_dist, mode="lines", name="Training Distribution",
            line=dict(color=PINN_BLUE, width=3), fill="tozeroy",
            fillcolor="rgba(41,182,246,0.2)",
        ))
        fig_shift.add_trace(go.Scatter(
            x=x_range, y=deploy_dist, mode="lines", name=f"Deployment ({drift_months}mo)",
            line=dict(color=CORAL, width=3), fill="tozeroy",
            fillcolor="rgba(255,107,107,0.2)",
        ))

        # MMD zone
        overlap_start = max(train_mean - 2, deploy_mean - 2 * deploy_std)
        overlap_end = min(train_mean + 2, deploy_mean + 2 * deploy_std)
        fig_shift.add_vrect(x0=overlap_start, x1=overlap_end + 2,
                            fillcolor=CORAL, opacity=0.06)

        # Shift arrow
        fig_shift.add_annotation(
            x=deploy_mean, y=max(deploy_dist) * 0.9,
            ax=train_mean, ay=max(train_dist) * 0.9,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=2.5,
            arrowcolor=AMBER,
        )
        fig_shift.add_annotation(
            x=(train_mean + deploy_mean) / 2, y=0.45,
            text=f"Covariate Shift<br>({drift_months} months)",
            showarrow=False, font=dict(color=AMBER, size=12),
        )

        # MMD metric
        mmd_value = 0.05 * drift_months + 0.02 * drift_months**0.5
        threshold = 0.5
        fig_shift.add_annotation(
            x=deploy_mean + 2, y=0.08,
            text=f"MMD = {mmd_value:.2f}<br>{'⚠ > threshold!' if mmd_value > threshold else '✓ within bounds'}",
            showarrow=False,
            font=dict(color=CORAL if mmd_value > threshold else GREEN, size=11),
            bgcolor=CARD_BG, bordercolor=CORAL if mmd_value > threshold else GREEN,
            borderwidth=1,
        )

        fig_shift.update_layout(**dark_layout(
            height=400, title_text="Distribution Shift Detection (MMD)",
            xaxis_title="Feature Space", yaxis_title="Density",
        ))
        dark_axes(fig_shift)
        st.plotly_chart(fig_shift, use_container_width=True)

    # Calibration drift simulation
    st.markdown("#### Calibration Drift Over Time")
    months = np.arange(1, drift_months + 1)
    clock_drift = np.cumsum(np.random.normal(0.3, 0.15, drift_months))
    antenna_degradation = 100 - 0.4 * months - np.random.normal(0, 0.3, drift_months)
    firmware_events = [6, 12, 18] if drift_months >= 6 else []

    fig_drift = make_subplots(rows=1, cols=2,
                               subplot_titles=("Clock Error Accumulation", "Antenna Gain Degradation"))

    fig_drift.add_trace(go.Scatter(
        x=months, y=clock_drift, mode="lines+markers",
        name="Clock Drift (ns)", line=dict(color=AMBER, width=2),
        marker=dict(size=5, color=AMBER),
    ), row=1, col=1)

    fig_drift.add_trace(go.Scatter(
        x=months, y=antenna_degradation, mode="lines+markers",
        name="Antenna Gain (%)", line=dict(color=TEAL, width=2),
        marker=dict(size=5, color=TEAL),
    ), row=1, col=2)

    for fw in firmware_events:
        if fw <= drift_months:
            fig_drift.add_vline(x=fw, line_dash="dash", line_color=CORAL, row=1, col=1)
            fig_drift.add_vline(x=fw, line_dash="dash", line_color=CORAL, row=1, col=2)
            fig_drift.add_annotation(x=fw, y=max(clock_drift) * 0.9, text="FW Update",
                                     font=dict(color=CORAL, size=9), showarrow=False, row=1, col=1)

    fig_drift.update_layout(**dark_layout(height=350))
    fig_drift.update_xaxes(title_text="Months Deployed", gridcolor=GRID_COLOR)
    fig_drift.update_yaxes(gridcolor=GRID_COLOR)
    st.plotly_chart(fig_drift, use_container_width=True)

    # Defense protocol
    st.markdown("""
    <div class="success-card">
        <h4 style="color:#009688; margin-top:0;">🛡 Defense Protocol</h4>
        <p style="color:#E0E7EE;"><b>Deploy MMD Monitors</b> on input feature pipelines. Continuously compare live distributions against training data.</p>
        <p style="color:#E0E7EE;">When statistical distance exceeds threshold → <b>trigger retraining</b> or <b>fallback to classical Kalman filtering</b> for immediate safety.</p>
        <p style="color:#009688;"><b>→ Silent failures become detectable failures.</b></p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 5 — Compute Economics
# ══════════════════════════════════════════════
with tab5:
    st.subheader("Compute Economics: Why FLOPs Lie")
    st.markdown(
        "Matrix multiplication dominates FLOPs — but **FLOPs ≠ wall-clock speed**. "
        f"Current model: **{model_size}M parameters**."
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(f"""
        <div class="info-card">
            <h4 style="color:#29B6F6; margin-top:0;">Training (CAPEX — One-Time)</h4>
            <ul style="color:#E0E7EE;">
                <li>Compute-bound — parallelizable across GPUs</li>
                <li>≈ 6N FLOPs/token | ~12 bytes/param</li>
                <li>{model_size}M model = <b>{12 * model_size / 1000:.1f} GB</b> training state</li>
            </ul>
            <p style="color:#FFB74D;"><b>PINN overhead:</b> +20–40%/epoch for physics loss — but converges with far less data, so total cost is often lower.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="danger-card">
            <h4 style="color:#FF6B6B; margin-top:0;">Inference (OPEX — Ongoing)</h4>
            <ul style="color:#E0E7EE;">
                <li>Memory-bandwidth-bound — GPU idle waiting for weights</li>
                <li>≈ 2N FLOPs/token — but latency, not throughput, matters</li>
                <li>10 Hz GNSS = <100 ms per correction</li>
                <li>Scales linearly with every vehicle in the fleet</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    col_train, col_roof = st.columns(2)

    with col_train:
        # Training cost comparison
        sizes = np.arange(5, 205, 5)
        nn_flops = 6 * sizes  # GFLOPs simplified
        pinn_flops = nn_flops * 1.3
        mem_gb = 12 * sizes * 1e6 / 1e9

        fig_cost = make_subplots(specs=[[{"secondary_y": True}]])

        fig_cost.add_trace(go.Bar(
            x=[f"{s}M" for s in sizes[::4]], y=nn_flops[::4],
            name="NN (6N FLOPs)", marker_color=NN_RED, opacity=0.8,
        ), secondary_y=False)
        fig_cost.add_trace(go.Bar(
            x=[f"{s}M" for s in sizes[::4]], y=pinn_flops[::4],
            name="PINN (7.8N, +30%)", marker_color=PINN_BLUE, opacity=0.8,
        ), secondary_y=False)
        fig_cost.add_trace(go.Scatter(
            x=[f"{s}M" for s in sizes[::4]], y=mem_gb[::4],
            name="Memory (GB)", mode="lines+markers",
            line=dict(color=AMBER, width=2.5), marker=dict(size=7, color=AMBER),
        ), secondary_y=True)

        # Mark selected
        sel_idx = np.argmin(np.abs(sizes - model_size))
        fig_cost.add_trace(go.Scatter(
            x=[f"{model_size}M"], y=[nn_flops[sel_idx]], mode="markers",
            marker=dict(size=16, color=GOLD, symbol="star"), name=f"Selected: {model_size}M",
            showlegend=True,
        ), secondary_y=False)

        fig_cost.update_layout(**dark_layout(
            height=400, title_text="Training Cost: NN vs PINN",
            barmode="group",
        ))
        fig_cost.update_xaxes(title_text="Model Size", gridcolor=GRID_COLOR)
        fig_cost.update_yaxes(title_text="GFLOPs/Token", gridcolor=GRID_COLOR, secondary_y=False)
        fig_cost.update_yaxes(title_text="Memory (GB)", gridcolor=GRID_COLOR, secondary_y=True,
                              title_font=dict(color=AMBER), tickfont=dict(color=AMBER))
        st.plotly_chart(fig_cost, use_container_width=True)

    with col_roof:
        # Roofline model
        ai = np.logspace(-2, 2, 500)
        bw_limit = 900
        peak_flops = 10000
        roofline = np.minimum(peak_flops, bw_limit * ai)

        fig_roof = go.Figure()
        fig_roof.add_trace(go.Scatter(
            x=ai, y=roofline, mode="lines", name="Hardware Roofline",
            line=dict(color=CYAN, width=3), fill="tozeroy",
            fillcolor="rgba(0,188,212,0.05)",
        ))
        # Training point
        fig_roof.add_trace(go.Scatter(
            x=[15], y=[peak_flops * 0.7], mode="markers",
            marker=dict(size=14, color=GREEN, symbol="circle",
                        line=dict(width=2, color="white")),
            name="Training (compute-bound)",
        ))
        # Inference point
        fig_roof.add_trace(go.Scatter(
            x=[0.5], y=[450], mode="markers",
            marker=dict(size=14, color=CORAL, symbol="circle",
                        line=dict(width=2, color="white")),
            name="Inference (bandwidth-bound)",
        ))

        fig_roof.update_layout(**dark_layout(
            height=400, title_text="Roofline Model — Why FLOPs Lie",
            xaxis_title="Arithmetic Intensity (FLOPs/Byte)",
            yaxis_title="Performance (GFLOPS)",
            xaxis_type="log", yaxis_type="log",
        ))
        dark_axes(fig_roof)
        st.plotly_chart(fig_roof, use_container_width=True)

    # Key insight
    st.markdown("""
    <div class="info-card">
        <b>Key insight:</b> Training dominates CAPEX. Inference dominates OPEX.
        For fleet-scale GNSS, <span style="color:#FF6B6B;"><b>inference cost is the binding constraint</b></span>.
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    sel_flops_nn = 6 * model_size * 1e6 / 1e9
    sel_flops_pinn = sel_flops_nn * 1.3
    sel_mem = 12 * model_size * 1e6 / 1e9
    m1.metric("Training FLOPs/token (NN)", f"{sel_flops_nn:.1f} GFLOPs")
    m2.metric("Training FLOPs/token (PINN)", f"{sel_flops_pinn:.1f} GFLOPs")
    m3.metric("Training memory", f"{sel_mem:.2f} GB")


# ══════════════════════════════════════════════
# TAB 6 — Edge Deployment
# ══════════════════════════════════════════════
with tab6:
    st.subheader("Solving the Inference Bottleneck for Edge Deployment")
    st.markdown(
        f"Below the compute roofline — spare FLOPs, not spare bandwidth. "
        f"**{precision}** | **Batch: {batch_size} vehicles**"
    )

    # Pipeline flow visualization
    st.markdown("""
    <div style="display:flex; gap:10px; justify-content:center; margin:15px 0;">
        <div style="background:#142A3E; border:2px solid #29B6F6; border-radius:10px; padding:15px 25px; text-align:center; flex:1;">
            <p style="color:#29B6F6; font-weight:bold; margin:0;">GPU Memory</p>
            <p style="color:#A0AEBB; margin:0; font-size:12px;">Large footprint — Weights</p>
        </div>
        <div style="display:flex; align-items:center; color:#00BCD4; font-size:28px;">→</div>
        <div style="background:#3E250A; border:2px solid #FFB74D; border-radius:10px; padding:15px 25px; text-align:center; flex:1;">
            <p style="color:#FFB74D; font-weight:bold; margin:0;">Transfer to Cores</p>
            <p style="color:#A0AEBB; margin:0; font-size:12px;">THE BOTTLENECK</p>
        </div>
        <div style="display:flex; align-items:center; color:#00BCD4; font-size:28px;">→</div>
        <div style="background:#0A3520; border:2px solid #66BB6A; border-radius:10px; padding:15px 25px; text-align:center; flex:1;">
            <p style="color:#66BB6A; font-weight:bold; margin:0;">Matrix Multiply</p>
            <p style="color:#A0AEBB; margin:0; font-size:12px;">~2N FLOPs (fast, spare capacity)</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_quant, col_lat = st.columns(2)

    with col_quant:
        # Quantization chart
        quant_levels = ["FP32", "FP16", "INT8", "INT4"]
        mem_red = [1, 2, 4, 8]
        nn_acc = [100.0, 99.8, 99.2, 96.5]
        pinn_acc = [100.0, 99.9, 99.5, 97.0]

        fig_quant = go.Figure()
        fig_quant.add_trace(go.Bar(
            x=quant_levels, y=nn_acc, name="Traditional NN",
            marker_color=NN_RED, opacity=0.85,
            text=[f"{a:.1f}%" for a in nn_acc], textposition="outside",
            textfont=dict(color=TEXT_COLOR, size=10),
        ))
        fig_quant.add_trace(go.Bar(
            x=quant_levels, y=pinn_acc, name="PINN",
            marker_color=PINN_BLUE, opacity=0.85,
            text=[f"{a:.1f}%" for a in pinn_acc], textposition="outside",
            textfont=dict(color=TEXT_COLOR, size=10),
        ))
        fig_quant.add_hline(y=99, line_dash="dash", line_color=GREEN,
                            annotation_text="99% threshold", annotation_font_color=GREEN)

        # Memory reduction labels
        for i, (ql, mr) in enumerate(zip(quant_levels, mem_red)):
            fig_quant.add_annotation(x=ql, y=95.3, text=f"{mr}×",
                                     showarrow=False, font=dict(color=AMBER, size=12, family="Calibri"))

        fig_quant.update_layout(**dark_layout(
            height=420, title_text="Quantization: Accuracy vs Memory Reduction",
            xaxis_title="Precision Level", yaxis_title="Accuracy Retained (%)",
            yaxis_range=[94.5, 101], barmode="group",
        ))
        dark_axes(fig_quant)

        # Highlight selected precision
        sel_quant = precision.split(" ")[0]
        for i, ql in enumerate(quant_levels):
            if ql == sel_quant:
                fig_quant.add_annotation(x=ql, y=101, text="▼ Selected",
                                         showarrow=False, font=dict(color=GOLD, size=11))
        st.plotly_chart(fig_quant, use_container_width=True)

    with col_lat:
        # Inference latency — reactive to precision + batch
        sizes_lat = np.arange(5, 205, 5)
        if "FP32" in precision:
            bpp = 4.0
        elif "FP16" in precision:
            bpp = 2.0
        elif "INT8" in precision:
            bpp = 1.0
        else:
            bpp = 0.5

        bw_gbs = 900
        model_bytes = sizes_lat * 1e6 * bpp
        base_lat = (model_bytes / (bw_gbs * 1e9)) * 1000
        compute_lat = (2 * sizes_lat * 1e6) / (10e12) * 1000
        batch_factor = 1 + 0.15 * np.log2(max(batch_size, 1))
        total_lat = (base_lat + compute_lat) * batch_factor + 2.0
        total_lat += np.random.normal(0, 0.2, len(sizes_lat))
        total_lat = np.clip(total_lat, 0.5, 500)

        fig_lat = go.Figure()
        fig_lat.add_trace(go.Scatter(
            x=sizes_lat, y=total_lat, mode="lines",
            name=f"{precision.split(' ')[0]} — batch {batch_size}",
            line=dict(color=PINN_BLUE, width=3),
            fill="tozeroy", fillcolor="rgba(41,182,246,0.08)",
        ))
        fig_lat.add_hline(y=100, line_dash="dash", line_color=CORAL,
                          annotation_text="100 ms deadline (10 Hz)", annotation_font_color=CORAL)
        fig_lat.add_hline(y=50, line_dash="dot", line_color=AMBER,
                          annotation_text="50 ms safety margin", annotation_font_color=AMBER)

        # Mark selected
        idx_sel = np.argmin(np.abs(sizes_lat - model_size))
        meets = total_lat[idx_sel] < 100
        fig_lat.add_trace(go.Scatter(
            x=[model_size], y=[total_lat[idx_sel]], mode="markers",
            marker=dict(size=14, color=GREEN if meets else CORAL, symbol="star",
                        line=dict(width=2, color="white")),
            name=f"{model_size}M → {total_lat[idx_sel]:.1f} ms {'✓' if meets else '✗'}",
        ))

        fig_lat.update_layout(**dark_layout(
            height=420, title_text="Inference Latency",
            xaxis_title="Model Size (M params)", yaxis_title="Latency (ms)",
            yaxis_range=[0, max(150, total_lat[idx_sel] * 1.3)],
        ))
        dark_axes(fig_lat)
        st.plotly_chart(fig_lat, use_container_width=True)

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latency", f"{total_lat[idx_sel]:.1f} ms")
    m2.metric("Meets 100 ms?", "Yes ✓" if meets else "No ✗")
    fleet_cps = (1000 / max(total_lat[idx_sel], 1)) * batch_size
    m3.metric("Fleet corrections/sec", f"{fleet_cps:,.0f}")
    m4.metric("Memory reduction", f"{int(4/bpp)}× vs FP32")

    # Summary bars
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div class="{'success-card' if meets else 'danger-card'}">
            <b>{'✓ INT8 Default:' if 'INT8' in precision else '⚠ ' + precision.split(' ')[0] + ':'}</b>
            {int(4/bpp)}× memory reduction.
            {'Meets real-time deadline.' if meets else 'Exceeds latency budget — consider stronger quantization.'}
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="info-card">
            <b>Parallelism + Batching =</b> Fleet-scale edge deployment without cloud dependency.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 7 — Head-to-Head
# ══════════════════════════════════════════════
with tab7:
    st.subheader("Head-to-Head: Traditional NN vs PINN")
    st.markdown("Summary comparison across all decision dimensions.")

    col_table, col_viz = st.columns([3, 2])

    with col_table:
        comparison_data = {
            "Dimension": [
                "Data efficiency", "Physics compliance", "Failure mode",
                "Training (CAPEX)", "Inference (OPEX)", "Edge deployment",
                "Shift resilience", "Drift detection", "Domain expertise",
                "Generalization", "Safety guarantee", "Best fit",
            ],
            "Traditional NN / Transformer": [
                "Low — large labeled datasets", "None — impossible outputs possible",
                "Silent 50 m jumps", "Standard ~6N FLOPs/token",
                "~2N FLOPs, bandwidth-bound", "Viable with INT8/INT4",
                "No guardrail when inputs drift", "Needs external MMD monitors",
                "ML engineering", "Poor without local retraining",
                "None", "Abundant RTK data, stable env",
            ],
            "PINN (Physics-Informed)": [
                "High — up to 250× less data", "Built-in — violations penalized",
                "Bounded by physics model quality", "+20–40%/epoch, less data needed",
                "Identical — physics is training-only", "Equally viable — same inference graph",
                "Physics limits damage, won't detect drift", "Needs external MMD monitors",
                "ML + physics/GNSS domain", "Strong — physics transfers globally",
                "Soft — bounded by physics quality", "Scarce data, variable env, safety-critical",
            ],
        }
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True, height=480)

    with col_viz:
        # Radar chart — multi-dimensional comparison
        categories = ["Data\nEfficiency", "Physics\nCompliance", "Safety",
                       "Generalization", "Ease of\nImplementation", "Edge\nViability"]
        nn_scores = [2, 1, 1, 2, 5, 4]
        pinn_scores = [5, 5, 4, 5, 3, 4]
        kalman_scores = [3, 3, 3, 2, 4, 5]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=nn_scores + [nn_scores[0]], theta=categories + [categories[0]],
            name="Traditional NN", line=dict(color=NN_RED, width=2.5),
            fill="toself", fillcolor="rgba(239,83,59,0.15)",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=pinn_scores + [pinn_scores[0]], theta=categories + [categories[0]],
            name="PINN (Hybrid)", line=dict(color=PINN_BLUE, width=2.5),
            fill="toself", fillcolor="rgba(41,182,246,0.15)",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=kalman_scores + [kalman_scores[0]], theta=categories + [categories[0]],
            name="Kalman + INS", line=dict(color=KALMAN_AMBER, width=2),
            fill="toself", fillcolor="rgba(255,183,77,0.1)",
        ))

        fig_radar.update_layout(
            polar=dict(
                bgcolor=CARD_BG,
                radialaxis=dict(visible=True, range=[0, 5], gridcolor=GRID_COLOR,
                                tickfont=dict(color=MUTED, size=9)),
                angularaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TEXT_COLOR, size=10)),
            ),
            **dark_layout(height=380, title_text="Multi-Dimensional Comparison"),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Environment performance bar chart
        envs_comp = ["Open Sky", "Urban Canyon", "Tunnel Exit", "Deep Urban"]
        nn_errs = [4.5, 45, 85, 95]
        pinn_errs = [2.5, 8, 25, 35]
        kalman_errs = [3.5, 35, 70, 90]

        fig_env = go.Figure()
        fig_env.add_trace(go.Bar(x=envs_comp, y=kalman_errs, name="Kalman + INS",
                                  marker_color=KALMAN_AMBER, opacity=0.85))
        fig_env.add_trace(go.Bar(x=envs_comp, y=nn_errs, name="Traditional NN",
                                  marker_color=NN_RED, opacity=0.85))
        fig_env.add_trace(go.Bar(x=envs_comp, y=pinn_errs, name="PINN (Hybrid)",
                                  marker_color=PINN_BLUE, opacity=0.85))
        fig_env.add_hline(y=5, line_dash="dash", line_color=GREEN,
                          annotation_text="5 m target", annotation_font_color=GREEN)

        # Highlight selected env
        if environment in envs_comp:
            sel_i = envs_comp.index(environment)
            fig_env.add_annotation(x=envs_comp[sel_i], y=max(nn_errs[sel_i], kalman_errs[sel_i]) + 8,
                                   text="▼ Selected", showarrow=False,
                                   font=dict(color=GOLD, size=11))

        fig_env.update_layout(**dark_layout(
            height=380, title_text="Positioning Error by Environment",
            yaxis_title="Horizontal Error (m)", barmode="group",
        ))
        dark_axes(fig_env)
        st.plotly_chart(fig_env, use_container_width=True)

    # Key insights
    st.markdown("""
    <div class="info-card">
        <b>Key Insights:</b><br>
        • Inference cost is identical — physics loss is training-only. PINN vs NN doesn't affect OPEX.<br>
        • Both need external MMD monitors — physics constraints don't detect distribution shift.<br>
        • PINN advantage widens in degraded environments where data is worst.<br>
        • Generalization: physics transfers across geographies; pure NNs need local retraining.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 8 — Verdict
# ══════════════════════════════════════════════
with tab8:
    st.subheader("Verdict: Hybrid PINN — With Conditions")

    st.markdown("""
    <div class="success-card">
        <h4 style="color:#66BB6A; margin-top:0;">✓ Recommendation</h4>
        <p style="color:#E0E7EE; font-size:16px;">Adopt <b>Hybrid Physics-Informed Architecture</b> for degraded GNSS environments.</p>
    </div>
    """, unsafe_allow_html=True)

    col_when, col_plan = st.columns(2)

    with col_when:
        st.markdown("""
        <div class="info-card">
            <h4 style="color:#29B6F6; margin-top:0;">Use PINNs / Hybrid When:</h4>
            <ul style="color:#E0E7EE;">
                <li>Data is scarce, noisy, or MNAR</li>
                <li>Kinematics & signal physics are well-characterized</li>
                <li>Safety-critical — no impossible 50 m jumps</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="danger-card">
            <h4 style="color:#EF553B; margin-top:0;">Use Pure Transformers When:</h4>
            <ul style="color:#E0E7EE;">
                <li>Abundant RTK-corrected logs available</li>
                <li>Multipath complexity exceeds available physics models</li>
                <li>Environment is mapped and stable</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="warn-card">
            <h4 style="color:#FFB74D; margin-top:0;">🔒 Non-Negotiable Conditions:</h4>
            <ol style="color:#E0E7EE;">
                <li><b>(a)</b> Validate λ against positioning error budgets on held-out trajectories</li>
                <li><b>(b)</b> Eliminate temporal leakage from all training pipelines</li>
                <li><b>(c)</b> Deploy MMD drift monitors from Day 1 — not phase two</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with col_plan:
        # 10-week action plan — Gantt chart
        plan_data = pd.DataFrame([
            dict(Task="Data Audit (5 dims)", Owner="Data Engineering", Start=1, End=2,
                 Deliverable="Readiness scorecard", Color=GREEN),
            dict(Task="Baseline PDEs", Owner="Physics / Signal", Start=2, End=4,
                 Deliverable="Validated loss module", Color=PINN_BLUE),
            dict(Task="Train Prototype", Owner="ML Team", Start=4, End=8,
                 Deliverable="Benchmark vs Kalman", Color=CYAN),
            dict(Task="MMD Monitors", Owner="MLOps", Start=1, End=10,
                 Deliverable="Drift alert dashboard", Color=AMBER),
            dict(Task="INT8/INT4 Quantization", Owner="ML / Edge", Start=8, End=10,
                 Deliverable="Latency test on edge HW", Color=TEAL),
        ])

        fig_gantt = go.Figure()
        for i, row in plan_data.iterrows():
            fig_gantt.add_trace(go.Bar(
                x=[row["End"] - row["Start"]],
                y=[row["Task"]],
                base=[row["Start"]],
                orientation="h",
                marker_color=row["Color"],
                marker_line=dict(color="white", width=1),
                text=f"Wk {row['Start']}–{row['End']} | {row['Owner']}",
                textposition="inside",
                textfont=dict(color="white", size=10),
                name=row["Task"],
                showlegend=False,
                hovertemplate=f"<b>{row['Task']}</b><br>Owner: {row['Owner']}<br>Weeks: {row['Start']}–{row['End']}<br>→ {row['Deliverable']}<extra></extra>",
            ))

        fig_gantt.update_layout(**dark_layout(
            height=350, title_text="10-Week Action Plan",
            xaxis_title="Week", barmode="stack",
        ))
        fig_gantt.update_xaxes(gridcolor=GRID_COLOR, range=[0, 11],
                                dtick=1, tickvals=list(range(11)))
        fig_gantt.update_yaxes(gridcolor=GRID_COLOR, autorange="reversed")
        st.plotly_chart(fig_gantt, use_container_width=True)

        # Deliverables table
        with st.expander("Deliverables Detail", expanded=True):
            plan_display = plan_data[["Task", "Owner", "Deliverable"]].copy()
            plan_display.insert(0, "Week", [f"Wk {r['Start']}–{r['End']}" for _, r in plan_data.iterrows()])
            st.dataframe(plan_display, use_container_width=True, hide_index=True)

    # Closing bar
    st.markdown("""
    <div style="background:#142A3E; border:1px solid #00BCD4; border-radius:10px; padding:12px 24px; text-align:center; margin-top:15px;">
        <span style="color:#00BCD4; font-size:15px;"><b>The path from hype to engineering:</b></span>
        <span style="color:#E0E7EE; font-size:14px;">
            audit first → physics module → prototype → drift monitors (day 1) → quantization.
        </span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
st.caption(
    "Assignment 2 — AI & Large Models | "
    "Physics-Informed GNSS Positioning Error Correction in Degraded Environments | "
    "All data is synthetic/mock for illustration purposes."
)
