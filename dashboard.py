# -*- coding: utf-8 -*-
"""
GNSS Error Correction -- Data & Compute Trade-off Dashboard
Assignment 2: Physics-Informed GNSS Positioning in Degraded Environments

Run:  streamlit run dashboard.py
Requires:  pip install streamlit plotly numpy pandas
"""

import streamlit as st
import streamlit.components.v1 as components
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
# Design tokens
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

def dark_layout(**kwargs):
    base = dict(
        paper_bgcolor=PLOT_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_COLOR, family="Calibri, sans-serif", size=13),
        legend=dict(bgcolor="rgba(20,42,62,0.85)", bordercolor=GRID_COLOR, borderwidth=1),
        margin=dict(l=60, r=30, t=50, b=50),
    )
    base.update(kwargs)
    return base

def dark_axes(fig):
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    return fig

def scene_dark():
    return dict(
        bgcolor=PLOT_BG,
        xaxis=dict(backgroundcolor=CARD_BG, gridcolor=GRID_COLOR, showbackground=True, color=TEXT_COLOR),
        yaxis=dict(backgroundcolor=CARD_BG, gridcolor=GRID_COLOR, showbackground=True, color=TEXT_COLOR),
        zaxis=dict(backgroundcolor=CARD_BG, gridcolor=GRID_COLOR, showbackground=True, color=TEXT_COLOR),
    )

def render_autoplay(fig, height=500):
    """Render animated Plotly figure with auto-play via HTML component (use sparingly)."""
    html = fig.to_html(include_plotlyjs="cdn", full_html=True, auto_play=True,
                       config={"displayModeBar": True, "scrollZoom": True})
    html = html.replace("<body>",
        '<body style="background-color:#0D1B2A;margin:0;padding:0;overflow:hidden;">')
    components.html(html, height=height, scrolling=False)

def render_3d_auto(fig, height=500, speed=0.35):
    """Render 3D Plotly figure with auto-rotation via HTML component (use sparingly)."""
    html = fig.to_html(include_plotlyjs="cdn", full_html=True,
                       config={"displayModeBar": True, "scrollZoom": True})
    html = html.replace("<body>",
        '<body style="background-color:#0D1B2A;margin:0;padding:0;overflow:hidden;">')
    js = f"""<script>
    (function(){{
        function go(){{
            var g=document.querySelector('.js-plotly-plot');
            if(!g||!g.layout){{setTimeout(go,300);return;}}
            var a=0,d=false;
            g.addEventListener('mousedown',function(){{d=true;}});
            document.addEventListener('mouseup',function(){{d=false;}});
            setInterval(function(){{
                if(d)return;a+={speed};
                var r=a*Math.PI/180;
                Plotly.relayout(g,{{'scene.camera.eye':{{x:1.6*Math.cos(r),y:1.6*Math.sin(r),z:0.8}}}});
            }},50);
        }}
        setTimeout(go,800);
    }})();
    </script>"""
    html = html.replace("</body>", js + "</body>")
    components.html(html, height=height, scrolling=False)


# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0D1B2A; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #142A3E; color: #E0E7EE; border-radius: 6px 6px 0 0;
        padding: 8px 16px; border: 1px solid #1E3A5F;
    }
    .stTabs [aria-selected="true"] { background-color: #0A2E50; border-bottom: 2px solid #00BCD4; }
    h1, h2, h3, .stMarkdown p, .stMarkdown li { color: #E0E7EE; }
    .stMetric label { color: #A0AEBB !important; }
    .stMetric [data-testid="stMetricValue"] { color: #E0E7EE !important; }
    div[data-testid="stExpander"] { background-color: #142A3E; border: 1px solid #1E3A5F; border-radius: 8px; }
    .info-card { background: linear-gradient(135deg, #142A3E, #1A354C); border: 1px solid #1E3A5F;
        border-radius: 10px; padding: 16px 20px; margin: 8px 0; border-left: 4px solid #00BCD4; }
    .warn-card { background: linear-gradient(135deg, #2A1A0A, #3E250A); border: 1px solid #5C3A0A;
        border-radius: 10px; padding: 16px 20px; margin: 8px 0; border-left: 4px solid #FFB74D; }
    .danger-card { background: linear-gradient(135deg, #2A0A0A, #3E1515); border: 1px solid #5C1A1A;
        border-radius: 10px; padding: 16px 20px; margin: 8px 0; border-left: 4px solid #FF6B6B; }
    .success-card { background: linear-gradient(135deg, #0A2A15, #0A3E20); border: 1px solid #0A5C2A;
        border-radius: 10px; padding: 16px 20px; margin: 8px 0; border-left: 4px solid #66BB6A; }
</style>
""", unsafe_allow_html=True)

st.title("Data & Compute Trade-offs: Traditional NNs vs PINNs")
st.markdown(
    "*Physics-Informed GNSS Positioning Error Correction in Degraded Environments* "
    "— Assignment 2: The Data & Compute Reality Check"
)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
st.sidebar.header("Model & Environment Controls")
model_size = st.sidebar.slider("Model parameters (millions)", 5, 200, 50, 5,
    help="Reference: 50M ~ 600 MB training state.")
lambda_weight = st.sidebar.slider("λ — Physics loss weight", 0.0, 2.0, 0.5, 0.05,
    help="L(θ) = L_data + λ·L_physics")
precision = st.sidebar.selectbox("Quantization level",
    ["FP32 (baseline)", "FP16 (training)", "INT8 (default)", "INT4 (aggressive)"], index=2)
environment = st.sidebar.selectbox("Deployment environment",
    ["Open Sky", "Urban Canyon", "Tunnel Exit", "Deep Urban"], index=1)
batch_size = st.sidebar.slider("Inference batch size (vehicles)", 1, 64, 8, 1)
drift_months = st.sidebar.slider("Deployment duration (months)", 1, 24, 6, 1,
    help="Affects calibration drift and distribution shift.")

np.random.seed(42)

ENV_CONFIG = {
    "Open Sky":     {"sats": 10, "cn0": 42, "error_base": 3.5,  "idx": 0},
    "Urban Canyon": {"sats": 3,  "cn0": 22, "error_base": 55,   "idx": 1},
    "Tunnel Exit":  {"sats": 1,  "cn0": 10, "error_base": 70,   "idx": 2},
    "Deep Urban":   {"sats": 2,  "cn0": 15, "error_base": 90,   "idx": 3},
}
env = ENV_CONFIG[environment]

# ═════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📡 GNSS Degradation", "🏗️ Architecture", "📊 Data Reality", "⚠️ Silent Killers",
    "💰 Compute Economics", "🚀 Edge Deployment", "⚔️ Head-to-Head", "✅ Verdict",
])


# ══════════════════════════════════════════════
# TAB 1 — GNSS Fails Where It Matters Most
# ══════════════════════════════════════════════
with tab1:
    st.subheader("GNSS Fails Where It Matters Most")
    col_3d, col_info = st.columns([3, 2])

    with col_3d:
        st.markdown("#### 3D Satellite Constellation (drag to rotate)")
        inc = 55 * np.pi / 180
        sat_x, sat_y, sat_z, sat_colors, sat_labels = [], [], [], [], []
        visible_count = env["sats"]
        np.random.seed(7)
        for p in range(6):
            raan = p * 60 * np.pi / 180
            for s in range(4):
                nu = s * 90 * np.pi / 180 + p * 15 * np.pi / 180
                x = np.cos(raan)*np.cos(nu) - np.sin(raan)*np.sin(nu)*np.cos(inc)
                y = np.sin(raan)*np.cos(nu) + np.cos(raan)*np.sin(nu)*np.cos(inc)
                z = np.sin(nu)*np.sin(inc)
                sat_x.append(x); sat_y.append(y); sat_z.append(z)
                idx = p*4 + s
                vis = idx < visible_count
                sat_colors.append(GREEN if vis else CORAL)
                sat_labels.append(f"SV{idx+1}: {'Visible' if vis else 'Blocked'}")

        u_s = np.linspace(0, 2*np.pi, 30)
        v_s = np.linspace(0, np.pi, 20)
        ex = 0.35*np.outer(np.cos(u_s), np.sin(v_s))
        ey = 0.35*np.outer(np.sin(u_s), np.sin(v_s))
        ez = 0.35*np.outer(np.ones_like(u_s), np.cos(v_s))

        fig_sat = go.Figure()
        fig_sat.add_trace(go.Surface(x=ex, y=ey, z=ez, colorscale=[[0,"#1A5276"],[1,"#2E86C1"]],
                                      showscale=False, opacity=0.6, hoverinfo="skip"))
        for p in range(6):
            raan = p*60*np.pi/180
            th = np.linspace(0, 2*np.pi, 60)
            fig_sat.add_trace(go.Scatter3d(
                x=np.cos(raan)*np.cos(th)-np.sin(raan)*np.sin(th)*np.cos(inc),
                y=np.sin(raan)*np.cos(th)+np.cos(raan)*np.sin(th)*np.cos(inc),
                z=np.sin(th)*np.sin(inc), mode="lines",
                line=dict(color=GRID_COLOR, width=1), showlegend=False, hoverinfo="skip"))
        fig_sat.add_trace(go.Scatter3d(
            x=sat_x, y=sat_y, z=sat_z, mode="markers",
            marker=dict(size=5, color=sat_colors, symbol="diamond", line=dict(width=1, color="white")),
            text=sat_labels, hoverinfo="text", name="Satellites"))
        fig_sat.add_trace(go.Scatter3d(
            x=[0.36], y=[0], z=[0], mode="markers",
            marker=dict(size=7, color=GOLD, symbol="circle", line=dict(width=2, color="white")),
            name="Receiver"))
        fig_sat.update_layout(
            **dark_layout(height=480, title_text=f"GPS Constellation — {environment} ({visible_count} SVs)"),
            scene=dict(**scene_dark(), xaxis_title="", yaxis_title="", zaxis_title="",
                       aspectmode="cube", camera=dict(eye=dict(x=1.6, y=1.6, z=0.8))))
        # Use auto-rotation for this key visual
        render_3d_auto(fig_sat, height=500)

    with col_info:
        # Gauges
        fig_g = make_subplots(rows=1, cols=3,
            specs=[[{"type":"indicator"},{"type":"indicator"},{"type":"indicator"}]])
        fig_g.add_trace(go.Indicator(
            mode="gauge+number", value=env["sats"],
            title={"text":"Satellites","font":{"size":13,"color":TEXT_COLOR}},
            number={"font":{"color":TEXT_COLOR}},
            gauge={"axis":{"range":[0,12],"tickcolor":MUTED},"bar":{"color":PINN_BLUE},"bgcolor":CARD_BG,
                   "steps":[{"range":[0,3],"color":"#3E1515"},{"range":[3,6],"color":"#3E250A"},
                            {"range":[6,12],"color":"#0A3E20"}],
                   "threshold":{"line":{"color":GREEN,"width":3},"value":4,"thickness":0.8}}
        ), row=1, col=1)
        fig_g.add_trace(go.Indicator(
            mode="gauge+number", value=env["cn0"],
            title={"text":"C/N₀ (dB-Hz)","font":{"size":13,"color":TEXT_COLOR}},
            number={"font":{"color":TEXT_COLOR}},
            gauge={"axis":{"range":[0,50],"tickcolor":MUTED},"bar":{"color":CYAN},"bgcolor":CARD_BG,
                   "steps":[{"range":[0,15],"color":"#3E1515"},{"range":[15,30],"color":"#3E250A"},
                            {"range":[30,50],"color":"#0A3E20"}],
                   "threshold":{"line":{"color":GREEN,"width":3},"value":35,"thickness":0.8}}
        ), row=1, col=2)
        fig_g.add_trace(go.Indicator(
            mode="gauge+number", value=env["error_base"],
            title={"text":"Error (m)","font":{"size":13,"color":TEXT_COLOR}},
            number={"font":{"color":TEXT_COLOR}},
            gauge={"axis":{"range":[0,100],"tickcolor":MUTED},"bar":{"color":CORAL},"bgcolor":CARD_BG,
                   "steps":[{"range":[0,5],"color":"#0A3E20"},{"range":[5,30],"color":"#3E250A"},
                            {"range":[30,100],"color":"#3E1515"}],
                   "threshold":{"line":{"color":GREEN,"width":3},"value":5,"thickness":0.8}}
        ), row=1, col=3)
        fig_g.update_layout(**dark_layout(height=240, title_text=f"Environment: {environment}"))
        st.plotly_chart(fig_g, use_container_width=True)

        st.markdown(f"""
        <div class="{'success-card' if env['sats']>=6 else 'warn-card' if env['sats']>=2 else 'danger-card'}">
            <b>Status:</b> {'Acceptable for navigation' if env['sats']>=6 else 'Errors exceed lane width' if env['sats']>=2 else 'Signal denied — classical methods fail'}
        </div>
        """, unsafe_allow_html=True)

        # Degradation bars
        envs_all = ["Open Sky", "Urban Canyon", "Tunnel Exit", "Deep Urban"]
        errors_all = [3.5, 55, 70, 90]
        sats_all = [10, 3, 1, 2]
        cn0_all = [42, 22, 10, 15]

        fig_deg = make_subplots(rows=1, cols=2, subplot_titles=("Position Error", "Signal Quality"),
                                horizontal_spacing=0.12)
        fig_deg.add_trace(go.Bar(x=envs_all, y=errors_all,
            marker_color=[GREEN, AMBER, CORAL, "#E53935"],
            text=[f"{e}m" for e in errors_all], textposition="outside",
            textfont=dict(color=TEXT_COLOR, size=10), showlegend=False), row=1, col=1)
        fig_deg.add_hline(y=5, line_dash="dash", line_color=GREEN, row=1, col=1,
                          annotation_text="5m target", annotation_font_color=GREEN)
        fig_deg.add_trace(go.Bar(x=envs_all, y=sats_all, name="Satellites", marker_color=PINN_BLUE), row=1, col=2)
        fig_deg.add_trace(go.Bar(x=envs_all, y=cn0_all, name="C/N₀", marker_color=CYAN), row=1, col=2)
        sel_i = envs_all.index(environment)
        fig_deg.add_annotation(x=envs_all[sel_i], y=errors_all[sel_i]+8, text="▼ Selected",
                               showarrow=False, font=dict(color=GOLD, size=11), row=1, col=1)
        fig_deg.update_layout(**dark_layout(height=320, barmode="group"))
        fig_deg.update_xaxes(gridcolor=GRID_COLOR)
        fig_deg.update_yaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig_deg, use_container_width=True)

        st.markdown("""<div class="info-card"><b>Classical Baseline:</b> Kalman filters + INS — hand-tuned, brittle in novel environments.</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — Architecture
# ══════════════════════════════════════════════
with tab2:
    st.subheader("Choose Architecture by Problem Structure, Not Hype")
    st.markdown(f"**Wrong architecture = 10× more data.** Current λ = **{lambda_weight:.2f}**")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""<div class="danger-card"><h4 style="color:#EF553B;margin-top:0;">Standard NNs / Transformers</h4>
        <ul style="color:#E0E7EE;"><li>Purely data-driven sequence modeling</li><li>Requires massive labeled datasets</li></ul>
        <p style="color:#FF6B6B;"><b>⚠ Risk:</b> No physics guardrail — impossible 50 m jumps.</p></div>""", unsafe_allow_html=True)
    with col_r:
        st.markdown("""<div class="info-card"><h4 style="color:#29B6F6;margin-top:0;">Physics-Informed NNs (PINNs)</h4>
        <ul style="color:#E0E7EE;"><li>Physics in loss function — up to 250× data-efficient</li><li>Generalizes across geographies</li></ul>
        <p style="color:#FFB74D;"><b>⚠ Risks:</b> Incomplete physics = wrong constraint. Needs ML + GNSS staffing.</p></div>""", unsafe_allow_html=True)

    st.markdown("""<div style="background:#0A2E50;border:1px solid #00BCD4;border-radius:8px;padding:10px 20px;text-align:center;margin:10px 0;">
        <span style="color:#E0E7EE;font-size:16px;font-style:italic;">L(θ) = L<sub>data</sub>(θ) + λ · L<sub>physics</sub>(θ)</span>
        <span style="color:#FFB74D;font-size:13px;margin-left:20px;"><b>λ validated against error budgets</b></span></div>""", unsafe_allow_html=True)

    col_3d, col_lam = st.columns([3, 2])

    with col_3d:
        st.markdown("#### 3D Error Landscape: Data × λ → Error")
        log_s = np.linspace(2, 6, 50)
        lam_v = np.linspace(0, 2.0, 50)
        X, Y = np.meshgrid(log_s, lam_v)
        eff = np.clip(Y, 0.1, 1.5)
        ps = 150000 / (1 + 200 * eff)
        Zp = 80 * np.exp(-10**X / ps) + 2 + 15 * np.maximum(0, Y - 0.8)**1.5
        Zp += np.where(Y < 0.1, 8*(0.1-Y)/0.1, 0)
        Zp = np.clip(Zp, 1, 90)
        Zn = np.clip(80 * np.exp(-10**X / 150000) + 5, 2, 90)

        fig_3d = go.Figure()
        fig_3d.add_trace(go.Surface(x=log_s, y=lam_v, z=Zp,
            colorscale=[[0,"#0A3E20"],[0.3,PINN_BLUE],[0.6,AMBER],[1,CORAL]],
            name="PINN", opacity=0.85,
            colorbar=dict(title=dict(text="Error(m)", font=dict(color=TEXT_COLOR)), tickfont=dict(color=TEXT_COLOR)),
            contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
            hovertemplate="Samples: 10^%{x:.1f}<br>λ: %{y:.2f}<br>Error: %{z:.1f}m<extra>PINN</extra>"))
        fig_3d.add_trace(go.Surface(x=log_s, y=lam_v, z=Zn,
            colorscale=[[0,"rgba(239,83,59,0.3)"],[1,"rgba(239,83,59,0.6)"]],
            opacity=0.35, showscale=False, name="NN",
            hovertemplate="Samples: 10^%{x:.1f}<br>Error: %{z:.1f}m<extra>NN</extra>"))
        li = np.argmin(np.abs(lam_v - lambda_weight))
        fig_3d.add_trace(go.Scatter3d(
            x=[log_s[25]], y=[lambda_weight], z=[Zp[li, 25]],
            mode="markers", marker=dict(size=7, color=GOLD, symbol="diamond",
                                         line=dict(width=2, color="white")),
            name=f"Current λ={lambda_weight:.2f}"))
        fig_3d.update_layout(
            **dark_layout(height=500, title_text="PINN surface vs NN plane"),
            scene=dict(**scene_dark(), xaxis_title="log₁₀(Samples)", yaxis_title="λ",
                       zaxis_title="Error (m)", camera=dict(eye=dict(x=1.8, y=-1.2, z=0.9))))
        st.plotly_chart(fig_3d, use_container_width=True)

    with col_lam:
        lambdas = np.linspace(0, 2.0, 150)
        np.random.seed(42)
        lam_err = 3 + 40*np.exp(-8*lambdas) + 15*np.maximum(0, lambdas-0.8)**1.5
        lam_err += np.random.normal(0, 0.3, 150)
        lam_err = np.clip(lam_err, 1, 55)

        fig_lam = go.Figure()
        fig_lam.add_trace(go.Scatter(x=lambdas, y=lam_err, mode="lines", name="PINN Error",
            line=dict(color=PINN_BLUE, width=3), fill="tozeroy", fillcolor="rgba(41,182,246,0.08)"))
        fig_lam.add_vrect(x0=0.3, x1=0.7, fillcolor=GREEN, opacity=0.1,
                          annotation_text="Optimal Zone", annotation_font_color=GREEN, annotation_position="top")
        ci = np.argmin(np.abs(lambdas - lambda_weight))
        fig_lam.add_trace(go.Scatter(x=[lambda_weight], y=[lam_err[ci]], mode="markers",
            marker=dict(size=14, color=GOLD, symbol="star", line=dict(width=2, color="white")),
            name=f"λ={lambda_weight:.2f} → {lam_err[ci]:.1f}m"))
        fig_lam.add_annotation(x=0.05, y=40, text="λ too low<br>→ No guardrail", showarrow=False,
                               font=dict(color=CORAL, size=10))
        fig_lam.add_annotation(x=1.7, y=40, text="λ too high<br>→ Ignores data", showarrow=False,
                               font=dict(color=CORAL, size=10))
        fig_lam.update_layout(**dark_layout(height=380, title_text="λ Sensitivity",
            xaxis_title="λ", yaxis_title="Error (m)", yaxis_range=[0, 55]))
        dark_axes(fig_lam)
        st.plotly_chart(fig_lam, use_container_width=True)

        if lambda_weight < 0.1:
            st.markdown('<div class="danger-card"><b>⚠ λ too low</b> — guardrail disabled.</div>', unsafe_allow_html=True)
        elif lambda_weight > 1.2:
            st.markdown('<div class="warn-card"><b>⚠ λ too high</b> — over-constrains.</div>', unsafe_allow_html=True)
        elif 0.3 <= lambda_weight <= 0.7:
            st.markdown('<div class="success-card"><b>✓ λ in optimal range.</b></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-card"><b>ℹ λ outside optimal</b> — validate.</div>', unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        m1.metric("Current λ error", f"{lam_err[ci]:.1f} m")
        oi = np.argmin(lam_err)
        m2.metric("Optimal λ", f"{lambdas[oi]:.2f} → {lam_err[oi]:.1f} m")


# ══════════════════════════════════════════════
# TAB 3 — Data Reality
# ══════════════════════════════════════════════
with tab3:
    st.subheader("Data Reality: Quality, Missingness, and Leakage")
    st.markdown("**Data is signal, not fuel.** 80% of model failures trace to data, not architecture.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="success-card"><h4 style="color:#66BB6A;margin-top:0;">📊 Quality > Quantity</h4>
        <p style="color:#E0E7EE;">10K clean × 1.5 bits outweigh 1M noisy × 0.2 bits.</p>
        <p style="color:#66BB6A;"><b>Stop collecting. Start auditing.</b></p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="warn-card"><h4 style="color:#FFB74D;margin-top:0;">⚠ MNAR Missingness</h4>
        <p style="color:#E0E7EE;">Signal drops = informative features. Imputation destroys this.</p>
        <p style="color:#FFB74D;"><b>Preserve dropout patterns.</b></p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="danger-card"><h4 style="color:#FF6B6B;margin-top:0;">⛔ Temporal Leakage</h4>
        <p style="color:#E0E7EE;">Training on post-processed RTK = cheating. Deployed accuracy collapses.</p>
        <p style="color:#FF6B6B;"><b>Strict temporal splits.</b></p></div>""", unsafe_allow_html=True)

    col_sig, col_val = st.columns(2)

    with col_sig:
        # Animated signal trace — auto-play
        st.markdown("#### Live GNSS Signal (auto-play)")
        np.random.seed(42)
        t_f = np.linspace(0, 60, 200)
        sig_f = 35 + 5*np.sin(0.3*t_f) + np.random.normal(0, 1.5, 200)
        tmask = (t_f > 20) & (t_f < 35)

        frames_s = []
        for fi in np.linspace(10, 200, 20, dtype=int):
            t_sl = t_f[:fi]
            s_sl = sig_f[:fi].copy()
            s_sl[tmask[:fi]] = np.nan
            frames_s.append(go.Frame(
                data=[go.Scatter(x=t_sl, y=s_sl, mode="lines", line=dict(color=CYAN, width=2))],
                name=str(fi)))

        fig_sig = go.Figure(data=[go.Scatter(x=[], y=[], mode="lines", line=dict(color=CYAN, width=2))],
                            frames=frames_s)
        fig_sig.add_vrect(x0=20, x1=35, fillcolor=CORAL, opacity=0.12,
                          annotation_text="Tunnel (MNAR)", annotation_font_color=CORAL, annotation_position="top left")
        fig_sig.update_layout(
            **dark_layout(height=320, title_text="Signal Trace — Dropout is Information"),
            xaxis=dict(title="Time (s)", range=[0, 62], gridcolor=GRID_COLOR),
            yaxis=dict(title="C/N₀ (dB-Hz)", range=[20, 50], gridcolor=GRID_COLOR),
            updatemenus=[dict(type="buttons", showactive=False, visible=False,
                buttons=[dict(method="animate", args=[None, {"frame":{"duration":150,"redraw":True},"fromcurrent":True}])])])
        render_autoplay(fig_sig, height=340)

    with col_val:
        st.markdown("#### Information Value: Clean vs Noisy")
        noise_level = st.slider("Noise bits/sample", 0.05, 1.5, 0.2, 0.05, key="noise")
        clean_bits = 10_000 * 1.5
        noisy_bits = 1_000_000 * noise_level

        fig_info = go.Figure()
        fig_info.add_trace(go.Bar(
            x=["10K Clean (1.5 b/s)", f"1M Noisy ({noise_level:.2f} b/s)"],
            y=[clean_bits, noisy_bits], marker_color=[GREEN, CORAL],
            text=[f"{clean_bits:,.0f}", f"{noisy_bits:,.0f}"], textposition="outside",
            textfont=dict(color=TEXT_COLOR, size=13)))
        v = "Noisy wins on volume" if noisy_bits > clean_bits else "Clean wins — quality matters"
        fig_info.update_layout(**dark_layout(height=320, title_text="Total Information Content",
            yaxis_title="Total Bits",
            annotations=[dict(x=0.5, y=1.05, xref="paper", yref="paper", showarrow=False,
                text=v, font=dict(color=AMBER if noisy_bits > clean_bits else GREEN, size=13))]))
        dark_axes(fig_info)
        st.plotly_chart(fig_info, use_container_width=True)

    # Temporal leakage
    fig_leak = go.Figure()
    fig_leak.add_trace(go.Bar(x=[6], y=["Timeline"], orientation="h", base=[0],
        marker_color="rgba(10,62,32,0.7)", marker_line=dict(color=GREEN, width=2),
        name="Training", text=["Training Data"], textposition="inside", textfont=dict(color=GREEN, size=12)))
    fig_leak.add_trace(go.Bar(x=[4], y=["Timeline"], orientation="h", base=[6],
        marker_color="rgba(62,21,21,0.7)", marker_line=dict(color=CORAL, width=2),
        name="Inference", text=["Future / Deploy"], textposition="inside", textfont=dict(color=CORAL, size=12)))
    fig_leak.add_annotation(x=5, y="Timeline", ax=7.5, ay="Timeline", xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=3, arrowcolor=CORAL)
    fig_leak.add_annotation(x=6.25, y="Timeline", text="⚠ LEAKAGE", showarrow=False,
        font=dict(color=CORAL, size=14), bgcolor="#3E1515", bordercolor=CORAL, borderwidth=1, yshift=-40)
    fig_leak.add_vline(x=6, line_color=AMBER, line_width=3)
    fig_leak.add_annotation(x=6, y="Timeline", text="▼ Temporal Boundary", showarrow=False,
        yshift=40, font=dict(color=AMBER, size=12))
    fig_leak.update_layout(**dark_layout(height=180, barmode="stack", showlegend=True))
    fig_leak.update_xaxes(showticklabels=False, gridcolor=GRID_COLOR)
    fig_leak.update_yaxes(showticklabels=False, gridcolor=GRID_COLOR)
    st.plotly_chart(fig_leak, use_container_width=True)

    st.markdown("""<div class="info-card"><b>Pre-Training Audit:</b>
    <span style="color:#00BCD4;">Completeness · Consistency · Accuracy · Timeliness · Relevance</span></div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 4 — Silent Killers
# ══════════════════════════════════════════════
with tab4:
    st.subheader("Silent Killers: Drift, Shift, and Distribution Collapse")
    st.markdown(f"**Distribution is destiny.** Deployment: **{drift_months} months**.")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""<div class="danger-card"><h4 style="color:#FF6B6B;margin-top:0;">⚠ Covariate Shift</h4>
        <p style="color:#E0E7EE;"><b>Trap:</b> Suburban model deployed downtown. <b>Reality:</b> Satellite geometry, multipath, rain all change silently.</p></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="danger-card"><h4 style="color:#FF6B6B;margin-top:0;">⚠ Calibration Drift</h4>
        <p style="color:#E0E7EE;"><b>Trap:</b> Clocks accumulate error. <b>Reality:</b> Antenna degrades, firmware alters pseudorange.</p></div>""", unsafe_allow_html=True)

    with col_r:
        # Auto-animated distribution shift
        st.markdown("#### Distribution Shift Animation")
        x_r = np.linspace(-5, 12, 200)
        td = np.exp(-0.5*x_r**2) / np.sqrt(2*np.pi)
        frames_sh = []
        for mo in range(0, drift_months+1):
            dm = 0.15*mo; ds = 1+0.05*mo
            dd = np.exp(-0.5*((x_r-dm)/ds)**2) / (ds*np.sqrt(2*np.pi))
            mmd = 0.05*mo + 0.02*mo**0.5
            frames_sh.append(go.Frame(
                data=[
                    go.Scatter(x=x_r, y=td, mode="lines", line=dict(color=PINN_BLUE, width=3),
                        fill="tozeroy", fillcolor="rgba(41,182,246,0.2)", name="Training"),
                    go.Scatter(x=x_r, y=dd, mode="lines", line=dict(color=CORAL, width=3),
                        fill="tozeroy", fillcolor="rgba(255,107,107,0.2)", name=f"Deploy ({mo}mo)")],
                name=str(mo),
                layout=go.Layout(title_text=f"Month {mo} | MMD={mmd:.2f} {'⚠ ALERT' if mmd>0.5 else '✓'}")))

        fig_sh = go.Figure(data=frames_sh[0].data if frames_sh else [], frames=frames_sh)
        fig_sh.update_layout(
            **dark_layout(height=350, title_text="Month 0 | MMD=0.00 ✓"),
            xaxis=dict(title="Feature Space", range=[-5, 12], gridcolor=GRID_COLOR),
            yaxis=dict(title="Density", range=[0, 0.5], gridcolor=GRID_COLOR),
            updatemenus=[dict(type="buttons", showactive=False, visible=False,
                buttons=[dict(method="animate", args=[None, {"frame":{"duration":500,"redraw":True},"fromcurrent":True}])])])
        render_autoplay(fig_sh, height=370)

    # 3D error surface: shift × drift
    st.markdown("#### 3D Error Surface: Shift × Drift → Error (drag to rotate)")
    sh_r = np.linspace(0, 3, 30)
    dr_r = np.linspace(0, 5, 30)
    SH, DR = np.meshgrid(sh_r, dr_r)
    np.random.seed(42)
    ERR = 3 + 8*SH + 5*DR + 3*SH*DR + np.random.normal(0, 0.5, SH.shape)
    ERR = np.clip(ERR, 2, 100)

    cur_sh = min(0.15*drift_months, 3)
    cur_dr = min(0.3*drift_months, 5)
    cur_e = 3 + 8*cur_sh + 5*cur_dr + 3*cur_sh*cur_dr

    fig_3de = go.Figure()
    fig_3de.add_trace(go.Surface(x=sh_r, y=dr_r, z=ERR,
        colorscale=[[0,"#0A3E20"],[0.3,PINN_BLUE],[0.6,AMBER],[1,CORAL]],
        colorbar=dict(title=dict(text="Error(m)", font=dict(color=TEXT_COLOR)), tickfont=dict(color=TEXT_COLOR)),
        contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
        hovertemplate="Shift:%{x:.1f}<br>Drift:%{y:.1f}ns<br>Error:%{z:.1f}m<extra></extra>"))
    fig_3de.add_trace(go.Scatter3d(x=[cur_sh], y=[cur_dr], z=[cur_e], mode="markers",
        marker=dict(size=7, color=GOLD, symbol="diamond", line=dict(width=2, color="white")),
        name=f"Current ({drift_months}mo)"))
    fig_3de.update_layout(
        **dark_layout(height=450, title_text="Error grows multiplicatively"),
        scene=dict(**scene_dark(), xaxis_title="Covariate Shift", yaxis_title="Clock Drift (ns)",
                   zaxis_title="Error (m)", camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0))))
    st.plotly_chart(fig_3de, use_container_width=True)

    st.markdown("""<div class="success-card"><h4 style="color:#009688;margin-top:0;">🛡 Defense Protocol</h4>
    <p style="color:#E0E7EE;"><b>Deploy MMD Monitors</b> on input pipelines. Threshold exceeded → <b>retrain</b> or <b>fallback to Kalman</b>.</p>
    <p style="color:#009688;"><b>→ Silent failures become detectable.</b></p></div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 5 — Compute Economics
# ══════════════════════════════════════════════
with tab5:
    st.subheader("Compute Economics: Why FLOPs Lie")
    st.markdown(f"**FLOPs ≠ wall-clock speed.** Model: **{model_size}M params**.")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(f"""<div class="info-card"><h4 style="color:#29B6F6;margin-top:0;">Training (CAPEX)</h4>
        <ul style="color:#E0E7EE;"><li>Compute-bound — ≈ 6N FLOPs/token</li>
        <li>{model_size}M = <b>{12*model_size/1000:.1f} GB</b></li></ul>
        <p style="color:#FFB74D;"><b>PINN:</b> +20–40%/epoch, but less data needed.</p></div>""", unsafe_allow_html=True)
    with col_r:
        st.markdown("""<div class="danger-card"><h4 style="color:#FF6B6B;margin-top:0;">Inference (OPEX)</h4>
        <ul style="color:#E0E7EE;"><li>Bandwidth-bound — ≈ 2N FLOPs/token</li>
        <li>10 Hz = <100 ms per correction</li><li>Scales linearly with fleet</li></ul></div>""", unsafe_allow_html=True)

    col_cost, col_roof = st.columns(2)
    with col_cost:
        sizes = np.arange(5, 205, 5)
        nn_f = 6*sizes; pinn_f = nn_f*1.3; mem = 12*sizes*1e6/1e9
        fig_c = make_subplots(specs=[[{"secondary_y": True}]])
        fig_c.add_trace(go.Bar(x=[f"{s}M" for s in sizes[::4]], y=nn_f[::4],
            name="NN (6N)", marker_color=NN_RED, opacity=0.8), secondary_y=False)
        fig_c.add_trace(go.Bar(x=[f"{s}M" for s in sizes[::4]], y=pinn_f[::4],
            name="PINN (7.8N)", marker_color=PINN_BLUE, opacity=0.8), secondary_y=False)
        fig_c.add_trace(go.Scatter(x=[f"{s}M" for s in sizes[::4]], y=mem[::4],
            name="Memory (GB)", mode="lines+markers", line=dict(color=AMBER, width=2.5),
            marker=dict(size=7, color=AMBER)), secondary_y=True)
        si = np.argmin(np.abs(sizes-model_size))
        fig_c.add_trace(go.Scatter(x=[f"{model_size}M"], y=[nn_f[si]], mode="markers",
            marker=dict(size=16, color=GOLD, symbol="star"), name=f"Selected: {model_size}M"), secondary_y=False)
        fig_c.update_layout(**dark_layout(height=400, title_text="Training Cost", barmode="group"))
        fig_c.update_xaxes(title_text="Model Size", gridcolor=GRID_COLOR)
        fig_c.update_yaxes(title_text="GFLOPs/Token", gridcolor=GRID_COLOR, secondary_y=False)
        fig_c.update_yaxes(title_text="Memory (GB)", gridcolor=GRID_COLOR, secondary_y=True,
            title_font=dict(color=AMBER), tickfont=dict(color=AMBER))
        st.plotly_chart(fig_c, use_container_width=True)

    with col_roof:
        # 3D Roofline
        st.markdown("#### 3D Roofline (drag to rotate)")
        ms3 = np.linspace(5, 200, 25)
        ai3 = np.logspace(-1, 2, 25)
        MS, AI = np.meshgrid(ms3, ai3)
        PERF = np.clip(np.minimum(10000, 900*AI)*(1-0.001*MS), 1, 10000)

        fig_rf = go.Figure()
        fig_rf.add_trace(go.Surface(x=ms3, y=np.log10(ai3), z=np.log10(PERF),
            colorscale=[[0,CARD_BG],[0.5,CYAN],[1,GREEN]], opacity=0.7, showscale=False,
            hovertemplate="Size:%{x:.0f}M<br>log₁₀(AI):%{y:.1f}<br>log₁₀(GFLOPS):%{z:.1f}<extra></extra>"))
        fig_rf.add_trace(go.Scatter3d(x=[model_size], y=[np.log10(15)], z=[np.log10(7000)],
            mode="markers+text", text=["Training"], textposition="top center",
            marker=dict(size=7, color=GREEN, line=dict(width=2, color="white")),
            textfont=dict(color=GREEN, size=10), name="Training"))
        fig_rf.add_trace(go.Scatter3d(x=[model_size], y=[np.log10(0.5)], z=[np.log10(450)],
            mode="markers+text", text=["Inference"], textposition="top center",
            marker=dict(size=7, color=CORAL, line=dict(width=2, color="white")),
            textfont=dict(color=CORAL, size=10), name="Inference"))
        fig_rf.update_layout(**dark_layout(height=430, title_text="3D Roofline"),
            scene=dict(**scene_dark(), xaxis_title="Model Size (M)", yaxis_title="log₁₀(Arith. Int.)",
                       zaxis_title="log₁₀(GFLOPS)", camera=dict(eye=dict(x=1.5, y=-1.3, z=0.9))))
        st.plotly_chart(fig_rf, use_container_width=True)

    st.markdown("""<div class="info-card"><b>Key insight:</b> Training = CAPEX. Inference = OPEX.
    <span style="color:#FF6B6B;"><b>Inference cost is the binding constraint</b></span> for fleet-scale GNSS.</div>""", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("NN FLOPs/token", f"{6*model_size*1e6/1e9:.1f} GFLOPs")
    m2.metric("PINN FLOPs/token", f"{6*model_size*1e6/1e9*1.3:.1f} GFLOPs")
    m3.metric("Training memory", f"{12*model_size*1e6/1e9:.2f} GB")


# ══════════════════════════════════════════════
# TAB 6 — Edge Deployment
# ══════════════════════════════════════════════
with tab6:
    st.subheader("Solving the Inference Bottleneck for Edge Deployment")
    st.markdown(f"Spare FLOPs, not spare bandwidth. **{precision}** | **Batch: {batch_size}**")

    st.markdown("""<div style="display:flex;gap:10px;justify-content:center;margin:15px 0;">
    <div style="background:#142A3E;border:2px solid #29B6F6;border-radius:10px;padding:15px 25px;text-align:center;flex:1;">
        <p style="color:#29B6F6;font-weight:bold;margin:0;">GPU Memory</p><p style="color:#A0AEBB;margin:0;font-size:12px;">Weights</p></div>
    <div style="display:flex;align-items:center;color:#00BCD4;font-size:28px;">→</div>
    <div style="background:#3E250A;border:2px solid #FFB74D;border-radius:10px;padding:15px 25px;text-align:center;flex:1;">
        <p style="color:#FFB74D;font-weight:bold;margin:0;">Transfer to Cores</p><p style="color:#A0AEBB;margin:0;font-size:12px;">THE BOTTLENECK</p></div>
    <div style="display:flex;align-items:center;color:#00BCD4;font-size:28px;">→</div>
    <div style="background:#0A3520;border:2px solid #66BB6A;border-radius:10px;padding:15px 25px;text-align:center;flex:1;">
        <p style="color:#66BB6A;font-weight:bold;margin:0;">Matrix Multiply</p><p style="color:#A0AEBB;margin:0;font-size:12px;">~2N FLOPs (spare)</p></div>
    </div>""", unsafe_allow_html=True)

    col_q, col_lat = st.columns(2)

    with col_q:
        ql = ["FP32", "FP16", "INT8", "INT4"]
        mr = [1, 2, 4, 8]
        na = [100.0, 99.8, 99.2, 96.5]; pa = [100.0, 99.9, 99.5, 97.0]
        fig_q = go.Figure()
        fig_q.add_trace(go.Bar(x=ql, y=na, name="NN", marker_color=NN_RED, opacity=0.85,
            text=[f"{a:.1f}%" for a in na], textposition="outside", textfont=dict(color=TEXT_COLOR, size=10)))
        fig_q.add_trace(go.Bar(x=ql, y=pa, name="PINN", marker_color=PINN_BLUE, opacity=0.85,
            text=[f"{a:.1f}%" for a in pa], textposition="outside", textfont=dict(color=TEXT_COLOR, size=10)))
        fig_q.add_hline(y=99, line_dash="dash", line_color=GREEN, annotation_text="99%", annotation_font_color=GREEN)
        for i, (q, m) in enumerate(zip(ql, mr)):
            fig_q.add_annotation(x=q, y=95.3, text=f"{m}×", showarrow=False, font=dict(color=AMBER, size=12))
        sq = precision.split(" ")[0]
        for q in ql:
            if q == sq:
                fig_q.add_annotation(x=q, y=101, text="▼ Selected", showarrow=False, font=dict(color=GOLD, size=11))
        fig_q.update_layout(**dark_layout(height=400, title_text="Quantization: Accuracy vs Memory",
            yaxis_range=[94.5, 101.5], barmode="group"))
        dark_axes(fig_q)
        st.plotly_chart(fig_q, use_container_width=True)

    with col_lat:
        # 3D Latency surface
        st.markdown("#### 3D Latency Surface (drag to rotate)")
        s3 = np.arange(5, 205, 10)
        b3 = np.arange(1, 65, 4)
        SZ, BA = np.meshgrid(s3, b3)
        bpp_map = {"FP32":4.0, "FP16":2.0, "INT8":1.0, "INT4":0.5}
        bpp = bpp_map.get(sq, 1.0)
        mb = SZ*1e6*bpp
        bl = (mb/(900*1e9))*1000
        cl = (2*SZ*1e6)/(10e12)*1000
        bf = 1 + 0.15*np.log2(np.maximum(BA, 1))
        LAT = np.clip((bl+cl)*bf+2.0, 0.5, 500)

        clv = float(LAT[np.argmin(np.abs(b3-batch_size)), np.argmin(np.abs(s3-model_size))])

        fig_l3 = go.Figure()
        fig_l3.add_trace(go.Surface(x=s3, y=b3, z=LAT,
            colorscale=[[0,"#0A3E20"],[0.3,PINN_BLUE],[0.7,AMBER],[1,CORAL]],
            colorbar=dict(title=dict(text="ms", font=dict(color=TEXT_COLOR)), tickfont=dict(color=TEXT_COLOR)),
            contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
            hovertemplate="Size:%{x}M<br>Batch:%{y}<br>%{z:.1f}ms<extra></extra>"))
        fig_l3.add_trace(go.Surface(x=s3, y=b3, z=np.full_like(LAT, 100),
            colorscale=[[0,"rgba(255,107,107,0.3)"],[1,"rgba(255,107,107,0.3)"]],
            showscale=False, opacity=0.3, hovertemplate="100ms deadline<extra></extra>"))
        fig_l3.add_trace(go.Scatter3d(x=[model_size], y=[batch_size], z=[clv],
            mode="markers", marker=dict(size=7, color=GOLD, symbol="diamond", line=dict(width=2, color="white")),
            name=f"Current: {clv:.1f}ms"))
        fig_l3.update_layout(**dark_layout(height=430, title_text=f"Latency ({sq})"),
            scene=dict(**scene_dark(), xaxis_title="Model (M)", yaxis_title="Batch",
                       zaxis_title="Latency (ms)", camera=dict(eye=dict(x=1.6, y=-1.4, z=0.9))))
        st.plotly_chart(fig_l3, use_container_width=True)

    meets = clv < 100
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latency", f"{clv:.1f} ms")
    m2.metric("Meets 100ms?", "Yes ✓" if meets else "No ✗")
    m3.metric("Fleet corr/sec", f"{(1000/max(clv,1))*batch_size:,.0f}")
    m4.metric("Memory reduction", f"{int(4/bpp)}× vs FP32")

    ca, cb = st.columns(2)
    with ca:
        st.markdown(f"""<div class="{'success-card' if meets else 'danger-card'}">
        <b>{sq}:</b> {int(4/bpp)}× reduction. {'Meets deadline.' if meets else 'Exceeds budget.'}</div>""", unsafe_allow_html=True)
    with cb:
        st.markdown("""<div class="info-card"><b>Parallelism + Batching =</b> Fleet-scale edge without cloud.</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 7 — Head-to-Head
# ══════════════════════════════════════════════
with tab7:
    st.subheader("Head-to-Head: Traditional NN vs PINN")

    col_t, col_v = st.columns([3, 2])
    with col_t:
        cmp = {
            "Dimension": ["Data efficiency","Physics compliance","Failure mode","Training (CAPEX)",
                "Inference (OPEX)","Edge deployment","Shift resilience","Drift detection",
                "Domain expertise","Generalization","Safety guarantee","Best fit"],
            "Traditional NN / Transformer": [
                "Low — large datasets","None — impossible outputs","Silent 50 m jumps","~6N FLOPs/token",
                "~2N FLOPs, BW-bound","Viable INT8/INT4","No guardrail","Needs MMD monitors",
                "ML engineering","Poor w/o retraining","None","Abundant RTK, stable"],
            "PINN (Physics-Informed)": [
                "High — up to 250× less","Built-in — violations penalized","Bounded by physics quality",
                "+20–40%/epoch, less data","Identical — training-only","Same inference graph",
                "Physics limits damage","Needs MMD monitors","ML + physics/GNSS",
                "Strong — physics global","Soft — physics quality","Scarce data, safety-critical"],
        }
        st.dataframe(pd.DataFrame(cmp), use_container_width=True, hide_index=True, height=480)

    with col_v:
        # Radar chart
        cats = ["Data\nEfficiency","Physics\nCompliance","Safety","Generalization","Ease of\nImpl.","Edge\nViability"]
        nn_s = [2,1,1,2,5,4]; pi_s = [5,5,4,5,3,4]; ka_s = [3,3,3,2,4,5]

        fig_rad = go.Figure()
        fig_rad.add_trace(go.Scatterpolar(r=nn_s+[nn_s[0]], theta=cats+[cats[0]], name="NN",
            line=dict(color=NN_RED, width=2.5), fill="toself", fillcolor="rgba(239,83,59,0.15)"))
        fig_rad.add_trace(go.Scatterpolar(r=pi_s+[pi_s[0]], theta=cats+[cats[0]], name="PINN",
            line=dict(color=PINN_BLUE, width=2.5), fill="toself", fillcolor="rgba(41,182,246,0.15)"))
        fig_rad.add_trace(go.Scatterpolar(r=ka_s+[ka_s[0]], theta=cats+[cats[0]], name="Kalman",
            line=dict(color=KALMAN_AMBER, width=2), fill="toself", fillcolor="rgba(255,183,77,0.1)"))
        fig_rad.update_layout(
            polar=dict(bgcolor=CARD_BG,
                radialaxis=dict(visible=True, range=[0,5], gridcolor=GRID_COLOR, tickfont=dict(color=MUTED, size=9)),
                angularaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TEXT_COLOR, size=10))),
            **dark_layout(height=380, title_text="Capability Comparison"))
        st.plotly_chart(fig_rad, use_container_width=True)

        # 3D method scatter
        st.markdown("#### 3D Method Comparison")
        methods = ["Kalman+INS","Trad. NN","PINN Hybrid"]
        de = [3,2,5]; sf = [3,1,4]; ge = [2,2,5]
        mc = [KALMAN_AMBER, NN_RED, PINN_BLUE]; ms_3d = [18,18,22]

        fig_3m = go.Figure()
        for i, (m, d, s, g, c, sz) in enumerate(zip(methods, de, sf, ge, mc, ms_3d)):
            fig_3m.add_trace(go.Scatter3d(x=[d], y=[s], z=[g], mode="markers+text",
                marker=dict(size=sz, color=c, opacity=0.9, line=dict(width=2, color="white")),
                text=[m], textposition="top center", textfont=dict(color=c, size=11), name=m))
        fig_3m.update_layout(**dark_layout(height=380, title_text="Efficiency × Safety × Generalization"),
            scene=dict(**scene_dark(), xaxis_title="Data Eff.", yaxis_title="Safety",
                       zaxis_title="Generalization", camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0))))
        st.plotly_chart(fig_3m, use_container_width=True)

    # Environment bars
    envs = ["Open Sky","Urban Canyon","Tunnel Exit","Deep Urban"]
    ne = [4.5,45,85,95]; pe = [2.5,8,25,35]; ke = [3.5,35,70,90]
    fig_e = go.Figure()
    fig_e.add_trace(go.Bar(x=envs, y=ke, name="Kalman+INS", marker_color=KALMAN_AMBER, opacity=0.85))
    fig_e.add_trace(go.Bar(x=envs, y=ne, name="Traditional NN", marker_color=NN_RED, opacity=0.85))
    fig_e.add_trace(go.Bar(x=envs, y=pe, name="PINN (Hybrid)", marker_color=PINN_BLUE, opacity=0.85))
    fig_e.add_hline(y=5, line_dash="dash", line_color=GREEN, annotation_text="5m target", annotation_font_color=GREEN)
    if environment in envs:
        ei = envs.index(environment)
        fig_e.add_annotation(x=envs[ei], y=max(ne[ei], ke[ei])+8, text="▼ Selected",
            showarrow=False, font=dict(color=GOLD, size=11))
    fig_e.update_layout(**dark_layout(height=380, title_text="Error by Environment", yaxis_title="Error (m)", barmode="group"))
    dark_axes(fig_e)
    st.plotly_chart(fig_e, use_container_width=True)

    st.markdown("""<div class="info-card"><b>Key:</b> Inference cost identical (physics = training-only). Both need MMD monitors.
    PINN advantage widens in degraded environments. Physics transfers globally.</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 8 — Verdict
# ══════════════════════════════════════════════
with tab8:
    st.subheader("Verdict: Hybrid PINN — With Conditions")

    st.markdown("""<div class="success-card"><h4 style="color:#66BB6A;margin-top:0;">✓ Recommendation</h4>
    <p style="color:#E0E7EE;font-size:16px;">Adopt <b>Hybrid Physics-Informed Architecture</b> for degraded GNSS.</p></div>""", unsafe_allow_html=True)

    col_w, col_p = st.columns(2)
    with col_w:
        st.markdown("""<div class="info-card"><h4 style="color:#29B6F6;margin-top:0;">Use PINNs When:</h4>
        <ul style="color:#E0E7EE;"><li>Data scarce, noisy, or MNAR</li><li>Physics well-characterized</li><li>Safety-critical</li></ul></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="danger-card"><h4 style="color:#EF553B;margin-top:0;">Use Transformers When:</h4>
        <ul style="color:#E0E7EE;"><li>Abundant RTK logs</li><li>Multipath exceeds physics models</li><li>Stable environment</li></ul></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="warn-card"><h4 style="color:#FFB74D;margin-top:0;">🔒 Non-Negotiable:</h4>
        <ol style="color:#E0E7EE;"><li>Validate λ against error budgets</li><li>Eliminate temporal leakage</li><li>MMD monitors from Day 1</li></ol></div>""", unsafe_allow_html=True)

    with col_p:
        # Animated Gantt — auto-play
        plan = [("Data Audit","Data Eng.",1,2,GREEN),("Baseline PDEs","Physics",2,4,PINN_BLUE),
                ("Train Prototype","ML Team",4,8,CYAN),("MMD Monitors","MLOps",1,10,AMBER),
                ("INT8/INT4 Quant","ML/Edge",8,10,TEAL)]
        tasks = [p[0] for p in plan]
        tasks_r = list(reversed(tasks))

        frames_g = []
        for wk in range(1, 11):
            bx, by, bb, bc, bt = [], [], [], [], []
            for (t, o, s, e, c) in plan:
                if s <= wk:
                    ve = min(e, wk)
                    bx.append(ve-s); by.append(t); bb.append(s); bc.append(c); bt.append(f"Wk{s}-{ve}")
            frames_g.append(go.Frame(
                data=[go.Bar(x=bx, y=by, base=bb, orientation="h", marker_color=bc,
                    marker_line=dict(color="white", width=1), text=bt, textposition="inside",
                    textfont=dict(color="white", size=10), showlegend=False)],
                name=str(wk),
                layout=go.Layout(title_text=f"10-Week Action Plan — Week {wk}")))

        fig_gn = go.Figure(data=[go.Bar(x=[], y=[], orientation="h")], frames=frames_g)
        fig_gn.update_layout(
            **dark_layout(height=350, title_text="10-Week Action Plan (auto-play)"),
            xaxis=dict(title="Week", range=[0, 11], dtick=1, gridcolor=GRID_COLOR),
            yaxis=dict(gridcolor=GRID_COLOR, categoryorder="array", categoryarray=tasks_r),
            barmode="stack",
            updatemenus=[dict(type="buttons", showactive=False, visible=False,
                buttons=[dict(method="animate", args=[None, {"frame":{"duration":500,"redraw":True},"fromcurrent":True}])])])
        render_autoplay(fig_gn, height=370)

        with st.expander("Deliverables", expanded=True):
            st.dataframe(pd.DataFrame([
                {"Week":"Wk 1-2","Task":"Data Audit","Owner":"Data Eng.","Deliverable":"Readiness scorecard"},
                {"Week":"Wk 2-4","Task":"Baseline PDEs","Owner":"Physics","Deliverable":"Validated loss module"},
                {"Week":"Wk 4-8","Task":"Train Prototype","Owner":"ML Team","Deliverable":"Benchmark vs Kalman"},
                {"Week":"Wk 1-10","Task":"MMD Monitors","Owner":"MLOps","Deliverable":"Drift alert dashboard"},
                {"Week":"Wk 8-10","Task":"INT8/INT4 Quant","Owner":"ML/Edge","Deliverable":"Latency test on edge HW"},
            ]), use_container_width=True, hide_index=True)

    st.markdown("""<div style="background:#142A3E;border:1px solid #00BCD4;border-radius:10px;padding:12px 24px;text-align:center;margin-top:15px;">
    <span style="color:#00BCD4;font-size:15px;"><b>The path from hype to engineering:</b></span>
    <span style="color:#E0E7EE;font-size:14px;"> audit → physics module → prototype → drift monitors (day 1) → quantization.</span></div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
st.divider()
st.caption("Assignment 2 — AI & Large Models | Physics-Informed GNSS Positioning | All data synthetic/mock.")
