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

# ─────────────────────────────────────────────
# Plotly helpers
# ─────────────────────────────────────────────
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

def dark_axes(fig):
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    return fig

def scene_dark(title=""):
    return dict(
        bgcolor=PLOT_BG,
        xaxis=dict(backgroundcolor=CARD_BG, gridcolor=GRID_COLOR,
                    showbackground=True, color=TEXT_COLOR),
        yaxis=dict(backgroundcolor=CARD_BG, gridcolor=GRID_COLOR,
                    showbackground=True, color=TEXT_COLOR),
        zaxis=dict(backgroundcolor=CARD_BG, gridcolor=GRID_COLOR,
                    showbackground=True, color=TEXT_COLOR),
    )


def render_animated(fig, height=500, key=None):
    """Render a Plotly figure with auto-playing animation via HTML component."""
    html = fig.to_html(
        include_plotlyjs="cdn", full_html=True, auto_play=True,
        config={"displayModeBar": True, "scrollZoom": True},
    )
    html = html.replace("<body>",
        '<body style="background-color:#0D1B2A; margin:0; padding:0; overflow:hidden;">')
    components.html(html, height=height, scrolling=False)


def render_3d_rotating(fig, height=550, speed=0.4, key=None):
    """Render a 3D Plotly figure that auto-rotates."""
    html = fig.to_html(
        include_plotlyjs="cdn", full_html=True,
        config={"displayModeBar": True, "scrollZoom": True},
    )
    html = html.replace("<body>",
        '<body style="background-color:#0D1B2A; margin:0; padding:0; overflow:hidden;">')
    rotate_js = f"""
    <script>
    (function() {{
        function tryRotate() {{
            var gd = document.querySelector('.js-plotly-plot');
            if (!gd || !gd.layout) {{ setTimeout(tryRotate, 200); return; }}
            var angle = 0;
            var dragging = false;
            gd.addEventListener('mousedown', function() {{ dragging = true; }});
            gd.addEventListener('mouseup', function() {{ dragging = false; }});
            setInterval(function() {{
                if (dragging) return;
                angle += {speed};
                var rad = angle * Math.PI / 180;
                var eye = {{x: 1.6*Math.cos(rad), y: 1.6*Math.sin(rad), z: 0.8}};
                Plotly.relayout(gd, {{'scene.camera.eye': eye}});
            }}, 50);
        }}
        setTimeout(tryRotate, 800);
    }})();
    </script>
    """
    html = html.replace("</body>", rotate_js + "</body>")
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
)
drift_months = st.sidebar.slider(
    "Deployment duration (months)",
    min_value=1, max_value=24, value=6, step=1,
    help="How long the model has been deployed — affects calibration drift.",
)

np.random.seed(42)

ENV_CONFIG = {
    "Open Sky":      {"sats": 10, "cn0": 42, "error_base": 3.5,  "multipath": 0.1, "idx": 0},
    "Urban Canyon":  {"sats": 3,  "cn0": 22, "error_base": 55,   "multipath": 0.7, "idx": 1},
    "Tunnel Exit":   {"sats": 1,  "cn0": 10, "error_base": 70,   "multipath": 0.3, "idx": 2},
    "Deep Urban":    {"sats": 2,  "cn0": 15, "error_base": 90,   "multipath": 0.9, "idx": 3},
}
env = ENV_CONFIG[environment]

# ═════════════════════════════════════════════
# TABS
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

    col_3d, col_info = st.columns([3, 2])

    with col_3d:
        # 3D satellite constellation — auto-rotating
        st.markdown("#### 3D Satellite Constellation")
        num_sats = 24
        np.random.seed(7)
        # Generate satellite positions on orbital shell (~26,600 km radius)
        inc = 55 * np.pi / 180
        planes = 6
        sats_per_plane = num_sats // planes
        sat_x, sat_y, sat_z = [], [], []
        sat_colors, sat_labels = [], []
        visible_count = env["sats"]

        for p in range(planes):
            raan = p * 60 * np.pi / 180
            for s in range(sats_per_plane):
                nu = s * (360 / sats_per_plane) * np.pi / 180 + p * 15 * np.pi / 180
                # Orbital position
                r = 1.0  # normalized radius
                x_orb = r * (np.cos(raan) * np.cos(nu) - np.sin(raan) * np.sin(nu) * np.cos(inc))
                y_orb = r * (np.sin(raan) * np.cos(nu) + np.cos(raan) * np.sin(nu) * np.cos(inc))
                z_orb = r * np.sin(nu) * np.sin(inc)
                sat_x.append(x_orb)
                sat_y.append(y_orb)
                sat_z.append(z_orb)
                idx = p * sats_per_plane + s
                visible = idx < visible_count
                sat_colors.append(GREEN if visible else CORAL)
                sat_labels.append(f"SV{idx+1}: {'Visible' if visible else 'Blocked'}")

        # Earth sphere
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 30)
        ex = 0.35 * np.outer(np.cos(u), np.sin(v))
        ey = 0.35 * np.outer(np.sin(u), np.sin(v))
        ez = 0.35 * np.outer(np.ones_like(u), np.cos(v))

        fig_3d_sat = go.Figure()
        fig_3d_sat.add_trace(go.Surface(
            x=ex, y=ey, z=ez, colorscale=[[0, "#1A5276"], [1, "#2E86C1"]],
            showscale=False, opacity=0.6, name="Earth",
            hoverinfo="skip",
        ))
        # Orbital rings
        for p in range(planes):
            raan = p * 60 * np.pi / 180
            theta = np.linspace(0, 2 * np.pi, 100)
            rx = np.cos(raan) * np.cos(theta) - np.sin(raan) * np.sin(theta) * np.cos(inc)
            ry = np.sin(raan) * np.cos(theta) + np.cos(raan) * np.sin(theta) * np.cos(inc)
            rz = np.sin(theta) * np.sin(inc)
            fig_3d_sat.add_trace(go.Scatter3d(
                x=rx, y=ry, z=rz, mode="lines",
                line=dict(color=GRID_COLOR, width=1),
                showlegend=False, hoverinfo="skip",
            ))
        # Satellites
        fig_3d_sat.add_trace(go.Scatter3d(
            x=sat_x, y=sat_y, z=sat_z, mode="markers+text",
            marker=dict(size=6, color=sat_colors, symbol="diamond",
                        line=dict(width=1, color="white")),
            text=sat_labels, textposition="top center",
            textfont=dict(size=8, color=TEXT_COLOR),
            name="Satellites", hoverinfo="text",
        ))
        # Receiver on Earth surface
        fig_3d_sat.add_trace(go.Scatter3d(
            x=[0.36], y=[0], z=[0], mode="markers",
            marker=dict(size=8, color=GOLD, symbol="circle",
                        line=dict(width=2, color="white")),
            name="Receiver", hoverinfo="name",
        ))

        fig_3d_sat.update_layout(
            **dark_layout(height=500, title_text=f"GPS Constellation — {environment} ({visible_count} SVs visible)"),
            scene=dict(
                **scene_dark(),
                xaxis_title="", yaxis_title="", zaxis_title="",
                aspectmode="cube",
                camera=dict(eye=dict(x=1.6, y=1.6, z=0.8)),
            ),
            showlegend=True,
        )
        render_3d_rotating(fig_3d_sat, height=520)

    with col_info:
        # Gauge indicators
        fig_gauge = make_subplots(
            rows=1, cols=3,
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
        )
        fig_gauge.add_trace(go.Indicator(
            mode="gauge+number", value=env["sats"],
            title={"text": "Satellites", "font": {"size": 13, "color": TEXT_COLOR}},
            number={"font": {"color": TEXT_COLOR}},
            gauge={"axis": {"range": [0, 12], "tickcolor": MUTED},
                   "bar": {"color": PINN_BLUE}, "bgcolor": CARD_BG,
                   "steps": [{"range": [0, 3], "color": "#3E1515"},
                             {"range": [3, 6], "color": "#3E250A"},
                             {"range": [6, 12], "color": "#0A3E20"}],
                   "threshold": {"line": {"color": GREEN, "width": 3}, "value": 4, "thickness": 0.8}},
        ), row=1, col=1)
        fig_gauge.add_trace(go.Indicator(
            mode="gauge+number", value=env["cn0"],
            title={"text": "C/N₀ (dB-Hz)", "font": {"size": 13, "color": TEXT_COLOR}},
            number={"font": {"color": TEXT_COLOR}},
            gauge={"axis": {"range": [0, 50], "tickcolor": MUTED},
                   "bar": {"color": CYAN}, "bgcolor": CARD_BG,
                   "steps": [{"range": [0, 15], "color": "#3E1515"},
                             {"range": [15, 30], "color": "#3E250A"},
                             {"range": [30, 50], "color": "#0A3E20"}],
                   "threshold": {"line": {"color": GREEN, "width": 3}, "value": 35, "thickness": 0.8}},
        ), row=1, col=2)
        fig_gauge.add_trace(go.Indicator(
            mode="gauge+number", value=env["error_base"],
            title={"text": "Error (m)", "font": {"size": 13, "color": TEXT_COLOR}},
            number={"font": {"color": TEXT_COLOR}},
            gauge={"axis": {"range": [0, 100], "tickcolor": MUTED},
                   "bar": {"color": CORAL}, "bgcolor": CARD_BG,
                   "steps": [{"range": [0, 5], "color": "#0A3E20"},
                             {"range": [5, 30], "color": "#3E250A"},
                             {"range": [30, 100], "color": "#3E1515"}],
                   "threshold": {"line": {"color": GREEN, "width": 3}, "value": 5, "thickness": 0.8}},
        ), row=1, col=3)
        fig_gauge.update_layout(**dark_layout(height=250, title_text=f"Environment: {environment}"))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"""
        <div class="{'success-card' if env['sats'] >= 6 else 'warn-card' if env['sats'] >= 2 else 'danger-card'}">
            <b>Status:</b> {'Acceptable for navigation' if env['sats'] >= 6 else 'Errors exceed lane width — multipath dominant' if env['sats'] >= 2 else 'Signal denied — classical methods fail'}
        </div>
        """, unsafe_allow_html=True)

        # Animated degradation bar chart — sweeps through environments
        envs_all = ["Open Sky", "Urban Canyon", "Tunnel Exit", "Deep Urban"]
        errors_all = [3.5, 55, 70, 90]
        colors_bar = [GREEN, AMBER, CORAL, "#E53935"]

        frames_deg = []
        for step in range(1, len(envs_all) + 1):
            frames_deg.append(go.Frame(
                data=[go.Bar(
                    x=envs_all[:step], y=errors_all[:step],
                    marker_color=colors_bar[:step],
                    text=[f"{e} m" for e in errors_all[:step]], textposition="outside",
                    textfont=dict(color=TEXT_COLOR, size=11),
                )],
                name=str(step),
            ))

        fig_deg = go.Figure(
            data=[go.Bar(x=[], y=[], marker_color=[])],
            frames=frames_deg,
            layout=go.Layout(
                **dark_layout(height=320, title_text="Error by Environment (auto-play)"),
                xaxis=dict(gridcolor=GRID_COLOR, range=[-0.5, 3.5]),
                yaxis=dict(gridcolor=GRID_COLOR, title="Error (m)", range=[0, 110]),
                updatemenus=[dict(type="buttons", showactive=False, visible=False,
                                  buttons=[dict(method="animate",
                                                args=[None, {"frame": {"duration": 800, "redraw": True},
                                                             "fromcurrent": True}])])],
            ),
        )
        fig_deg.add_hline(y=5, line_dash="dash", line_color=GREEN,
                          annotation_text="5 m target", annotation_font_color=GREEN)
        render_animated(fig_deg, height=340)

        st.markdown("""
        <div class="info-card">
            <b>Classical Baseline:</b> Kalman filters + INS aiding — hand-tuned and brittle in novel environments.
        </div>
        """, unsafe_allow_html=True)


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
            <p style="color:#FF6B6B;"><b>⚠ Critical Risk:</b> No physics guardrail — will predict physically impossible 50 m jumps.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_right:
        st.markdown("""
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

    st.markdown("""
    <div style="background:#0A2E50; border:1px solid #00BCD4; border-radius:8px; padding:10px 20px; text-align:center; margin:10px 0;">
        <span style="color:#E0E7EE; font-size:16px; font-style:italic;">L(θ) = L<sub>data</sub>(θ) + λ · L<sub>physics</sub>(θ)</span>
        <span style="color:#FFB74D; font-size:13px; margin-left:20px;"><b>λ validated against error budgets — not guesswork</b></span>
    </div>
    """, unsafe_allow_html=True)

    col_3d, col_lam = st.columns([3, 2])

    with col_3d:
        # 3D error surface: Error = f(log_samples, lambda)
        st.markdown("#### 3D Trade-off Surface: Data × Lambda → Error")
        log_samples = np.linspace(2, 6, 60)
        lambda_vals = np.linspace(0, 2.0, 60)
        X, Y = np.meshgrid(log_samples, lambda_vals)

        eff_lam = np.clip(Y, 0.1, 1.5)
        pinn_scale = 150000 / (1 + 200 * eff_lam)
        Z_pinn = 80 * np.exp(-10**X / pinn_scale) + 2
        Z_pinn += 15 * np.maximum(0, Y - 0.8)**1.5
        penalty = np.where(Y < 0.1, 8 * (0.1 - Y) / 0.1, 0)
        Z_pinn += penalty
        Z_pinn = np.clip(Z_pinn, 1, 90)

        Z_nn = 80 * np.exp(-10**X / 150000) + 5
        Z_nn = np.clip(Z_nn, 2, 90)

        fig_3d_surf = go.Figure()
        fig_3d_surf.add_trace(go.Surface(
            x=log_samples, y=lambda_vals, z=Z_pinn,
            colorscale=[[0, "#0A3E20"], [0.3, "#29B6F6"], [0.6, "#FFB74D"], [1, "#FF6B6B"]],
            name="PINN Error", opacity=0.85,
            colorbar=dict(title="Error (m)", tickfont=dict(color=TEXT_COLOR),
                          titlefont=dict(color=TEXT_COLOR)),
            contours=dict(z=dict(show=True, usecolormap=True, project_z=True, highlightcolor="white")),
            hovertemplate="Samples: 10^%{x:.1f}<br>λ: %{y:.2f}<br>Error: %{z:.1f} m<extra>PINN</extra>",
        ))
        # NN plane (flat in lambda dimension)
        fig_3d_surf.add_trace(go.Surface(
            x=log_samples, y=lambda_vals, z=Z_nn,
            colorscale=[[0, "rgba(239,83,59,0.3)"], [1, "rgba(239,83,59,0.6)"]],
            name="NN Error", opacity=0.4, showscale=False,
            hovertemplate="Samples: 10^%{x:.1f}<br>Error: %{z:.1f} m<extra>NN (no λ)</extra>",
        ))
        # Mark current lambda
        cur_samples_idx = 30
        fig_3d_surf.add_trace(go.Scatter3d(
            x=[log_samples[cur_samples_idx]], y=[lambda_weight],
            z=[Z_pinn[np.argmin(np.abs(lambda_vals - lambda_weight)), cur_samples_idx]],
            mode="markers", marker=dict(size=8, color=GOLD, symbol="diamond",
                                         line=dict(width=2, color="white")),
            name=f"Current λ={lambda_weight:.2f}",
        ))

        fig_3d_surf.update_layout(
            **dark_layout(height=520, title_text="Error Landscape: PINN (surface) vs NN (plane)"),
            scene=dict(
                **scene_dark(),
                xaxis_title="log₁₀(Training Samples)",
                yaxis_title="λ (Physics Weight)",
                zaxis_title="Position Error (m)",
                camera=dict(eye=dict(x=1.8, y=-1.2, z=0.9)),
            ),
        )
        render_3d_rotating(fig_3d_surf, height=540, speed=0.25)

    with col_lam:
        # Lambda sensitivity (2D, interactive)
        lambdas = np.linspace(0, 2.0, 150)
        lam_error = 3 + 40 * np.exp(-8 * lambdas) + 15 * np.maximum(0, lambdas - 0.8)**1.5
        np.random.seed(42)
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
        cur_idx = np.argmin(np.abs(lambdas - lambda_weight))
        fig_lam.add_trace(go.Scatter(
            x=[lambda_weight], y=[lam_error[cur_idx]], mode="markers",
            marker=dict(size=14, color=GOLD, symbol="star",
                        line=dict(width=2, color="white")),
            name=f"λ={lambda_weight:.2f} → {lam_error[cur_idx]:.1f} m",
        ))
        fig_lam.add_annotation(x=0.05, y=40, text="λ too low<br>→ No physics<br>guardrail",
                               showarrow=False, font=dict(color=CORAL, size=10))
        fig_lam.add_annotation(x=1.7, y=40, text="λ too high<br>→ Ignores data",
                               showarrow=False, font=dict(color=CORAL, size=10))

        fig_lam.update_layout(**dark_layout(
            height=380, title_text="λ Sensitivity",
            xaxis_title="λ", yaxis_title="Error (m)", yaxis_range=[0, 55],
        ))
        dark_axes(fig_lam)
        st.plotly_chart(fig_lam, use_container_width=True)

        if lambda_weight < 0.1:
            st.markdown('<div class="danger-card"><b>⚠ λ too low</b> — physics guardrail disabled.</div>', unsafe_allow_html=True)
        elif lambda_weight > 1.2:
            st.markdown('<div class="warn-card"><b>⚠ λ too high</b> — over-constrains to physics.</div>', unsafe_allow_html=True)
        elif 0.3 <= lambda_weight <= 0.7:
            st.markdown('<div class="success-card"><b>✓ λ in optimal range.</b> Validate on held-out trajectories.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-card"><b>ℹ λ outside optimal zone</b> — validate carefully.</div>', unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        m1.metric("Current λ error", f"{lam_error[cur_idx]:.1f} m")
        opt_idx = np.argmin(lam_error)
        m2.metric("Optimal λ", f"{lambdas[opt_idx]:.2f} → {lam_error[opt_idx]:.1f} m")


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
            <p style="color:#E0E7EE;">10,000 clean samples × 1.5 bits outweigh 1,000,000 noisy × 0.2 bits.</p>
            <p style="color:#66BB6A;"><b>Stop collecting. Start auditing.</b></p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="warn-card">
            <h4 style="color:#FFB74D; margin-top:0;">⚠ MNAR Missingness</h4>
            <p style="color:#E0E7EE;">Signal drops in tunnels = informative features. Standard imputation destroys this.</p>
            <p style="color:#FFB74D;"><b>Preserve dropout patterns.</b></p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="danger-card">
            <h4 style="color:#FF6B6B; margin-top:0;">⛔ Temporal Leakage</h4>
            <p style="color:#E0E7EE;">Training on post-processed RTK = model cheats. Deployed accuracy collapses.</p>
            <p style="color:#FF6B6B;"><b>Strict temporal splits. Non-negotiable.</b></p>
        </div>
        """, unsafe_allow_html=True)

    col_sig, col_val = st.columns(2)

    with col_sig:
        # Auto-animated GNSS signal trace with dropout
        st.markdown("#### Live GNSS Signal Simulation (MNAR Dropout)")
        t_full = np.linspace(0, 60, 300)
        signal_full = 35 + 5 * np.sin(0.3 * t_full) + np.random.normal(0, 1.5, 300)
        tunnel_mask = (t_full > 20) & (t_full < 35)

        n_frames = 30
        frame_indices = np.linspace(10, 300, n_frames, dtype=int)
        frames_sig = []
        for fi in frame_indices:
            t_slice = t_full[:fi]
            s_slice = signal_full[:fi].copy()
            mask_slice = tunnel_mask[:fi]
            s_display = s_slice.copy()
            s_display[mask_slice] = np.nan

            frames_sig.append(go.Frame(
                data=[
                    go.Scatter(x=t_slice, y=s_display, mode="lines",
                               line=dict(color=CYAN, width=2), name="C/N₀ Signal"),
                ],
                name=str(fi),
            ))

        fig_sig = go.Figure(
            data=[go.Scatter(x=[], y=[], mode="lines", line=dict(color=CYAN, width=2), name="C/N₀ Signal")],
            frames=frames_sig,
        )
        fig_sig.add_vrect(x0=20, x1=35, fillcolor=CORAL, opacity=0.12,
                          annotation_text="Tunnel (MNAR)", annotation_font_color=CORAL,
                          annotation_position="top left")
        fig_sig.update_layout(
            **dark_layout(height=350, title_text="Signal Trace — Dropout is Information"),
            xaxis=dict(title="Time (s)", range=[0, 62], gridcolor=GRID_COLOR),
            yaxis=dict(title="C/N₀ (dB-Hz)", range=[20, 50], gridcolor=GRID_COLOR),
            updatemenus=[dict(type="buttons", showactive=False, visible=False,
                              buttons=[dict(method="animate",
                                            args=[None, {"frame": {"duration": 120, "redraw": True},
                                                         "fromcurrent": True, "mode": "immediate"}])])],
        )
        render_animated(fig_sig, height=370)

    with col_val:
        # Information value with interactive slider
        st.markdown("#### Information Value: Clean vs Noisy")
        noise_level = st.slider("Noise level (bits/sample for noisy data)", 0.05, 1.5, 0.2, 0.05, key="noise")
        clean_bits = 10_000 * 1.5
        noisy_bits = 1_000_000 * noise_level

        fig_info = go.Figure()
        fig_info.add_trace(go.Bar(
            x=["10K Clean<br>(1.5 bits/sample)", f"1M Noisy<br>({noise_level:.2f} bits/sample)"],
            y=[clean_bits, noisy_bits],
            marker_color=[GREEN, CORAL],
            text=[f"{clean_bits:,.0f} bits", f"{noisy_bits:,.0f} bits"],
            textposition="outside", textfont=dict(color=TEXT_COLOR, size=13),
        ))
        verdict = "Noisy wins on volume" if noisy_bits > clean_bits else "Clean wins — quality matters"
        fig_info.update_layout(**dark_layout(
            height=350, title_text="Total Information Content",
            yaxis_title="Total Bits",
            annotations=[dict(x=0.5, y=1.05, xref="paper", yref="paper", showarrow=False,
                              text=verdict, font=dict(color=AMBER if noisy_bits > clean_bits else GREEN, size=13))],
        ))
        dark_axes(fig_info)
        st.plotly_chart(fig_info, use_container_width=True)

    # Temporal leakage timeline
    st.markdown("#### Temporal Leakage Timeline")
    fig_leak = go.Figure()
    fig_leak.add_trace(go.Bar(
        x=[6], y=["Timeline"], orientation="h", base=[0],
        marker_color="rgba(10,62,32,0.7)", marker_line=dict(color=GREEN, width=2),
        name="Training Window", text=["Training Data"], textposition="inside",
        textfont=dict(color=GREEN, size=12),
    ))
    fig_leak.add_trace(go.Bar(
        x=[4], y=["Timeline"], orientation="h", base=[6],
        marker_color="rgba(62,21,21,0.7)", marker_line=dict(color=CORAL, width=2),
        name="Inference Window", text=["Future / Deployment"], textposition="inside",
        textfont=dict(color=CORAL, size=12),
    ))
    fig_leak.add_annotation(x=5, y="Timeline", ax=7.5, ay="Timeline",
                            xref="x", yref="y", axref="x", ayref="y",
                            showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=3, arrowcolor=CORAL)
    fig_leak.add_annotation(x=6.25, y="Timeline", text="⚠ LEAKAGE", showarrow=False,
                            font=dict(color=CORAL, size=14), bgcolor="#3E1515", bordercolor=CORAL,
                            borderwidth=1, yshift=-40)
    fig_leak.add_vline(x=6, line_color=AMBER, line_width=3)
    fig_leak.add_annotation(x=6, y="Timeline", text="▼ Temporal Boundary", showarrow=False,
                            yshift=40, font=dict(color=AMBER, size=12))
    fig_leak.update_layout(**dark_layout(height=200, barmode="stack", showlegend=True))
    fig_leak.update_xaxes(showticklabels=False, gridcolor=GRID_COLOR)
    fig_leak.update_yaxes(showticklabels=False, gridcolor=GRID_COLOR)
    st.plotly_chart(fig_leak, use_container_width=True)

    st.markdown("""
    <div class="info-card">
        <b>Pre-Training Audit:</b>
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
            <p style="color:#E0E7EE;"><b>The Trap:</b> Suburban model deployed downtown.</p>
            <p style="color:#E0E7EE;"><b>Reality:</b> Satellite geometry, multipath, rain all change. Distribution shifts silently.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="danger-card">
            <h4 style="color:#FF6B6B; margin-top:0;">⚠ Calibration Drift (Hardware)</h4>
            <p style="color:#E0E7EE;"><b>The Trap:</b> Receiver clocks accumulate error over months.</p>
            <p style="color:#E0E7EE;"><b>Reality:</b> Antenna degrades, firmware updates alter pseudorange calculations.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        # Auto-animated distribution shift over months
        st.markdown("#### Distribution Shift Animation (auto-play)")
        x_range = np.linspace(-5, 12, 300)
        train_dist = np.exp(-0.5 * x_range**2) / np.sqrt(2 * np.pi)

        frames_shift = []
        for mo in range(0, drift_months + 1):
            d_mean = 0.15 * mo
            d_std = 1 + 0.05 * mo
            dep_dist = np.exp(-0.5 * ((x_range - d_mean)/d_std)**2) / (d_std * np.sqrt(2 * np.pi))
            mmd = 0.05 * mo + 0.02 * mo**0.5

            frames_shift.append(go.Frame(
                data=[
                    go.Scatter(x=x_range, y=train_dist, mode="lines",
                               line=dict(color=PINN_BLUE, width=3), fill="tozeroy",
                               fillcolor="rgba(41,182,246,0.2)", name="Training"),
                    go.Scatter(x=x_range, y=dep_dist, mode="lines",
                               line=dict(color=CORAL, width=3), fill="tozeroy",
                               fillcolor="rgba(255,107,107,0.2)", name=f"Deploy ({mo}mo)"),
                ],
                name=str(mo),
                layout=go.Layout(
                    title_text=f"Distribution Shift — Month {mo}  |  MMD = {mmd:.2f} {'⚠ ALERT' if mmd > 0.5 else '✓'}",
                ),
            ))

        fig_shift = go.Figure(
            data=frames_shift[0].data if frames_shift else [],
            frames=frames_shift,
        )
        fig_shift.update_layout(
            **dark_layout(height=380, title_text="Distribution Shift — Month 0"),
            xaxis=dict(title="Feature Space", range=[-5, 12], gridcolor=GRID_COLOR),
            yaxis=dict(title="Density", range=[0, 0.5], gridcolor=GRID_COLOR),
            updatemenus=[dict(type="buttons", showactive=False, visible=False,
                              buttons=[dict(method="animate",
                                            args=[None, {"frame": {"duration": 400, "redraw": True},
                                                         "fromcurrent": True}])])],
        )
        render_animated(fig_shift, height=400)

    # 3D: Error surface from shift × drift
    st.markdown("#### 3D Error Surface: Covariate Shift × Calibration Drift → Positioning Error")
    shift_range = np.linspace(0, 3, 40)
    drift_range = np.linspace(0, 5, 40)
    SH, DR = np.meshgrid(shift_range, drift_range)
    # Error grows with both shift and drift, multiplicatively
    ERR = 3 + 8 * SH + 5 * DR + 3 * SH * DR + np.random.normal(0, 0.5, SH.shape)
    ERR = np.clip(ERR, 2, 100)

    fig_3d_err = go.Figure()
    fig_3d_err.add_trace(go.Surface(
        x=shift_range, y=drift_range, z=ERR,
        colorscale=[[0, "#0A3E20"], [0.3, "#29B6F6"], [0.6, "#FFB74D"], [1, "#FF6B6B"]],
        colorbar=dict(title="Error (m)", tickfont=dict(color=TEXT_COLOR),
                      titlefont=dict(color=TEXT_COLOR)),
        contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
        hovertemplate="Shift: %{x:.1f}<br>Drift: %{y:.1f} ns<br>Error: %{z:.1f} m<extra></extra>",
    ))
    # Mark current deployment point
    cur_shift = 0.15 * drift_months
    cur_drift_ns = 0.3 * drift_months
    cur_err = 3 + 8 * min(cur_shift, 3) + 5 * min(cur_drift_ns, 5) + 3 * min(cur_shift, 3) * min(cur_drift_ns, 5)
    fig_3d_err.add_trace(go.Scatter3d(
        x=[min(cur_shift, 3)], y=[min(cur_drift_ns, 5)], z=[cur_err],
        mode="markers", marker=dict(size=8, color=GOLD, symbol="diamond",
                                     line=dict(width=2, color="white")),
        name=f"Current ({drift_months}mo)",
    ))

    fig_3d_err.update_layout(
        **dark_layout(height=480, title_text="Error grows multiplicatively with shift × drift"),
        scene=dict(
            **scene_dark(),
            xaxis_title="Covariate Shift",
            yaxis_title="Clock Drift (ns)",
            zaxis_title="Position Error (m)",
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
        ),
    )
    render_3d_rotating(fig_3d_err, height=500, speed=0.3)

    st.markdown("""
    <div class="success-card">
        <h4 style="color:#009688; margin-top:0;">🛡 Defense Protocol</h4>
        <p style="color:#E0E7EE;"><b>Deploy MMD Monitors</b> on input feature pipelines. When statistical distance exceeds threshold →
        <b>trigger retraining</b> or <b>fallback to classical Kalman filtering</b>.</p>
        <p style="color:#009688;"><b>→ Silent failures become detectable failures.</b></p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 5 — Compute Economics
# ══════════════════════════════════════════════
with tab5:
    st.subheader("Compute Economics: Why FLOPs Lie")
    st.markdown(f"Matrix multiplication dominates FLOPs — but **FLOPs ≠ wall-clock speed**. Model: **{model_size}M params**.")

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(f"""
        <div class="info-card">
            <h4 style="color:#29B6F6; margin-top:0;">Training (CAPEX — One-Time)</h4>
            <ul style="color:#E0E7EE;">
                <li>Compute-bound — parallelizable</li>
                <li>≈ 6N FLOPs/token | ~12 bytes/param</li>
                <li>{model_size}M = <b>{12 * model_size / 1000:.1f} GB</b> training state</li>
            </ul>
            <p style="color:#FFB74D;"><b>PINN:</b> +20–40%/epoch, but converges with far less data.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_right:
        st.markdown("""
        <div class="danger-card">
            <h4 style="color:#FF6B6B; margin-top:0;">Inference (OPEX — Ongoing)</h4>
            <ul style="color:#E0E7EE;">
                <li>Memory-bandwidth-bound — GPU idle waiting for weights</li>
                <li>≈ 2N FLOPs/token — latency is the real constraint</li>
                <li>10 Hz GNSS = <100 ms per correction</li>
                <li>Scales linearly with fleet size</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    col_cost, col_roof = st.columns(2)

    with col_cost:
        # Animated training cost growth
        sizes = np.arange(5, 205, 5)
        nn_flops = 6 * sizes
        pinn_flops = nn_flops * 1.3
        mem_gb = 12 * sizes * 1e6 / 1e9

        fig_cost = make_subplots(specs=[[{"secondary_y": True}]])
        fig_cost.add_trace(go.Bar(
            x=[f"{s}M" for s in sizes[::4]], y=nn_flops[::4],
            name="NN (6N)", marker_color=NN_RED, opacity=0.8,
        ), secondary_y=False)
        fig_cost.add_trace(go.Bar(
            x=[f"{s}M" for s in sizes[::4]], y=pinn_flops[::4],
            name="PINN (7.8N)", marker_color=PINN_BLUE, opacity=0.8,
        ), secondary_y=False)
        fig_cost.add_trace(go.Scatter(
            x=[f"{s}M" for s in sizes[::4]], y=mem_gb[::4],
            name="Memory (GB)", mode="lines+markers",
            line=dict(color=AMBER, width=2.5), marker=dict(size=7, color=AMBER),
        ), secondary_y=True)

        sel_idx = np.argmin(np.abs(sizes - model_size))
        fig_cost.add_trace(go.Scatter(
            x=[f"{model_size}M"], y=[nn_flops[sel_idx]], mode="markers",
            marker=dict(size=16, color=GOLD, symbol="star"), name=f"Selected: {model_size}M",
        ), secondary_y=False)

        fig_cost.update_layout(**dark_layout(height=400, title_text="Training Cost: NN vs PINN", barmode="group"))
        fig_cost.update_xaxes(title_text="Model Size", gridcolor=GRID_COLOR)
        fig_cost.update_yaxes(title_text="GFLOPs/Token", gridcolor=GRID_COLOR, secondary_y=False)
        fig_cost.update_yaxes(title_text="Memory (GB)", gridcolor=GRID_COLOR, secondary_y=True,
                              title_font=dict(color=AMBER), tickfont=dict(color=AMBER))
        st.plotly_chart(fig_cost, use_container_width=True)

    with col_roof:
        # 3D Roofline: model size × arithmetic intensity → performance
        st.markdown("#### 3D Roofline Model")
        model_sizes_3d = np.linspace(5, 200, 30)
        ai_3d = np.logspace(-1, 2, 30)
        MS, AI = np.meshgrid(model_sizes_3d, ai_3d)
        bw = 900
        peak = 10000
        PERF = np.minimum(peak, bw * AI) * (1 - 0.001 * MS)
        PERF = np.clip(PERF, 1, peak)

        fig_roof_3d = go.Figure()
        fig_roof_3d.add_trace(go.Surface(
            x=model_sizes_3d, y=np.log10(ai_3d), z=np.log10(PERF),
            colorscale=[[0, CARD_BG], [0.5, CYAN], [1, GREEN]],
            opacity=0.7, showscale=False,
            hovertemplate="Size: %{x:.0f}M<br>log₁₀(AI): %{y:.1f}<br>log₁₀(GFLOPS): %{z:.1f}<extra></extra>",
        ))
        # Training point
        fig_roof_3d.add_trace(go.Scatter3d(
            x=[model_size], y=[np.log10(15)], z=[np.log10(peak * 0.7)],
            mode="markers+text", text=["Training"], textposition="top center",
            marker=dict(size=8, color=GREEN, symbol="circle", line=dict(width=2, color="white")),
            textfont=dict(color=GREEN, size=10), name="Training",
        ))
        # Inference point
        fig_roof_3d.add_trace(go.Scatter3d(
            x=[model_size], y=[np.log10(0.5)], z=[np.log10(450)],
            mode="markers+text", text=["Inference"], textposition="top center",
            marker=dict(size=8, color=CORAL, symbol="circle", line=dict(width=2, color="white")),
            textfont=dict(color=CORAL, size=10), name="Inference",
        ))

        fig_roof_3d.update_layout(
            **dark_layout(height=450, title_text="3D Roofline — Compute vs Bandwidth"),
            scene=dict(
                **scene_dark(),
                xaxis_title="Model Size (M)",
                yaxis_title="log₁₀(Arith. Intensity)",
                zaxis_title="log₁₀(GFLOPS)",
                camera=dict(eye=dict(x=1.5, y=-1.3, z=0.9)),
            ),
        )
        render_3d_rotating(fig_roof_3d, height=470, speed=0.3)

    st.markdown("""
    <div class="info-card">
        <b>Key insight:</b> Training dominates CAPEX. Inference dominates OPEX.
        For fleet-scale GNSS, <span style="color:#FF6B6B;"><b>inference cost is the binding constraint</b></span>.
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("NN FLOPs/token", f"{6 * model_size * 1e6 / 1e9:.1f} GFLOPs")
    m2.metric("PINN FLOPs/token", f"{6 * model_size * 1e6 / 1e9 * 1.3:.1f} GFLOPs")
    m3.metric("Training memory", f"{12 * model_size * 1e6 / 1e9:.2f} GB")


# ══════════════════════════════════════════════
# TAB 6 — Edge Deployment
# ══════════════════════════════════════════════
with tab6:
    st.subheader("Solving the Inference Bottleneck for Edge Deployment")
    st.markdown(f"Spare FLOPs, not spare bandwidth. **{precision}** | **Batch: {batch_size}**")

    st.markdown("""
    <div style="display:flex; gap:10px; justify-content:center; margin:15px 0;">
        <div style="background:#142A3E; border:2px solid #29B6F6; border-radius:10px; padding:15px 25px; text-align:center; flex:1;">
            <p style="color:#29B6F6; font-weight:bold; margin:0;">GPU Memory</p>
            <p style="color:#A0AEBB; margin:0; font-size:12px;">Weights storage</p>
        </div>
        <div style="display:flex; align-items:center; color:#00BCD4; font-size:28px;">→</div>
        <div style="background:#3E250A; border:2px solid #FFB74D; border-radius:10px; padding:15px 25px; text-align:center; flex:1;">
            <p style="color:#FFB74D; font-weight:bold; margin:0;">Transfer to Cores</p>
            <p style="color:#A0AEBB; margin:0; font-size:12px;">THE BOTTLENECK</p>
        </div>
        <div style="display:flex; align-items:center; color:#00BCD4; font-size:28px;">→</div>
        <div style="background:#0A3520; border:2px solid #66BB6A; border-radius:10px; padding:15px 25px; text-align:center; flex:1;">
            <p style="color:#66BB6A; font-weight:bold; margin:0;">Matrix Multiply</p>
            <p style="color:#A0AEBB; margin:0; font-size:12px;">~2N FLOPs (spare capacity)</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_quant, col_lat = st.columns(2)

    with col_quant:
        quant_levels = ["FP32", "FP16", "INT8", "INT4"]
        mem_red = [1, 2, 4, 8]
        nn_acc = [100.0, 99.8, 99.2, 96.5]
        pinn_acc = [100.0, 99.9, 99.5, 97.0]

        fig_quant = go.Figure()
        fig_quant.add_trace(go.Bar(x=quant_levels, y=nn_acc, name="Traditional NN",
                                    marker_color=NN_RED, opacity=0.85,
                                    text=[f"{a:.1f}%" for a in nn_acc], textposition="outside",
                                    textfont=dict(color=TEXT_COLOR, size=10)))
        fig_quant.add_trace(go.Bar(x=quant_levels, y=pinn_acc, name="PINN",
                                    marker_color=PINN_BLUE, opacity=0.85,
                                    text=[f"{a:.1f}%" for a in pinn_acc], textposition="outside",
                                    textfont=dict(color=TEXT_COLOR, size=10)))
        fig_quant.add_hline(y=99, line_dash="dash", line_color=GREEN,
                            annotation_text="99% threshold", annotation_font_color=GREEN)
        for i, (ql, mr) in enumerate(zip(quant_levels, mem_red)):
            fig_quant.add_annotation(x=ql, y=95.3, text=f"{mr}×", showarrow=False,
                                     font=dict(color=AMBER, size=12))
        sel_quant = precision.split(" ")[0]
        for ql in quant_levels:
            if ql == sel_quant:
                fig_quant.add_annotation(x=ql, y=101, text="▼ Selected", showarrow=False,
                                         font=dict(color=GOLD, size=11))
        fig_quant.update_layout(**dark_layout(
            height=420, title_text="Quantization: Accuracy vs Memory",
            xaxis_title="Precision", yaxis_title="Accuracy Retained (%)",
            yaxis_range=[94.5, 101.5], barmode="group",
        ))
        dark_axes(fig_quant)
        st.plotly_chart(fig_quant, use_container_width=True)

    with col_lat:
        # 3D Latency surface: model size × batch size → latency
        st.markdown("#### 3D Latency: Model Size × Batch → Latency")
        sizes_3d = np.arange(5, 205, 10)
        batches_3d = np.arange(1, 65, 4)
        SZ, BA = np.meshgrid(sizes_3d, batches_3d)

        bpp_map = {"FP32": 4.0, "FP16": 2.0, "INT8": 1.0, "INT4": 0.5}
        bpp = bpp_map.get(sel_quant, 1.0)
        bw_gbs = 900
        model_bytes_3d = SZ * 1e6 * bpp
        base_lat = (model_bytes_3d / (bw_gbs * 1e9)) * 1000
        compute_lat = (2 * SZ * 1e6) / (10e12) * 1000
        batch_factor = 1 + 0.15 * np.log2(np.maximum(BA, 1))
        LAT = (base_lat + compute_lat) * batch_factor + 2.0
        LAT = np.clip(LAT, 0.5, 500)

        fig_lat_3d = go.Figure()
        fig_lat_3d.add_trace(go.Surface(
            x=sizes_3d, y=batches_3d, z=LAT,
            colorscale=[[0, "#0A3E20"], [0.3, "#29B6F6"], [0.7, "#FFB74D"], [1, "#FF6B6B"]],
            colorbar=dict(title="ms", tickfont=dict(color=TEXT_COLOR),
                          titlefont=dict(color=TEXT_COLOR)),
            contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
            hovertemplate="Size: %{x}M<br>Batch: %{y}<br>Latency: %{z:.1f} ms<extra></extra>",
        ))
        # 100ms deadline plane
        fig_lat_3d.add_trace(go.Surface(
            x=sizes_3d, y=batches_3d, z=np.full_like(LAT, 100),
            colorscale=[[0, "rgba(255,107,107,0.3)"], [1, "rgba(255,107,107,0.3)"]],
            showscale=False, opacity=0.3, name="100ms deadline",
            hovertemplate="100 ms deadline<extra></extra>",
        ))
        # Current operating point
        cur_lat_val = float(LAT[np.argmin(np.abs(batches_3d - batch_size)),
                                np.argmin(np.abs(sizes_3d - model_size))])
        fig_lat_3d.add_trace(go.Scatter3d(
            x=[model_size], y=[batch_size], z=[cur_lat_val],
            mode="markers", marker=dict(size=8, color=GOLD, symbol="diamond",
                                         line=dict(width=2, color="white")),
            name=f"Current: {cur_lat_val:.1f} ms",
        ))

        fig_lat_3d.update_layout(
            **dark_layout(height=480, title_text=f"Latency Surface ({sel_quant})"),
            scene=dict(
                **scene_dark(),
                xaxis_title="Model Size (M)",
                yaxis_title="Batch Size",
                zaxis_title="Latency (ms)",
                camera=dict(eye=dict(x=1.6, y=-1.4, z=0.9)),
            ),
        )
        render_3d_rotating(fig_lat_3d, height=500, speed=0.3)

    meets = cur_lat_val < 100
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latency", f"{cur_lat_val:.1f} ms")
    m2.metric("Meets 100 ms?", "Yes ✓" if meets else "No ✗")
    fleet_cps = (1000 / max(cur_lat_val, 1)) * batch_size
    m3.metric("Fleet corrections/sec", f"{fleet_cps:,.0f}")
    m4.metric("Memory reduction", f"{int(4/bpp)}× vs FP32")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div class="{'success-card' if meets else 'danger-card'}">
            <b>{sel_quant}:</b> {int(4/bpp)}× memory reduction.
            {'Meets real-time deadline.' if meets else 'Exceeds budget — consider stronger quantization.'}
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
                "Identical — physics is training-only", "Equally viable — same graph",
                "Physics limits damage, won't detect drift", "Needs external MMD monitors",
                "ML + physics/GNSS domain", "Strong — physics transfers globally",
                "Soft — bounded by physics quality", "Scarce data, variable env, safety-critical",
            ],
        }
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True, height=480)

    with col_viz:
        # Animated radar chart — sweeps through dimensions
        categories = ["Data\nEfficiency", "Physics\nCompliance", "Safety",
                       "Generalization", "Ease of\nImpl.", "Edge\nViability"]
        nn_scores = [2, 1, 1, 2, 5, 4]
        pinn_scores = [5, 5, 4, 5, 3, 4]
        kalman_scores = [3, 3, 3, 2, 4, 5]

        frames_radar = []
        for step in range(1, len(categories) + 1):
            cats_s = categories[:step] + [categories[0]]
            nn_s = nn_scores[:step] + [nn_scores[0]]
            pinn_s = pinn_scores[:step] + [pinn_scores[0]]
            kal_s = kalman_scores[:step] + [kalman_scores[0]]
            frames_radar.append(go.Frame(
                data=[
                    go.Scatterpolar(r=nn_s, theta=cats_s, name="NN",
                                     line=dict(color=NN_RED, width=2.5),
                                     fill="toself", fillcolor="rgba(239,83,59,0.15)"),
                    go.Scatterpolar(r=pinn_s, theta=cats_s, name="PINN",
                                     line=dict(color=PINN_BLUE, width=2.5),
                                     fill="toself", fillcolor="rgba(41,182,246,0.15)"),
                    go.Scatterpolar(r=kal_s, theta=cats_s, name="Kalman",
                                     line=dict(color=KALMAN_AMBER, width=2),
                                     fill="toself", fillcolor="rgba(255,183,77,0.1)"),
                ],
                name=str(step),
            ))

        fig_radar = go.Figure(
            data=frames_radar[0].data if frames_radar else [],
            frames=frames_radar,
        )
        fig_radar.update_layout(
            polar=dict(
                bgcolor=CARD_BG,
                radialaxis=dict(visible=True, range=[0, 5], gridcolor=GRID_COLOR,
                                tickfont=dict(color=MUTED, size=9)),
                angularaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TEXT_COLOR, size=10)),
            ),
            **dark_layout(height=400, title_text="Capability Comparison (auto-build)"),
            updatemenus=[dict(type="buttons", showactive=False, visible=False,
                              buttons=[dict(method="animate",
                                            args=[None, {"frame": {"duration": 600, "redraw": True},
                                                         "fromcurrent": True}])])],
        )
        render_animated(fig_radar, height=420)

        # 3D multi-dimensional scatter
        st.markdown("#### 3D Method Comparison")
        methods = ["Kalman+INS", "Trad. NN", "PINN Hybrid"]
        data_eff = [3, 2, 5]
        safety = [3, 1, 4]
        gen = [2, 2, 5]
        m_colors = [KALMAN_AMBER, NN_RED, PINN_BLUE]
        m_sizes = [20, 20, 25]

        fig_3d_comp = go.Figure()
        for i, (m, d, s, g, c, sz) in enumerate(zip(methods, data_eff, safety, gen, m_colors, m_sizes)):
            fig_3d_comp.add_trace(go.Scatter3d(
                x=[d], y=[s], z=[g], mode="markers+text",
                marker=dict(size=sz, color=c, opacity=0.9,
                            line=dict(width=2, color="white")),
                text=[m], textposition="top center",
                textfont=dict(color=c, size=11), name=m,
            ))

        fig_3d_comp.update_layout(
            **dark_layout(height=380, title_text="3D: Efficiency × Safety × Generalization"),
            scene=dict(
                **scene_dark(),
                xaxis_title="Data Efficiency",
                yaxis_title="Safety",
                zaxis_title="Generalization",
                camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
            ),
        )
        render_3d_rotating(fig_3d_comp, height=400, speed=0.5)

    # Environment bars
    envs_comp = ["Open Sky", "Urban Canyon", "Tunnel Exit", "Deep Urban"]
    nn_errs = [4.5, 45, 85, 95]
    pinn_errs = [2.5, 8, 25, 35]
    kalman_errs = [3.5, 35, 70, 90]

    fig_env = go.Figure()
    fig_env.add_trace(go.Bar(x=envs_comp, y=kalman_errs, name="Kalman+INS", marker_color=KALMAN_AMBER, opacity=0.85))
    fig_env.add_trace(go.Bar(x=envs_comp, y=nn_errs, name="Traditional NN", marker_color=NN_RED, opacity=0.85))
    fig_env.add_trace(go.Bar(x=envs_comp, y=pinn_errs, name="PINN (Hybrid)", marker_color=PINN_BLUE, opacity=0.85))
    fig_env.add_hline(y=5, line_dash="dash", line_color=GREEN, annotation_text="5 m target", annotation_font_color=GREEN)
    if environment in envs_comp:
        si = envs_comp.index(environment)
        fig_env.add_annotation(x=envs_comp[si], y=max(nn_errs[si], kalman_errs[si]) + 8,
                               text="▼ Selected", showarrow=False, font=dict(color=GOLD, size=11))
    fig_env.update_layout(**dark_layout(height=380, title_text="Positioning Error by Environment",
                                         yaxis_title="Horizontal Error (m)", barmode="group"))
    dark_axes(fig_env)
    st.plotly_chart(fig_env, use_container_width=True)

    st.markdown("""
    <div class="info-card">
        <b>Key Insights:</b> Inference cost is identical (physics loss = training-only). Both need MMD monitors.
        PINN advantage widens in degraded environments. Physics transfers across geographies.
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
                <li>Multipath exceeds available physics models</li>
                <li>Environment is mapped and stable</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="warn-card">
            <h4 style="color:#FFB74D; margin-top:0;">🔒 Non-Negotiable Conditions:</h4>
            <ol style="color:#E0E7EE;">
                <li><b>(a)</b> Validate λ against positioning error budgets</li>
                <li><b>(b)</b> Eliminate temporal leakage from all pipelines</li>
                <li><b>(c)</b> Deploy MMD drift monitors from Day 1</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with col_plan:
        # Animated Gantt chart — builds week by week
        plan_data = [
            ("Data Audit", "Data Eng.", 1, 2, GREEN),
            ("Baseline PDEs", "Physics", 2, 4, PINN_BLUE),
            ("Train Prototype", "ML Team", 4, 8, CYAN),
            ("MMD Monitors", "MLOps", 1, 10, AMBER),
            ("INT8/INT4 Quant", "ML/Edge", 8, 10, TEAL),
        ]
        tasks = [p[0] for p in plan_data]
        tasks_r = list(reversed(tasks))

        frames_gantt = []
        for week in range(1, 11):
            bars_x, bars_y, bars_base, bars_color, bars_text = [], [], [], [], []
            for (task, owner, start, end, color) in plan_data:
                if start <= week:
                    visible_end = min(end, week)
                    bars_x.append(visible_end - start)
                    bars_y.append(task)
                    bars_base.append(start)
                    bars_color.append(color)
                    bars_text.append(f"Wk {start}-{visible_end}")

            frames_gantt.append(go.Frame(
                data=[go.Bar(
                    x=bars_x, y=bars_y, base=bars_base, orientation="h",
                    marker_color=bars_color, marker_line=dict(color="white", width=1),
                    text=bars_text, textposition="inside", textfont=dict(color="white", size=10),
                    hoverinfo="text", showlegend=False,
                )],
                name=str(week),
                layout=go.Layout(title_text=f"10-Week Action Plan — Week {week}"),
            ))

        fig_gantt = go.Figure(
            data=[go.Bar(x=[], y=[], orientation="h")],
            frames=frames_gantt,
        )
        fig_gantt.update_layout(
            **dark_layout(height=380, title_text="10-Week Action Plan (auto-play)"),
            xaxis=dict(title="Week", range=[0, 11], dtick=1, gridcolor=GRID_COLOR),
            yaxis=dict(gridcolor=GRID_COLOR, categoryorder="array", categoryarray=tasks_r),
            barmode="stack",
            updatemenus=[dict(type="buttons", showactive=False, visible=False,
                              buttons=[dict(method="animate",
                                            args=[None, {"frame": {"duration": 500, "redraw": True},
                                                         "fromcurrent": True}])])],
        )
        render_animated(fig_gantt, height=400)

        with st.expander("Deliverables Detail", expanded=True):
            plan_df = pd.DataFrame([
                {"Week": f"Wk {s}-{e}", "Task": t, "Owner": o,
                 "Deliverable": d}
                for t, o, s, e, _ in [
                    ("Data Audit", "Data Eng.", 1, 2, "Readiness scorecard"),
                    ("Baseline PDEs", "Physics", 2, 4, "Validated loss module"),
                    ("Train Prototype", "ML Team", 4, 8, "Benchmark vs Kalman"),
                    ("MMD Monitors", "MLOps", 1, 10, "Drift alert dashboard"),
                    ("INT8/INT4 Quant", "ML/Edge", 8, 10, "Latency test on edge HW"),
                ]
                for d in [{"Data Audit": "Readiness scorecard", "Baseline PDEs": "Validated loss module",
                           "Train Prototype": "Benchmark vs Kalman", "MMD Monitors": "Drift alert dashboard",
                           "INT8/INT4 Quant": "Latency test on edge HW"}[t]]
            ])
            st.dataframe(plan_df, use_container_width=True, hide_index=True)

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
