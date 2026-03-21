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

st.title("GNSS Error Correction: Traditional NN vs PINN Trade-offs")
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
    help="Balance between data loss and physics loss in the PINN loss function.",
)

precision = st.sidebar.selectbox(
    "Quantization level",
    ["FP32 (baseline)", "INT8 (default)", "INT4 (aggressive)"],
    index=1,
)

environment = st.sidebar.selectbox(
    "Deployment environment",
    ["Open Sky", "Urban Canyon", "Tunnel / Indoor"],
    index=1,
)

batch_size = st.sidebar.slider(
    "Inference batch size (vehicles)",
    min_value=1, max_value=64, value=8, step=1,
    help="Number of vehicles batched per inference call.",
)

np.random.seed(42)

# ─────────────────────────────────────────────
# Helper: mock data generators
# ─────────────────────────────────────────────

def generate_accuracy_vs_data(lambda_w: float):
    """Simulate accuracy curves for NN and PINN as a function of training data size."""
    data_sizes = np.logspace(2, 6, 50)  # 100 to 1,000,000 samples

    # Traditional NN: slow convergence, needs lots of data
    nn_error = 80 * np.exp(-data_sizes / 150_000) + 5 + np.random.normal(0, 1.5, 50)
    nn_error = np.clip(nn_error, 2, 100)

    # PINN: fast convergence due to physics prior, modulated by lambda
    effective_lambda = np.clip(lambda_w, 0.1, 1.5)
    pinn_scale = 150_000 / (1 + 200 * effective_lambda)
    pinn_error = 80 * np.exp(-data_sizes / pinn_scale) + 2 + np.random.normal(0, 1.0, 50)

    # Bad lambda (too high or too low) degrades PINN performance
    if lambda_w < 0.1 or lambda_w > 1.5:
        pinn_error += 8
    pinn_error = np.clip(pinn_error, 1, 100)

    return data_sizes, nn_error, pinn_error


def generate_training_cost(model_params_m: int):
    """Compute training FLOPs and memory for a range of model sizes."""
    sizes = np.arange(5, 205, 5)
    flops_per_token = 6 * sizes * 1e6  # 6N
    memory_gb = (12 * sizes * 1e6) / 1e9  # 12 bytes/param → GB

    # PINN overhead: +30% per epoch
    pinn_flops = flops_per_token * 1.3

    return sizes, flops_per_token, pinn_flops, memory_gb


def generate_inference_latency(model_params_m: int, quant: str, batch: int):
    """Simulate inference latency as a function of model size, quantization, batch size."""
    sizes = np.arange(5, 205, 5)

    # Bytes per parameter based on quantization
    if "FP32" in quant:
        bytes_per_param = 4.0
        label = "FP32"
    elif "INT8" in quant:
        bytes_per_param = 1.0
        label = "INT8"
    else:
        bytes_per_param = 0.5
        label = "INT4"

    # Memory bandwidth bottleneck: latency ~ (model_bytes / bandwidth)
    # Assume ~900 GB/s bandwidth (typical edge GPU)
    bandwidth_gbs = 900
    model_bytes = sizes * 1e6 * bytes_per_param
    # Base latency in ms from memory transfer
    base_latency_ms = (model_bytes / (bandwidth_gbs * 1e9)) * 1000

    # Add compute time (small relative to transfer)
    compute_ms = (2 * sizes * 1e6) / (10e12) * 1000  # assume 10 TFLOPS

    # Batch amortization: latency grows sub-linearly with batch
    batch_factor = 1 + 0.15 * np.log2(max(batch, 1))

    total_ms = (base_latency_ms + compute_ms) * batch_factor
    # Add realistic overhead (kernel launch, preprocessing)
    total_ms += 2.0 + np.random.normal(0, 0.3, len(sizes))
    total_ms = np.clip(total_ms, 0.5, 500)

    return sizes, total_ms, label


def generate_quantization_tradeoff():
    """Show accuracy retention vs memory reduction for different quantization levels."""
    quant_levels = ["FP32", "FP16", "INT8", "INT4", "INT2"]
    memory_reduction = [1, 2, 4, 8, 16]
    nn_accuracy_retained = [100.0, 99.8, 99.2, 96.5, 88.0]
    pinn_accuracy_retained = [100.0, 99.9, 99.5, 97.0, 89.5]
    return quant_levels, memory_reduction, nn_accuracy_retained, pinn_accuracy_retained


def generate_lambda_sensitivity():
    """PINN positioning error as a function of lambda."""
    lambdas = np.linspace(0, 2.0, 100)
    # Optimal around 0.3–0.7
    error = 3 + 40 * np.exp(-8 * lambdas) + 15 * np.maximum(0, lambdas - 0.8) ** 1.5
    error += np.random.normal(0, 0.5, 100)
    error = np.clip(error, 1, 60)
    return lambdas, error


def generate_environment_comparison():
    """Error by environment for NN vs PINN."""
    envs = ["Open Sky", "Urban Canyon", "Tunnel Exit", "Deep Urban"]
    nn_errors = [4.5, 45, 85, 95]
    pinn_errors = [2.5, 8, 25, 35]
    kalman_errors = [3.5, 35, 70, 90]
    return envs, nn_errors, pinn_errors, kalman_errors


# ─────────────────────────────────────────────
# Tab layout
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Accuracy vs Data",
    "💰 Training Cost",
    "⏱️ Inference Latency",
    "🔧 Quantization",
    "⚖️ λ Sensitivity",
    "🌍 Environment",
])

# ── Tab 1: Accuracy vs Training Data Size ────
with tab1:
    st.subheader("Positioning Error vs Training Data Size")
    st.markdown(
        f"PINN data efficiency advantage at **λ = {lambda_weight:.2f}**. "
        "PINNs reach target accuracy with ~250× less data in physics-rich regimes."
    )

    data_sizes, nn_err, pinn_err = generate_accuracy_vs_data(lambda_weight)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=data_sizes, y=nn_err, mode="lines",
        name="Traditional NN",
        line=dict(color="#EF553B", width=3),
    ))
    fig1.add_trace(go.Scatter(
        x=data_sizes, y=pinn_err, mode="lines",
        name=f"PINN (λ={lambda_weight:.2f})",
        line=dict(color="#636EFA", width=3),
    ))
    fig1.add_hline(y=5, line_dash="dash", line_color="green",
                   annotation_text="5 m target accuracy")
    fig1.update_layout(
        xaxis_title="Training Samples",
        yaxis_title="Horizontal Position Error (m)",
        xaxis_type="log",
        template="plotly_white",
        height=500,
        legend=dict(x=0.65, y=0.95),
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Show the data efficiency ratio
    nn_threshold_idx = np.argmax(nn_err < 10)
    pinn_threshold_idx = np.argmax(pinn_err < 10)
    if nn_threshold_idx > 0 and pinn_threshold_idx > 0:
        ratio = data_sizes[nn_threshold_idx] / data_sizes[pinn_threshold_idx]
        col1, col2, col3 = st.columns(3)
        col1.metric("NN samples to reach <10 m", f"{data_sizes[nn_threshold_idx]:,.0f}")
        col2.metric("PINN samples to reach <10 m", f"{data_sizes[pinn_threshold_idx]:,.0f}")
        col3.metric("Data efficiency ratio", f"{ratio:,.0f}×")


# ── Tab 2: Training Cost ─────────────────────
with tab2:
    st.subheader("Training Compute & Memory vs Model Size")
    st.markdown(
        f"Current model: **{model_size}M parameters** → "
        f"**{12 * model_size / 1000:.1f} GB** training state. "
        "PINN adds ~30% per-epoch overhead for physics loss evaluation."
    )

    sizes, nn_flops, pinn_flops, mem_gb = generate_training_cost(model_size)

    fig2 = make_subplots(
        rows=1, cols=2,
        subplot_titles=("FLOPs per Token", "Training Memory (GB)"),
    )

    fig2.add_trace(go.Scatter(
        x=sizes, y=nn_flops / 1e9, mode="lines",
        name="NN — 6N FLOPs", line=dict(color="#EF553B", width=2),
    ), row=1, col=1)

    fig2.add_trace(go.Scatter(
        x=sizes, y=pinn_flops / 1e9, mode="lines",
        name="PINN — ~7.8N FLOPs (+30%)", line=dict(color="#636EFA", width=2),
    ), row=1, col=1)

    fig2.add_trace(go.Bar(
        x=sizes, y=mem_gb,
        name="Memory (12 bytes/param)",
        marker_color="#00CC96", opacity=0.6,
    ), row=1, col=2)

    # Mark the selected model size
    selected_flops_nn = 6 * model_size * 1e6 / 1e9
    selected_flops_pinn = selected_flops_nn * 1.3
    selected_mem = 12 * model_size * 1e6 / 1e9

    fig2.add_trace(go.Scatter(
        x=[model_size], y=[selected_flops_nn], mode="markers",
        marker=dict(size=14, color="#EF553B", symbol="diamond"),
        name=f"Selected: {model_size}M (NN)", showlegend=True,
    ), row=1, col=1)

    fig2.add_trace(go.Scatter(
        x=[model_size], y=[selected_flops_pinn], mode="markers",
        marker=dict(size=14, color="#636EFA", symbol="diamond"),
        name=f"Selected: {model_size}M (PINN)", showlegend=True,
    ), row=1, col=1)

    fig2.update_xaxes(title_text="Model Size (M params)", row=1, col=1)
    fig2.update_xaxes(title_text="Model Size (M params)", row=1, col=2)
    fig2.update_yaxes(title_text="GFLOPs per Token", row=1, col=1)
    fig2.update_yaxes(title_text="Memory (GB)", row=1, col=2)

    fig2.update_layout(template="plotly_white", height=450)
    st.plotly_chart(fig2, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Training FLOPs/token (NN)", f"{selected_flops_nn:.1f} GFLOPs")
    c2.metric("Training FLOPs/token (PINN)", f"{selected_flops_pinn:.1f} GFLOPs")
    c3.metric("Training memory", f"{selected_mem:.2f} GB")

    if selected_mem < 8:
        st.success(f"✅ {model_size}M params fits on a single consumer GPU ({selected_mem:.1f} GB < 8 GB VRAM)")
    elif selected_mem < 24:
        st.info(f"ℹ️ {model_size}M params requires a workstation GPU ({selected_mem:.1f} GB)")
    else:
        st.warning(f"⚠️ {model_size}M params requires multi-GPU or cloud ({selected_mem:.1f} GB)")


# ── Tab 3: Inference Latency ─────────────────
with tab3:
    st.subheader("Inference Latency vs Model Size")
    st.markdown(
        f"**Quantization:** {precision} | **Batch size:** {batch_size} vehicles | "
        "Inference is memory-bandwidth-bound — FLOPs ≠ wall-clock speed."
    )

    sizes, latency, quant_label = generate_inference_latency(model_size, precision, batch_size)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=sizes, y=latency, mode="lines+markers",
        name=f"{quant_label} — batch {batch_size}",
        line=dict(color="#636EFA", width=3),
        marker=dict(size=4),
    ))
    fig3.add_hline(y=100, line_dash="dash", line_color="red",
                   annotation_text="100 ms deadline (10 Hz GNSS)")
    fig3.add_hline(y=50, line_dash="dot", line_color="orange",
                   annotation_text="50 ms target (safety margin)")

    # Mark selected model size
    idx = np.argmin(np.abs(sizes - model_size))
    fig3.add_trace(go.Scatter(
        x=[model_size], y=[latency[idx]], mode="markers",
        marker=dict(size=16, color="red", symbol="star"),
        name=f"Selected: {model_size}M → {latency[idx]:.1f} ms",
    ))

    fig3.update_layout(
        xaxis_title="Model Size (M params)",
        yaxis_title="Inference Latency (ms)",
        template="plotly_white",
        height=500,
        legend=dict(x=0.02, y=0.98),
    )
    st.plotly_chart(fig3, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Latency per correction", f"{latency[idx]:.1f} ms")
    c2.metric("Meets 100 ms deadline?", "Yes ✅" if latency[idx] < 100 else "No ❌")
    fleet_cost_per_sec = (1000 / max(latency[idx], 1)) * batch_size
    c3.metric("Corrections/sec (fleet)", f"{fleet_cost_per_sec:,.0f}")

    st.info(
        "**Key insight:** Inference cost is identical for NN and PINN — "
        "the physics loss only exists during training. PINN vs Transformer "
        "choice does not affect your OPEX."
    )


# ── Tab 4: Quantization Trade-off ────────────
with tab4:
    st.subheader("Quantization: Memory Reduction vs Accuracy Retention")
    st.markdown(
        "Below the compute roofline — spare FLOPs, not spare bandwidth. "
        "Quantization relieves the bandwidth bottleneck without hitting a compute wall."
    )

    quant_levels, mem_red, nn_acc, pinn_acc = generate_quantization_tradeoff()

    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        x=quant_levels, y=nn_acc, name="Traditional NN",
        marker_color="#EF553B", opacity=0.8,
    ))
    fig4.add_trace(go.Bar(
        x=quant_levels, y=pinn_acc, name="PINN",
        marker_color="#636EFA", opacity=0.8,
    ))
    fig4.add_hline(y=99, line_dash="dash", line_color="green",
                   annotation_text="99% accuracy threshold")
    fig4.update_layout(
        xaxis_title="Quantization Level",
        yaxis_title="Positioning Accuracy Retained (%)",
        yaxis_range=[85, 101],
        barmode="group",
        template="plotly_white",
        height=450,
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Memory savings table
    quant_df = pd.DataFrame({
        "Quantization": quant_levels,
        "Memory Reduction": [f"{r}×" for r in mem_red],
        "NN Accuracy Retained": [f"{a:.1f}%" for a in nn_acc],
        "PINN Accuracy Retained": [f"{a:.1f}%" for a in pinn_acc],
        "Recommendation": [
            "Baseline",
            "Safe default for training",
            "✅ Default for inference",
            "⚠️ Conditional — validate first",
            "❌ Not recommended for safety-critical",
        ],
    })
    st.dataframe(quant_df, use_container_width=True, hide_index=True)


# ── Tab 5: Lambda Sensitivity ────────────────
with tab5:
    st.subheader("PINN Positioning Error vs λ (Physics Loss Weight)")
    st.markdown(
        f"Current λ = **{lambda_weight:.2f}**. "
        "Too low → physics guardrail lost. Too high → model ignores valid data anomalies. "
        "Optimal range is typically 0.3–0.7 — validated against positioning error budgets."
    )

    lambdas, errors = generate_lambda_sensitivity()

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=lambdas, y=errors, mode="lines",
        name="PINN Error",
        line=dict(color="#636EFA", width=3),
        fill="tozeroy", fillcolor="rgba(99, 110, 250, 0.1)",
    ))

    # Mark current lambda
    current_idx = np.argmin(np.abs(lambdas - lambda_weight))
    fig5.add_trace(go.Scatter(
        x=[lambda_weight], y=[errors[current_idx]], mode="markers",
        marker=dict(size=16, color="red", symbol="star"),
        name=f"Current λ = {lambda_weight:.2f} → {errors[current_idx]:.1f} m error",
    ))

    # Optimal zone
    fig5.add_vrect(x0=0.3, x1=0.7, fillcolor="green", opacity=0.08,
                   annotation_text="Optimal zone", annotation_position="top left")

    fig5.update_layout(
        xaxis_title="λ (Physics Loss Weight)",
        yaxis_title="Horizontal Position Error (m)",
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(fig5, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Current λ error", f"{errors[current_idx]:.1f} m")
    optimal_idx = np.argmin(errors)
    c2.metric("Optimal λ", f"{lambdas[optimal_idx]:.2f} → {errors[optimal_idx]:.1f} m")

    if lambda_weight < 0.1:
        st.warning("⚠️ λ too low — physics guardrail effectively disabled. Model behaves like a standard NN.")
    elif lambda_weight > 1.2:
        st.warning("⚠️ λ too high — model over-constrains to physics and ignores valid data patterns.")
    elif 0.3 <= lambda_weight <= 0.7:
        st.success("✅ λ within optimal range. Validate against error budgets on held-out trajectories.")
    else:
        st.info("ℹ️ λ outside optimal zone but may be acceptable. Validate carefully.")


# ── Tab 6: Environment Comparison ────────────
with tab6:
    st.subheader("Positioning Error by Deployment Environment")
    st.markdown(
        f"Selected environment: **{environment}**. "
        "PINN advantage grows as conditions degrade — physics constraints matter most when data is worst."
    )

    envs, nn_errs, pinn_errs, kalman_errs = generate_environment_comparison()

    fig6 = go.Figure()
    fig6.add_trace(go.Bar(
        x=envs, y=kalman_errs, name="Classical Kalman + INS",
        marker_color="#FFA15A", opacity=0.8,
    ))
    fig6.add_trace(go.Bar(
        x=envs, y=nn_errs, name="Traditional NN",
        marker_color="#EF553B", opacity=0.8,
    ))
    fig6.add_trace(go.Bar(
        x=envs, y=pinn_errs, name="PINN (Hybrid)",
        marker_color="#636EFA", opacity=0.8,
    ))
    fig6.add_hline(y=5, line_dash="dash", line_color="green",
                   annotation_text="5 m lane-level target")
    fig6.update_layout(
        xaxis_title="Environment",
        yaxis_title="Horizontal Position Error (m)",
        barmode="group",
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(fig6, use_container_width=True)

    # Improvement ratios
    env_df = pd.DataFrame({
        "Environment": envs,
        "Kalman (m)": kalman_errs,
        "Trad. NN (m)": nn_errs,
        "PINN (m)": pinn_errs,
        "PINN vs NN Improvement": [f"{(1 - p / n) * 100:.0f}%" for n, p in zip(nn_errs, pinn_errs)],
        "PINN vs Kalman Improvement": [f"{(1 - p / k) * 100:.0f}%" for k, p in zip(kalman_errs, pinn_errs)],
    })
    st.dataframe(env_df, use_container_width=True, hide_index=True)

    st.info(
        "**Pattern:** PINN advantage widens in degraded environments. "
        "In open sky (easy case), all methods perform similarly. "
        "In urban canyons and tunnel exits (hard cases), physics constraints "
        "prevent the catastrophic failures seen with unconstrained NNs."
    )


# ─────────────────────────────────────────────
# Comparison table (always visible at bottom)
# ─────────────────────────────────────────────
st.divider()
st.subheader("Summary: Traditional NN vs PINN — Head-to-Head")

comparison_data = {
    "Dimension": [
        "Data efficiency",
        "Physics compliance",
        "Failure mode",
        "Training cost (CAPEX)",
        "Inference cost (OPEX)",
        "Edge deployment",
        "Quantization tolerance",
        "Shift resilience",
        "Drift detection",
        "Domain expertise",
        "Generalization",
        "Safety guarantee",
        "Best-fit scenario",
    ],
    "Traditional NN / Transformer": [
        "Low — large labeled datasets",
        "None — can output impossible positions",
        "Silent 50 m jumps",
        "Standard ~6N FLOPs/token",
        "~2N FLOPs, bandwidth-bound",
        "Viable with INT8/INT4",
        "Standard trade-offs",
        "No guardrail when inputs drift",
        "Needs external MMD monitors",
        "ML engineering",
        "Poor without local retraining",
        "None",
        "Abundant RTK data, stable env",
    ],
    "PINN (Physics-Informed)": [
        "High — up to 250× less data",
        "Built-in — violations penalized",
        "Bounded by physics model quality",
        "+20–40%/epoch, less data needed",
        "Identical — physics is training-only",
        "Equally viable — same inference graph",
        "Same — quantization affects weights, not loss",
        "Physics limits damage, won't detect drift",
        "Needs external MMD monitors",
        "ML + physics/GNSS domain",
        "Strong — physics transfers across geographies",
        "Soft — bounded by physics quality",
        "Scarce data, variable env, safety-critical",
    ],
}

comparison_df = pd.DataFrame(comparison_data)
st.dataframe(comparison_df, use_container_width=True, hide_index=True, height=500)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
st.caption(
    "Assignment 2 — AI & Large Models | "
    "Physics-Informed GNSS Positioning Error Correction in Degraded Environments | "
    "All data is synthetic/mock for illustration purposes."
)
