# app.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time

st.set_page_config(page_title="PSO Robotic Arm - Live", layout="wide")

# --------------------------
# Helper: forward kinematics
# --------------------------
def fk(theta1_deg, theta2_deg, L1=225, L2=175):
    """Return (x0,y0,z0), (x1,y1,z1), (x2,y2,z2) for shoulder, elbow, end-effector.
       z coordinate uses 'd' separately (z = 144 - d). Here we return just XY for links."""
    t1 = np.radians(theta1_deg)
    t2 = np.radians(theta2_deg)
    x1 = L1 * np.cos(t1)
    y1 = L1 * np.sin(t1)
    x2 = x1 + L2 * np.cos(t1 + t2)
    y2 = y1 + L2 * np.sin(t1 + t2)
    return (0.0, 0.0, 144.0), (x1, y1, 144.0), (x2, y2, 144.0)

# --------------------------
# UI controls (left column)
# --------------------------
st.sidebar.title("PSO Controls")
with st.sidebar.form("pso_form"):
    N_PARTICLES = st.number_input("Number of particles", min_value=3, max_value=200, value=10, step=1)
    ITERATIONS = st.number_input("Max iterations", min_value=1, max_value=5000, value=200, step=1)
    W = st.number_input("Inertia (W)", min_value=0.0, max_value=2.0, value=0.65, step=0.05)
    C1 = st.number_input("Cognitive (C1)", min_value=0.0, max_value=3.0, value=1.5, step=0.1)
    C2 = st.number_input("Social (C2)", min_value=0.0, max_value=3.0, value=1.5, step=0.1)
    L1 = st.number_input("Link L1", min_value=1.0, value=225.0)
    L2 = st.number_input("Link L2", min_value=1.0, value=175.0)
    XE = st.number_input("Target XE", value=350.0)
    YE = st.number_input("Target YE", value=150.0)
    ZE = st.number_input("Target ZE", value=100.0)
    submit = st.form_submit_button("Apply & Set")

# Display some explanation
st.sidebar.markdown("---")
st.sidebar.markdown("Tips:\n- Use **Stop** during a run to interrupt.\n- The 3D plot shows the best particle's arm and target.")
st.sidebar.markdown("---")

# --------------------------
# Main layout
# --------------------------
st.title("🔵 PSO — Robot Arm Positioning (Live)")
col1, col2 = st.columns([2, 3])

with col1:
    st.header("Run Controls")
    start_btn = st.button("▶ Start PSO", key="start_pso")
    stop_btn = st.button("⏹ Stop PSO", key="stop_pso")
    st.write("Adjust parameters in the sidebar and click **Start PSO**.")
    st.markdown("---")
    st.subheader("Best-so-far (live)")
    best_text = st.empty()  # will update each iter

    st.markdown("---")
    st.subheader("Convergence")
    conv_placeholder = st.empty()  # line chart placeholder

with col2:
    st.header("3D Robot Arm (best particle)")
    fig3d_ph = st.empty()
    st.markdown("---")
    st.subheader("Particles Table (live, updates each iteration)")
    table_ph = st.empty()

# --------------------------
# State initialization
# --------------------------
if "running" not in st.session_state:
    st.session_state.running = False
if "pso_state" not in st.session_state:
    st.session_state.pso_state = None

# Stop button handler
if stop_btn:
    st.session_state.running = False
    st.warning("PSO stop requested. Finishing current loop...")

# Start button handler: initialize PSO state
if start_btn:
    # create new PSO state
    np.random.seed()  # allow randomness (or set a seed if reproducibility needed)
    theta1 = np.random.uniform(-130, 130, int(N_PARTICLES)).astype(float)
    theta2 = np.random.uniform(-130, 130, int(N_PARTICLES)).astype(float)
    d = np.random.uniform(0, 118, int(N_PARTICLES)).astype(float)

    # velocities
    v1 = np.random.uniform(-1, 1, int(N_PARTICLES)).astype(float)
    v2 = np.random.uniform(-1, 1, int(N_PARTICLES)).astype(float)
    v3 = np.random.uniform(-1, 1, int(N_PARTICLES)).astype(float)

    # compute initial pos
    t1_rad = np.radians(theta1)
    t2_rad = np.radians(theta2)
    x = L1 * np.cos(t1_rad) + L2 * np.cos(t1_rad + t2_rad)
    y = L1 * np.sin(t1_rad) + L2 * np.sin(t1_rad + t2_rad)
    z = 144.0 - d

    # compute fitness (distance to target)
    f = np.sqrt((x - XE) ** 2 + (y - YE) ** 2 + (z - ZE) ** 2)

    pso_state = {
        "theta1": theta1,
        "theta2": theta2,
        "d": d,
        "v1": v1,
        "v2": v2,
        "v3": v3,
        "x": x,
        "y": y,
        "z": z,
        "f": f,
        "pbest_theta1": theta1.copy(),
        "pbest_theta2": theta2.copy(),
        "pbest_d": d.copy(),
        "pbest_f": f.copy(),
        "gbest_idx": int(np.argmin(f)),
        "history_best": [float(np.min(f))],
        "iteration": 0
    }
    pso_state["gbest_theta1"] = float(pso_state["theta1"][pso_state["gbest_idx"]])
    pso_state["gbest_theta2"] = float(pso_state["theta2"][pso_state["gbest_idx"]])
    pso_state["gbest_d"] = float(pso_state["d"][pso_state["gbest_idx"]])
    pso_state["gbest_f"] = float(pso_state["f"][pso_state["gbest_idx"]])

    st.session_state.pso_state = pso_state
    st.session_state.running = True
    st.success("PSO initialized — running...")

# --------------------------
# PSO main loop (runs inside the Streamlit script)
# --------------------------
if st.session_state.running and st.session_state.pso_state is not None:
    p = st.session_state.pso_state

    # placeholders for plots were created above; we'll update them inside the loop
    # ensure we keep consistent placeholders and don't create duplicate elements

    for it in range(p["iteration"], int(ITERATIONS)):
        if not st.session_state.running:
            # interrupted by Stop button
            break

        # update iteration count in state (so restart continues where left off if desired)
        p["iteration"] = it

        # random factors
        r1 = np.random.rand(len(p["theta1"]))
        r2 = np.random.rand(len(p["theta1"]))

        # velocity updates
        p["v1"] = W * p["v1"] + C1 * r1 * (p["pbest_theta1"] - p["theta1"]) + C2 * r2 * (p["gbest_theta1"] - p["theta1"])
        p["v2"] = W * p["v2"] + C1 * r1 * (p["pbest_theta2"] - p["theta2"]) + C2 * r2 * (p["gbest_theta2"] - p["theta2"])
        p["v3"] = W * p["v3"] + C1 * r1 * (p["pbest_d"] - p["d"]) + C2 * r2 * (p["gbest_d"] - p["d"])

        # update positions
        p["theta1"] += p["v1"]
        p["theta2"] += p["v2"]
        p["d"] += p["v3"]

        # constraints
        p["theta1"] = np.clip(p["theta1"], -130, 130)
        p["theta2"] = np.clip(p["theta2"], -130, 130)
        p["d"] = np.clip(p["d"], 0, 118)

        # compute positions & fitness
        t1_rad = np.radians(p["theta1"])
        t2_rad = np.radians(p["theta2"])
        p["x"] = L1 * np.cos(t1_rad) + L2 * np.cos(t1_rad + t2_rad)
        p["y"] = L1 * np.sin(t1_rad) + L2 * np.sin(t1_rad + t2_rad)
        p["z"] = 144.0 - p["d"]
        p["f"] = np.sqrt((p["x"] - XE) ** 2 + (p["y"] - YE) ** 2 + (p["z"] - ZE) ** 2)

        # update personal bests
        improved = p["f"] < p["pbest_f"]
        if np.any(improved):
            p["pbest_theta1"][improved] = p["theta1"][improved]
            p["pbest_theta2"][improved] = p["theta2"][improved]
            p["pbest_d"][improved] = p["d"][improved]
            p["pbest_f"][improved] = p["f"][improved]

        # update global best
        min_idx = int(np.argmin(p["f"]))
        min_val = float(p["f"][min_idx])
        if min_val < p["gbest_f"]:
            p["gbest_idx"] = min_idx
            p["gbest_theta1"] = float(p["theta1"][min_idx])
            p["gbest_theta2"] = float(p["theta2"][min_idx])
            p["gbest_d"] = float(p["d"][min_idx])
            p["gbest_f"] = min_val
        
        # ============================
        # 🔥 EARLY STOP CONDITION HERE
        # ============================
        if p["gbest_f"] <= 0.00022:
            st.warning(f"Early Stop Triggered — gbest_f reached {p['gbest_f']:.2f}")
            break

        # record history
        p["history_best"].append(float(p["gbest_f"]))

        # --------------------------
        # Update visuals (single placeholder updates)
        # --------------------------
        # update best-so-far text
        best_text.markdown(
            f"**Best so far -> Theta1:** {p['gbest_theta1']:.2f}°, "
            f"**Theta2:** {p['gbest_theta2']:.2f}°, **D:** {p['gbest_d']:.2f}, "
            f"**best function value:** {p['gbest_f']:.2f}"
        )

        # update convergence chart (use pandas for line)
        conv_df = pd.DataFrame({"Best": p["history_best"]})
        conv_chart = conv_placeholder.line_chart(conv_df)

        # update particles table (full)
        df = pd.DataFrame({
            "V1": p["v1"], "V2": p["v2"], "V3": p["v3"],
            "Theta1": p["theta1"], "Theta2": p["theta2"], "D": p["d"],
            "X": p["x"], "Y": p["y"], "Z": p["z"],
            "F": p["f"], "Pbest1": p["pbest_theta1"], "Pbest2": p["pbest_theta2"], "Pbest3": p["pbest_d"]
        })
        df.index.name = "SR.NO"
        df.index = np.arange(1, len(df) + 1)
        table_ph.dataframe(df.style.format(precision=3), use_container_width=True)

        # update 3D arm (best particle)
        bi = p["gbest_idx"]
        # shoulder, elbow, end
        s, e, ee = fk(p["gbest_theta1"], p["gbest_theta2"], L1=L1, L2=L2)
        # build 3D figure
        fig = go.Figure()

        # arm (two segments) in XY plane, show Z as well (we'll use actual Z = 144-d)
        xs = [s[0], e[0], ee[0]]
        ys = [s[1], e[1], ee[1]]
        zs = [s[2] - 0.0, e[2] - 0.0, (144.0 - p["gbest_d"])]

        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines+markers",
            marker=dict(size=4),
            line=dict(width=6),
            name="Arm (best)"
        ))

        # show all particle end-effectors as small points
        fig.add_trace(go.Scatter3d(
            x=p["x"], y=p["y"], z=p["z"],
            mode="markers",
            marker=dict(size=3),
            name="Particles (EE)"
        ))

        # target point
        fig.add_trace(go.Scatter3d(
            x=[XE], y=[YE], z=[ZE],
            mode="markers",
            marker=dict(size=6, color="red"),
            name="Target"
        ))

        fig.update_layout(
            scene=dict(
                xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                aspectmode="data"
            ),
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=True,
            height=480
        )

        fig3d_ph.plotly_chart(fig, use_container_width=True)

        # small delay so UI updates and shows animation
        time.sleep(0.05)

    # loop finished (either completed or stopped)
    st.session_state.running = False
    st.session_state.pso_state = p  # save final state

    # final message
    st.success(
        f"PSO finished. Best so far -> Theta1: {p['gbest_theta1']:.2f}, "
        f"Theta2: {p['gbest_theta2']:.2f}, D: {p['gbest_d']:.2f}, "
        f"best function value: {p['gbest_f']:.2f}"
    )

# --------------------------
# If not running, show final saved result if exists
# --------------------------
if not st.session_state.running and st.session_state.pso_state is not None:
    p = st.session_state.pso_state
    st.sidebar.markdown("---")
    st.sidebar.subheader("Last run summary")
    st.sidebar.write(f"Iterations run: {len(p['history_best']) - 1}")
    st.sidebar.write(f"Best value: {p['gbest_f']:.3f}")
    st.sidebar.write(f"Theta1: {p['gbest_theta1']:.2f}, Theta2: {p['gbest_theta2']:.2f}, D: {p['gbest_d']:.2f}")
    if st.sidebar.button("🔁 Re-run last config"):
        # re-initialize iteration pointer to 0 and start
        p["iteration"] = 0
        st.session_state.pso_state = p
        st.session_state.running = True
        st.rerun()

