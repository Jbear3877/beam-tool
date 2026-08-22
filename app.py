import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(page_title="副车架梁模型分析工具", layout="wide", initial_sidebar_state="expanded")

# ============================================================
# 修复标题裁切的CSS
# ============================================================
st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem !important;
    }
    [data-testid="stHeader"] {
        background: transparent !important;
        height: auto !important;
        min-height: 0 !important;
        overflow: visible !important;
    }
    h1 {
        line-height: 1.6 !important;
        overflow: visible !important;
        padding-top: 0.3rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 材料数据库
# ============================================================
MATERIALS = {
    "SAPH440":  {"E": 206000, "sigma_y": 305},
    "SAPH590":  {"E": 206000, "sigma_y": 450},
    "Q235":     {"E": 206000, "sigma_y": 235},
    "Q345":     {"E": 206000, "sigma_y": 345},
    "AL6061-T6": {"E": 68900,  "sigma_y": 276},
}


# ============================================================
# 截面特性计算（矩形空心截面）
# ============================================================
def calc_section_props(h, b, t):
    ho = max(h - 2 * t, 1e-6)
    bo = max(b - 2 * t, 1e-6)
    A  = h * b - ho * bo
    Iy = (b * h**3 - bo * ho**3) / 12.0
    Wy = 2.0 * Iy / h
    J  = 2.0 * t * (h - t)**2 * (b - t)**2 / (h + b - 2 * t)
    return A, Iy, J, Wy


# ============================================================
# 梁结构分析 — 直接刚度法
# ============================================================
def analyze_beam(nodes, EI, F, support_left, support_right, seg_h, seg_b, seg_t, sigma_y):
    n_nodes = len(nodes)
    n_elem  = n_nodes - 1

    def beam_k(ei, L):
        k = np.array([
            [ 12*ei/L**3,   6*ei/L**2, -12*ei/L**3,   6*ei/L**2],
            [  6*ei/L**2,   4*ei/L,     -6*ei/L**2,   2*ei/L],
            [-12*ei/L**3,  -6*ei/L**2,  12*ei/L**3,  -6*ei/L**2],
            [  6*ei/L**2,   2*ei/L,     -6*ei/L**2,   4*ei/L]
        ])
        return k

    n_dof = 2 * n_nodes
    K = np.zeros((n_dof, n_dof))
    for i in range(n_elem):
        L = nodes[i+1] - nodes[i]
        ke = beam_k(EI[i], L)
        dofs = [2*i, 2*i+1, 2*i+2, 2*i+3]
        for r in range(4):
            for c in range(4):
                K[dofs[r], dofs[c]] += ke[r, c]

    fixed_dofs = set()
    if support_left == "fixed":
        fixed_dofs.update([0, 1])
    else:
        fixed_dofs.add(0)
    if support_right == "fixed":
        fixed_dofs.update([2*n_nodes-2, 2*n_nodes-1])
    else:
        fixed_dofs.add(2*n_nodes-2)

    free_dofs  = sorted(set(range(n_dof)) - fixed_dofs)
    fixed_dofs = sorted(fixed_dofs)

    Kff = K[np.ix_(free_dofs, free_dofs)]
    Kfc = K[np.ix_(free_dofs, fixed_dofs)]
    Ff  = np.array([F[d//2] for d in free_dofs])

    U = np.zeros(n_dof)
    U[free_dofs] = np.linalg.solve(Kff, Ff - Kfc @ U[fixed_dofs])

    elem_forces = []
    for i in range(n_elem):
        L = nodes[i+1] - nodes[i]
        dofs = [2*i, 2*i+1, 2*i+2, 2*i+3]
        fe = beam_k(EI[i], L) @ U[dofs]
        elem_forces.append(fe)

    nodal_v = np.zeros(n_nodes)
    nodal_m = np.zeros(n_nodes)
    for i in range(n_elem):
        nodal_v[i]   += elem_forces[i][0]
        nodal_m[i]   += elem_forces[i][1]
        nodal_v[i+1] += elem_forces[i][2]
        nodal_m[i+1] += elem_forces[i][3]

    w_vals = U[0::2]
    max_deflection = float(np.max(np.abs(w_vals)))
    max_moment     = float(np.max(np.abs(nodal_m)))
    max_shear      = float(np.max(np.abs(nodal_v)))

    max_sigma = 0.0
    for i in range(n_elem):
        _, Iy_i, _, Wy_i = calc_section_props(seg_h[i], seg_b[i], seg_t[i])
        if Wy_i > 0:
            sig = abs(nodal_m[i]) / Wy_i * 1e3
            if sig > max_sigma:
                max_sigma = sig

    safety_factor = 0.0
    stress_status = "不满足"
    if max_sigma > 1e-10:
        safety_factor = sigma_y / max_sigma
        if safety_factor >= 1.0:
            stress_status = "满足"

    return {
        "U": U, "w": w_vals,
        "elem_forces": elem_forces,
        "nodal_v": nodal_v, "nodal_m": nodal_m,
        "max_moment": max_moment, "max_shear": max_shear,
        "max_deflection": max_deflection,
        "max_sigma": max_sigma,
        "safety_factor": safety_factor, "stress_status": stress_status,
    }


# ============================================================
# 绘图：结构示意图
# ============================================================
def plot_beam(nodes, seg_starts, seg_ends, supports, loads,
              seg_h, seg_b, seg_t, res, n_elem, support_labels):

    fig = go.Figure()

    colors = ["#8CC4F0", "#A6CEE8"]

    for i in range(n_elem):
        x0, x1 = seg_starts[i] * 1e3, seg_ends[i] * 1e3
        h_vis = max(seg_h[i] * 0.20, 4)
        label = f"段{i+1}\n{seg_h[i]:.1f}×{seg_b[i]:.1f}×{seg_t[i]:.1f}mm"
        fig.add_trace(go.Scatter(
            x=[x0, x1, x1, x0, x0],
            y=[h_vis, h_vis, -h_vis, -h_vis, h_vis],
            fill="toself", fillcolor=colors[i % 2],
            line=dict(color="#4A90D9", width=1.5),
            name=label, showlegend=True,
            legendgroup=f"seg{i}", hoverinfo="skip"
        ))

    x_s = np.array([s[0] for s in supports]) * 1e3
    for idx, sx in enumerate(x_s):
        if "固支" in support_labels[idx]:
            fig.add_trace(go.Scatter(
                x=[sx, sx+14, sx, sx-14, sx], y=[-20, -30, -40, -30, -20],
                fill="toself", fillcolor="#2ECC71", line=dict(color="#1B9E4B", width=1.5),
                showlegend=False, hoverinfo="skip"
            ))
            for j in range(4):
                xs = sx - 14 + j * 9
                fig.add_trace(go.Scatter(
                    x=[xs, xs-7], y=[-40, -50], mode="lines",
                    line=dict(color="#1B9E4B", width=1), showlegend=False, hoverinfo="skip"
                ))
        else:
            fig.add_trace(go.Scatter(
                x=[sx-15, sx, sx+15, sx-15], y=[-20, -38, -20, -20],
                fill="toself", fillcolor="#2ECC71", line=dict(color="#1B9E4B", width=1.5),
                showlegend=False, hoverinfo="skip"
            ))

    for lx in np.array([l[0] for l in loads]) * 1e3:
        arrow_h = 42
        fig.add_trace(go.Scatter(
            x=[lx, lx], y=[arrow_h, 20], mode="lines",
            line=dict(color="red", width=3), showlegend=False, hoverinfo="skip"
        ))
        fig.add_annotation(
            x=lx, y=20, ax=0, ay=22, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=3, arrowcolor="red"
        )
        fig.add_annotation(x=lx, y=arrow_h+7, text="5.0 kN", showarrow=False,
                           font=dict(size=11, color="red", family="Arial Black"))

    for nx in nodes:
        fig.add_trace(go.Scatter(
            x=[nx*1e3], y=[0], mode="markers",
            marker=dict(size=6, color="#E74C3C"), showlegend=False, hoverinfo="skip"
        ))

    x_node_vals = [nx*1e3 for nx in nodes]
    fig.add_trace(go.Scatter(
        x=x_node_vals, y=[-55]*len(x_node_vals), mode="text",
        text=[f"{v:.0f}" for v in x_node_vals],
        textfont=dict(size=10, color="#555"), showlegend=False, hoverinfo="skip"
    ))

    x_total = np.array(nodes).max() * 1e3
    fig.add_annotation(x=x_total/2, y=-70, text="位置 x (mm)", showarrow=False,
                       font=dict(size=11, color="#666"))

    fig.update_layout(
        height=340,
        template="plotly_white",
        title=dict(text="<b>梁结构示意图（结构与载荷可视化）</b>", x=0.01, font=dict(size=14)),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center", font=dict(size=10)),
        margin=dict(l=40, r=20, t=60, b=30),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(visible=False), yaxis=dict(visible=False, range=[-75, 65]),
        showlegend=True,
    )
    return fig


# ============================================================
# 侧边栏参数
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ 参数设置")

    st.markdown("### 材料")
    mat_name = st.selectbox("材料", list(MATERIALS.keys()), index=0)
    mat = MATERIALS[mat_name]
    E   = st.number_input("E (MPa)", value=float(mat["E"]), step=1000.0, format="%.2f")
    sigma_y = st.number_input("σy (MPa)", value=float(mat["sigma_y"]), step=1.0, format="%.2f")

    st.markdown("---")
    st.markdown("### 梁分段")
    n_seg = st.number_input("分段数", min_value=1, max_value=10, value=3, step=1, format="%.0f")
    n_seg = int(n_seg)

    segments = []
    for i in range(n_seg):
        with st.expander(f"段{i+1}", expanded=(i < 2)):
            c1, c2 = st.columns(2)
            with c1:
                L = st.number_input("长度 mm", value=300.0, step=10.0, format="%.2f", key=f"L{i}")
            with c2:
                t = st.number_input("壁厚 mm", value=2.50, step=0.1, format="%.2f", key=f"t{i}")
            c3, c4 = st.columns(2)
            with c3:
                h = st.number_input("高度 h mm", value=80.0, step=1.0, format="%.2f", key=f"h{i}")
            with c4:
                b = st.number_input("宽度 b mm", value=55.0, step=1.0, format="%.2f", key=f"b{i}")
            segments.append(dict(L=L, h=h, b=b, t=t))

    st.markdown("---")
    st.markdown("### 支撑")
    support_left_type  = st.selectbox("左端支撑", ["fixed", "pinned"], index=0,
                                       format_func=lambda x: {"fixed": "固支 (Fixed)", "pinned": "铰支 (Pinned)"}[x])
    support_right_type = st.selectbox("右端支撑", ["fixed", "pinned"], index=0,
                                       format_func=lambda x: {"fixed": "固支 (Fixed)", "pinned": "铰支 (Pinned)"}[x])

    st.markdown("---")
    st.markdown("### 载荷")
    load_positions = []
    for i in range(n_seg - 1):
        pos_mm = sum(s["L"] for s in segments[:i+1])
        with st.expander(f"载荷{i+1}（段{i+1}与段{i+2}交界）", expanded=True):
            px = st.number_input("位置 mm", value=float(pos_mm), step=1.0, format="%.2f", key=f"px{i}")
            pf = st.number_input("力 N（向下为正）", value=5000.0, step=100.0, format="%.2f", key=f"pf{i}")
            load_positions.append((px / 1000.0, pf))

    st.markdown("---")
    run = st.button("▶  运行分析", use_container_width=True, type="primary")


# ============================================================
# 主界面
# ============================================================
st.markdown("# 🛠️ 副车架梁模型分析工具")

if not run:
    st.info("👈 请在左侧设置参数后点击「运行分析」")
    st.stop()

nodes      = [0.0]
seg_starts = []
seg_ends   = []
seg_h_list = []
seg_b_list = []
seg_t_list = []
EI_list    = []

for seg in segments:
    seg_starts.append(nodes[-1])
    seg_h_list.append(seg["h"])
    seg_b_list.append(seg["b"])
    seg_t_list.append(seg["t"])
    L_m = seg["L"] / 1000.0
    _, Iy, _, _ = calc_section_props(seg["h"], seg["b"], seg["t"])
    EI_list.append(E * Iy)
    nodes.append(nodes[-1] + L_m)
    seg_ends.append(nodes[-1])

n_nodes = len(nodes)
n_elem  = n_nodes - 1

F = np.zeros(n_nodes)
for px, pf in load_positions:
    idx = min(range(n_nodes), key=lambda i: abs(nodes[i] - px))
    F[idx] += pf

res = analyze_beam(nodes, EI_list, F, support_left_type, support_right_type,
                   seg_h_list, seg_b_list, seg_t_list, sigma_y)

supports = [
    (0.0, f"支撑 固支@N0" if support_left_type == "fixed" else f"支撑 铰支@N0"),
    (nodes[-1], f"支撑 固支@N{n_nodes-1}" if support_right_type == "fixed" else f"支撑 铰支@N{n_nodes-1}"),
]
support_labels = [s[1] for s in supports]

fig = plot_beam(nodes, seg_starts, seg_ends, supports, load_positions,
                seg_h_list, seg_b_list, seg_t_list, res, n_elem, support_labels)
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 分析结果
# ============================================================
st.markdown("### 📊 分析结果")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("最大弯矩", f"{res['max_moment']/1000:.3f} kN·m")
with c2:
    st.metric("最大剪力", f"{res['max_shear']/1000:.3f} kN")
with c3:
    st.metric("最大挠度", f"{res['max_deflection']:.4f} mm")
with c4:
    st.metric("安全系数", f"{res['safety_factor']:.2f}")
    if res["stress_status"] == "满足":
        st.success("✔ 满足")
    else:
        st.error("⬆ 不满足")

# ============================================================
# 各段截面特性表
# ============================================================
st.markdown("### 📋 各段截面特性")

rows = []
for i, seg in enumerate(segments):
    A_i, Iy_i, J_i, Wy_i = calc_section_props(seg["h"], seg["b"], seg["t"])
    rows.append({
        "段号": f"段{i+1}",
        "长度 (mm)": int(seg["L"]),
        "h×b×t (mm)": f"{seg['h']:.1f}×{seg['b']:.1f}×{seg['t']:.1f}",
        "A (mm²)": int(A_i),
        "Iy (mm⁴)": int(Iy_i),
        "J (mm⁴)": int(J_i),
        "Wy (mm³)": int(Wy_i),
    })

df_sec = pd.DataFrame(rows)
st.dataframe(df_sec, use_container_width=True, hide_index=True)

# ============================================================
# 支撑反力表
# ============================================================
st.markdown("### 📌 支撑反力")

react_rows = []
for dof_idx, label in [(0, "N0"), (2*n_nodes-2, f"N{n_nodes-1}")]:
    node_i = dof_idx // 2
    react_rows.append({
        "节点": label,
        "x (mm)": f"{nodes[node_i]*1e3:.0f}",
        "Fx (N)": f"{0.0:.1f}",
        "Fy (N)": f"{res['nodal_v'][node_i]:.1f}",
        "Mz (N·mm)": f"{res['nodal_m'][node_i]:.1f}",
    })

df_react = pd.DataFrame(react_rows)
st.dataframe(df_react, use_container_width=True, hide_index=True)
