"""
副车架梁模型快速分析工具 - 可视化交互版
运行方式: py -3.14 -m streamlit run app.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from beam_solver import BeamSolver

# ========== 页面配置 ==========
st.set_page_config(
    page_title="副车架梁分析工具",
    page_icon="🔧",
    layout="wide"
)

# ========== 自定义样式 ==========
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem !important;
    }
    [data-testid="stHeader"] {
        background: transparent !important;
        height: auto !important;
        min-height: 0 !important;
        overflow: visible !important;
    }
    h1 {
        font-size: 1.5rem !important;
        line-height: 1.6 !important;
        overflow: visible !important;
        padding-top: 0.5rem !important;
    }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1rem !important; }
    .stMetric { background: #f8f9fa; border-radius: 8px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🔧 副车架梁模型分析工具")


# ========== 初始化 Session State ==========
if 'segments' not in st.session_state:
    st.session_state.segments = [
        {'length': 250.0, 'h': 80.0, 'b': 55.0, 't': 2.5, 'type': '矩形管'},
        {'length': 300.0, 'h': 85.0, 'b': 58.0, 't': 2.5, 'type': '矩形管'},
        {'length': 250.0, 'h': 80.0, 'b': 55.0, 't': 2.5, 'type': '矩形管'},
    ]
if 'supports' not in st.session_state:
    st.session_state.supports = [
        {'node': 0.0, 'type': 'fixed'},
        {'node': 3.0, 'type': 'fixed'},
    ]
if 'point_loads' not in st.session_state:
    st.session_state.point_loads = [
        {'node': 2.0, 'Fx': 0.0, 'Fy': -5000.0, 'Mz': 0.0},
    ]


# ========== 辅助函数 ==========
def calc_rect_Iy(h, b, t):
    Iy = (b * h**3 - (b - 2*t) * (h - 2*t)**3) / 12
    return max(Iy, 0.0)

def calc_rect_A(h, b, t):
    A = h * b - (h - 2*t) * (b - 2*t)
    return max(A, 0.0)

def calc_rect_J(h, b, t):
    A_m = (h - t) * (b - t)
    perimeter = 2.0 * ((h - t) + (b - t))
    J = 4.0 * A_m**2 * t / perimeter
    return J

def get_node_positions(segments):
    positions = [0.0]
    for seg in segments:
        positions.append(positions[-1] + seg['length'])
    return positions

def draw_beam_structure(segments, supports, point_loads, dist_loads=None):
    node_pos = get_node_positions(segments)
    total_length = node_pos[-1]

    fig = go.Figure()

    # ========== 绘制梁段 ==========
    max_h = max(seg['h'] for seg in segments)
    if max_h <= 0:
        max_h = 80.0

    for i, seg in enumerate(segments):
        x0 = node_pos[i]
        x1 = node_pos[i + 1]
        h_ratio = seg['h'] / max_h
        bar_h = h_ratio * 20.0

        fig.add_shape(
            type="rect",
            x0=x0, x1=x1,
            y0=-bar_h/2, y1=bar_h/2,
            fillcolor="rgba(52, 152, 219, 0.6)",
            line=dict(color="rgba(41, 128, 185, 1)", width=2),
            layer="below"
        )

        mid_x = (x0 + x1) / 2.0
        fig.add_annotation(
            x=mid_x, y=bar_h/2 + 8,
            text=f"段{i+1}<br>{seg['h']}×{seg['b']}×{seg['t']}mm",
            showarrow=False,
            font=dict(size=10, color="#2c3e50"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#bdc3c7",
            borderwidth=1,
            borderpad=3
        )

    # ========== 绘制节点 ==========
    fig.add_trace(go.Scatter(
        x=node_pos,
        y=[0.0] * len(node_pos),
        mode='markers+text',
        marker=dict(size=12, color='#e74c3c', symbol='circle',
                    line=dict(width=2, color='white')),
        text=[f"N{i}" for i in range(len(node_pos))],
        textposition='bottom center',
        textfont=dict(size=10, color='#e74c3c'),
        hovertext=[f"节点{i}<br>x = {p:.0f} mm" for i, p in enumerate(node_pos)],
        hoverinfo='text',
        name='节点',
        showlegend=False
    ))

    # ========== 绘制支撑 ==========
    support_colors = {'fixed': '#2ecc71', 'pinned': '#f39c12', 'roller_y': '#9b59b6'}
    support_symbols = {'fixed': 'diamond', 'pinned': 'triangle-up', 'roller_y': 'circle'}
    support_labels = {'fixed': '固支', 'pinned': '铰支', 'roller_y': '竖向约束'}

    for sup in supports:
        nid = int(sup['node'])
        if nid < len(node_pos):
            x = node_pos[nid]
            color = support_colors.get(sup['type'], '#2ecc71')
            symbol = support_symbols.get(sup['type'], 'diamond')
            triangle_size = 12.0

            fig.add_trace(go.Scatter(
                x=[x], y=[-triangle_size - 5],
                mode='markers',
                marker=dict(size=18, color=color, symbol=symbol,
                            line=dict(width=2, color='white')),
                name=f"支撑: {support_labels.get(sup['type'], sup['type'])} @N{nid}",
                hovertext=f"节点{nid} | {support_labels.get(sup['type'], sup['type'])}",
                hoverinfo='text'
            ))

            fig.add_shape(
                type="line",
                x0=x-10, x1=x+10,
                y0=-triangle_size - 14, y1=-triangle_size - 14,
                line=dict(color=color, width=2, dash="solid")
            )
            for offset in [-8, -4, 0, 4, 8]:
                fig.add_shape(
                    type="line",
                    x0=x+offset, x1=x+offset-4,
                    y0=-triangle_size - 14, y1=-triangle_size - 20,
                    line=dict(color=color, width=1)
                )

    # ========== 绘制集中力 ==========
    for pl in point_loads:
        nid = int(pl['node'])
        if nid < len(node_pos):
            x = node_pos[nid]
            fy = pl['Fy']

            if abs(fy) > 0:
                direction = -1.0 if fy < 0 else 1.0
                arrow_length = min(abs(fy) / 5000.0 * 25.0, 40.0)

                fig.add_annotation(
                    x=x, y=0.0,
                    ax=x, ay=direction * arrow_length,
                    xref="x", yref="y",
                    axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=3, arrowsize=1.5, arrowwidth=3,
                    arrowcolor='#e74c3c'
                )
                fig.add_annotation(
                    x=x, y=direction * (arrow_length + 8),
                    text=f"{abs(fy)/1000:.1f} kN",
                    showarrow=False,
                    font=dict(size=11, color='#e74c3c', family='Arial Black'),
                    bgcolor="rgba(255,255,255,0.8)"
                )

            mz = pl.get('Mz', 0.0)
            if abs(mz) > 0:
                fig.add_annotation(
                    x=x, y=25,
                    text=f"M={mz/1000:.1f} kN·m",
                    showarrow=False,
                    font=dict(size=10, color='#e67e22'),
                    bgcolor="rgba(255,255,255,0.8)"
                )

    # ========== 绘制分布载荷 ==========
    if dist_loads:
        for dl in dist_loads:
            eid = int(dl['elem_id'])
            w1 = dl['w_start']
            w2 = dl['w_end']
            if eid < len(segments):
                x0 = node_pos[eid]
                x1 = node_pos[eid + 1]
                n_pts = 30
                x_dist = np.linspace(x0, x1, n_pts)
                w_dist = np.linspace(w1, w2, n_pts)

                scale = 20.0 / max(abs(max(w_dist)), abs(min(w_dist)), 1.0)
                y_top = 0.0
                y_load = [-w * scale for w in w_dist]

                x_fill = np.concatenate([x_dist, x_dist[::-1]])
                y_fill = np.concatenate([[y_top]*n_pts, [yl for yl in y_load[::-1]]])

                fig.add_trace(go.Scatter(
                    x=x_fill, y=y_fill,
                    fill='toself',
                    fillcolor='rgba(46, 204, 113, 0.3)',
                    line=dict(color='rgba(39, 174, 96, 1)', width=1),
                    name=f"分布载荷 @段{eid+1}",
                    hoverinfo='name'
                ))

                for j in range(0, n_pts, max(n_pts // 6, 1)):
                    fig.add_annotation(
                        x=x_dist[j], y=0.0,
                        ax=x_dist[j], ay=y_load[j],
                        xref="x", yref="y",
                        axref="x", ayref="y",
                        showarrow=True,
                        arrowhead=2, arrowsize=1, arrowwidth=1.5,
                        arrowcolor='rgba(39, 174, 96, 0.8)'
                    )

    fig.update_layout(
        title=dict(text="梁结构示意图（结构与载荷可视化）", font=dict(size=14)),
        xaxis=dict(
            title="位置 x (mm)",
            range=[-30, total_length + 30],
            showgrid=True, gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(title="", range=[-40, 35], showticklabels=False, showgrid=False),
        height=350,
        margin=dict(l=20, r=20, t=40, b=30),
        plot_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=9)),
        hovermode='closest'
    )

    return fig


def draw_results_plot(internal_forces, supports, node_pos):
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=("弯矩图 M (kN·m)", "剪力图 V (kN)", "挠度图 δ (mm)"),
        vertical_spacing=0.08
    )

    colors = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6', '#e74c3c']

    for i, r in enumerate(internal_forces):
        c = colors[i % len(colors)]
        label = f"段{r['elem_id']+1}"

        fig.add_trace(go.Scatter(
            x=r['x_global'], y=r['moment'],
            mode='lines', name=label,
            line=dict(color=c, width=2),
            fill='tozeroy',
            hovertemplate='x=%{x:.0f}mm<br>M=%{y:.3f} kN·m<extra></extra>'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=r['x_global'], y=r['shear'],
            mode='lines', name=label,
            line=dict(color=c, width=2),
            fill='tozeroy',
            showlegend=False,
            hovertemplate='x=%{x:.0f}mm<br>V=%{y:.3f} kN<extra></extra>'
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=r['x_global'], y=r['deflection'],
            mode='lines', name=label,
            line=dict(color=c, width=2),
            fill='tozeroy',
            showlegend=False,
            hovertemplate='x=%{x:.0f}mm<br>δ=%{y:.6f} mm<extra></extra>'
        ), row=3, col=1)

    for sup in supports:
        if int(sup['node']) < len(node_pos):
            sx = node_pos[int(sup['node'])]
            for row in [1, 2, 3]:
                fig.add_vline(x=sx, line=dict(color='red', width=1, dash='dot'), opacity=0.5, row=row, col=1)

    fig.update_layout(
        height=600,
        margin=dict(l=50, r=20, t=60, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        hovermode='x unified'
    )

    fig.update_xaxes(title_text="位置 x (mm)", row=3, col=1)
    fig.update_yaxes(title_text="M (kN·m)", row=1, col=1)
    fig.update_yaxes(title_text="V (kN)", row=2, col=1)
    fig.update_yaxes(title_text="δ (mm)", row=3, col=1)

    return fig


# ========== 左侧边栏 ==========
with st.sidebar:
    st.header("📋 参数设置")

    st.subheader("材料")
    material = st.selectbox("选择材料", ["SAPH440", "DP780", "6061-T6铝合金", "自定义"], label_visibility="collapsed")
    mat_data = {
        "SAPH440": {"E": 206000.0, "sy": 305.0},
        "DP780": {"E": 206000.0, "sy": 780.0},
        "6061-T6铝合金": {"E": 68900.0, "sy": 276.0},
        "自定义": {"E": 206000.0, "sy": 305.0},
    }
    E = st.number_input("E (MPa)", value=mat_data[material]["E"], step=1000.0)
    sigma_y = st.number_input("σy (MPa)", value=mat_data[material]["sy"], step=10.0)

    st.divider()

    st.subheader("梁分段")
    n_seg = st.number_input("分段数", min_value=1.0, max_value=10.0, value=float(len(st.session_state.segments)), step=1.0)
    n_seg = int(n_seg)

    while len(st.session_state.segments) < n_seg:
        st.session_state.segments.append({'length': 200.0, 'h': 80.0, 'b': 55.0, 't': 2.5, 'type': '矩形管'})
    while len(st.session_state.segments) > n_seg:
        st.session_state.segments.pop()

    for i in range(n_seg):
        seg = st.session_state.segments[i]
        with st.expander(f"段{i+1}", expanded=(i == 0)):
            c1, c2 = st.columns(2)
            with c1:
                seg['length'] = st.number_input("长度 mm", value=float(seg['length']), min_value=10.0, step=10.0, key=f"sl_{i}")
            with c2:
                seg['t'] = st.number_input("壁厚 mm", value=float(seg['t']), min_value=0.5, step=0.1, key=f"st_{i}")
            c3, c4 = st.columns(2)
            with c3:
                seg['h'] = st.number_input("高度 h mm", value=float(seg['h']), min_value=10.0, step=5.0, key=f"sh_{i}")
            with c4:
                seg['b'] = st.number_input("宽度 b mm", value=float(seg['b']), min_value=10.0, step=5.0, key=f"sb_{i}")

    st.divider()

    st.subheader("支撑条件")
    n_sup = st.number_input("支撑数", min_value=2.0, max_value=10.0, value=float(len(st.session_state.supports)), step=1.0, key="n_sup")
    n_sup = int(n_sup)

    while len(st.session_state.supports) < n_sup:
        st.session_state.supports.append({'node': 0.0, 'type': 'fixed'})
    while len(st.session_state.supports) > n_sup:
        st.session_state.supports.pop()

    total_nodes = n_seg + 1
    for i in range(n_sup):
        c1, c2 = st.columns(2)
        with c1:
            st.session_state.supports[i]['node'] = st.number_input(
                "节点", value=float(st.session_state.supports[i]['node']),
                min_value=0.0, max_value=float(total_nodes - 1), step=1.0, key=f"sup_n_{i}"
            )
        with c2:
            st.session_state.supports[i]['type'] = st.selectbox(
                "类型",
                options=['fixed', 'pinned', 'roller_y'],
                index=['fixed', 'pinned', 'roller_y'].index(st.session_state.supports[i]['type']),
                format_func=lambda x: {'fixed': '固支', 'pinned': '铰支', 'roller_y': '竖向约束'}.get(x, x),
                key=f"sup_t_{i}"
            )

    st.divider()

    st.subheader("载荷")
    n_pl = st.number_input("集中力数", min_value=0.0, max_value=10.0, value=float(len(st.session_state.point_loads)), step=1.0, key="n_pl")
    n_pl = int(n_pl)

    while len(st.session_state.point_loads) < n_pl:
        st.session_state.point_loads.append({'node': 1.0, 'Fx': 0.0, 'Fy': -5000.0, 'Mz': 0.0})
    while len(st.session_state.point_loads) > n_pl:
        st.session_state.point_loads.pop()

    for i in range(n_pl):
        with st.expander(f"集中力{i+1}"):
            c1, c2 = st.columns(2)
            with c1:
                st.session_state.point_loads[i]['node'] = st.number_input(
                    "节点", value=float(st.session_state.point_loads[i]['node']),
                    min_value=0.0, max_value=float(total_nodes - 1), step=1.0, key=f"pln_{i}"
                )
            with c2:
                st.session_state.point_loads[i]['Fy'] = st.number_input(
                    "Fy (N)", value=float(st.session_state.point_loads[i]['Fy']), step=500.0, key=f"plfy_{i}"
                )
            st.session_state.point_loads[i]['Mz'] = st.number_input(
                "Mz (N·mm)", value=float(st.session_state.point_loads[i].get('Mz', 0.0)), step=100.0, key=f"plmz_{i}"
            )

    st.subheader("分布载荷（可选）")
    use_dist = st.checkbox("添加分布载荷", value=False)
    dist_loads = []
    if use_dist:
        n_dl = st.number_input("分布载荷数", min_value=1.0, max_value=5.0, value=1.0, step=1.0, key="n_dl")
        n_dl = int(n_dl)
        for i in range(n_dl):
            c1, c2, c3 = st.columns(3)
            with c1:
                de = st.number_input("单元号", value=0.0, min_value=0.0, max_value=float(n_seg - 1), step=1.0, key=f"de_{i}")
            with c2:
                dw1 = st.number_input("w₁ (N/mm)", value=0.0, step=0.1, key=f"dw1_{i}")
            with c3:
                dw2 = st.number_input("w₂ (N/mm)", value=-1.0, step=0.1, key=f"dw2_{i}")
            dist_loads.append({'elem_id': de, 'w_start': dw1, 'w_end': dw2})

    st.divider()
    solve_clicked = st.button("🚀 求解", type="primary", use_container_width=True)


# ========== 主界面 ==========
node_pos = get_node_positions(st.session_state.segments)

fig_struct = draw_beam_structure(
    st.session_state.segments,
    st.session_state.supports,
    st.session_state.point_loads,
    dist_loads if use_dist else None
)
st.plotly_chart(fig_struct, use_container_width=True)

if solve_clicked:
    solver = BeamSolver()

    node_ids = []
    for i, pos in enumerate(node_pos):
        node_ids.append(solver.add_node(pos))

    for i, seg in enumerate(st.session_state.segments):
        Iy = calc_rect_Iy(seg['h'], seg['b'], seg['t'])
        A = calc_rect_A(seg['h'], seg['b'], seg['t'])
        solver.add_element(node_ids[i], node_ids[i + 1], Iy, A, E)

    for sup in st.session_state.supports:
        nid = int(sup['node'])
        if nid < len(node_ids):
            solver.add_support(nid, sup['type'])

    for pl in st.session_state.point_loads:
        nid = int(pl['node'])
        solver.add_point_load(nid, pl['Fx'], pl['Fy'], pl['Mz'])

    if use_dist:
        for dl in dist_loads:
            eid = int(dl['elem_id'])
            if eid < len(solver.elements):
                solver.add_dist_load(eid, dl['w_start'], dl['w_end'])

    try:
        U = solver.solve()
        internal = solver.get_internal_forces(n_points=200)
        max_results = solver.get_max_results(internal)
    except Exception as e:
        st.error(f"求解失败: {str(e)}")
        st.stop()

    st.divider()
    st.subheader("📊 分析结果")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最大弯矩", f"{max_results['max_moment']:.3f} kN·m")
    c2.metric("最大剪力", f"{max_results['max_shear']:.3f} kN")
    c3.metric("最大挠度", f"{max_results['max_deflection']:.4f} mm")

    max_seg = max(st.session_state.segments, key=lambda s: s['h'])
    h_half = max_seg['h'] / 2.0
    Iy_max = calc_rect_Iy(max_seg['h'], max_seg['b'], max_seg['t'])
    stress = max_results['max_moment'] * 1e6 * h_half / Iy_max if Iy_max > 0 else 0.0
    safety = sigma_y / stress if stress > 0 else 999.0
    c4.metric("安全系数", f"{safety:.2f}", delta="满足" if safety > 1.2 else "不满足", delta_color="normal" if safety > 1.2 else "inverse")

    st.subheader("📋 各段截面特性")
    seg_data = []
    for i, seg in enumerate(st.session_state.segments):
        Iy = calc_rect_Iy(seg['h'], seg['b'], seg['t'])
        A = calc_rect_A(seg['h'], seg['b'], seg['t'])
        J = calc_rect_J(seg['h'], seg['b'], seg['t'])
        Wy = Iy / (seg['h'] / 2.0)
        seg_data.append({
            '段号': f"段{i+1}",
            '长度 (mm)': seg['length'],
            'h×b×t (mm)': f"{seg['h']}×{seg['b']}×{seg['t']}",
            'A (mm²)': f"{A:.0f}",
            'Iy (mm⁴)': f"{Iy:.0f}",
            'J (mm⁴)': f"{J:.0f}",
            'Wy (mm³)': f"{Wy:.0f}",
        })
    st.dataframe(pd.DataFrame(seg_data), use_container_width=True, hide_index=True)

    st.subheader("📐 支撑反力")
    rxn_data = []
    for nid, comps in solver.reactions.items():
        rxn_data.append({
            '节点': f"N{nid}",
            'x (mm)': f"{node_pos[nid]:.0f}",
            'Fx (N)': f"{comps.get('Fx', 0.0):.1f}",
            'Fy (N)': f"{comps.get('Fy', 0.0):.1f}",
            'Mz (N·mm)': f"{comps.get('Mz', 0.0):.1f}",
        })
    st.dataframe(pd.DataFrame(rxn_data), use_container_width=True, hide_index=True)

    st.subheader("📈 内力图与变形图")
    fig_results = draw_results_plot(internal, st.session_state.supports, node_pos)
    st.plotly_chart(fig_results, use_container_width=True)

    st.subheader("📋 节点位移")
    disp_data = []
    for i in range(len(node_ids)):
        disp_data.append({
            '节点': f"N{i}",
            'x (mm)': f"{node_pos[i]:.0f}",
            'Ux (mm)': f"{U[i*3]:.6f}",
            'Uy (mm)': f"{U[i*3+1]:.6f}",
            'Rz (°)': f"{np.degrees(U[i*3+2]):.4f}",
        })
    st.dataframe(pd.DataFrame(disp_data), use_container_width=True, hide_index=True)

else:
    st.info("👆 在左侧设置参数，点击「求解」查看分析结果")
