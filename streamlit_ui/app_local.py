"""
streamlit_ui/app_local.py
Equipment Utilization & Activity Classification — Dashboard
"""
from __future__ import annotations

import os
import sqlite3
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DB_PATH     = os.getenv("LOCAL_DB", "local_dev.db")
CLIP_PATH   = "latest_clip.mp4"
REFRESH_SEC = 5
MIN_FRAMES  = 100

st.set_page_config(
    page_title = "Equipment Monitor",
    page_icon  = "🏗️",
    layout     = "wide",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding: 1.5rem 2rem 1rem; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Top header bar */
.top-bar {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.25rem 1.75rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.top-bar-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.02em;
}
.top-bar-sub {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 2px;
}
.top-bar-badge {
    background: #1e3a5f;
    color: #60a5fa;
    border: 1px solid #1d4ed8;
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}

/* Status banner */
.banner {
    border-radius: 8px;
    padding: 0.65rem 1.1rem;
    font-size: 0.8rem;
    font-weight: 500;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.banner-processing { background:#1e293b; color:#94a3b8; border:1px solid #334155; }
.banner-done       { background:#052e16; color:#4ade80; border:1px solid #166534; }
.banner-waiting    { background:#1c1917; color:#78716c; border:1px solid #292524; }
.banner-error      { background:#450a0a; color:#f87171; border:1px solid #991b1b; }

/* KPI cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.85rem;
    margin-bottom: 1.5rem;
}
.kpi {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    position: relative;
    overflow: hidden;
}
.kpi::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent, #3b82f6);
    border-radius: 10px 10px 0 0;
}
.kpi-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1;
}
.kpi-sub { font-size: 0.72rem; color: #475569; margin-top: 4px; }

/* Machine card */
.mc {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.85rem;
    border-left: 4px solid var(--act-color, #334155);
}
.mc-id   { font-size: 1rem; font-weight: 700; color: #f1f5f9; }
.mc-type { font-size: 0.67rem; color: #475569; text-transform: uppercase;
           letter-spacing: 0.05em; margin-bottom: 0.5rem; }
.mc-act  {
    display: inline-block;
    background: var(--act-bg, #1e293b);
    color: var(--act-color, #94a3b8);
    border: 1px solid var(--act-border, #334155);
    border-radius: 5px;
    padding: 3px 10px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}
.mc-state-active   { display:inline-block; background:#052e16; color:#4ade80;
                     border:1px solid #166534; border-radius:4px;
                     padding:2px 8px; font-size:0.65rem; font-weight:600;
                     letter-spacing:0.06em; text-transform:uppercase; margin-left:6px; }
.mc-state-inactive { display:inline-block; background:#1c1917; color:#78716c;
                     border:1px solid #292524; border-radius:4px;
                     padding:2px 8px; font-size:0.65rem; font-weight:600;
                     letter-spacing:0.06em; text-transform:uppercase; margin-left:6px; }
.mc-util { font-size:1.6rem; font-weight:700; color:#f1f5f9; line-height:1; }
.mc-bar-bg { background:#1e293b; border-radius:3px; height:5px; margin:6px 0 4px; }
.mc-meta { font-size:0.7rem; color:#475569; line-height:1.7; }

/* Section label */
.sec { font-size:0.68rem; font-weight:600; letter-spacing:0.08em;
       text-transform:uppercase; color:#64748b;
       margin-bottom:0.75rem; padding-bottom:0.4rem;
       border-bottom:1px solid #1e293b; }

/* Video placeholder */
.vid-placeholder {
    background: #0a0f1e;
    border: 1px solid #1e293b;
    border-radius: 10px;
    height: 320px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    color: #334155;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_resource
def _conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def _q(sql):
    try:
        return pd.read_sql_query(sql, _conn())
    except Exception:
        return pd.DataFrame()

def _status():
    try:
        df = pd.read_sql_query(
            "SELECT key,value FROM processing_status", _conn())
        return dict(zip(df["key"], df["value"]))
    except Exception:
        return {}

def _chart(fig):
    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font=dict(color="#94a3b8", size=11),
        margin=dict(l=0, r=0, t=32, b=0),
        xaxis=dict(gridcolor="#1e293b", linecolor="#334155"),
        yaxis=dict(gridcolor="#1e293b", linecolor="#334155"),
        title_font=dict(size=12, color="#94a3b8"),
        legend=dict(bgcolor="#1e293b", bordercolor="#334155",
                    borderwidth=1, font=dict(size=10)),
    )
    return fig

ACT_COLORS = {
    "DIGGING":  ("#3b82f6", "#1e3a5f", "#1d4ed8"),
    "SWINGING": ("#a78bfa", "#2e1065", "#6d28d9"),
    "DUMPING":  ("#fbbf24", "#451a03", "#b45309"),
    "MOVING":   ("#22d3ee", "#083344", "#0e7490"),
    "WAITING":  ("#64748b", "#1e293b", "#334155"),
    "MIXING":   ("#34d399", "#022c22", "#065f46"),
}


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="top-bar">
  <div>
    <div class="top-bar-title">🏗️ Equipment Utilization Monitor</div>
    <div class="top-bar-sub">YOLOv11 · MOG2 · LSTM · Real-time Activity Classification</div>
  </div>
  <div class="top-bar-badge">PROTOTYPE v1.0</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STATUS BANNER
# ══════════════════════════════════════════════════════════════════════════════
s = _status()
ps = s.get("status", "waiting")

if ps in ("waiting", "starting"):
    st.markdown(
        '<div class="banner banner-waiting">⏳ &nbsp;Pipeline not started — '
        'run: <code>python run_local.py --video data/input.mp4 --fresh</code></div>',
        unsafe_allow_html=True)
elif ps == "processing":
    pct = float(s.get("progress_pct", 0))
    cf  = int(s.get("current_frame", 0))
    tf  = int(s.get("total_frames", 1))
    m   = s.get("machines", "0")
    st.markdown(
        f'<div class="banner banner-processing">⚙️ &nbsp;Processing &nbsp;—&nbsp; '
        f'{pct:.0f}% &nbsp;({cf:,} / {tf:,} frames &nbsp;|&nbsp; '
        f'{m} machine(s) detected)</div>',
        unsafe_allow_html=True)
    st.progress(pct / 100)
elif ps == "done":
    st.markdown(
        '<div class="banner banner-done">✅ &nbsp;Processing complete — '
        'all results loaded</div>', unsafe_allow_html=True)
elif ps == "error":
    st.markdown(
        f'<div class="banner banner-error">❌ &nbsp;Error: '
        f'{s.get("error","unknown")}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════════════════
df_sum = _q(f"""
    SELECT equipment_id, equipment_class,
           ROUND(MAX(total_active_sec),1) AS active_sec,
           ROUND(MAX(total_idle_sec),  1) AS idle_sec,
           ROUND(MAX(utilization_pct), 1) AS util_pct,
           COUNT(*) AS frames
    FROM equipment_telemetry
    GROUP BY equipment_id, equipment_class
    HAVING COUNT(*) > {MIN_FRAMES}
    ORDER BY util_pct DESC
""")

n_machines   = len(df_sum) if not df_sum.empty else 0
avg_util     = float(df_sum["util_pct"].mean()) if not df_sum.empty else 0.0
total_active = float(df_sum["active_sec"].sum()) if not df_sum.empty else 0.0
total_idle   = float(df_sum["idle_sec"].sum())   if not df_sum.empty else 0.0

k1, k2, k3, k4 = st.columns(4, gap="small")
def _kpi(col, label, val, sub="", accent="#3b82f6"):
    col.markdown(
        f'<div class="kpi" style="--accent:{accent};">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{val}</div>'
        f'{"" if not sub else f"<div class=kpi-sub>{sub}</div>"}'
        f'</div>', unsafe_allow_html=True)

_kpi(k1, "Machines Tracked",  str(n_machines),
     "confirmed detections", "#3b82f6")
_kpi(k2, "Avg Utilization",   f"{avg_util:.1f}%",
     "active / total tracked", "#a78bfa")
_kpi(k3, "Total Active",      f"{total_active/60:.1f} min",
     f"{total_active:.0f}s across all machines", "#22d3ee")
_kpi(k4, "Total Idle",        f"{total_idle/60:.1f} min",
     f"{total_idle:.0f}s across all machines", "#64748b")

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO + MACHINE CARDS
# ══════════════════════════════════════════════════════════════════════════════
col_vid, col_cards = st.columns([3, 1], gap="medium")

with col_vid:
    st.markdown('<div class="sec">Video Playback</div>',
                unsafe_allow_html=True)
    if os.path.exists(CLIP_PATH) and os.path.getsize(CLIP_PATH) > 10000:
        mtime = os.path.getmtime(CLIP_PATH)
        st.video(CLIP_PATH)
        st.markdown(
            f'<div style="font-size:0.7rem;color:#475569;margin-top:4px;">'
            f'Last clip: {time.strftime("%H:%M:%S", time.localtime(mtime))}'
            f' &nbsp;—&nbsp; updates every minute of processing</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="vid-placeholder">'
            '<span style="font-size:2rem;">▶</span>'
            '<span style="font-size:0.85rem;">Processing first minute...</span>'
            '<span style="font-size:0.72rem;color:#1e293b;">'
            'Video appears here after 1 min</span>'
            '</div>', unsafe_allow_html=True)

with col_cards:
    st.markdown('<div class="sec">Live Machine Status</div>',
                unsafe_allow_html=True)

    # Get latest state per machine
    df_live = _q(f"""
        SELECT t.equipment_id, t.equipment_class,
               t.current_state, t.current_activity,
               t.utilization_pct,
               t.total_active_sec, t.total_idle_sec
        FROM equipment_telemetry t
        INNER JOIN (
            SELECT equipment_id, MAX(frame_id) AS mf
            FROM equipment_telemetry
            GROUP BY equipment_id
            HAVING COUNT(*) > {MIN_FRAMES}
        ) l ON t.equipment_id = l.equipment_id
           AND t.frame_id     = l.mf
        ORDER BY t.equipment_id
    """)

    if df_live.empty:
        st.markdown(
            '<div style="color:#334155;font-size:0.8rem;padding:1rem;">'
            'No machines yet.</div>', unsafe_allow_html=True)
    else:
        for _, row in df_live.iterrows():
            act     = row["current_activity"]
            state   = row["current_state"]
            util    = float(row["utilization_pct"])
            colors  = ACT_COLORS.get(act, ACT_COLORS["WAITING"])
            act_c, act_bg, act_border = colors
            bar_c   = "#3b82f6" if util >= 70 else "#f59e0b" if util >= 40 else "#f87171"
            s_cls   = "mc-state-active" if state == "ACTIVE" else "mc-state-inactive"
            eq_type = str(row["equipment_class"]).replace("_", " ").title()
            a_min   = float(row["total_active_sec"]) / 60
            i_min   = float(row["total_idle_sec"])   / 60

            st.markdown(f"""
<div class="mc" style="--act-color:{act_c};">
  <div class="mc-id">{row["equipment_id"]}</div>
  <div class="mc-type">{eq_type}</div>
  <span class="mc-act"
    style="--act-color:{act_c};--act-bg:{act_bg};--act-border:{act_border};">
    {act}
  </span>
  <span class="{s_cls}">{state}</span>
  <div style="margin-top:0.5rem;">
    <span class="mc-util">{util:.1f}<span style="font-size:0.9rem;color:#475569;font-weight:400">%</span></span>
  </div>
  <div class="mc-bar-bg">
    <div style="height:5px;border-radius:3px;background:{bar_c};width:{min(util,100):.0f}%;"></div>
  </div>
  <div class="mc-meta">
    Active: {a_min:.1f} min &nbsp;|&nbsp; Idle: {i_min:.1f} min
  </div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════
if not df_sum.empty:
    st.markdown('<div class="sec">Analytics</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")

    with c1:
        fig = go.Figure()
        for _, row in df_sum.iterrows():
            col = "#3b82f6" if row["util_pct"] >= 70 else "#f59e0b"
            fig.add_trace(go.Bar(
                x=[row["equipment_id"]], y=[row["util_pct"]],
                marker_color=col, showlegend=False,
                text=[f"{row['util_pct']:.1f}%"], textposition="outside",
            ))
        fig.add_hline(y=70, line_dash="dot", line_color="#475569",
                      annotation_text="Target 70%",
                      annotation_font_color="#475569",
                      annotation_font_size=10)
        fig.update_layout(
            title="Utilization per Machine",
            yaxis_range=[0, 115], yaxis_title="Utilization %")
        st.plotly_chart(_chart(fig), use_container_width=True)

    with c2:
        fig2 = go.Figure(go.Pie(
            labels=["Active", "Idle"],
            values=[max(total_active, 0.001), max(total_idle, 0.001)],
            hole=0.68,
            marker=dict(colors=["#3b82f6", "#1e293b"],
                        line=dict(color="#0f172a", width=2)),
            textinfo="label+percent",
            textfont=dict(color="#94a3b8", size=11),
        ))
        fig2.update_layout(
            title="Fleet Active vs Idle",
            annotations=[dict(
                text=f"{avg_util:.0f}%", x=0.5, y=0.5,
                showarrow=False, font=dict(size=30, color="#f1f5f9"),
            )])
        st.plotly_chart(_chart(fig2), use_container_width=True)

    # Activity breakdown
    df_act = _q(f"""
        SELECT equipment_id, current_activity, COUNT(*) AS frames
        FROM equipment_telemetry
        WHERE equipment_id IN (
            SELECT equipment_id FROM equipment_telemetry
            GROUP BY equipment_id HAVING COUNT(*) > {MIN_FRAMES}
        )
        GROUP BY equipment_id, current_activity
        ORDER BY equipment_id, frames DESC
    """)

    if not df_act.empty:
        c3, c4 = st.columns(2, gap="medium")
        with c3:
            fig3 = go.Figure()
            for act, (col, _, _) in ACT_COLORS.items():
                sub = df_act[df_act["current_activity"] == act]
                if sub.empty:
                    continue
                fig3.add_trace(go.Bar(
                    name=act, x=sub["equipment_id"],
                    y=sub["frames"], marker_color=col,
                ))
            fig3.update_layout(
                title="Activity Breakdown by Machine",
                barmode="group",
                xaxis_title="Machine", yaxis_title="Frames",
                legend=dict(orientation="h", y=-0.3))
            st.plotly_chart(_chart(fig3), use_container_width=True)

        with c4:
            display = df_sum.rename(columns={
                "equipment_id":   "Machine",
                "equipment_class":"Type",
                "active_sec":     "Active (s)",
                "idle_sec":       "Idle (s)",
                "util_pct":       "Util %",
                "frames":         "Frames",
            }).copy()
            display["Type"] = (
                display["Type"].str.replace("_", " ").str.title()
            )
            st.markdown(
                '<div class="sec">Summary Table</div>',
                unsafe_allow_html=True)
            st.dataframe(display, use_container_width=True,
                         hide_index=True, height=280)

    # Utilization over time
    df_ts = _q(f"""
        SELECT ts, equipment_id, utilization_pct
        FROM equipment_telemetry
        WHERE equipment_id IN (
            SELECT equipment_id FROM equipment_telemetry
            GROUP BY equipment_id HAVING COUNT(*) > {MIN_FRAMES}
        )
        ORDER BY ts
    """)
    if not df_ts.empty:
        df_ts["ts"] = pd.to_datetime(df_ts["ts"])
        palette = ["#3b82f6","#a78bfa","#fbbf24","#34d399"]
        pfill   = ["rgba(59,130,246,0.07)","rgba(167,139,250,0.07)",
                   "rgba(251,191,36,0.07)","rgba(52,211,153,0.07)"]
        fig4 = go.Figure()
        for i, eid in enumerate(df_ts["equipment_id"].unique()):
            sub = df_ts[df_ts["equipment_id"] == eid]
            fig4.add_trace(go.Scatter(
                x=sub["ts"], y=sub["utilization_pct"],
                mode="lines", name=eid,
                line=dict(width=1.5, color=palette[i % len(palette)]),
                fill="tozeroy",
                fillcolor=pfill[i % len(pfill)],
            ))
        fig4.add_hline(y=70, line_dash="dot", line_color="#475569",
                       annotation_text="Target 70%",
                       annotation_font_color="#475569",
                       annotation_font_size=10)
        fig4.update_layout(
            title="Utilization Over Time",
            yaxis_range=[0, 105],
            xaxis_title="Time", yaxis_title="Utilization %")
        st.plotly_chart(_chart(fig4), use_container_width=True)

else:
    st.markdown(
        '<div style="color:#334155;font-size:0.85rem;padding:3rem;text-align:center;">'
        'No data yet — start the pipeline to see analytics.</div>',
        unsafe_allow_html=True)


# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div style="border-top:1px solid #1e293b;margin-top:2rem;'
    f'padding-top:0.75rem;display:flex;justify-content:space-between;'
    f'align-items:center;">'
    f'<span style="font-size:0.65rem;color:#334155;">'
    f'Equipment Utilization &amp; Activity Classification — Prototype v1.0</span>'
    f'<span style="font-size:0.65rem;color:#334155;">'
    f'Auto-refresh every {REFRESH_SEC}s &nbsp;·&nbsp; '
    f'{pd.Timestamp.now().strftime("%d %b %Y  %H:%M:%S")}</span>'
    f'</div>',
    unsafe_allow_html=True)

time.sleep(REFRESH_SEC)
st.rerun()