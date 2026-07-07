"""
dashboard/app_streamlit.py — PomodorIA · Dashboard di Monitoraggio

Design premium: SVG icons, gauge charts, histogram, benchmark comparator,
radar sensor, system monitor. Nessun emoji nei titoli di sezione.

Avvio:
    streamlit run dashboard/app_streamlit.py
"""
import sys, os, time
from datetime import datetime
from collections import Counter
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import yaml, psutil

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="PomodorIA · Serra Domotica",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🍅</text></svg>",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "PomodorIA — Edge AI Serra Domotica · Univ. Federico II"},
)

# ══════════════════════════════════════════════════════════════
# COSTANTI
# ══════════════════════════════════════════════════════════════
CLASS_LABELS = {
    "Tomato___Bacterial_spot":                        "Bacterial Spot",
    "Tomato___Early_blight":                          "Early Blight",
    "Tomato___Late_blight":                           "Late Blight",
    "Tomato___Leaf_Mold":                             "Leaf Mold",
    "Tomato___Septoria_leaf_spot":                    "Septoria",
    "Tomato___Spider_mites Two-spotted_spider_mite":  "Spider Mites",
    "Tomato___Target_Spot":                           "Target Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":         "TYLCV",
    "Tomato___Tomato_mosaic_virus":                   "Mosaic Virus",
    "Tomato___healthy":                               "Healthy",
}
DISEASE_COLORS = {
    "healthy": "#00b894", "fungal": "#fd9644", "viral": "#ee5a24",
    "bacterial": "#7c5cfc", "pest": "#ff9f43", "unknown": "#636e72",
}
CLASS_COLORS = {
    "Healthy": "#00b894", "Early Blight": "#fd9644", "Late Blight": "#e17055",
    "Leaf Mold": "#fdcb6e", "Septoria": "#a29bfe", "Spider Mites": "#ff9f43",
    "Target Spot": "#ff7675", "TYLCV": "#ee5a24", "Mosaic Virus": "#d63031",
    "Bacterial Spot": "#7c5cfc",
}
MODE_LABELS = {
    "full_precision":               "FULL PRECISION (float32)",
    "optimized_onnx":                "OPTIMIZED · ONNX INT8",
    "optimized_pruned_quantized":    "OPTIMIZED · PRUNED+QUANT INT8",
    "optimized_pruned":              "OPTIMIZED · PRUNED (float32)",
}
BIAS_OPTIONS = ["— Random —"] + list(CLASS_LABELS.keys())
_C = dict(teal="#00d4aa", purple="#7c5cfc", amber="#f59e0b",
          red="#ee5a24", green="#00b894", muted="rgba(255,255,255,0.38)")

# ══════════════════════════════════════════════════════════════
# SVG ICON LIBRARY  (Heroicons stroke style)
# ══════════════════════════════════════════════════════════════
_SVG = {
"activity":    "M22 12h-4l-3 9L9 3l-3 9H2",
"target":      "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20zm0-14a4 4 0 1 0 0 8 4 4 0 0 0 0-8zm0 2a2 2 0 1 1 0 4 2 2 0 0 1 0-4z",
"zap":         "M13 2 3 14h9l-1 8 10-12h-9z",
"database":    "M12 2c4.97 0 9 1.34 9 3v14c0 1.66-4.03 3-9 3S3 20.66 3 19V5c0-1.66 4.03-3 9-3zm9 5c0 1.66-4.03 3-9 3S3 12.66 3 12m18 4c0 1.66-4.03 3-9 3S3 16.66 3 16",
"alert-tri":   "M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0zM12 9v4m0 4h.01",
"thermometer": "M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z",
"droplet":     "M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z",
"sun":         "M12 7a5 5 0 1 0 0 10A5 5 0 0 0 12 7zm0-5v2m0 14v2M4.22 4.22l1.42 1.42m12.72 12.72 1.42 1.42M2 12h2m14 0h4M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42",
"cpu":         "M9 1v3m6-3v3M9 20v3m6-3v3M1 9h3m-3 6h3m20-6h-3m3 6h-3M4 4h16a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2zm5 5h6v6H9z",
"refresh":     "M23 4v6h-6M1 20v-6h6m16.73-6A10 10 0 0 0 3.27 10.27M.27 16A10 10 0 0 0 20.73 13.73",
"bell":        "M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9m-4.27 13a2 2 0 0 1-3.46 0",
"check":       "M20 6 9 17 4 12",
"x-mark":      "M18 6 6 18M6 6l12 12",
"leaf":        "M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10zm-9 1c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12",
"wind":        "M9.59 4.59A2 2 0 1 1 11 8H2m10.59 11.41A2 2 0 1 0 14 16H2m15.73-8.27A2.5 2.5 0 1 1 19.5 12H2",
"server":      "M2 2h20a2 2 0 0 1 2 2v4a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2zm0 12h20a2 2 0 0 1 2 2v4a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-4a2 2 0 0 1 2-2zm4 2h.01M6 6h.01",
"bar-chart":   "M18 20V10M12 20V4M6 20v-6",
"settings":    "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z",
"microscope":  "M6 18h8M3 22h18M14 22a7 7 0 1 0 0-14h-1M9 14h2M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2zm3-6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3",
"clock":       "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20zm0-6V12l4 2",
"layers":      "M12 2 2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5",
}

def _icon(name: str, size: int = 16, color: str = "currentColor") -> str:
    d = _SVG.get(name, "")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" '
        f'stroke-linecap="round" stroke-linejoin="round">'
        f'<path d="{d}"/></svg>'
    )

def _sec(icon_name: str, label: str, color: str = _C["teal"]) -> str:
    return (
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:14px">'
        f'<span style="color:{color}">{_icon(icon_name, 15, color)}</span>'
        f'<span style="font-size:.65rem;font-weight:700;letter-spacing:.12em;'
        f'text-transform:uppercase;color:{_C["muted"]}">{label}</span>'
        f'<div style="flex:1;height:1px;background:rgba(255,255,255,0.05);margin-left:6px"></div>'
        f'</div>'
    )

# ══════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
*,[class*="css"]{font-family:'Inter',sans-serif!important;}
code,pre{font-family:'JetBrains Mono',monospace!important;}

.stApp{background:linear-gradient(160deg,#070b14 0%,#0e1420 50%,#070b14 100%);}
.block-container{padding:1rem 1.5rem 2rem;max-width:1500px;}
#MainMenu,footer,.stDeployButton{visibility:hidden;}
header[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important;}

/* ─ Cards ─ */
.card{
  background:linear-gradient(145deg,rgba(255,255,255,0.025) 0%,rgba(255,255,255,0.01) 100%);
  border:1px solid rgba(255,255,255,0.07);border-radius:18px;
  padding:20px 22px;position:relative;overflow:hidden;
}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(0,212,170,0.3),transparent);}
.card-accent-purple::before{background:linear-gradient(90deg,transparent,rgba(124,92,252,0.4),transparent);}
.card-accent-amber::before{background:linear-gradient(90deg,transparent,rgba(245,158,11,0.4),transparent);}
.card-accent-red::before{background:linear-gradient(90deg,transparent,rgba(238,90,36,0.4),transparent);}

/* ─ Header ─ */
.page-header{
  background:linear-gradient(135deg,rgba(0,212,170,0.08) 0%,rgba(124,92,252,0.08) 100%);
  border:1px solid rgba(0,212,170,0.15);border-radius:20px;
  padding:22px 30px;display:flex;align-items:center;justify-content:space-between;
  margin-bottom:22px;position:relative;overflow:hidden;
}
.page-header::after{content:'';position:absolute;right:-60px;top:-60px;
  width:200px;height:200px;border-radius:50%;
  background:radial-gradient(circle,rgba(0,212,170,0.06) 0%,transparent 70%);}
.ph-title{font-size:1.55rem;font-weight:800;color:#fff;letter-spacing:-.03em;}
.ph-sub{font-size:.72rem;color:rgba(255,255,255,0.35);letter-spacing:.08em;text-transform:uppercase;margin-top:3px;}

/* ─ Status pill ─ */
.pill{display:inline-flex;align-items:center;gap:6px;padding:6px 16px;
  border-radius:20px;font-size:.68rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;}
.pill-run{background:rgba(0,184,148,0.12);border:1px solid rgba(0,184,148,0.3);color:#00b894;}
.pill-stop{background:rgba(99,110,114,0.12);border:1px solid rgba(99,110,114,0.25);color:#b2bec3;}
.dot{width:7px;height:7px;border-radius:50%;}
.dot-run{background:#00b894;box-shadow:0 0 6px #00b894;animation:pulse 1.4s infinite;}
.dot-stop{background:#636e72;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}

/* ─ KPI ─ */
.kpi{background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);
  border-radius:16px;padding:18px 14px;text-align:center;transition:border-color .2s,transform .15s;}
.kpi:hover{border-color:rgba(0,212,170,0.2);transform:translateY(-1px);}
.kpi-icon{margin-bottom:6px;opacity:.7;}
.kpi-v{font-size:2rem;font-weight:800;line-height:1;letter-spacing:-.03em;}
.kpi-l{font-size:.6rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
  color:rgba(255,255,255,0.35);margin-top:5px;}
.kpi-s{font-size:.68rem;color:rgba(255,255,255,0.28);margin-top:3px;}

/* ─ Actuators ─ */
.act{display:flex;align-items:center;gap:10px;padding:11px 14px;
  border-radius:11px;margin-bottom:8px;transition:all .2s;}
.act-on{background:rgba(0,184,148,0.06);border:1px solid rgba(0,184,148,0.2);}
.act-off{background:rgba(255,255,255,0.015);border:1px solid rgba(255,255,255,0.055);}
.act-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}
.ad-on{background:#00b894;box-shadow:0 0 9px rgba(0,184,148,.65);}
.ad-off{background:rgba(255,255,255,0.15);}
.act-name{font-size:.82rem;font-weight:600;color:rgba(255,255,255,.88);}
.act-sub{font-size:.66rem;color:rgba(255,255,255,.35);}
.act-st-on{font-size:.63rem;font-weight:700;color:#00b894;letter-spacing:.08em;margin-left:auto;}
.act-st-off{font-size:.63rem;font-weight:600;color:rgba(255,255,255,.2);letter-spacing:.08em;margin-left:auto;}

/* ─ Reasoning ─ */
.reason-item{font-size:.72rem;color:rgba(255,255,255,.62);padding:6px 10px;
  border-left:2px solid rgba(0,212,170,.4);border-radius:0 6px 6px 0;
  background:rgba(0,212,170,0.03);margin-bottom:6px;line-height:1.5;}
.reason-item.warn{border-left-color:rgba(253,150,68,.5);background:rgba(253,150,68,0.03);}
.reason-item.crit{border-left-color:rgba(238,90,36,.5);background:rgba(238,90,36,0.03);}

/* ─ Prediction badge ─ */
.pred-badge{display:inline-block;padding:5px 14px;border-radius:20px;
  font-size:.75rem;font-weight:700;letter-spacing:.05em;}

/* ─ Sidebar ─ */
[data-testid="stSidebar"]{background:rgba(7,11,20,.97);border-right:1px solid rgba(255,255,255,0.055);}
[data-testid="stSidebarContent"]{padding-top:.8rem;}

/* ─ RPI panel ─ */
.rpi{background:linear-gradient(135deg,rgba(124,92,252,0.07),rgba(0,212,170,0.07));
  border:1px solid rgba(124,92,252,0.18);border-radius:13px;padding:14px;margin-top:8px;}
.rpi-row{display:flex;justify-content:space-between;margin-bottom:6px;font-size:.73rem;}
.rpi-k{color:rgba(255,255,255,.38);}
.rpi-v{color:rgba(255,255,255,.85);font-weight:600;}
.rpi-bar-bg{background:rgba(255,255,255,0.06);border-radius:3px;height:4px;margin-top:3px;overflow:hidden;}
.rpi-bar-fill{height:100%;border-radius:3px;transition:width .5s;}

/* ─ Buttons ─ */
div.stButton>button{width:100%;border-radius:11px;font-weight:700;
  letter-spacing:.03em;border:none!important;transition:all .2s;}
div.stButton>button:first-child{
  background:linear-gradient(135deg,#00b894,#00cec9)!important;color:#070b14!important;}
div.stButton>button:first-child:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(0,184,148,.3);}

/* ─ Divider ─ */
.hdiv{height:1px;background:rgba(255,255,255,0.055);margin:14px 0;}

/* ─ Stat line (sidebar) ─ */
.sl{display:flex;justify-content:space-between;align-items:center;
  padding:7px 0;border-bottom:1px solid rgba(255,255,255,0.04);font-size:.73rem;}
.sl-k{color:rgba(255,255,255,.38);display:flex;align-items:center;gap:6px;}
.sl-v{color:rgba(255,255,255,.85);font-weight:600;font-variant-numeric:tabular-nums;}

/* ─ Benchmark table ─ */
.bm-row{display:flex;align-items:center;gap:10px;padding:9px 12px;
  border-radius:9px;margin-bottom:6px;background:rgba(255,255,255,0.02);
  border:1px solid rgba(255,255,255,0.05);}
.bm-label{font-size:.77rem;font-weight:600;flex:1;color:rgba(255,255,255,.8);}
.bm-bar-bg{flex:2;background:rgba(255,255,255,0.05);border-radius:3px;height:6px;overflow:hidden;}
.bm-bar{height:100%;border-radius:3px;}
.bm-val{font-size:.72rem;font-variant-numeric:tabular-nums;color:rgba(255,255,255,.55);width:60px;text-align:right;}

/* ─ Log table ─ */
.stDataFrame [data-testid="stDataFrameResizable"]{border-radius:12px!important;}

/* ─ Charts ─ */
.js-plotly-plot .plotly{border-radius:14px;}
</style>
"""

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def _cfg() -> dict:
    with open(os.path.join(ROOT, "config.yaml")) as f:
        return yaml.safe_load(f)

# ══════════════════════════════════════════════════════════════
# INIT COMPONENTS
# ══════════════════════════════════════════════════════════════
def _init(config: dict, mode: str, bias: Optional[str], variant: str = "auto"):
    import copy
    from sensors.virtual_camera import VirtualCameraSensor
    from sensors.environment_simulator import EnvironmentSensorSimulator
    from edge.inference_engine import InferenceEngine
    from agent.decision_agent import DecisionAgent
    from actuators.actuators import ActuatorBank

    cfg = copy.deepcopy(config)
    cfg["model"]["mode"] = mode
    cfg["model"]["optimized_variant"] = variant
    cfg.setdefault("virtual_camera", {})["bias_class"] = bias

    ss = st.session_state
    ss.camera    = VirtualCameraSensor(cfg)
    ss.env_sim   = EnvironmentSensorSimulator(cfg)
    ss.engine    = InferenceEngine(cfg)
    ss.actuators = ActuatorBank()
    ss.agent     = DecisionAgent(cfg, ss.actuators)
    ss.cycles    = []
    ss.init      = True
    ss.cfg_mode  = mode
    ss.cfg_bias  = bias
    ss.cfg_variant = variant

# ══════════════════════════════════════════════════════════════
# CYCLE RUNNER
# ══════════════════════════════════════════════════════════════
def _cycle():
    ss = st.session_state
    try:
        cap = ss.camera.capture()
        env = ss.env_sim.read(disease_label=cap.true_label)
        inf = ss.engine.predict(cap.image_tensor)
        dec = ss.agent.decide(inf, env, cap.true_label)
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.Process().memory_info().rss / 1024**2

        ss.cycles.append({
            "n":          len(ss.cycles),
            "ts":         datetime.now().strftime("%H:%M:%S"),
            "true":       cap.true_label,
            "true_s":     CLASS_LABELS.get(cap.true_label, cap.true_label),
            "pred":       inf.predicted_label,
            "pred_s":     CLASS_LABELS.get(inf.predicted_label, inf.predicted_label),
            "conf":       inf.confidence,
            "probs":      inf.all_probabilities,
            "ms":         inf.inference_time_ms,
            "ram":        ram,
            "cpu":        cpu,
            "mode":       inf.model_mode,
            "temp":       env.temperature_c,
            "hum":        env.humidity_pct,
            "soil":       env.soil_moisture_pct,
            "lux":        env.light_lux,
            "cat":        dec.disease_category,
            "consec":     dec.consecutive_alerts,
            "irr":        ss.actuators.irrigation.is_active,
            "vent":       ss.actuators.ventilation.is_active,
            "alarm":      ss.actuators.alarm.is_active,
            "notif":      ss.actuators.notification.total_notifications,
            "ok":         cap.true_label == inf.predicted_label,
            "reasoning":  dec.reasoning,
            "img":        cap.image_path,
        })
        ss.err = None
    except Exception as e:
        ss.err = str(e)
        ss.running = False

# ══════════════════════════════════════════════════════════════
# CHART FUNCTIONS
# ══════════════════════════════════════════════════════════════
_PB = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
           plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color="rgba(255,255,255,0.6)"))

def _chart_confidence(probs, classes, pred_label) -> go.Figure:
    short  = [CLASS_LABELS.get(c, c) for c in classes]
    pairs  = sorted(zip(probs, short, classes), reverse=True)
    sp, ss_l, sc = zip(*pairs) if pairs else ([], [], [])
    cols   = ["#00d4aa" if c == pred_label else "rgba(255,255,255,0.06)" for c in sc]
    tcols  = ["rgba(255,255,255,0.9)" if c == pred_label else "rgba(255,255,255,0.3)" for c in sc]
    fig = go.Figure(go.Bar(
        x=list(sp), y=list(ss_l), orientation="h",
        marker=dict(color=cols, line=dict(width=0)),
        text=[f"{p:.1%}" for p in sp], textposition="outside",
        textfont=dict(size=10, color=tcols),
        hovertemplate="%{y}: %{x:.2%}<extra></extra>",
    ))
    fig.update_layout(**_PB, height=290,
        margin=dict(l=4, r=60, t=10, b=6),
        xaxis=dict(range=[0,1.28], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=10.5)),
        showlegend=False,
    )
    return fig

def _chart_env_gauges(latest: dict) -> go.Figure:
    specs = [[{"type": "indicator"}] * 4]
    fig = make_subplots(rows=1, cols=4, specs=specs,
                        subplot_titles=["Temperatura", "Umidità Aria", "Umidità Suolo", "Luminosità"])
    gauges = [
        (latest["temp"],  0,  50,  "°C",  "#ee5a24" if latest["temp"]>35 else "#00d4aa",
         [{"range":[0,15],"color":"rgba(124,92,252,0.12)"},
          {"range":[15,30],"color":"rgba(0,212,170,0.12)"},
          {"range":[30,50],"color":"rgba(238,90,36,0.12)"}]),
        (latest["hum"],   0, 100,  "%",   "#ee5a24" if latest["hum"]>85 else ("#fd9644" if latest["hum"]>70 else "#00d4aa"),
         [{"range":[0,40],"color":"rgba(0,212,170,0.08)"},
          {"range":[40,70],"color":"rgba(0,212,170,0.12)"},
          {"range":[70,100],"color":"rgba(238,90,36,0.12)"}]),
        (latest["soil"],  0, 100,  "%",   "#ee5a24" if latest["soil"]<25 else ("#fd9644" if latest["soil"]<40 else "#00d4aa"),
         [{"range":[0,30],"color":"rgba(238,90,36,0.12)"},
          {"range":[30,60],"color":"rgba(0,212,170,0.12)"},
          {"range":[60,100],"color":"rgba(0,184,148,0.1)"}]),
        (min(latest["lux"],80000), 0, 80000, " lux", "#00d4aa" if latest["lux"]>3000 else "#fd9644",
         [{"range":[0,2000],"color":"rgba(238,90,36,0.1)"},
          {"range":[2000,40000],"color":"rgba(0,212,170,0.1)"},
          {"range":[40000,80000],"color":"rgba(253,203,110,0.1)"}]),
    ]
    for i, (val, lo, hi, sfx, col, steps) in enumerate(gauges, 1):
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=val,
            number={"suffix": sfx, "font": {"size": 20, "color": "#fff", "family": "Inter"}, "valueformat": ".1f" if sfx != " lux" else ".0f"},
            gauge={
                "axis": {"range":[lo,hi], "tickwidth":0, "showticklabels": False},
                "bar":  {"color": col, "thickness": 0.28},
                "bgcolor": "rgba(255,255,255,0.03)",
                "borderwidth": 0,
                "steps": steps,
            },
        ), row=1, col=i)
    fig.update_layout(**_PB, height=190,
        margin=dict(l=10, r=10, t=28, b=4),
        title_font=dict(size=11),
    )
    fig.update_annotations(font_size=10, font_color="rgba(255,255,255,0.45)")
    return fig

def _chart_system_monitor(cycles: list) -> go.Figure:
    latest = cycles[-1] if cycles else {}
    cpu  = latest.get("cpu", 0)
    ram  = latest.get("ram", 0)
    lat  = latest.get("ms", 0)
    specs = [[{"type":"indicator"},{"type":"indicator"},{"type":"indicator"}]]
    fig = make_subplots(rows=1, cols=3, specs=specs,
                        subplot_titles=["CPU Utilizzo", "RAM Processo", "Latenza Inferenza"])
    for col_i, (val, lo, hi, sfx, col, steps) in enumerate([
        (cpu, 0, 100, "%", "#7c5cfc" if cpu<50 else "#ee5a24",
         [{"range":[0,40],"color":"rgba(0,212,170,0.08)"},
          {"range":[40,70],"color":"rgba(253,150,68,0.08)"},
          {"range":[70,100],"color":"rgba(238,90,36,0.08)"}]),
        (min(ram,2000), 0, 2000, " MB", "#7c5cfc",
         [{"range":[0,500],"color":"rgba(0,212,170,0.08)"},
          {"range":[500,1200],"color":"rgba(124,92,252,0.1)"},
          {"range":[1200,2000],"color":"rgba(238,90,36,0.08)"}]),
        (min(lat,100), 0, 100, " ms", "#00d4aa" if lat<20 else "#fd9644",
         [{"range":[0,15],"color":"rgba(0,212,170,0.12)"},
          {"range":[15,40],"color":"rgba(253,150,68,0.1)"},
          {"range":[40,100],"color":"rgba(238,90,36,0.1)"}]),
    ], 1):
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=val,
            number={"suffix": sfx, "font":{"size":20,"color":"#fff","family":"Inter"}, "valueformat":".0f"},
            gauge={
                "axis":{"range":[lo,hi],"tickwidth":0,"showticklabels":False},
                "bar":{"color":col,"thickness":0.28},
                "bgcolor":"rgba(255,255,255,0.03)","borderwidth":0,"steps":steps,
            },
        ), row=1, col=col_i)
    fig.update_layout(**_PB, height=190, margin=dict(l=10,r=10,t=28,b=4))
    fig.update_annotations(font_size=10, font_color="rgba(255,255,255,0.45)")
    return fig

def _chart_accuracy_timeline(cycles: list) -> go.Figure:
    if not cycles:
        return go.Figure(layout=dict(**_PB, height=230,
            title=dict(text="Accuracy & Latenza nel tempo", font=dict(size=13), x=0.01),
            margin=dict(l=8,r=8,t=36,b=8)))
    idxs   = [c["n"] for c in cycles]
    cumAcc = [sum(1 for cc in cycles[:i+1] if cc["ok"])/(i+1)*100 for i in range(len(cycles))]
    lats   = [c["ms"] for c in cycles]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idxs, y=cumAcc, name="Accuracy %",
        mode="lines", line=dict(color="#00d4aa", width=2.5, shape="spline"),
        fill="tozeroy", fillcolor="rgba(0,212,170,0.06)",
        hovertemplate="Ciclo %{x}: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=idxs, y=lats, name="Latenza ms",
        mode="lines", yaxis="y2", line=dict(color="#7c5cfc", width=1.8, dash="dot"),
        hovertemplate="Ciclo %{x}: %{y:.1f}ms<extra></extra>"))
    wrong_x = [c["n"] for c in cycles if not c["ok"]]
    wrong_y = [cumAcc[c["n"]] for c in cycles if not c["ok"]]
    if wrong_x:
        fig.add_trace(go.Scatter(x=wrong_x, y=wrong_y, mode="markers", name="Errore",
            marker=dict(color="#ee5a24", size=9, symbol="x", line=dict(width=2)),
            hovertemplate="Errore ciclo %{x}<extra></extra>"))
    fig.update_layout(**_PB, height=230,
        title=dict(text="Accuracy & Latenza nel tempo", font=dict(size=13), x=0.01),
        margin=dict(l=8,r=8,t=36,b=8),
        xaxis=dict(title="Ciclo", showgrid=True, gridcolor="rgba(255,255,255,0.04)", zeroline=False),
        yaxis=dict(title="Acc %", range=[0,108], showgrid=True, gridcolor="rgba(255,255,255,0.04)"),
        yaxis2=dict(overlaying="y", side="right", showgrid=False, zeroline=False,
                    tickfont=dict(color="#7c5cfc", size=10)),
        legend=dict(orientation="h", y=1.08, font=dict(size=10)),
    )
    return fig

def _chart_distribution(cycles: list) -> go.Figure:
    if not cycles:
        return go.Figure(layout=dict(**_PB, height=230,
            title=dict(text="Distribuzione predizioni", font=dict(size=13), x=0.01),
            margin=dict(l=8,r=8,t=36,b=8)))
    counts = Counter(c["pred_s"] for c in cycles)
    labels = list(counts.keys())
    vals   = list(counts.values())
    colors = [CLASS_COLORS.get(l, "#636e72") for l in labels]
    n = len(cycles)
    acc = sum(1 for c in cycles if c["ok"])/n*100 if n else 0
    fig = go.Figure(go.Pie(
        labels=labels, values=vals, hole=0.62,
        marker=dict(colors=colors, line=dict(color="rgba(7,11,20,0.9)", width=2)),
        textinfo="percent", textfont=dict(size=10),
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        pull=[0.05 if l == max(counts, key=counts.get) else 0 for l in labels],
    ))
    fig.add_annotation(text=f"<b>{acc:.0f}%</b><br><span style='font-size:10px;opacity:.6'>ACC</span>",
        x=0.5, y=0.5, showarrow=False, font=dict(size=18, color="#00d4aa"), align="center")
    fig.update_layout(**_PB, height=230,
        title=dict(text="Distribuzione predizioni", font=dict(size=13), x=0.01),
        margin=dict(l=8,r=8,t=36,b=8),
        legend=dict(font=dict(size=9), x=1.01, y=0.5),
    )
    return fig

def _chart_latency_hist(cycles: list) -> go.Figure:
    if len(cycles) < 3:
        return go.Figure(layout=dict(**_PB, height=220,
            title=dict(text="Distribuzione Latenza", font=dict(size=13), x=0.01),
            margin=dict(l=8,r=8,t=36,b=8)))
    lats = [c["ms"] for c in cycles]
    fig = go.Figure(go.Histogram(
        x=lats, nbinsx=20,
        marker=dict(color="#7c5cfc", opacity=0.8,
                    line=dict(color="rgba(255,255,255,0.1)", width=0.5)),
        hovertemplate="Latenza %{x:.1f}ms — %{y} cicli<extra></extra>",
    ))
    avg = sum(lats)/len(lats)
    fig.add_vline(x=avg, line=dict(color="#00d4aa", dash="dash", width=2),
        annotation=dict(text=f"media {avg:.1f}ms", font=dict(size=10, color="#00d4aa"),
                        bgcolor="rgba(0,0,0,0)"))
    fig.update_layout(**_PB, height=220,
        title=dict(text="Distribuzione Latenza (ms)", font=dict(size=13), x=0.01),
        margin=dict(l=8,r=8,t=36,b=8),
        xaxis=dict(title="ms", showgrid=True, gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(title="Cicli", showgrid=True, gridcolor="rgba(255,255,255,0.04)"),
        showlegend=False,
        bargap=0.08,
    )
    return fig

def _chart_sensor_radar(cycles: list) -> go.Figure:
    if len(cycles) < 2:
        return go.Figure(layout=dict(**_PB, height=220,
            title=dict(text="Radar Sensori (ultimi 5)", font=dict(size=13), x=0.01),
            margin=dict(l=8,r=8,t=36,b=8)))
    cats = ["Temperatura", "Umidità Aria", "Umidità Suolo", "Luminosità (n)"]
    recent = cycles[-5:]
    fig = go.Figure()
    for c in recent:
        vals = [
            c["temp"] / 50 * 100,
            c["hum"],
            c["soil"],
            min(c["lux"], 80000) / 80000 * 100,
        ]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]],
            fill="toself", fillcolor="rgba(0,212,170,0.05)",
            line=dict(color="rgba(0,212,170,0.4)", width=1.2),
            name=f"Ciclo {c['n']}",
            hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
        ))
    fig.update_layout(**_PB, height=220,
        title=dict(text="Radar Sensori — ultimi 5 cicli", font=dict(size=13), x=0.01),
        margin=dict(l=8,r=8,t=36,b=8),
        polar=dict(
            radialaxis=dict(visible=True, range=[0,100], showticklabels=False,
                            gridcolor="rgba(255,255,255,0.07)"),
            angularaxis=dict(tickfont=dict(size=9), gridcolor="rgba(255,255,255,0.07)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
    )
    return fig

def _chart_benchmark() -> Optional[go.Figure]:
    path = os.path.join(ROOT, "benchmarks", "benchmark_results.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    # Size bars
    cols = ["#00d4aa", "#7c5cfc", "#fd9644", "#ee5a24"]
    names = df["variant"].tolist()
    sizes = pd.to_numeric(df["file_size_mb"], errors="coerce").fillna(0).tolist()
    lats  = pd.to_numeric(df["avg_inference_ms"].astype(str).str.replace("ms",""), errors="coerce").fillna(0).tolist()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Size (MB)", x=names, y=sizes,
        marker=dict(color=[cols[i % len(cols)] for i in range(len(names))], opacity=0.85,
                    line=dict(width=0)),
        text=[f"{s:.1f}MB" for s in sizes], textposition="outside",
        textfont=dict(size=10), yaxis="y",
        hovertemplate="%{x}: %{y:.1f} MB<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        name="Latenza (ms)", x=names, y=lats, mode="markers+lines",
        marker=dict(size=9, color="#f59e0b", line=dict(width=1.5, color="rgba(0,0,0,0.3)")),
        line=dict(color="#f59e0b", width=1.8, dash="dot"),
        yaxis="y2",
        hovertemplate="%{x}: %{y:.1f}ms<extra></extra>",
    ))
    fig.update_layout(**_PB, height=220,
        title=dict(text="Confronto Modelli — Size vs Latenza", font=dict(size=13), x=0.01),
        margin=dict(l=8,r=8,t=36,b=8),
        xaxis=dict(showgrid=False, tickangle=-15, tickfont=dict(size=9)),
        yaxis=dict(title="Size MB", showgrid=True, gridcolor="rgba(255,255,255,0.04)"),
        yaxis2=dict(title="ms", overlaying="y", side="right", showgrid=False,
                    tickfont=dict(color="#f59e0b", size=10)),
        legend=dict(orientation="h", y=1.08, font=dict(size=10)),
        bargap=0.35,
    )
    return fig

# ══════════════════════════════════════════════════════════════
# HTML HELPERS
# ══════════════════════════════════════════════════════════════
def _hex_rgb(h):
    h = h.lstrip("#")
    return f"{int(h[:2],16)},{int(h[2:4],16)},{int(h[4:],16)}"

def _badge(label: str, cat: str) -> str:
    c = DISEASE_COLORS.get(cat, "#636e72")
    return (f'<span class="pred-badge" style="background:rgba({_hex_rgb(c)},.12);'
            f'border:1px solid rgba({_hex_rgb(c)},.35);color:{c}">{label.upper()}</span>')

def _kpi(icon_n, val, label, sub="", color=_C["teal"]) -> str:
    return (f'<div class="kpi"><div class="kpi-icon" style="color:{color}">'
            f'{_icon(icon_n,18,color)}</div>'
            f'<div class="kpi-v" style="color:{color}">{val}</div>'
            f'<div class="kpi-l">{label}</div>'
            + (f'<div class="kpi-s">{sub}</div>' if sub else "") +
            f'</div>')

def _act_row(icon_n, name, on, sub="") -> str:
    cls   = "act-on" if on else "act-off"
    dcls  = "ad-on"  if on else "ad-off"
    scls  = "act-st-on" if on else "act-st-off"
    return (f'<div class="act {cls}">'
            f'<div class="act-dot {dcls}"></div>'
            f'<span style="color:rgba(255,255,255,0.5)">{_icon(icon_n,14,"currentColor")}</span>'
            f'<div><div class="act-name">{name}</div>'
            f'<div class="act-sub">{sub if on else "inattivo"}</div></div>'
            f'<div class="{scls}">{"ATTIVO" if on else "OFF"}</div>'
            f'</div>')

def _rpi_bar(pct: float, color: str) -> str:
    return (f'<div class="rpi-bar-bg"><div class="rpi-bar-fill" '
            f'style="width:{min(pct,100):.0f}%;background:{color}"></div></div>')

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
def _sidebar(config: dict):
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:10px 0 20px">'
            f'<div style="color:#00d4aa;margin-bottom:6px">{_icon("leaf",44,"#00d4aa")}</div>'
            '<div style="font-size:1.15rem;font-weight:800;color:#fff;letter-spacing:-.02em">PomodorIA</div>'
            '<div style="font-size:.65rem;color:rgba(255,255,255,.35);letter-spacing:.1em;text-transform:uppercase;margin-top:2px">Serra Domotica · Edge AI</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(_sec("settings", "Configurazione"), unsafe_allow_html=True)
        mode = st.selectbox("Modello",
            ["full_precision","optimized"],
            format_func=lambda x: "Full Precision (float32)" if x=="full_precision" else "Ottimizzato (INT8)",
            key="sel_mode")

        variant = "auto"
        if mode == "optimized":
            variant = st.selectbox(
                "Variante ottimizzata",
                ["auto", "onnx", "pruned_quantized", "pruned"],
                format_func=lambda v: {
                    "auto": "Auto (priorità: ONNX > pruned+quant > pruned)",
                    "onnx": "ONNX quantizzato (~stessa accuracy baseline)",
                    "pruned_quantized": "Pruned + Quantizzato INT8 (più compresso, accuracy più bassa)",
                    "pruned": "Solo Pruned (float32)",
                }[v],
                key="sel_variant",
                help="Forza quale file in models/optimized/ usare, invece di lasciare "
                     "che il sistema scelga automaticamente il primo disponibile.",
            )

        bias_raw = st.selectbox("Classe forzata", BIAS_OPTIONS, key="sel_bias")
        bias = None if bias_raw.startswith("—") else bias_raw
        interval = st.slider("Intervallo cicli (sec)", 0.0, 10.0, 1.0, 0.5, key="sel_int")
        max_cyc  = st.number_input("Max cicli (0 = ∞)", 0, 10000, 50, key="sel_max")

        st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)
        st.markdown(_sec("refresh", "Controlli"), unsafe_allow_html=True)

        ss = st.session_state
        running = ss.get("running", False)
        if not running:
            if st.button("Avvia Simulazione", key="btn_start"):
                if (not ss.get("init") or ss.get("cfg_mode")!=mode
                        or ss.get("cfg_bias")!=bias or ss.get("cfg_variant")!=variant):
                    with st.spinner("Inizializzazione..."):
                        _init(config, mode, bias, variant)
                ss.running = True; ss.interval = interval; ss.max_cyc = max_cyc
                st.rerun()
        else:
            if st.button("Ferma", key="btn_stop"):
                ss.running = False; st.rerun()

        if ss.get("init"):
            if st.button("Reset sessione", key="btn_reset"):
                ss.running = False; ss.cycles = []
                ss.actuators.deactivate_all("reset")
                st.rerun()

        if ss.get("err"):
            st.error(f"Errore: {ss.err}")

        # ── Raspberry Pi panel ──
        st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)
        st.markdown(_sec("server", "Simulazione Raspberry Pi", _C["purple"]), unsafe_allow_html=True)

        import torch
        cpu_now = psutil.cpu_percent(interval=None)
        ram_now = psutil.Process().memory_info().rss / 1024**2
        threads = torch.get_num_threads()
        model_mb = ss.engine.get_model_size_mb() if ss.get("engine") else 0.0
        variant_loaded = ss.engine.loaded_variant if ss.get("engine") else "—"
        variant_label = MODE_LABELS.get(variant_loaded, variant_loaded)

        cycles = ss.get("cycles", [])
        avg_ms = sum(c["ms"] for c in cycles)/len(cycles) if cycles else 0

        st.markdown(f"""
        <div class="rpi">
          <div class="sl"><span class="sl-k">{_icon("cpu",12,"rgba(255,255,255,0.4)")} Thread attivi</span><span class="sl-v">{threads} / {psutil.cpu_count()} core</span></div>
          <div class="sl"><span class="sl-k">{_icon("layers",12,"rgba(255,255,255,0.4)")} Device</span><span class="sl-v">CPU only</span></div>
          <div class="sl"><span class="sl-k">{_icon("activity",12,"rgba(255,255,255,0.4)")} CPU attuale</span><span class="sl-v">{cpu_now:.0f}%</span></div>
          {_rpi_bar(cpu_now, "#7c5cfc")}
          <div class="sl" style="margin-top:6px"><span class="sl-k">{_icon("database",12,"rgba(255,255,255,0.4)")} RAM processo</span><span class="sl-v">{ram_now:.0f} MB</span></div>
          {_rpi_bar(ram_now/4000*100, "#00d4aa")}
          <div class="sl" style="margin-top:6px"><span class="sl-k">{_icon("layers",12,"rgba(255,255,255,0.4)")} Dimensione file</span><span class="sl-v">{model_mb:.1f} MB</span></div>
          <div class="sl"><span class="sl-k">{_icon("zap",12,"rgba(255,255,255,0.4)")} Lat. media</span><span class="sl-v">{avg_ms:.1f} ms</span></div>
          <div class="sl" style="margin-top:6px;border-bottom:none"><span class="sl-k">{_icon("microscope",12,"rgba(255,255,255,0.4)")} Modello in uso</span></div>
          <div style="font-size:.72rem;font-weight:700;color:#00d4aa;text-align:right;margin-top:-4px">{variant_label}</div>
          <div style="font-size:.6rem;color:rgba(255,255,255,.2);text-align:center;margin-top:8px">RPi 4: 4 core ARM · 2–8 GB RAM · inference on CPU</div>
        </div>
        """, unsafe_allow_html=True)

    return interval, max_cyc

# ══════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════
def _main(config: dict):
    ss      = st.session_state
    cycles  = ss.get("cycles", [])
    running = ss.get("running", False)
    latest  = cycles[-1] if cycles else None
    classes = config["model"]["classes"]
    n       = len(cycles)

    # ── Header ──────────────────────────────────────────────────
    pill_cls  = "pill-run" if running else "pill-stop"
    dot_cls   = "dot-run"  if running else "dot-stop"
    acc_txt   = f"{sum(c['ok'] for c in cycles)/n:.1%}" if n else "—"
    avg_ms    = f"{sum(c['ms'] for c in cycles)/n:.1f}" if n else "—"

    st.markdown(f"""
    <div class="page-header">
      <div>
        <div class="ph-title">PomodorIA · Serra Domotica con Edge AI</div>
        <div class="ph-sub">Università Federico II &nbsp;·&nbsp; Corso Internet of Things &nbsp;·&nbsp; Edge Inference su CPU</div>
      </div>
      <div style="display:flex;align-items:center;gap:20px">
        <div style="text-align:right">
          <div style="font-size:.62rem;color:rgba(255,255,255,.32);text-transform:uppercase;letter-spacing:.08em">Cicli</div>
          <div style="font-size:1.4rem;font-weight:800;color:#fff">{n}</div>
        </div>
        <div style="text-align:right">
          <div style="font-size:.62rem;color:rgba(255,255,255,.32);text-transform:uppercase;letter-spacing:.08em">Accuracy</div>
          <div style="font-size:1.4rem;font-weight:800;color:#00d4aa">{acc_txt}</div>
        </div>
        <div style="text-align:right">
          <div style="font-size:.62rem;color:rgba(255,255,255,.32);text-transform:uppercase;letter-spacing:.08em">Latenza avg</div>
          <div style="font-size:1.4rem;font-weight:800;color:#7c5cfc">{avg_ms}ms</div>
        </div>
        <span class="pill {pill_cls}">
          <span class="dot {dot_cls}"></span>{"IN ESECUZIONE" if running else "FERMO"}
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row ─────────────────────────────────────────────────
    consec = latest["consec"] if latest else 0
    notif  = latest["notif"]  if latest else 0
    ram    = latest["ram"]    if latest else 0
    cpu    = latest["cpu"]    if latest else 0
    acc_c  = _C["green"] if n and sum(c["ok"] for c in cycles)/n > 0.9 else _C["amber"]
    lat_c  = _C["teal"]  if n and sum(c["ms"] for c in cycles)/n < 20  else _C["amber"]
    alr_c  = _C["red"]   if consec >= 3 else (_C["amber"] if consec > 0 else "rgba(255,255,255,0.4)")

    k1,k2,k3,k4,k5 = st.columns(5, gap="small")
    k1.markdown(_kpi("refresh",  str(n),     "Cicli totali", f"max: {ss.get('max_cyc','∞')}"), unsafe_allow_html=True)
    k2.markdown(_kpi("target",   acc_txt,    "Accuracy live", color=acc_c), unsafe_allow_html=True)
    k3.markdown(_kpi("zap",      f"{avg_ms}ms", "Latenza media", color=lat_c), unsafe_allow_html=True)
    k4.markdown(_kpi("database", f"{ram:.0f}MB", "RAM processo", f"CPU {cpu:.0f}%", _C["purple"]), unsafe_allow_html=True)
    k5.markdown(_kpi("alert-tri",str(consec),"Alert consecutivi", f"notif: {notif}", alr_c), unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Row 1: Diagnosis + Environment Gauges ───────────────────
    col_d, col_e = st.columns([5, 7], gap="medium")

    with col_d:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(_sec("microscope", "Diagnosi Pianta"), unsafe_allow_html=True)
        if latest:
            ci, cp = st.columns([2,3], gap="small")
            with ci:
                if os.path.exists(latest["img"]):
                    try:
                        img = Image.open(latest["img"]).convert("RGB")
                        st.image(img, width="stretch")
                    except Exception:
                        st.markdown(f'<div style="width:100%;aspect-ratio:1;background:rgba(255,255,255,.03);border-radius:10px;display:flex;align-items:center;justify-content:center;color:rgba(255,255,255,.2)">{_icon("leaf",40,"currentColor")}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="width:100%;aspect-ratio:1;background:rgba(255,255,255,.03);border-radius:10px;display:flex;align-items:center;justify-content:center;color:rgba(255,255,255,.2)">{_icon("leaf",40,"currentColor")}</div>', unsafe_allow_html=True)
            with cp:
                cat   = latest["cat"]
                color = DISEASE_COLORS.get(cat, "#636e72")
                conf  = latest["conf"]
                ok    = latest["ok"]
                st.markdown(
                    f'<div style="font-size:.6rem;color:{_C["muted"]};text-transform:uppercase;letter-spacing:.08em">Predizione CNN</div>'
                    f'<div style="font-size:1rem;font-weight:700;color:#fff;margin:5px 0 6px">{latest["pred_s"]}</div>'
                    f'{_badge(cat, cat)}'
                    f'<div style="margin-top:12px;font-size:.6rem;color:{_C["muted"]};text-transform:uppercase;letter-spacing:.08em">Confidenza</div>'
                    f'<div style="font-size:2rem;font-weight:800;color:{color};line-height:1;letter-spacing:-.03em">{conf:.1%}</div>'
                    f'<div style="margin-top:8px;font-size:.68rem;color:{"#00b894" if ok else "#ee5a24"}">'
                    f'{_icon("check" if ok else "x-mark", 12, "#00b894" if ok else "#ee5a24")} Vera: <b>{latest["true_s"]}</b></div>',
                    unsafe_allow_html=True,
                )
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            pred_idx = classes.index(latest["pred"]) if latest["pred"] in classes else 0
            st.plotly_chart(_chart_confidence(latest["probs"], classes, latest["pred"]),
                            width="stretch", config={"displayModeBar": False})
        else:
            st.markdown(
                f'<div style="text-align:center;padding:60px 20px;color:rgba(255,255,255,.18)">'
                f'{_icon("leaf",48,"currentColor")}'
                f'<div style="font-size:.88rem;margin-top:12px">Premi <b>Avvia</b> per iniziare</div></div>',
                unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_e:
        # ─ Gauges ambientali ─
        st.markdown('<div class="card card-accent-purple">', unsafe_allow_html=True)
        st.markdown(_sec("thermometer", "Sensori Ambientali", _C["purple"]), unsafe_allow_html=True)
        if latest:
            st.plotly_chart(_chart_env_gauges(latest), width="stretch", config={"displayModeBar": False})
            # mini trend
            if len(cycles) >= 3:
                nr = min(20, len(cycles)); rec = cycles[-nr:]
                ft = go.Figure()
                ft.add_trace(go.Scatter(x=list(range(nr)), y=[c["temp"] for c in rec],
                    name="T°C", line=dict(color="#ee5a24", width=1.8),
                    fill="tozeroy", fillcolor="rgba(238,90,36,0.05)",
                    hovertemplate="Ciclo %{x}: %{y:.1f}°C<extra>T</extra>"))
                ft.add_trace(go.Scatter(x=list(range(nr)), y=[c["hum"] for c in rec],
                    name="Hum%", line=dict(color="#7c5cfc", width=1.8),
                    hovertemplate="Ciclo %{x}: %{y:.0f}%<extra>Hum</extra>"))
                ft.add_trace(go.Scatter(x=list(range(nr)), y=[c["soil"] for c in rec],
                    name="Soil%", line=dict(color="#00b894", width=1.8, dash="dot"),
                    hovertemplate="Ciclo %{x}: %{y:.0f}%<extra>Soil</extra>"))
                ft.update_layout(**_PB, height=110,
                    margin=dict(l=4, r=4, t=14, b=4),
                    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=9)),
                    legend=dict(font=dict(size=9), orientation="h", y=1.12),
                )
                st.plotly_chart(ft, width="stretch", config={"displayModeBar": False})
        else:
            st.markdown('<div style="text-align:center;padding:40px;color:rgba(255,255,255,.18)">Nessun dato</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Row 2: Actuators + Reasoning + System Monitor ───────────
    col_a, col_sys = st.columns([5, 7], gap="medium")

    with col_a:
        st.markdown('<div class="card card-accent-amber">', unsafe_allow_html=True)
        st.markdown(_sec("bell", "Stato Attuatori", _C["amber"]), unsafe_allow_html=True)
        if ss.get("actuators"):
            acts = ss.actuators
            st.markdown(
                _act_row("droplet", "Irrigazione",    acts.irrigation.is_active,  "pompa suolo attiva")
                +_act_row("wind",   "Ventilazione",   acts.ventilation.is_active, "aerazione forzata")
                +_act_row("alert-tri","Allarme LED",  acts.alarm.is_active,       "emergenza rilevata")
                +_act_row("bell",   "Notifiche",      latest and latest["notif"]>0 if latest else False,
                          f"{latest['notif']} inviate" if latest and latest['notif'] else ""),
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div style="text-align:center;padding:30px;color:rgba(255,255,255,.18)">—</div>', unsafe_allow_html=True)

        if latest and latest["reasoning"]:
            st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)
            st.markdown(_sec("activity", "Reasoning Agente"), unsafe_allow_html=True)
            for r in latest["reasoning"][:4]:
                lvl = "crit" if "CRITICO" in r or "ALLARME" in r or "VIRALE" in r else ("warn" if "FUNGINA" in r or "BATTERICA" in r else "")
                st.markdown(f'<div class="reason-item {lvl}">{r}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_sys:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(_sec("cpu", "System Monitor — Simulazione Edge", _C["teal"]), unsafe_allow_html=True)
        st.plotly_chart(_chart_system_monitor(cycles), width="stretch", config={"displayModeBar": False})

        # Timestamp last cycle + mode badge
        if latest:
            mode_c = "#00d4aa" if "full" in latest["mode"] else "#7c5cfc"
            mode_label = MODE_LABELS.get(latest["mode"], latest["mode"].upper())
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'margin-top:8px;padding:8px 12px;background:rgba(255,255,255,0.02);'
                f'border-radius:8px;border:1px solid rgba(255,255,255,0.05)">'
                f'<span style="font-size:.7rem;color:{_C["muted"]}">{_icon("clock",12,"currentColor")} Ultimo ciclo: <b style="color:rgba(255,255,255,.7)">{latest["ts"]}</b></span>'
                f'<span style="font-size:.68rem;font-weight:700;color:{mode_c};letter-spacing:.05em">'
                f'{_icon("layers",12,mode_c)} {mode_label}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Row 3: Timeline + Distribution ──────────────────────────
    st.markdown(_sec("bar-chart", "Analytics — Timeline e Distribuzione"), unsafe_allow_html=True)
    ct, cd = st.columns([6, 4], gap="medium")
    ct.plotly_chart(_chart_accuracy_timeline(cycles), width="stretch", config={"displayModeBar": False})
    cd.plotly_chart(_chart_distribution(cycles),      width="stretch", config={"displayModeBar": False})

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Row 4: Latency hist + Radar + Benchmark ─────────────────
    st.markdown(_sec("layers", "Analisi Avanzata"), unsafe_allow_html=True)
    ch, cr, cb = st.columns([4, 4, 4], gap="medium")
    ch.plotly_chart(_chart_latency_hist(cycles),   width="stretch", config={"displayModeBar": False})
    cr.plotly_chart(_chart_sensor_radar(cycles),   width="stretch", config={"displayModeBar": False})
    bm = _chart_benchmark()
    if bm:
        cb.plotly_chart(bm, width="stretch", config={"displayModeBar": False})
    else:
        cb.markdown('<div style="height:220px;display:flex;align-items:center;justify-content:center;color:rgba(255,255,255,.2);font-size:.8rem">Benchmark CSV non trovato</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Log table ───────────────────────────────────────────────
    if cycles:
        st.markdown(_sec("clock", f"Storico Cicli — ultimi 30 di {n} totali"), unsafe_allow_html=True)
        rec = cycles[-30:][::-1]
        df  = pd.DataFrame([{
            "#":       c["n"], "Ora": c["ts"],
            "Vera":    c["true_s"], "Predetta": c["pred_s"],
            "Conf.":   f"{c['conf']:.1%}", "Categoria": c["cat"],
            "ms":      f"{c['ms']:.1f}", "CPU%": f"{c['cpu']:.0f}",
            "T°C":     f"{c['temp']:.0f}", "Hum%": f"{c['hum']:.0f}",
            "Soil%":   f"{c['soil']:.0f}", "Lux": f"{c['lux']:.0f}",
            "Esito":   "Corretto" if c["ok"] else "Errato",
            "Irr":     "ON" if c["irr"]   else "—",
            "Vent":    "ON" if c["vent"]  else "—",
            "Alrm":    "ON" if c["alarm"] else "—",
        } for c in rec])
        st.dataframe(df, width="stretch",
                     height=min(360, 38+len(df)*35), hide_index=True)

# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    config = _cfg()

    for k, v in [("running",False),("cycles",[]),("init",False),("err",None),
                 ("interval",1.0),("max_cyc",50)]:
        if k not in st.session_state:
            st.session_state[k] = v

    _sidebar(config)
    _main(config)

    ss = st.session_state
    if ss.running:
        max_c = ss.get("max_cyc", 0)
        if max_c > 0 and len(ss.cycles) >= max_c:
            ss.running = False
            st.rerun()
        else:
            _cycle()
            ivl = ss.get("interval", 1.0)
            if ivl > 0:
                time.sleep(ivl)
            st.rerun()

if __name__ == "__main__":
    main()
