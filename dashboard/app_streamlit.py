"""
dashboard/app_streamlit.py — PomodorIA · Dashboard di Monitoraggio

Interfaccia Streamlit che esegue lo stesso ciclo sense-think-act-log
dell'orchestratore da riga di comando (orchestrator/main_loop.py), un
ciclo per ogni rerun, mostrando in tempo reale diagnosi, ambiente,
attuatori e metriche aggregate.

Avvio:
    streamlit run dashboard/app_streamlit.py
"""
import os
import sys
import time
from collections import Counter
from datetime import datetime
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd
import plotly.graph_objects as go
import psutil
import streamlit as st
import torch
import yaml
from PIL import Image

# ══════════════════════════════════════════════════════════════
# Costanti
# ══════════════════════════════════════════════════════════════
CLASS_LABELS = {
    "Tomato___Bacterial_spot": "Bacterial Spot",
    "Tomato___Early_blight": "Early Blight",
    "Tomato___Late_blight": "Late Blight",
    "Tomato___Leaf_Mold": "Leaf Mold",
    "Tomato___Septoria_leaf_spot": "Septoria",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Spider Mites",
    "Tomato___Target_Spot": "Target Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "TYLCV",
    "Tomato___Tomato_mosaic_virus": "Mosaic Virus",
    "Tomato___healthy": "Healthy",
}
CLASS_COLORS = {
    "Healthy": "#00d4aa", "Early Blight": "#fd9644", "Late Blight": "#e17055",
    "Leaf Mold": "#fdcb6e", "Septoria": "#a29bfe", "Spider Mites": "#ff9f43",
    "Target Spot": "#ff7675", "TYLCV": "#ee5a24", "Mosaic Virus": "#d63031",
    "Bacterial Spot": "#7c5cfc",
}
CATEGORY_COLORS = {
    "healthy": "#00d4aa", "fungal": "#fd9644", "viral": "#ee5a24",
    "bacterial": "#7c5cfc", "pest": "#f5a623", "unknown": "#8a8f98",
}
ACCENT = dict(teal="#00d4aa", purple="#7c5cfc", amber="#f5a623", red="#ee5a24")

# Layout comune per i grafici Plotly: tema scuro, sfondo trasparente
# (si fonde con il tema dell'app impostato in .streamlit/config.toml)
DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c9ccd1"),
)
MODE_LABELS = {
    "full_precision": "Full Precision (float32)",
    "optimized_onnx": "Optimized · ONNX INT8",
    "optimized_pruned_quantized": "Optimized · Pruned+Quant INT8",
    "optimized_pruned": "Optimized · Pruned (float32)",
}
VARIANT_LABELS = {
    "auto": "Auto (priorità: ONNX > pruned+quant > pruned)",
    "onnx": "ONNX quantizzato (~stessa accuracy della baseline)",
    "pruned_quantized": "Pruned + Quantizzato INT8 (più compresso, accuracy più bassa)",
    "pruned": "Solo Pruned (float32)",
}
BIAS_OPTIONS = ["— Random —"] + list(CLASS_LABELS.keys())

st.set_page_config(page_title="PomodorIA · Serra Domotica", page_icon="", layout="wide")

# Piccolo ritocco estetico: nasconde il chrome di Streamlit
# (il tema scuro/colori vengono da .streamlit/config.toml).
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# Set minimo di icone SVG (stroke-based, stile Heroicons): solo quelle
# effettivamente usate in dashboard. Il colore dello stroke è passato come
# parametro, così ogni icona si intona sempre con l'accento della card che
# la contiene (a differenza delle emoji, che hanno un colore fisso).
_ICONS = {
    "settings": "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z",
    "gamepad": "M6 12h4m-2-2v4m7-1h.01M17 9h.01M2 15V9a4 4 0 0 1 4-4h12a4 4 0 0 1 4 4v6a4 4 0 0 1-4 4c-1 0-1.5-.5-2-1l-1-1H9l-1 1c-.5.5-1 1-2 1a4 4 0 0 1-4-4z",
    "server": "M2 2h20a2 2 0 0 1 2 2v4a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2zm0 12h20a2 2 0 0 1 2 2v4a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-4a2 2 0 0 1 2-2zm4 2h.01M6 6h.01",
    "microscope": "M6 18h8M3 22h18M14 22a7 7 0 1 0 0-14h-1M9 14h2M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2zm3-6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3",
    "cloud": "M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9z",
    "wrench": "M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z",
    "cpu": "M9 1v3m6-3v3M9 20v3m6-3v3M1 9h3m-3 6h3m20-6h-3m3 6h-3M4 4h16a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2zm5 5h6v6H9z",
    "bar-chart": "M18 20V10M12 20V4M6 20v-6",
    "list": "M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01",
    "refresh": "M23 4v6h-6M1 20v-6h6m16.73-6A10 10 0 0 0 3.27 10.27M.27 16A10 10 0 0 0 20.73 13.73",
    "target": "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20zm0-14a4 4 0 1 0 0 8 4 4 0 0 0 0-8zm0 2a2 2 0 1 1 0 4 2 2 0 0 1 0-4z",
    "zap": "M13 2 3 14h9l-1 8 10-12h-9z",
    "alert-tri": "M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0zM12 9v4m0 4h.01",
    "thermometer": "M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z",
    "droplet": "M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z",
    "sun": "M12 7a5 5 0 1 0 0 10A5 5 0 0 0 12 7zm0-5v2m0 14v2M4.22 4.22l1.42 1.42m12.72 12.72 1.42 1.42M2 12h2m14 0h4M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42",
    "leaf": "M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10zm-9 1c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12",
    "wind": "M9.59 4.59A2 2 0 1 1 11 8H2m10.59 11.41A2 2 0 1 0 14 16H2m15.73-8.27A2.5 2.5 0 1 1 19.5 12H2",
    "database": "M12 2c4.97 0 9 1.34 9 3v14c0 1.66-4.03 3-9 3S3 20.66 3 19V5c0-1.66 4.03-3 9-3zm9 5c0 1.66-4.03 3-9 3S3 12.66 3 12m18 4c0 1.66-4.03 3-9 3S3 16.66 3 16",
}


def icon_svg(name: str, size: int = 20, color: str = "currentColor") -> str:
    """Icona SVG in linea, colorata secondo il parametro `color` — a
    differenza delle emoji, il colore si adatta sempre all'accento usato."""
    path = _ICONS.get(name, "")
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="1.8" stroke-linecap="round" '
        f'stroke-linejoin="round"><path d="{path}"/></svg>'
    )


def metric_card(icon: str, label: str, value: str, color: str = ACCENT["teal"]) -> None:
    """Card colorata con icona, usata al posto di st.metric per evitare il
    troncamento del testo ("...") che Streamlit applica nei suoi metric
    nativi quando lo spazio in colonna è ristretto, e per un impatto
    visivo più marcato (icona + colore d'accento)."""
    st.markdown(
        f'<div style="background:rgba(255,255,255,0.03);border:1px solid {color}40;'
        f'border-radius:12px;padding:14px 10px;text-align:center">'
        f'<div style="line-height:1">{icon_svg(icon, 24, color)}</div>'
        f'<div style="font-size:1.35rem;font-weight:700;color:{color};'
        f'margin-top:6px;white-space:nowrap">{value}</div>'
        f'<div style="font-size:0.68rem;color:rgba(255,255,255,0.5);'
        f'text-transform:uppercase;letter-spacing:0.05em;margin-top:2px">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_header(icon: str, label: str, color: str = ACCENT["teal"]) -> None:
    """Titolo di sezione con icona SVG colorata, al posto di st.subheader
    con emoji (che avrebbero un colore fisso non intonato al tema)."""
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;margin:8px 0 4px">'
        f'{icon_svg(icon, 20, color)}'
        f'<span style="font-size:1.15rem;font-weight:600;color:#e8e8e8">{label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
# Configurazione e inizializzazione componenti
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_config() -> dict:
    with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def init_components(config: dict, mode: str, bias: Optional[str], variant: str) -> None:
    """Istanzia i 5 componenti di dominio e li salva in session_state."""
    import copy

    from actuators.actuators import ActuatorBank
    from agent.decision_agent import DecisionAgent
    from edge.inference_engine import InferenceEngine
    from sensors.environment_simulator import EnvironmentSensorSimulator
    from sensors.virtual_camera import VirtualCameraSensor

    cfg = copy.deepcopy(config)
    cfg["model"]["mode"] = mode
    cfg["model"]["optimized_variant"] = variant
    cfg.setdefault("virtual_camera", {})["bias_class"] = bias

    ss = st.session_state
    ss.camera = VirtualCameraSensor(cfg)
    ss.env_sim = EnvironmentSensorSimulator(cfg)
    ss.engine = InferenceEngine(cfg)
    ss.actuators = ActuatorBank()
    ss.agent = DecisionAgent(cfg, ss.actuators)
    ss.cycles = []
    ss.init = True
    ss.cfg_mode, ss.cfg_bias, ss.cfg_variant = mode, bias, variant


def run_cycle() -> None:
    """Esegue un ciclo sense-think-act-log e appende il risultato a ss.cycles."""
    ss = st.session_state
    try:
        capture = ss.camera.capture()
        env = ss.env_sim.read(disease_label=capture.true_label)
        inference = ss.engine.predict(capture.image_tensor)
        decision = ss.agent.decide(inference, env, capture.true_label)

        ss.cycles.append({
            "n": len(ss.cycles),
            "ts": datetime.now().strftime("%H:%M:%S"),
            "true": capture.true_label,
            "true_label": CLASS_LABELS.get(capture.true_label, capture.true_label),
            "pred": inference.predicted_label,
            "pred_label": CLASS_LABELS.get(inference.predicted_label, inference.predicted_label),
            "confidence": inference.confidence,
            "probabilities": inference.all_probabilities,
            "latency_ms": inference.inference_time_ms,
            "ram_mb": psutil.Process().memory_info().rss / 1024 ** 2,
            "cpu_pct": psutil.cpu_percent(interval=None),
            "model_mode": inference.model_mode,
            "temperature": env.temperature_c,
            "humidity": env.humidity_pct,
            "soil_moisture": env.soil_moisture_pct,
            "light": env.light_lux,
            "category": decision.disease_category,
            "consecutive_alerts": decision.consecutive_alerts,
            "irrigation_on": ss.actuators.irrigation.is_active,
            "ventilation_on": ss.actuators.ventilation.is_active,
            "alarm_on": ss.actuators.alarm.is_active,
            "notifications_sent": ss.actuators.notification.total_notifications,
            "correct": capture.true_label == inference.predicted_label,
            "reasoning": decision.reasoning,
            "image_path": capture.image_path,
        })
        ss.error = None
    except Exception as exc:
        ss.error = str(exc)
        ss.running = False


# ══════════════════════════════════════════════════════════════
# Grafici
# ══════════════════════════════════════════════════════════════
def confidence_chart(probabilities, classes, predicted_label) -> go.Figure:
    """Barre orizzontali con le probabilità softmax delle 10 classi."""
    labels = [CLASS_LABELS.get(c, c) for c in classes]
    rows = sorted(zip(probabilities, labels, classes), reverse=True)
    probs, short_labels, class_names = zip(*rows) if rows else ((), (), ())
    colors = [ACCENT["teal"] if c == predicted_label else "rgba(255,255,255,0.12)" for c in class_names]

    fig = go.Figure(go.Bar(
        x=list(probs), y=list(short_labels), orientation="h",
        marker_color=colors,
        text=[f"{p:.1%}" for p in probs], textposition="outside",
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        height=320, margin=dict(l=4, r=40, t=10, b=10),
        xaxis=dict(range=[0, 1.15], visible=False),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    return fig


def accuracy_timeline_chart(cycles: list) -> go.Figure:
    """Accuratezza cumulativa e latenza nel tempo, con marcatori sugli errori."""
    fig = go.Figure()
    if cycles:
        idx = [c["n"] for c in cycles]
        cum_acc = [sum(1 for c in cycles[:i + 1] if c["correct"]) / (i + 1) * 100
                   for i in range(len(cycles))]
        fig.add_trace(go.Scatter(x=idx, y=cum_acc, name="Accuracy %",
                                  line=dict(color=ACCENT["teal"], width=2),
                                  fill="tozeroy", fillcolor="rgba(0,212,170,0.07)"))
        fig.add_trace(go.Scatter(x=idx, y=[c["latency_ms"] for c in cycles], name="Latenza (ms)",
                                  yaxis="y2", line=dict(color=ACCENT["purple"], width=1.5, dash="dot")))
        errors = [(c["n"], cum_acc[c["n"]]) for c in cycles if not c["correct"]]
        if errors:
            ex, ey = zip(*errors)
            fig.add_trace(go.Scatter(x=ex, y=ey, mode="markers", name="Errore",
                                      marker=dict(color=ACCENT["red"], size=8, symbol="x")))
    fig.update_layout(
        **DARK_LAYOUT,
        title="Accuracy cumulativa e latenza", height=280,
        margin=dict(l=8, r=8, t=40, b=8),
        xaxis=dict(title="Ciclo"), yaxis=dict(title="Accuracy %", range=[0, 105]),
        yaxis2=dict(title="ms", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.15),
    )
    return fig


def prediction_distribution_chart(cycles: list) -> go.Figure:
    """Torta con la distribuzione delle classi predette."""
    fig = go.Figure()
    if cycles:
        counts = Counter(c["pred_label"] for c in cycles)
        fig.add_trace(go.Pie(
            labels=list(counts.keys()), values=list(counts.values()), hole=0.55,
            marker=dict(colors=[CLASS_COLORS.get(l, "#888") for l in counts],
                        line=dict(color="#0e1420", width=2)),
        ))
    fig.update_layout(**DARK_LAYOUT, title="Distribuzione predizioni", height=280,
                       margin=dict(l=8, r=8, t=40, b=8))
    return fig


def benchmark_comparison_chart() -> Optional[go.Figure]:
    """Confronto dimensione/latenza fra le varianti compresse (da benchmarks/benchmark_results.csv)."""
    path = os.path.join(ROOT, "benchmarks", "benchmark_results.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None

    sizes = pd.to_numeric(df["file_size_mb"], errors="coerce").fillna(0)
    latencies = pd.to_numeric(
        df["avg_inference_ms"].astype(str).str.replace("ms", ""), errors="coerce"
    ).fillna(0)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Dimensione (MB)", x=df["variant"], y=sizes, marker_color=ACCENT["teal"]))
    fig.add_trace(go.Scatter(name="Latenza (ms)", x=df["variant"], y=latencies,
                              yaxis="y2", mode="markers+lines", marker_color=ACCENT["amber"]))
    fig.update_layout(
        **DARK_LAYOUT,
        title="Confronto varianti compresse: dimensione vs latenza", height=300,
        margin=dict(l=8, r=8, t=40, b=8),
        xaxis=dict(tickangle=-15),
        yaxis=dict(title="MB"), yaxis2=dict(title="ms", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.15),
    )
    return fig


# ══════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════
def render_sidebar(config: dict) -> tuple[float, int]:
    ss = st.session_state
    with st.sidebar:
        st.title("PomodorIA")
        st.caption("Serra Domotica · Edge AI")

        section_header("settings", "Configurazione")
        mode = st.selectbox(
            "Modello", ["full_precision", "optimized"],
            format_func=lambda x: "Full Precision (float32)" if x == "full_precision" else "Ottimizzato (INT8)",
            key="sel_mode",
        )
        variant = "auto"
        if mode == "optimized":
            variant = st.selectbox(
                "Variante ottimizzata", list(VARIANT_LABELS),
                format_func=lambda v: VARIANT_LABELS[v], key="sel_variant",
                help="Forza quale file in models/optimized/ usare, invece della scelta automatica.",
            )
        bias_choice = st.selectbox("Classe forzata", BIAS_OPTIONS, key="sel_bias")
        bias = None if bias_choice.startswith("—") else bias_choice
        interval = st.slider("Intervallo cicli (sec)", 0.0, 10.0, 1.0, 0.5, key="sel_interval")
        max_cycles = st.number_input("Max cicli (0 = infinito)", 0, 10000, 50, key="sel_max_cycles")

        st.divider()
        section_header("gamepad", "Controlli")
        running = ss.get("running", False)
        if not running:
            if st.button("▶ Avvia simulazione", use_container_width=True):
                config_changed = (
                    not ss.get("init")
                    or ss.get("cfg_mode") != mode
                    or ss.get("cfg_bias") != bias
                    or ss.get("cfg_variant") != variant
                )
                if config_changed:
                    with st.spinner("Inizializzazione..."):
                        init_components(config, mode, bias, variant)
                ss.running, ss.interval, ss.max_cycles = True, interval, max_cycles
                st.rerun()
        elif st.button("Ferma", use_container_width=True):
            ss.running = False
            st.rerun()

        if ss.get("init") and st.button("Reset sessione", use_container_width=True):
            ss.running, ss.cycles = False, []
            ss.actuators.deactivate_all("reset")
            st.rerun()

        if ss.get("error"):
            st.error(f"Errore: {ss.error}")

        st.divider()
        section_header("server", "Simulazione Raspberry Pi", ACCENT["purple"])
        engine = ss.get("engine")
        sc1, sc2 = st.columns(2)
        with sc1:
            metric_card("cpu", "CPU", f"{psutil.cpu_percent(interval=None):.0f} %", ACCENT["purple"])
        with sc2:
            metric_card("database", "RAM", f"{psutil.Process().memory_info().rss / 1024**2:.0f} MB", ACCENT["teal"])
        st.caption(f"Thread: {torch.get_num_threads()} / {psutil.cpu_count()} core · CPU only")
        #if engine:
        #    st.caption(f"Modello caricato: **{MODE_LABELS.get(engine.loaded_variant, engine.loaded_variant)}** "
        #               f"({engine.get_model_size_mb():.1f} MB)")

    return interval, max_cycles


# ══════════════════════════════════════════════════════════════
# Contenuto principale
# ══════════════════════════════════════════════════════════════
def render_diagnosis(latest: Optional[dict], classes: list) -> None:
    section_header("microscope", "Diagnosi pianta")
    if not latest:
        st.info("Premi **Avvia simulazione** per iniziare.")
        return

    col_img, col_info = st.columns([2, 3])
    with col_img:
        if os.path.exists(latest["image_path"]):
            st.image(Image.open(latest["image_path"]).convert("RGB"), use_container_width=True)
    with col_info:
        cat_color = CATEGORY_COLORS.get(latest["category"], "#8a8f98")
        st.markdown(f"**Predizione CNN:** {latest['pred_label']}")
        st.markdown(
            f'<span style="background:{cat_color}22;color:{cat_color};'
            f'border:1px solid {cat_color}55;border-radius:12px;padding:3px 12px;'
            f'font-size:0.8rem;font-weight:600">{latest["category"].upper()}</span>',
            unsafe_allow_html=True,
        )
        metric_card("target", "Confidenza", f"{latest['confidence']:.1%}", cat_color)
        if latest["correct"]:
            st.success(f"Corretta — vera classe: {latest['true_label']}")
        else:
            st.error(f"Errata — vera classe: {latest['true_label']}")

    st.plotly_chart(confidence_chart(latest["probabilities"], classes, latest["pred"]),
                     use_container_width=True, config={"displayModeBar": False})


def render_environment(latest: Optional[dict]) -> None:
    section_header("cloud", "Sensori ambientali", ACCENT["purple"])
    if not latest:
        st.info("Nessun dato disponibile.")
        return
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("thermometer", "Temperatura", f"{latest['temperature']:.1f} °C", ACCENT["red"])
    with c2:
        metric_card("droplet", "Umidità aria", f"{latest['humidity']:.0f} %", ACCENT["purple"])
    with c3:
        metric_card("leaf", "Umidità suolo", f"{latest['soil_moisture']:.0f} %", ACCENT["teal"])
    with c4:
        metric_card("sun", "Luminosità", f"{latest['light']:.0f} lux", ACCENT["amber"])


def render_actuators(ss, latest: Optional[dict]) -> None:
    section_header("wrench", "Attuatori", ACCENT["amber"])
    if not ss.get("actuators"):
        st.info("—")
        return
    actuators = ss.actuators
    labels = [
        ("droplet", "Irrigazione", actuators.irrigation.is_active),
        ("wind", "Ventilazione", actuators.ventilation.is_active),
        ("alert-tri", "Allarme", actuators.alarm.is_active),
    ]
    cols = st.columns(len(labels))
    for col, (icon, name, active) in zip(cols, labels):
        with col:
            metric_card(icon, name, "ATTIVO" if active else "off",
                        ACCENT["teal"] if active else "#5a5f68")

    if latest and latest["reasoning"]:
        st.caption("Ragionamento dell'agente:")
        for line in latest["reasoning"][:4]:
            st.write(f"- {line}")


def render_system_monitor(latest: Optional[dict]) -> None:
    section_header("cpu", "System Monitor")
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("cpu", "CPU", f"{latest['cpu_pct']:.0f} %" if latest else "—", ACCENT["purple"])
    with c2:
        metric_card("database", "RAM", f"{latest['ram_mb']:.0f} MB" if latest else "—", ACCENT["teal"])
    with c3:
        metric_card("zap", "Latenza", f"{latest['latency_ms']:.1f} ms" if latest else "—", ACCENT["amber"])
    if latest:
        st.caption(f"Ultimo ciclo: {latest['ts']} · Modello: "
                   f"{MODE_LABELS.get(latest['model_mode'], latest['model_mode'])}")


def render_log_table(cycles: list) -> None:
    if not cycles:
        return
    section_header("list", f"Storico cicli (ultimi 30 di {len(cycles)})")
    recent = cycles[-30:][::-1]
    df = pd.DataFrame([{
        "#": c["n"], "Ora": c["ts"],
        "Vera": c["true_label"], "Predetta": c["pred_label"],
        "Conf.": f"{c['confidence']:.1%}", "Categoria": c["category"],
        "ms": f"{c['latency_ms']:.1f}",
        "T°C": f"{c['temperature']:.0f}", "Hum%": f"{c['humidity']:.0f}",
        "Soil%": f"{c['soil_moisture']:.0f}", "Lux": f"{c['light']:.0f}",
        "Esito": "Corretto" if c["correct"] else "Errato",
        "Irr": "ON" if c["irrigation_on"] else "—",
        "Vent": "ON" if c["ventilation_on"] else "—",
        "Alrm": "ON" if c["alarm_on"] else "—",
    } for c in recent])
    st.dataframe(df, use_container_width=True, hide_index=True,
                 height=min(360, 38 + len(df) * 35))


def render_main(config: dict) -> None:
    ss = st.session_state
    cycles = ss.get("cycles", [])
    latest = cycles[-1] if cycles else None
    n = len(cycles)

    st.title("PomodorIA — Serra Domotica con Edge AI")
    st.caption("Università Federico II · Corso di Internet of Things")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        metric_card("refresh", "Cicli totali", str(n), ACCENT["teal"])
    with k2:
        metric_card("target", "Accuracy live", f"{sum(c['correct'] for c in cycles) / n:.1%}" if n else "—", ACCENT["teal"])
    with k3:
        metric_card("zap", "Latenza media", f"{sum(c['latency_ms'] for c in cycles) / n:.1f} ms" if n else "—", ACCENT["purple"])
    with k4:
        consec = latest["consecutive_alerts"] if latest else 0
        alert_color = ACCENT["red"] if consec >= 3 else (ACCENT["amber"] if consec else "#8a8f98")
        metric_card("alert-tri", "Alert consecutivi", str(consec), alert_color)

    st.divider()
    col_left, col_right = st.columns([5, 7])
    with col_left:
        render_diagnosis(latest, config["model"]["classes"])
    with col_right:
        render_environment(latest)
        st.divider()
        render_actuators(ss, latest)

    st.divider()
    render_system_monitor(latest)

    st.divider()
    section_header("bar-chart", "Analytics")
    col_a, col_b = st.columns([6, 4])
    col_a.plotly_chart(accuracy_timeline_chart(cycles), use_container_width=True,
                        config={"displayModeBar": False})
    col_b.plotly_chart(prediction_distribution_chart(cycles), use_container_width=True,
                        config={"displayModeBar": False})

    with st.expander("Confronto varianti di compressione del modello"):
        chart = benchmark_comparison_chart()
        if chart:
            st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("File benchmarks/benchmark_results.csv non trovato. "
                    "Esegui 'python models/compress.py' per generarlo.")

    st.divider()
    render_log_table(cycles)


# ══════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════
def main() -> None:
    config = load_config()
    for key, default in [("running", False), ("cycles", []), ("init", False),
                          ("error", None), ("interval", 1.0), ("max_cycles", 50)]:
        st.session_state.setdefault(key, default)

    interval, max_cycles = render_sidebar(config)
    render_main(config)

    ss = st.session_state
    if ss.running:
        if 0 < ss.get("max_cycles", 0) <= len(ss.cycles):
            ss.running = False
            st.rerun()
        else:
            run_cycle()
            if ss.get("interval", 1.0) > 0:
                time.sleep(ss.interval)
            st.rerun()


if __name__ == "__main__":
    main()
