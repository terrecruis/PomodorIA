"""
orchestrator/main_loop.py — Orchestratore (Ciclo sense → think → act → log)

Coordina tutti i componenti del sistema in un ciclo continuo:

    1. SENSE  → VirtualCamera.capture() + EnvironmentSensorSimulator.read()
    2. THINK  → InferenceEngine.predict() + DecisionAgent.decide()
    3. ACT    → ActuatorBank applica le azioni decise
    4. LOG    → PoC Logger registra ogni ciclo su CSV

Il main loop rispetta l'architettura IoTWF a 7 livelli:
    Livello 1 (Physical/Sensors) → VirtualCamera, EnvironmentSimulator
    Livello 3 (Edge Computing)   → InferenceEngine (CNN compressa su CPU)
    Livello 6 (Application)     → DecisionAgent (regole PEAS)
    Livello 4 (Data Accumulation)→ Logger CSV/SQLite

Modalità:
    demo_mode=True  → cicli veloci, stampa dettagliata a console
    demo_mode=False → cicli ogni N secondi, log silenzioso

Utilizzo:
    python orchestrator/main_loop.py
    python orchestrator/main_loop.py --cycles 50 --interval 2
    python orchestrator/main_loop.py --class Tomato___Early_blight --cycles 20
"""

import sys
import os
import time
import csv
import logging
import argparse
import signal
from datetime import datetime
from pathlib import Path

import yaml
import psutil

# ── Path setup ───────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sensors.virtual_camera import VirtualCameraSensor
from sensors.environment_simulator import EnvironmentSensorSimulator
from edge.inference_engine import InferenceEngine
from agent.decision_agent import DecisionAgent
from actuators.actuators import ActuatorBank


# ══════════════════════════════════════════════════════════════
# Logger strutturato su CSV
# ══════════════════════════════════════════════════════════════
class PoCLogger:
    """
    Registra ogni ciclo sense-think-act su file CSV.

    Ogni riga corrisponde a un ciclo e contiene:
        - timestamp, cycle_idx
        - image_path, true_label
        - temperatura, umidità, soil_moisture, luce
        - predicted_label, confidence, inference_ms, ram_mb
        - disease_category, consecutive_alerts
        - azioni attuatori (boolean)
        - is_correct (predizione == vera classe)
    """

    FIELDNAMES = [
        "timestamp", "cycle_idx",
        "image_path", "true_label",
        "temperature_c", "humidity_pct", "soil_moisture_pct", "light_lux",
        "predicted_label", "confidence", "inference_time_ms", "ram_used_mb",
        "cpu_percent",
        "model_mode",
        "disease_category", "consecutive_alerts",
        "irrigation_active", "ventilation_active", "alarm_active",
        "notifications_sent_total",
        "is_correct",
        "reasoning_summary",
    ]

    def __init__(self, log_dir: str):
        """
        Args:
            log_dir: cartella dove salvare i file di log
        """
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"run_{timestamp_str}.csv")

        self._file = open(self.log_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self._file.flush()

        self.rows_written = 0
        logging.info(f"PoCLogger: log su '{self.log_path}'")
        print(f"📄 Log CSV: {self.log_path}")

    def log(self, cycle_idx: int, capture, env, inference, decision,
             actuator_status: dict, cpu_percent: float = 0.0) -> None:
        """
        Scrive una riga nel CSV per il ciclo corrente.

        Args:
            cycle_idx:      indice del ciclo (0-based)
            capture:        CameraCapture dal VirtualCamera
            env:            EnvironmentReading dal simulatore
            inference:      InferenceResult dall'InferenceEngine
            decision:       AgentDecision dall'agente
            actuator_status: dict con stato degli attuatori (da ActuatorBank.get_status())
        """
        is_correct = (capture.true_label == inference.predicted_label)
        reasoning_short = " | ".join(decision.reasoning[:2])  # max 2 righe

        row = {
            "timestamp":             datetime.now().isoformat(),
            "cycle_idx":             cycle_idx,
            "image_path":            capture.image_path,
            "true_label":            capture.true_label,
            "temperature_c":         round(env.temperature_c, 2),
            "humidity_pct":          round(env.humidity_pct, 2),
            "soil_moisture_pct":     round(env.soil_moisture_pct, 2),
            "light_lux":             round(env.light_lux, 1),
            "predicted_label":       inference.predicted_label,
            "confidence":            round(inference.confidence, 4),
            "inference_time_ms":     round(inference.inference_time_ms, 2),
            "ram_used_mb":           round(inference.ram_used_mb, 1),
            "cpu_percent":           round(cpu_percent, 1),
            "model_mode":            inference.model_mode,
            "disease_category":      decision.disease_category,
            "consecutive_alerts":    decision.consecutive_alerts,
            "irrigation_active":     int(actuator_status.get("irrigation_active", False)),
            "ventilation_active":    int(actuator_status.get("ventilation_active", False)),
            "alarm_active":          int(actuator_status.get("alarm_active", False)),
            "notifications_sent_total": actuator_status.get("notifications_sent", 0),
            "is_correct":            int(is_correct),
            "reasoning_summary":     reasoning_short,
        }
        self._writer.writerow(row)
        self._file.flush()
        self.rows_written += 1

    def close(self) -> None:
        """Chiude il file di log."""
        if not self._file.closed:
            self._file.close()
        print(f"📄 Log chiuso: {self.rows_written} righe scritte → {self.log_path}")

    def __del__(self):
        self.close()


# ══════════════════════════════════════════════════════════════
# Formattazione output console
# ══════════════════════════════════════════════════════════════
def print_cycle_header(cycle_idx: int, total: int) -> None:
    """Stampa l'intestazione di un ciclo."""
    bar = "═" * 60
    print(f"\n{bar}")
    print(f"  CICLO {cycle_idx + 1}/{total if total > 0 else '∞'} "
          f"— {datetime.now().strftime('%H:%M:%S')}")
    print(bar)


def print_cycle_summary(capture, env, inference, decision) -> None:
    """Stampa il riepilogo di un ciclo in modo compatto e leggibile."""
    ok_icon = "✅" if capture.true_label == inference.predicted_label else "❌"

    print(f"\n📷 Immagine: {os.path.basename(capture.image_path)}")
    print(f"   Vera:      {capture.true_label}")
    print(f"   Predetta:  {inference.predicted_label} "
          f"(conf: {inference.confidence:.1%}) {ok_icon}")
    print(f"   Latenza:   {inference.inference_time_ms:.1f}ms | "
          f"RAM: {inference.ram_used_mb:.0f}MB | "
          f"Mode: {inference.model_mode}")

    print(f"\n🌡️  Ambiente: "
          f"T={env.temperature_c:.1f}°C | "
          f"Hum={env.humidity_pct:.0f}% | "
          f"Soil={env.soil_moisture_pct:.0f}% | "
          f"Lux={env.light_lux:.0f}")

    if decision.reasoning:
        print(f"\n🤖 Ragionamento agente:")
        for r in decision.reasoning:
            print(f"   {r}")

    if decision.actions:
        active_actions = [a for a in decision.actions if a.action != "none"]
        if active_actions:
            print(f"\n⚙️  Azioni eseguite:")
            for a in active_actions:
                print(f"   {a}")


# ══════════════════════════════════════════════════════════════
# Statistiche finali
# ══════════════════════════════════════════════════════════════
def print_final_stats(
    total_cycles: int,
    correct: int,
    total_ms: list[float],
    agent_stats: dict,
    log_path: str,
) -> None:
    """Stampa le statistiche finali della sessione."""
    avg_ms = sum(total_ms) / len(total_ms) if total_ms else 0
    accuracy = correct / total_cycles if total_cycles > 0 else 0

    print("\n" + "═" * 60)
    print("  STATISTICHE FINALI SESSIONE")
    print("═" * 60)
    print(f"  Cicli totali:         {total_cycles}")
    print(f"  Predizioni corrette:  {correct}/{total_cycles} ({accuracy:.1%})")
    print(f"  Latenza media CNN:    {avg_ms:.1f} ms")
    print(f"  Latenza min/max:      {min(total_ms):.1f}/{max(total_ms):.1f} ms")
    print(f"  Consecutive disease:  {agent_stats['consecutive_disease_count']}")
    print(f"  Behavior accuracy:    {agent_stats['behavior_accuracy']:.1%}")
    print(f"  Log CSV salvato in:   {log_path}")
    print("═" * 60 + "\n")


# ══════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════
class Orchestrator:
    """
    Coordina il ciclo sense → think → act → log.

    Parametri configurabili da config.yaml (sezione "orchestrator"):
        cycle_interval_sec: secondi di pausa tra cicli
        max_cycles:         0 = infinito
        demo_mode:          True = output dettagliato
    """

    def __init__(self, config: dict, forced_class: str | None = None):
        """
        Args:
            config:       dizionario di configurazione (config.yaml)
            forced_class: se impostato, la VirtualCamera pesca sempre
                          da questa classe (utile per demo mirate)
        """
        self.config = config
        cfg_orch = config.get("orchestrator", {})
        self.cycle_interval = cfg_orch.get("cycle_interval_sec", 5)
        self.max_cycles     = cfg_orch.get("max_cycles", 0)
        self.demo_mode      = cfg_orch.get("demo_mode", True)

        print("\n" + "═" * 60)
        print("  PomodorIA — Serra Domotica con Edge AI")
        print("  Ciclo sense → think → act → log")
        print("═" * 60)

        # ── Istanzia tutti i componenti ──────────────────────
        print("\n🔧 Inizializzazione componenti...")

        # Se forced_class è specificato, configura il bias nel config
        if forced_class:
            config.setdefault("virtual_camera", {})["bias_class"] = forced_class
            print(f"   Demo mode: classe forzata → '{forced_class}'")

        self.camera   = VirtualCameraSensor(config)
        self.env_sim  = EnvironmentSensorSimulator(config)
        self.engine   = InferenceEngine(config)
        self.actuators = ActuatorBank()
        self.agent    = DecisionAgent(config, self.actuators)
        self.poc_logger = PoCLogger(config["paths"]["log_dir"])

        # ── Gestione segnale CTRL+C ───────────────────────────
        self._running = True
        signal.signal(signal.SIGINT, self._handle_interrupt)

        print(f"\n✅ Sistema pronto | "
              f"max_cycles={self.max_cycles or '∞'} | "
              f"interval={self.cycle_interval}s | "
              f"demo={self.demo_mode}")

    def _handle_interrupt(self, signum, frame) -> None:
        """Intercetta CTRL+C per un arresto pulito."""
        print("\n\n⛔  Interruzione ricevuta — arresto in corso...")
        self._running = False

    def run(self) -> None:
        """
        Avvia il ciclo sense → think → act → log.

        Il ciclo continua finché:
        - max_cycles è raggiunto (se > 0)
        - l'utente preme CTRL+C
        """
        cycle_idx = 0
        correct   = 0
        latencies: list[float] = []

        total = self.max_cycles if self.max_cycles > 0 else 0

        while self._running:
            if self.max_cycles > 0 and cycle_idx >= self.max_cycles:
                break

            if self.demo_mode:
                print_cycle_header(cycle_idx, total)

            t_cycle_start = time.perf_counter()

            # ══════════════════════════════════════════════════
            # 1. SENSE
            # ══════════════════════════════════════════════════
            capture = self.camera.capture()
            env     = self.env_sim.read(disease_label=capture.true_label)

            # ══════════════════════════════════════════════════
            # 2. THINK (inferenza + decisione)
            # ══════════════════════════════════════════════════
            inference = self.engine.predict(capture.image_tensor)
            decision  = self.agent.decide(
                inference=inference,
                env=env,
                true_label=capture.true_label,
            )

            # ══════════════════════════════════════════════════
            # 3. ACT (già eseguito dentro decide())
            # ══════════════════════════════════════════════════
            actuator_status = self.actuators.get_status()
            cpu_pct = psutil.cpu_percent(interval=None)  # non-blocking

            # ══════════════════════════════════════════════════
            # 4. LOG
            # ══════════════════════════════════════════════════
            self.poc_logger.log(
                cycle_idx=cycle_idx,
                capture=capture,
                env=env,
                inference=inference,
                decision=decision,
                actuator_status=actuator_status,
                cpu_percent=cpu_pct,
            )

            # ── Aggiorna statistiche ─────────────────────────
            if capture.true_label == inference.predicted_label:
                correct += 1
            latencies.append(inference.inference_time_ms)

            # ── Output console ────────────────────────────────
            if self.demo_mode:
                print_cycle_summary(capture, env, inference, decision)
                t_cycle = (time.perf_counter() - t_cycle_start) * 1000
                print(f"\n⏱️  Ciclo completato in {t_cycle:.0f}ms | "
                      f"Accuracy live: {correct/(cycle_idx+1):.1%} | "
                      f"CPU: {cpu_pct:.0f}% | "
                      f"Attuatori: {self.actuators}")
            else:
                # Output minimo in modalità non-demo
                ok = "✅" if capture.true_label == inference.predicted_label else "❌"
                print(
                    f"Ciclo {cycle_idx+1:4d} {ok} | "
                    f"{inference.predicted_label:<45} "
                    f"conf={inference.confidence:.2f} | "
                    f"{inference.inference_time_ms:.1f}ms"
                )

            cycle_idx += 1

            # ── Pausa tra cicli ───────────────────────────────
            if self._running and (self.max_cycles == 0 or cycle_idx < self.max_cycles):
                if self.demo_mode:
                    print(f"\n⏳ Prossimo ciclo tra {self.cycle_interval}s "
                          "(CTRL+C per fermare)...")
                time.sleep(self.cycle_interval)

        # ── Fine sessione ─────────────────────────────────────
        self.actuators.deactivate_all("fine sessione")
        self.poc_logger.close()
        print_final_stats(
            total_cycles=cycle_idx,
            correct=correct,
            total_ms=latencies,
            agent_stats=self.agent.get_stats(),
            log_path=self.poc_logger.log_path,
        )


# ══════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════
def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = os.path.join(ROOT, "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="PomodorIA — Orchestratore ciclo sense-think-act-log"
    )
    parser.add_argument(
        "--cycles", type=int, default=None,
        help="Numero massimo di cicli (0 o assente = infinito)"
    )
    parser.add_argument(
        "--interval", type=float, default=None,
        help="Secondi tra un ciclo e l'altro (default: da config.yaml)"
    )
    parser.add_argument(
        "--class", dest="forced_class", type=str, default=None,
        help="Forza la VirtualCamera a pescare dalla classe specificata "
             "(es. 'Tomato___Early_blight') per demo mirate"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Disabilita la modalità demo (output minimo)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Percorso al file di configurazione (default: config.yaml)"
    )
    parser.add_argument(
        "--mode", choices=["full_precision", "optimized"], default=None,
        help="Forza la modalità del modello (default: da config.yaml)"
    )
    parser.add_argument(
        "--variant", choices=["auto", "onnx", "pruned_quantized", "pruned"], default=None,
        help="Se --mode=optimized, forza quale file compresso usare "
             "(default: da config.yaml, 'auto' se non specificato)"
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)

    # Override config da argomenti CLI
    if args.cycles is not None:
        config.setdefault("orchestrator", {})["max_cycles"] = args.cycles
    if args.interval is not None:
        config.setdefault("orchestrator", {})["cycle_interval_sec"] = args.interval
    if args.quiet:
        config.setdefault("orchestrator", {})["demo_mode"] = False
    if args.mode is not None:
        config.setdefault("model", {})["mode"] = args.mode
    if args.variant is not None:
        config.setdefault("model", {})["optimized_variant"] = args.variant

    orchestrator = Orchestrator(config, forced_class=args.forced_class)
    orchestrator.run()


if __name__ == "__main__":
    main()
