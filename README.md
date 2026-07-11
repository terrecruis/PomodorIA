
<div align="center">

# 🍅 PomodorIA
### Serra Domotica con Edge AI — Riconoscimento Patologie Fogliari del Pomodoro

*Proof of Concept software — simulazione locale del ciclo IoT sense → think → act, senza hardware reale*

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CNN-EE4C2C?logo=pytorch&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?logo=onnx&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-PoC%20Attivo-brightgreen)

**Autori:** Vellotti Gianmarco, Terrecuso Francesco
**Corso:** Operating Systems for mobile, cloud and IoT — Università degli Studi di Napoli Federico II
**Modello ML di partenza:** TomatoCNN (progetto P14, esame di Neural Networks)

</div>

## 1. Obiettivo del progetto

Realizzare un **Proof of Concept (PoC) interamente software** che simuli una serra
domotica intelligente per il pomodoro, capace di:

1. "Osservare" le foglie tramite una fotocamera **simulata** (che pesca immagini
   reali dal dataset PlantVillage-Tomato già usato per l'addestramento);
2. Classificare lo stato di salute della pianta con la **CNN già addestrata**
   (10 classi, incluso "healthy"), **compressa** per l'esecuzione su hardware
   a risorse limitate (Raspberry Pi);
3. Simulare sensori ambientali (temperatura, umidità, luminosità, umidità del
   suolo) e un **agente decisionale** che, combinando la diagnosi della CNN con
   il contesto ambientale, aziona **attuatori simulati** (irrigazione,
   ventilazione, allarme);
4. Far girare tutto questo **su un normale PC**, che nel PoC "si finge" un
   Raspberry Pi (niente hardware reale, niente sensori fisici).

Non si tratta quindi di un deployment reale, ma di una **simulazione end-to-end
del ciclo sense → think → act** tipico di un sistema IoT/Edge AI, usando dati
reali del dataset al posto degli stimoli fisici.

### 1.1 Punto di partenza: i risultati del progetto ML (P14)

Il PoC riusa direttamente il lavoro già fatto nell'esame di Neural Networks,
che ha confrontato una FCNN e una CNN sullo stesso task:

| | FCNN [2L-512-ReLU] | **CNN [64f-k3-3blk]** |
|---|---|---|
| Accuracy (holdout 80/20) | ~84% | **~96%** |
| Accuracy media (Stratified 5-Fold, Weighted CE) | — | **94.98%** (σ < 3.2%, range 93.32–96.52%) |
| Recall minimo per classe | 0.69 (Target Spot) | 0.90 |
| Punto debole | Non percepisce la struttura spaziale → confonde Early blight/Late blight/Bacterial spot | Classi rare (Tomato mosaic virus, 299 img) restano le più critiche |

La rete da portare in edge è quindi la **CNN**, nella configurazione migliore
individuata in Fase 1 e validata in Fase 2:

```python
# model.py — configurazione da usare per il PoC
model = TomatoCNN(
    n_filters=64,
    kernel_size=3,
    num_blocks=3,
    num_classes=10
)
model.load_state_dict(torch.load("best_cnn_64f_k3_3blk.pth"))
```

Pipeline di input da rispettare (identica al training, altrimenti la rete
non funziona in inferenza):

- resize a **64×64**, RGB (3×64×64 = 12.288 valori)
- normalizzazione ImageNet: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
- nessuna augmentation in inferenza (flip/rotazione solo in training)

Classi critiche da tenere d'occhio nel PoC (per scegliere immagini di test
"interessanti" da mostrare in demo): **Early blight** (confusa con Late
blight) e **Tomato mosaic virus** (classe più rara, recall più basso).

---

## 2. Architettura generale (mappata sul modello IoTWF a 7 livelli)

Riprendendo il modello **IoT World Forum** visto a lezione, il PoC copre così
i livelli:

| Livello IoTWF | Nel PoC reale (target) | Nel PoC implementato (software) |
|---|---|---|
| 1. Physical Devices & Controllers | Fotocamera, sensori DHT22/igrometro, pompa, ventola | **Moduli Python mock** (`VirtualCameraSensor`, `EnvironmentSensorSimulator`, `FakeGPIO`) |
| 2. Connectivity | Wi-Fi del Raspberry Pi | **Loopback locale** (comunicazione in-memory tra moduli Python via interfacce strutturate) |
| 3. Edge (Fog) Computing | Inferenza CNN quantizzata sul Raspberry Pi | Inferenza CNN quantizzata **su CPU** (con vincoli di risorse simulati: 1 core, solo CPU) |
| 4. Data Accumulation | Storage locale su SD card | **Log CSV strutturati** (`logs/run_*.csv`) |
| 5. Data Abstraction | Normalizzazione dei dati per il livello applicativo | **Classi e dataclass Python** (`SensorReading`, `InferenceResult`, `AgentDecision`) |
| 6. Application | Dashboard di monitoraggio | **Dashboard interattiva real-time** (Streamlit + Plotly in `dashboard/app_streamlit.py`) |
| 7. Collaboration & Processes | Notifiche all'agricoltore, integrazione gestionale | **Alert critici e notifiche** loggate ed esposte in UI (`NotificationActuator`) |

Questo mapping è utile anche per la relazione/presentazione del progetto,
perché lega esplicitamente le scelte implementative alla teoria del corso.

### 2.1 Schema a blocchi del PoC

```
┌───────────────────────────────────────────────────────────────────────┐
│                     "RASPBERRY PI VIRTUALE" (processo Python)         │
│                                                                        │
│  ┌────────────────┐    ┌──────────────────┐    ┌───────────────────┐ │
│  │  Sensor Layer   │    │   Edge AI Layer  │    │   Actuator Layer  │ │
│  │  (simulato)     │───▶│  (CNN compressa) │───▶│    (simulato)     │ │
│  │                 │    │                  │    │                   │ │
│  │ • Virtual Cam   │    │ • Preprocessing  │    │ • Irrigazione     │ │
│  │ • Temp/Umidità  │    │ • Inferenza      │    │ • Ventilazione    │ │
│  │ • Umidità suolo │    │ • Post-proc.     │    │ • Allarme/Alert   │ │
│  │ • Luminosità    │    │   (softmax→label)│    │ • Notifica log    │ │
│  └────────────────┘    └──────────────────┘    └───────────────────┘ │
│           │                      │                       ▲            │
│           └──────────────────────┼───────────────────────┘            │
│                                  ▼                                    │
│                     ┌─────────────────────────┐                      │
│                     │   Agente Decisionale     │                      │
│                     │   (regole PEAS)          │                      │
│                     └─────────────────────────┘                      │
│                                  │                                    │
│                                  ▼                                    │
│                     ┌─────────────────────────┐                      │
│                     │  Logging / Dashboard     │                      │
│                     └─────────────────────────┘                      │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 3. Componenti del PoC

### 3.1 Sensor Layer (simulato)

**Virtual Camera (`VirtualCameraSensor`)**
- Ad ogni ciclo, pesca in modo pseudo-casuale un'immagine dalla cartella del
  dataset PlantVillage-Tomato (riusando lo split di **test**, per non "barare"
  con immagini viste in training).
- Restituisce: immagine (tensor), classe reale (per calcolare accuratezza
  live nel PoC), timestamp.
- Parametro opzionale: `bias_class` per forzare la simulazione a pescare più
  spesso da una classe specifica (utile per demo mirate, es. simulare
  un'epidemia di Early blight).

**Sensori ambientali (`EnvironmentSensorSimulator`)**
- Genera in modo sintetico: temperatura (°C), umidità relativa (%), umidità
  del suolo (%), luminosità (lux).
- Per rendere la simulazione plausibile e "raccontabile" nella relazione, i
  valori non sono puramente casuali ma **correlati euristicamente alla
  patologia estratta dalla foto** (es. umidità alta → più probabile la
  patologia fungina Early/Late blight; questo riprende il concetto di **Data
  Fusion** visto a lezione: unire dati eterogenei — immagine + ambiente — in
  un'unica rappresentazione logica per la decisione).

```python
class SensorReading:
    timestamp: float
    image_path: str
    true_label: str          # per validazione nel PoC
    temperature_c: float
    humidity_pct: float
    soil_moisture_pct: float
    light_lux: float
```

### 3.2 Edge AI Layer

- Carica il modello **compresso** (vedi §4) all'avvio, una sola volta.
- Applica lo stesso preprocessing del training (resize 64×64 + normalizzazione
  ImageNet).
- Esegue l'inferenza e restituisce: classe predetta, confidenza (softmax),
  tempo di inferenza (ms) — utile per le metriche di performance edge.
- Gira in due modalità, selezionabili da config (`model.mode`), per
  confrontarle nella relazione:
  - `full_precision` → modello originale `.pth` (baseline)
  - `optimized` → modello compresso (target edge). In questa modalità,
    `model.optimized_variant` sceglie quale variante caricare:
    `onnx` (solo quantizzato), `pruned_quantized`, `pruned`, oppure `auto`
    (cascata: ONNX → pruned+quant → pruned → fallback a full_precision).

### 3.3 Agente Decisionale (PEAS)

Qui si applica direttamente il modello dell'**agente razionale** visto negli
appunti IoT: si definisce esplicitamente il PEAS del sistema.

| Componente PEAS | Definizione nel PoC |
|---|---|
| **P**erformance | Diagnosi corrette, tempo di reazione, risparmio idrico, riduzione falsi allarmi |
| **E**nvironment | Serra simulata (parzialmente osservabile, dinamica, continua per i sensori ambientali, discreta per le classi patologiche) |
| **A**ctuators | Pompa irrigazione, ventola, LED di allarme, notifica |
| **S**ensors | Fotocamera virtuale, sensori ambientali simulati |

L'agente è del tipo **reattivo basato su modello + basato su obiettivi**:
mantiene un piccolo stato interno (es. "N. rilevamenti Early blight
consecutivi") e decide le azioni in base a regole condizione→azione, per
esempio:

```
SE predizione == "healthy"                      → nessuna azione
SE predizione == malattia fungina (Early/Late blight)
   E umidità > soglia                            → attiva ventilazione + alert
SE predizione == malattia virale (es. TYLCV)      → alert "possibile vettore insetto"
SE soil_moisture < soglia                        → attiva irrigazione
SE confidenza_modello < soglia_minima            → alert "richiesta ispezione umana"
                                                    (human-in-the-loop, come da
                                                     Service Layer dello stack IoT)
```

### 3.4 Actuator Layer (simulato)

Ogni attuatore è una semplice classe con metodo `activate()/deactivate()` che
al posto di pilotare un pin GPIO reale stampa/logga l'azione:

```python
class SimulatedActuator:
    def activate(self, reason: str): ...
    def deactivate(self): ...
```

Per rendere il PoC più "vicino" a un reale deployment Raspberry Pi, si può
usare una libreria di mock delle GPIO (es. `RPi.GPIO` sostituito da un modulo
fittizio con la stessa interfaccia), così il codice è già pronto per essere
spostato su hardware reale senza riscritture.

### 3.5 Orchestratore (main loop)

Ciclo `sense → think → act → log`, eseguito a intervalli regolari (es. ogni
5–10 secondi simulati, accelerabile per demo):

```python
while True:
    reading = camera.capture()
    env = env_sensors.read()
    prediction = edge_model.infer(reading.image)
    action = agent.decide(prediction, env)
    actuators.apply(action)
    logger.log(reading, env, prediction, action)
    time.sleep(CYCLE_INTERVAL)
```

### 3.6 Logging / Dashboard

- **Log strutturato CSV** (`logs/run_*.csv`): ad ogni ciclo l'orchestratore registra tutte le metriche del sistema (timestamp, ground truth, predizione, confidenza, tempi di inferenza, CPU/RAM, valori ambientali, stato attuatori e ragionamento dell'agente).
- **Dashboard real-time con Streamlit e Plotly** (`dashboard/app_streamlit.py`): interfaccia grafica avanzata che esegue e visualizza il ciclo integrato sense-think-act. Include:
  - Visualizzazione live della foto analizzata e barra delle probabilità softmax (10 classi);
  - Gauge chart interattivi per i sensori ambientali e mini-trend nel tempo;
  - Indicatori LED di stato per il banco attuatori (`ActuatorBank`) e log del ragionamento PEAS;
  - System Monitor interattivo (CPU %, RAM, latenza ms) per il monitoraggio dei vincoli Edge;
  - Grafici di analisi temporale (accuracy cumulativa, distribuzione predizioni, istogramma latenze, radar sensori);
  - Tabella storico degli ultimi 30 cicli.

---

## 4. Compressione del modello per l'Edge (pruning + quantizzazione)

Obiettivo: partire dal miglior checkpoint (`.pth`, CNN 64f-k3-3blk, ~96%/94.98%)
e produrre una versione più leggera/veloce, misurando il trade-off
accuracy/dimensione/latenza — è il cuore "tecnico" della parte Edge AI.

### 4.1 Pruning

- **Unstructured pruning** (`torch.nn.utils.prune.l1_unstructured`): azzera i
  pesi meno significativi (magnitude-based) nei layer `Conv2d`/`Linear`.
  Riduce la dimensione "logica" ma non necessariamente quella su disco/i
  tempi, a meno di usare formati sparsi.
- **Structured pruning** (rimozione di interi filtri/canali convoluzionali,
  `ln_structured` su `dim=0`): più efficace su CPU (Raspberry Pi) perché
  riduce davvero le operazioni, non solo azzera pesi. **È la tecnica adottata
  nella pipeline** (`models/compress.py`), con norma L1 e sparsità del 30%.

### 4.2 Quantizzazione

- **Dynamic quantization** (`torch.quantization.quantize_dynamic`): la più
  semplice, converte i pesi dei layer `Linear` in INT8 a runtime. Buon primo
  esperimento, ma la CNN ha per lo più `Conv2d`, meno impattata.
- **Static quantization (post-training, PTQ)**: richiede una fase di
  calibrazione con un piccolo batch di immagini del training set per stimare
  i range di attivazione; quantizza sia pesi che attivazioni. Più efficace
  su CNN.
- **Quantization-Aware Training (QAT)**: se si vuole il massimo recupero di
  accuracy, si simula la quantizzazione già durante un breve re-training.
- Alternativa via **ONNX**: esportare il modello con `torch.onnx.export`,
  quindi quantizzare con `onnxruntime.quantization` ed eseguire l'inferenza
  con **ONNX Runtime** — spesso più realistico di PyTorch puro come target
  "simil-Raspberry Pi" perché è il motore di inferenza più diffuso in ambito
  embedded.

### 4.3 Risultati sperimentali di compressione

La suite di benchmark è implementata in `models/compress.py` (eseguita automaticamente al termine della pipeline di compressione) e salva i risultati in `benchmarks/benchmark_results.csv`. Le performance misurate sulla nostra architettura, sul test set corretto:

| Variante | Accuracy | Dimensione su disco | Latenza media (CPU, 1 thread) | Parametri totali / non nulli | Sparsità |
|---|---|---|---|---|---|
| **Baseline (float32)** | **95.0%** | 33.44 MB | 3.33 ms | 8,765,066 / 8,765,066 | 0.0% |
| **Pruned (Structured, 30%)** | 82.2% | 33.44 MB | 3.28 ms | 8,765,066 / 8,653,961 | 1.3% |
| **Pruned + Dynamic Quant (INT8)** | 82.2% | 9.43 MB | 4.32 ms | 370,816 / 259,711 | 30.0% |
| **ONNX Quantized (INT8)** | **95.0%** | **8.37 MB** | 6.10 ms | — | — |

> **Nota sui risultati Edge**: la **quantizzazione ONNX** è la scelta più equilibrata: riduce la dimensione del modello di **~4x** (da 33.4 MB a 8.4 MB) preservando l'accuratezza del modello originale (95.0%), perché applica la sola quantizzazione senza pruning. Il **pruning strutturato al 30%**, applicato in un unico passo senza fine-tuning di recupero, penalizza invece sensibilmente l'accuratezza (da 95.0% a 82.2%) — è il classico trade-off tra compressione aggressiva e qualità del modello. Le latenze restano tutte nello stesso ordine di grandezza (3–6 ms) data la ridotta dimensione della rete.

---

## 5. Struttura del repository

```
PomodorIA/
├── README.md
├── requirements.txt
├── config.yaml                  # soglie, intervalli, path dataset, modalità modello
├── models/
│   ├── CNN_64f-k3-3blk.pth      # checkpoint originale della CNN (float32)
│   ├── model.py                 # architettura TomatoCNN
│   ├── compress.py              # pruning, quantizzazione, export ONNX + benchmark
│   └── optimized/               # checkpoint compressi generati (ONNX, pth)
├── scripts/
│   └── extract_test_set.py      # estrae il test set (seed=42) per evitare data leakage
├── plantvillage_testset/        # test set estratto (dataset_root usato dal PoC)
├── sensors/
│   ├── virtual_camera.py        # fotocamera virtuale che campiona dal dataset
│   └── environment_simulator.py # simulatore sensori T/Hum/Soil/Lux (con Data Fusion)
├── edge/
│   └── inference_engine.py      # motore di inferenza (preprocessing, predict, timing)
├── agent/
│   └── decision_agent.py        # agente decisionale razionale (regole PEAS)
├── actuators/
│   ├── fake_gpio.py             # mock hardware livello basso (interfaccia RPi.GPIO)
│   └── actuators.py             # banco attuatori (irrigazione, ventola, allarme, notifiche)
├── orchestrator/
│   └── main_loop.py             # orchestratore del ciclo sense-think-act
├── dashboard/
│   └── app_streamlit.py         # dashboard real-time interattiva (Streamlit + Plotly)
├── logs/
│   └── run_*.csv                # log strutturati generati dai cicli
└── benchmarks/
    └── benchmark_results.csv    # risultati di performance misurati (generati da compress.py)
```

---

## 6. Guida rapida: Installazione e Avvio

### 6.1 Prerequisiti e Setup dell'ambiente

Assicurati di avere Python 3.10+ installato sul tuo sistema. Per configurare l'ambiente isolato e installare tutte le dipendenze:

```bash
# 1. Entra nella cartella del progetto
cd PomodorIA

# 2. Crea un virtual environment (raccomandato)
python3 -m venv .venv

# 3. Attiva l'ambiente virtuale
# Su macOS / Linux:
source .venv/bin/activate
# Su Windows:
# .venv\Scripts\activate

# 4. Installa le dipendenze richieste (PyTorch, Streamlit, Plotly, ecc.)
pip install -r requirements.txt
```

### 6.2 Configurazione del Dataset

Il progetto usa come `dataset_root` il **test set** (`plantvillage_testset/`, già
incluso nell'archivio consegnato), **non** il dataset completo. Questo è
essenziale: la CNN è stata allenata sull'80% delle immagini, quindi pescare
dall'intero dataset la valuterebbe su immagini già viste in training, mostrando
un'accuratezza fasulla vicina al **100%**. Verifica quindi che `config.yaml`
punti al test set:

```yaml
paths:
  dataset_root: "./plantvillage_testset"
  checkpoint: "./models/CNN_64f-k3-3blk.pth"
```

> Il test set (2.906 immagini, il 20% escluso dal training) è stato generato
> con `scripts/extract_test_set.py`, che riproduce lo stesso split casuale
> (`seed=42`) del progetto di training originale. Rigenerarlo è necessario solo
> se si parte da una copia locale del dataset completo:
> ```bash
> python scripts/extract_test_set.py \
>     --dataset-root "/percorso/al/dataset/plantvillage" \
>     --output-dir "./plantvillage_testset" --mode symlink
> ```

### 6.3 Avvio della Dashboard Real-Time

Per avviare la simulazione interattiva della serra domotica con l'interfaccia grafica real-time (Streamlit):

```bash
streamlit run dashboard/app_streamlit.py
```

Il comando aprirà automaticamente nel tuo browser la dashboard all'indirizzo **`http://localhost:8501`**, dalla quale potrai:
- Eseguire e monitorare il ciclo integrato `sense → think → act` passo dopo passo o in automatico;
- Osservare le diagnosi della CNN con le barre di probabilità softmax sulle 10 classi del pomodoro;
- Monitorare i sensori ambientali in tempo reale (temperatura, umidità suolo/aria, luminosità);
- Verificare il ragionamento dell'agente PEAS e lo scatto degli attuatori virtuali (irrigazione, ventola, allarmi);
- Consultare il System Monitor per misurare latenza di inferenza, consumo RAM e utilizzo CPU (su 1 thread, come su Raspberry Pi).

---

## 7. Stack tecnologico utilizzato

| Ambito | Strumenti |
|---|---|
| ML / Edge AI | PyTorch, `torch.nn.utils.prune`, `torch.quantization`, ONNX, ONNX Runtime |
| Simulazione risorse | `psutil`, `time`, `torch.set_num_threads` |
| Dashboard & UI | Streamlit, Plotly, PIL |
| Logging / Gestione Dati | `pandas`, file CSV strutturati |
| Configurazione | `pyyaml` (`config.yaml`) |
| Testing | Script integrati e test end-to-end |

---

## 8. Possibili estensioni

- **Federated Learning simulato**: più "serre virtuali" che addestrano
  localmente e aggregano i pesi, richiamando il paradigma citato nei tuoi
  appunti per l'Edge/Federated Learning.
- **Modello di degrado nel tempo**: simulare l'evoluzione di una malattia
  su più cicli, non solo singoli scatti indipendenti.

---

## 9. Riferimenti

- Dataset: **PlantVillage — Tomato** (14.529 immagini, 10 classi).
- Modello: `TomatoCNN` (n_filters=64, kernel_size=3, num_blocks=3),
  progetto P14 "FCNN vs CNN", corso di Neural Networks.
- Appunti del corso di Internet of Things (capitoli su IoT, architetture,
  Big Data, Smart Cities/CPS) per il framing teorico.
