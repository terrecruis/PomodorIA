
# Serra Domotica con Edge AI — Riconoscimento Patologie Fogliari del Pomodoro
### Proof of Concept software (simulazione locale, senza hardware reale)

**Autori:** Vellotti Gianmarco, Terrecuso Francesco
**Corso:** Internet of Things — Università degli Studi di Napoli Federico II
**Modello ML di partenza:** `TomatoCNN` (progetto P14, esame di Neural Networks)

---

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
- Deve poter girare in due modalità, selezionabili da config, per confrontarle
  nella relazione:
  - `full_precision` → modello originale `.pth` (baseline)
  - `optimized` → modello quantizzato/pruned (target edge)

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
  riduce davvero le operazioni, non solo azzera pesi.
- Consigliato: pruning **iterativo** con fine-tuning breve dopo ogni step
  (poche epoche), per recuperare l'accuracy persa.

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

Dalla suite di benchmark implementata (`benchmarks/benchmark_compression.py`), abbiamo misurato le seguenti performance sulla nostra architettura:

| Variante | Accuracy | Dimensione su disco | Latenza media (CPU, 1 thread) | Parametri totali / non nulli | Sparsità | RAM processo |
|---|---|---|---|---|---|---|
| **Baseline (float32)** | **99.2%** | 33.44 MB | 2.28 ms | 8,765,066 / 8,765,066 | 0.0% | 404 MB |
| **Pruned (L1 Unstructured, 30%)** | 86.0% | 33.44 MB | 2.28 ms | 8,765,066 / 8,653,961 | 1.3% | 293 MB |
| **Pruned + Dynamic Quant (INT8)** | 86.2% | **9.43 MB** | 2.98 ms | 370,816 / 259,711 | 30.0% | 339 MB |
| **ONNX Runtime (float32/INT8)** | **99.2%** | — | **1.58 ms** | — | — | 746 MB |

> **Nota sui risultati Edge**: La quantizzazione dinamica (`Pruned + Dynamic Quant`) riduce la dimensione del modello di **~3.5x** (da 33.4 MB a 9.4 MB), rendendolo ideale per il caricamento nella RAM limitata di microcontrollori o schede embedded. Per quanto riguarda la velocità di esecuzione, il motore **ONNX Runtime** si dimostra il più efficiente sul nostro target CPU, abbattendo la latenza a **~1.58 ms per scatto** preservando l'accuratezza del modello originale.

---

## 5. Simulare il "Raspberry Pi virtuale" sul PC

Per rendere credibile l'affermazione "il PC diventa esso stesso un Raspberry
Pi" senza avere l'hardware:

1. **Limitare le risorse usate dal processo**, per rendere i numeri di
   latenza/throughput più rappresentativi di un dispositivo embedded:
   - vincolare l'inferenza a **1 solo core CPU** (`torch.set_num_threads(1)`),
     rendendo il confronto più onesto rispetto a un laptop multi-core;
   - forzare `device = "cpu"` sempre (niente GPU/MPS), dato che il Raspberry
     Pi non ha una GPU CUDA;
   - misurare RAM/CPU usage con `psutil` durante l'inferenza, da riportare
     come proxy di "fattibilità su Raspberry Pi" (i modelli Raspberry Pi 4/5
     hanno 2–8 GB di RAM).
2. **Mock delle GPIO**: una classe `FakeGPIO` (`actuators/fake_gpio.py`) con la stessa interfaccia di
   `RPi.GPIO` (`setup()`, `output()`, `input()`), così il codice degli
   attuatori è immediatamente utilizzabile su hardware reale.
3. **Comunicazione modulare in-memory (Livello Connectivity & Abstraction)**: i moduli comunicano tramite un'architettura software disaccoppiata basata su dataclass e interfacce ben definite (`SensorReading`, `InferenceResult`, `AgentDecision`). Questo rispecchia fedelmente il livello di astrazione del dato (Livello 5 IoTWF), garantendo leggerezza e rendendo il codice pronto per essere agganciato a un bus reale (es. MQTT) su hardware fisico.

---

## 6. Struttura del repository

```
PomodorIA/
├── README.md
├── requirements.txt
├── config.yaml                  # soglie, intervalli, path dataset, modalità modello
├── models/
│   ├── CNN_64f-k3-3blk.pth      # checkpoint originale della CNN (float32)
│   ├── model.py                 # architettura TomatoCNN
│   ├── compress.py              # pruning, quantizzazione dinamica, export ONNX
│   └── optimized/               # checkpoint compressi generati (ONNX, ecc.)
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
    ├── benchmark_compression.py # suite di benchmark di quantizzazione e pruning
    └── benchmark_results.csv    # risultati di performance misurati
```

---

## 7. Guida rapida: Installazione e Avvio

### 7.1 Prerequisiti e Setup dell'ambiente

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

### 7.2 Configurazione del Dataset

Prima di avviare il sistema, verifica che il percorso del dataset nel file [`config.yaml`](file:///Users/francescoterrecuso/Desktop/PomodorIA/config.yaml) punti correttamente alla cartella di *PlantVillage-Tomato* presente sul tuo computer:

```yaml
paths:
  dataset_root: "/percorso/al/tuo/dataset/plantvillage"
```

### 7.3 Avvio della Dashboard Real-Time

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

## 8. Stack tecnologico utilizzato

| Ambito | Strumenti |
|---|---|
| ML / Edge AI | PyTorch, `torch.nn.utils.prune`, `torch.quantization`, ONNX, ONNX Runtime |
| Simulazione risorse | `psutil`, `time`, `torch.set_num_threads` |
| Dashboard & UI | Streamlit, Plotly, PIL |
| Logging / Gestione Dati | `pandas`, file CSV strutturati |
| Configurazione | `pyyaml` (`config.yaml`) |
| Testing | Script integrati e test end-to-end |

---

## 9. Roadmap e Fasi Implementate

1. **Setup progetto e Modello Baseline**: struttura del repository, file di configurazione (`config.yaml`), caricamento del checkpoint PyTorch (`CNN_64f-k3-3blk.pth`) e verifica del funzionamento dell'inferenza.
2. **Sensor Layer**: implementazione della fotocamera virtuale (`VirtualCameraSensor`) con aggancio al dataset e del simulatore di sensori ambientali (`EnvironmentSensorSimulator`) con correlazioni euristiche alle patologie (Data Fusion).
3. **Edge AI Layer**: sviluppo dell'`InferenceEngine` con supporto per modalità multiple (`full_precision` e `optimized`) e misurazione accurata dei tempi di inferenza.
4. **Compressione per l'Edge**: realizzazione della suite di compressione (`models/compress.py`) con Unstructured/Structured Pruning, Dynamic Quantization e quantizzazione ONNX Runtime, e salvataggio dei benchmark (`benchmarks/benchmark_results.csv`).
5. **Actuator Layer**: implementazione dell'infrastruttura di hardware virtuale (`FakeGPIO`) e del banco attuatori (`ActuatorBank`: irrigazione, ventilazione, allarmi LED, notifiche).
6. **Agente Decisionale PEAS**: sviluppo dell'agente razionale (`DecisionAgent`) basato su regole prioritarie, memoria di stato (arretramento/persistenza allarmi) e logica di Data Fusion.
7. **Orchestratore e Simulazione Hardware**: implementazione del ciclo integrato sense-think-act (`orchestrator/main_loop.py`) con monitoraggio delle risorse esterne (1 thread CPU, memoria RAM, CPU usage tramite `psutil`).
8. **Dashboard Real-Time**: realizzazione dell'interfaccia grafica avanzata in Streamlit e Plotly (`dashboard/app_streamlit.py`) per il controllo e la visualizzazione interattiva del sistema della serra domotica.

---

## 10. Metriche di valutazione complessive del PoC

- **Metriche ML**: accuracy, precision/recall/F1 per classe (specialmente
  Early blight e Tomato mosaic virus, già note come critiche), confusion
  matrix, confronto baseline vs modello compresso.
- **Metriche di sistema/Edge**: dimensione modello (MB), tempo di inferenza
  medio (ms), throughput (immagini/secondo), uso di RAM/CPU.
- **Metriche di comportamento dell'agente**: numero di azioni corrette vs
  errate rispetto alla "verità" nota dal dataset (es. ha attivato
  ventilazione quando la classe reale era davvero una fungina?), numero di
  falsi allarmi.

---

## 11. Collegamento con la teoria del corso

- **IoTWF a 7 livelli** → mapping esplicito di ogni componente del PoC (§2).
- **PEAS e tipologie di agente** → definizione formale dell'agente
  decisionale (§3.3).
- **Edge/Fog Computing** → motivazione della compressione del modello: ridurre
  latenza e non dipendere dal cloud, cruciale in serre spesso in zone con
  connettività scarsa.
- **Data Fusion** → combinazione della predizione visiva (CNN) con i dati
  ambientali per decisioni più robuste.
- **Smart Agriculture / Precision Farming / Agriculture 4.0** → contesto
  applicativo generale del progetto, citato esplicitamente nei tuoi appunti.
- **Sfide IoT (dispositivi vincolati, analisi in tempo reale)** → giustificano
  sia la scelta della CNN compressa sia l'architettura a cicli sense-think-act.

---

## 12. Possibili estensioni

- **Federated Learning simulato**: più "serre virtuali" che addestrano
  localmente e aggregano i pesi, richiamando il paradigma citato nei tuoi
  appunti per l'Edge/Federated Learning.
- **Digital Twin della serra**: rappresentazione virtuale continuamente
  aggiornata dai dati simulati, con possibilità di "replay" di scenari.
- **Modello di degrado nel tempo**: simulare l'evoluzione di una malattia
  su più cicli, non solo singoli scatti indipendenti.
- **Confronto energetico stimato**: stimare il consumo energetico (proxy via
  FLOPs) di baseline vs modello compresso, per discutere sostenibilità.

---

## 13. Riferimenti

- Dataset: **PlantVillage — Tomato** (14.529 immagini, 10 classi).
- Modello: `TomatoCNN` (n_filters=64, kernel_size=3, num_blocks=3),
  progetto P14 "FCNN vs CNN", corso di Neural Networks.
- Appunti del corso di Internet of Things (capitoli su IoT, architetture,
  Big Data, Smart Cities/CPS) per il framing teorico.
