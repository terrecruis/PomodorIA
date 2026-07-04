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

| Livello IoTWF | Nel PoC reale (target) | Nel PoC simulato (quello che costruiamo) |
|---|---|---|
| 1. Physical Devices & Controllers | Fotocamera, sensori DHT22/igrometro, pompa, ventola | **Moduli Python mock** che generano/leggono dati sintetici |
| 2. Connectivity | Wi-Fi del Raspberry Pi | Loopback locale / MQTT su `localhost` (opzionale) |
| 3. Edge (Fog) Computing | Inferenza CNN quantizzata sul Raspberry Pi | Inferenza CNN quantizzata **sul PC**, con vincoli di risorse simulati |
| 4. Data Accumulation | Storage locale su SD card | File SQLite/JSON o CSV di log |
| 5. Data Abstraction | Normalizzazione dei dati per il livello applicativo | Classi Python che restituiscono un "readings object" uniforme |
| 6. Application | Dashboard di monitoraggio | Dashboard Streamlit/Flask (opzionale) |
| 7. Collaboration & Processes | Notifiche all'agricoltore, integrazione gestionale | Log/alert testuali, notifica simulata |

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

### 3.6 Logging / Dashboard (opzionale ma consigliato)

- Log strutturato (CSV o SQLite) di ogni ciclo: utile per generare i grafici
  della relazione finale (accuracy live, distribuzione predizioni, tempi di
  inferenza).
- Dashboard semplice con **Streamlit** o **Flask**: mostra ultima immagine
  analizzata, predizione, stato sensori/attuatori, storico. Non indispensabile
  per la validità scientifica del PoC ma alza molto l'impatto della demo.

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

### 4.3 Metriche da confrontare (tabella da produrre nella relazione)

| Metrica | Baseline (.pth originale) | Pruned | Quantized | Pruned+Quantized |
|---|---|---|---|---|
| Accuracy (test set) | 96% | | | |
| Dimensione modello (MB) | | | | |
| Tempo medio di inferenza per immagine (ms, su CPU) | | | | |
| N. parametri non nulli | | | | |

Questa tabella è il "risultato principale" della parte Edge del progetto,
analoga a quella prodotta nel confronto FCNN vs CNN.

---

## 5. Simulare il "Raspberry Pi virtuale" sul PC

Per rendere credibile l'affermazione "il PC diventa esso stesso un Raspberry
Pi" senza avere l'hardware:

1. **Limitare le risorse usate dal processo**, per rendere i numeri di
   latenza/throughput più rappresentativi di un dispositivo embedded:
   - vincolare l'inferenza a **1 solo core CPU** (`torch.set_num_threads(1)`,
     oppure `taskset` su Linux) — un Raspberry Pi 4 ha 4 core ARM, ma
     limitare i thread rende il confronto più onesto rispetto a un laptop
     multi-core;
   - forzare `device = "cpu"` sempre (niente GPU/MPS), dato che il Raspberry
     Pi non ha una GPU CUDA;
   - misurare RAM/CPU usage con `psutil` durante l'inferenza, da riportare
     come proxy di "fattibilità su Raspberry Pi" (i modelli Raspberry Pi 4/5
     hanno 2–8 GB di RAM).
2. **Mock delle GPIO**: una classe `FakeGPIO` con la stessa interfaccia di
   `RPi.GPIO` (`setup()`, `output()`, `input()`), così il codice degli
   attuatori è "drop-in replaceable" su hardware reale.
3. **(Opzionale) Comunicazione IoT realistica**: montare un broker **MQTT**
   locale (Mosquitto) e far comunicare i moduli (sensori → topic, agente →
   subscribe, attuatori → topic comandi) invece di semplici chiamate a
   funzione. Rende il PoC molto più vicino a un vero sistema IoT (protocollo
   citato esplicitamente nei tuoi appunti come standard per dispositivi
   vincolati) e dimostra di aver capito il livello *Connectivity* dello
   stack, anche se qui gira tutto su `localhost`.

---

## 6. Struttura del repository proposta

```
tomato-edge-greenhouse/
├── README.md
├── requirements.txt
├── config.yaml                  # soglie, intervalli, path dataset, modalità modello
├── models/
│   ├── best_cnn_64f_k3_3blk.pth # checkpoint originale
│   ├── model.py                 # TomatoFCNN / TomatoCNN (quello già fornito)
│   ├── compress.py              # pruning + quantizzazione + export ONNX
│   └── optimized/                # checkpoint compressi generati
├── sensors/
│   ├── virtual_camera.py
│   └── environment_simulator.py
├── edge/
│   └── inference_engine.py      # carica modello, preprocessing, predict()
├── agent/
│   └── decision_agent.py        # regole PEAS
├── actuators/
│   ├── fake_gpio.py
│   └── actuators.py
├── orchestrator/
│   └── main_loop.py             # ciclo sense-think-act
├── dashboard/
│   └── app_streamlit.py         # opzionale
├── logs/
│   └── run_YYYYMMDD.csv
├── benchmarks/
│   └── benchmark_compression.py # produce la tabella accuracy/size/latenza
└── notebooks/
    └── analisi_risultati.ipynb
```

---

## 7. Stack tecnologico consigliato

| Ambito | Strumenti |
|---|---|
| ML / Edge AI | PyTorch, `torch.nn.utils.prune`, `torch.quantization`, ONNX, ONNX Runtime |
| Simulazione risorse | `psutil`, `time`, `torch.set_num_threads` |
| Comunicazione IoT (opzionale) | MQTT (`paho-mqtt`) + broker Mosquitto locale |
| Dashboard (opzionale) | Streamlit oppure Flask + Chart.js |
| Logging/dati | `pandas`, SQLite o CSV |
| Config | `pyyaml` |
| Testing | `pytest` per i moduli sensori/agente/attuatori |

---

## 8. Roadmap di sviluppo (fasi)

1. **Setup progetto**: struttura repo, config, caricamento checkpoint `.pth`,
   verifica che l'inferenza baseline riproduca ~96% sul test set.
2. **Sensor layer**: Virtual Camera + simulatore ambientale, con log delle
   letture.
3. **Edge AI layer**: wrapper di inferenza pulito (preprocessing identico al
   training), con timing.
4. **Compressione**: pruning, quantizzazione, export ONNX; produzione della
   tabella comparativa di §4.3.
5. **Agente decisionale**: regole PEAS, integrazione con attuatori simulati.
6. **Orchestratore**: main loop completo, prima demo end-to-end.
7. **Simulazione vincoli Raspberry Pi**: limitazione risorse, misure
   CPU/RAM/latenza.
8. **(Opzionale) MQTT + Dashboard**: per una demo più "IoT-vera".
9. **Valutazione finale e relazione**: metriche ML (accuracy pre/post
   compressione) + metriche di sistema (latenza, RAM, robustezza dell'agente).

---

## 9. Metriche di valutazione complessive del PoC

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

## 10. Collegamento con la teoria del corso (utile per la relazione)

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

## 11. Possibili estensioni (se vuoi alzare il livello del progetto)

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

## 12. Riferimenti

- Dataset: **PlantVillage — Tomato** (14.529 immagini, 10 classi).
- Modello: `TomatoCNN` (n_filters=64, kernel_size=3, num_blocks=3),
  progetto P14 "FCNN vs CNN", corso di Neural Networks.
- Appunti del corso di Internet of Things (capitoli su IoT, architetture,
  Big Data, Smart Cities/CPS) per il framing teorico.
