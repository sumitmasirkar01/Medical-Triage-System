# STAT-Triage AI: Multi-Modal Emergency Radiology Prioritization System

STAT-Triage AI is a high-throughput, polymorphic emergency triage engine designed to ingest 3D Volumetric Head CTs, 2D Planar Chest Radiographs, and Multi-Sequence Knee MRIs into a unified, starvation-free dynamic priority worklist.

Built in alignment with the **FDA CADt (Computer-Assisted Triage and Notification)** regulatory framework, the system operates as an intelligent worklist switchboard—quantifying risk, generating localized visual explainability, and dynamically escalating critical cases in real time.

---

## Key System Capabilities

* **Polymorphic Ingestion Engine:** Unified `BaseTriageAdapter` abstraction supporting multi-slice 3D CT volumes (Bi-GRU sequence aggregation), 2D planar radiographs (DenseNet-121), and multi-planar MRIs (EfficientNet-B0).
* **Calibrated DICOM Physics Engine:** Direct 16-bit binary `.dcm` parsing with automated rescale slope/intercept correction and multi-band Hounsfield Unit (HU) tissue windowing (Brain, Subdural, Bone).
* **Stateless Visual Explainability:** Memory-safe Gradient-Weighted Class Activation Mapping (Grad-CAM) paired with a Perturbation Stability Index (PSI) to mathematically verify decision robustness against noise.
* **Starvation-Free Priority Queue:** Dynamic aging function $S(t)$ that prevents non-urgent cases from waiting indefinitely while instantly escalating emergent life threats to Rank #1.
* **Streamlit PACS Clinical Workstation:** Real-time priority queue monitoring, axial slice scrubbers, Grad-CAM transparency blending, subtype risk distributions, and sovereign clinician override controls.

---

## Core Mathematical Formulations

### 1. Hounsfield Unit (HU) Physics Calibration
$$\text{HU} = \text{PixelValue} \times \text{RescaleSlope} + \text{RescaleIntercept}$$

$$W(\text{HU}; c, w) = \text{clip}\left( \frac{\text{HU} - (c - w/2)}{w}, \, 0.0, \, 1.0 \right)$$

### 2. Temperature-Scaled Logit Calibration
$$P_{\text{calibrated}} = \sigma\left( \frac{z}{T} \right) = \frac{1}{1 + \exp\left(-\frac{z}{T}\right)}, \quad T > 0$$

### 3. Perturbation Stability Index (PSI)
$$\text{PSI} = \max\left(0.0, \, 1.0 - \frac{\vert{}P(X) - P(X + \epsilon)\vert{}}{\delta_{\text{max}}}\right), \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

### 4. Dynamic Anti-Starvation Queue Score
$$S(t) = \underbrace{P_{\text{critical}} \cdot C}_{\text{Clinical Severity}} + \underbrace{\lambda \cdot t}_{\text{Anti-Starvation Aging}} + \underbrace{\gamma \cdot (1.0 - \text{PSI})}_{\text{Instability Surcharge}}$$
$$\text{where } C = 2 \cdot \vert{}P_{\text{critical}} - 0.5\vert{} \in [0, 1]$$

---

## Verified Production Benchmarks

Executed on standard single-core CPU execution engines:

| Benchmark Dimension | Measured Result | Production Target | Status |
| :--- | :---: | :---: | :---: |
| **Knee MRI Latency (CPU)** | **$108.92\text{ ms}$** ($9.2\text{ FPS}$) | $< 250\text{ ms}$ | **Passed** |
| **Chest X-Ray Latency (CPU)** | **$257.88\text{ ms}$** ($3.9\text{ FPS}$) | $< 500\text{ ms}$ | **Passed** |
| **32-Slice Head CT Latency (CPU)** | **$769.19\text{ ms}$** ($1.3\text{ FPS}$) | $< 1200\text{ ms}$ | **Passed** |
| **Queue Sort Time (250 Studies)** | **$0.2610\text{ ms}$** | $< 10.0\text{ ms}$ | **Passed** |
| **Queue Throughput** | **$957,792\text{ evals/sec}$** | $> 100,000\text{ evals/sec}$ | **Passed** |
| **Memory Growth (100 E2E Cycles)** | **$43.24\text{ KB}$** | $< 1000\text{ KB}$ | **Zero Leak** |

---

## Directory Structure

```text
Medical_Triage_system/
├── core/
│   ├── __init__.py
│   ├── base.py              # Polymorphic adapter contracts & schemas
│   ├── preprocessing.py     # 16-bit DICOM parsing & HU window normalization
│   ├── models.py            # Neural architectures, Bi-GRU, Grad-CAM & PSI
│   ├── calibration.py       # Temperature scaling & high-sensitivity thresholds
│   └── gateway.py           # Priority queue math & dynamic zone routing
├── docs/                    # Architecture diagrams, figures & clinical reports
├── tests/                   # Deterministic unit and integration test suites
├── weights/                 # Model state dictionaries (.pt files)
├── benchmark_pipeline.py    # Latency, FPS, and memory stress benchmark suite
├── create_sample_dicoms.py  # Standalone 16-bit DICOM phantom generator
├── dashboard.py             # PACS workstation user interface
└── requirements.txt         # Pinned production environment dependencies