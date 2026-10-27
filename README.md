#  Enterprise Anti-Money Laundering (AML) & Graph AI Engine

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch)
![PyTorch Geometric](https://img.shields.io/badge/PyG-Graph_Neural_Networks-3DDC84?style=flat)
![CUDA](https://img.shields.io/badge/CUDA-NVIDIA_RTX_3060-76B900?style=flat&logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

An industrial-grade, end-to-end **Graph Neural Network (GNN)** pipeline designed for real-time financial crime detection, suspicious topology identification (Smurfing/Layering), Explainable AI (XAI) feature attribution, and automated regulatory reporting (MASAK STR).

---

##  Architectural Overview

Traditional machine learning algorithms (e.g., XGBoost, Random Forest) treat financial transactions as isolated tabular rows. However, modern money laundering schemes use complex networks of mule accounts and rapid cyclic transfers to obfuscate funds.

This platform models financial ecosystems as **heterogeneous/directed graphs** where:
* **Nodes ($V$):** Customer Accounts (attributing age, balance, daily volume, internal risk scores).
* **Edges ($E$):** Financial Transactions & Wire Transfers between accounts.

```text
       [Smurfing / Fan-Out]                   [Layering / Cyclic Loop]
          (Main Mule)                              (Account A)
          /    |    \                                 /     \
      (M1)   (M2)   (M3)                         (Account C) -- (Account B)
```

---

##  Key Modules & Capabilities

1. **Hybrid GNN Topology (`GraphSAGE` + `GATConv`):**
   * **GraphSAGE Layer:** Aggregates structural multi-hop neighborhood features.
   * **Graph Attention Network (GAT):** Multi-head attention mechanism assigning higher weights to high-frequency or anomalous transaction channels.

2. **Imbalanced Data & Threshold Tuning:**
   * **Weighted Cross-Entropy Loss:** Penalizes missed suspicious accounts (False Negatives) heavier based on class distribution ($N_{normal} / N_{suspicious}$).
   * **Precision-Recall & $F_2$-Score Optimization:** Dynamic threshold selection prioritizing recall for financial regulatory compliance.

3. **Explainable AI (XAI) & Feature Attribution:**
   * Feature-masking sensitivity analysis quantifying exact node-level attribution (Account Age, Balance, Volume, Risk Score).

4. **Temporal Velocity & Streaming Ingestion:**
   * Calculates rolling timestamp variance to detect rapid-fire sequential transactions.
   * Real-time inference sub-millisecond execution loop for streaming transactions.

5. **Adversarial Robustness & Hybrid Decision Engine:**
   * Graph Poisoning stress tests (%15 camouflage edge insertion) evaluating decision stability.
   * Dual-layer decision engine merging deterministic business rules with GNN risk probabilities.

6. **Automated Compliance & Interactive Visualization:**
   * Automated **MASAK STR (Suspicious Transaction Report)** JSON payload generation.
   * Interactive **PyVis HTML** network graph rendering zoomable, clickable node relationships.

---

##  Performance & Evaluation Metrics

Executed on **NVIDIA GeForce RTX 3060 Laptop GPU** via PyTorch CUDA acceleration:

| Metric | Score / Value |
| :--- | :--- |
| **ROC-AUC Score** | **0.9984** |
| **Accuracy** | **%97.50** |
| **Optimal Threshold ($F_2$-Optimized)** | **~0.42** |
| **Real-time Inference Latency** | **< 3.5 ms** |
| **Adversarial Stability** | **>%95.0** |

---

##  Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/erdembrn34/enterprise-aml-graph-ai.git
   cd enterprise-aml-graph-ai
   ```

2. **Install required packages:**
   ```bash
   pip install torch torch-geometric networkx matplotlib seaborn scikit-learn pyvis
   ```

3. **Run the Jupyter Notebook:**
   ```bash
   jupyter notebook AML_Graph_Neural_Network.ipynb
   ```

---

## 📁 Repository Structure

```text
├── AML_Graph_Neural_Network.ipynb   # Complete 20-cell End-to-End Execution Notebook
├── AML_Risk_Report.csv              # Auto-generated Compliance & MASAK Audit Report
├── aml_interactive_network.html     # Interactive PyVis Web Graph Export
├── aml_gnn_model_v1.pt              # Trained PyTorch Model Checkpoint
└── README.md                        # Technical Documentation
```

---

## 📄 Automated MASAK STR Payload Sample

```json
[
    {
        "Bildirim_Tipi": "MASAK_ŞÜPHELİ_İŞLEM_BİLDİRİMİ_(STR)",
        "Bildiren_Kurum": "X_BANKASI_UYUM_BAŞKANLIĞI",
        "Hesap_Bilgileri": {
            "Müşteri_Hesap_ID": 42,
            "Model_Risk_Skoru": "%98.40",
            "Tespit_Edilen_Tipoloji": "Smurfing / Fan-Out (Ana Paravan Hesap)"
        },
        "Teknik_Detaylar": {
            "Algoritma": "PyTorch Geometric - GraphSAGE + GAT",
            "Zamansal_Anomali_Skoru": 0.892,
            "Açıklanabilirlik_Notu": "GNNExplainer ile yüksek dereceli komşuluk zinciri ve anormal transfer hızı tespit edilmiştir."
        }
    }
]
