# 🧠 Project PlantBrain: Industrial Knowledge Intelligence Engine

> **Submission for ET AI Hackathon 2026** > **Category:** Edge AI / Industrial Systems  
> **Architecture:** Fully Offline, Edge-Native Dual-Route Retrieval-Augmented Generation (RAG)

---

## 📌 Project Overview
Project PlantBrain is a production-ready, fully offline, edge-native RAG platform engineered specifically for mission-critical heavy processing industries (such as petrochemical plants, power generation stations, and remote manufacturing hubs). 

In these environments, data security policies completely prohibit open cloud access. PlantBrain runs **entirely localized within the workstation's RAM and CPU bounds with 0.0ms cloud dependency**. It ingests fragmented, unstructured files—specifically immutable asset technical specifications (**PDF**) and daily operational shift logs (**TXT**)—mapping design parameters right beside historical maintenance failure records on a single glass-morphic interface.

---

## 🏆 Evaluation Rubric Alignment (100% Core Focus)

This project has been architected strictly against the four core dimensions of the hackathon scoring matrix:

### 💡 1. Technological Innovation (25%)
Traditional RAG frameworks dump all data sources into a single, unorganized vector cluster, which causes loose text narratives to blur precise engineering values. PlantBrain introduces a novel **Deterministic Dual-Route Selection Routing Engine**. Using real-time file extension filters, it isolates technical manufacturer rules from colloquial shift diaries, rendering distinct streams side-by-side on screen simultaneously.

### 🏗️ 2. Architectural Depth & Stack Power (25%)
Built using a lightweight, high-performance edge-native embedding layer utilizing the neural **`all-MiniLM-L6-v2` Sentence Transformer**. The architecture tokenizes, processes, and stores 384-dimensional continuous dense vector representations completely offline in local memory cache lines, ensuring absolute data isolation and zero network vulnerability handshakes.

### ⏱️ 3. Operational Impact & MTTR Mitigation (25%)
Unplanned asset downtime introduces massive financial overheads. PlantBrain directly mitigates **Mean Time to Repair (MTTR)**. By evaluating semantic similarity via a local dot product calculation loop, it immediately connects live site emergency prompts (e.g., *"Suction side pressure drop at intake strainer"*) to exact physical tolerances and past successful human fixes in milliseconds.

### 🚀 4. Completeness & Production Readiness (25%)
Optimized against interface calculation lag common in local multi-megabyte text-parsing run loops. The application implements **Non-Blocking Container Mutation Injection** via `st.session_state` synchronization. Processing latency and chunk telemetry overwrite localized sidebar display blocks instantly without triggering global framework script re-runs (`st.rerun()`), providing a bulletproof user interface tailored for high-pressure emergency scenarios.

---

## 📁 Repository Directory Structure
```text
Project-PlantBrain/
├── app.py                      # Core optimized Streamlit application blueprint
├── requirements.txt            # Production package configuration manifest
├── PlantBrain_Whitepaper.pdf   # Complete detailed technical engineering specification
├── README.md                   # Project presentation dashboard reference
└── static/
    └── background.png          # High-resolution industrial frosted-glass UI visual asset
