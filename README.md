# AI-Based Early Warning and Landslide Risk Monitoring System for NER

## 🚀 Live Demo

👉 [Open the Working Demo](https://ai-based-early-warning-and-landslide-monitoring-system-tlzbc4o.streamlit.app/)

## Smart India Hackathon 2026 — SIH26001

An AI-based geospatial monitoring prototype designed to assess rainfall-driven landslide risk across the North Eastern Region (NER) of India and present location-specific risk and early-warning information through an interactive dashboard.

### NER States Covered

- Assam
- Meghalaya
- Manipur
- Mizoram
- Nagaland
- Sikkim
- Tripura

> Detailed spatial prototype monitoring is demonstrated for Sikkim.

---

## Problem

The North Eastern Region is vulnerable to landslides because of heavy rainfall, steep terrain and fragile geological conditions.

Environmental information such as rainfall, terrain, geology and historical landslide data can come from different sources, making it difficult to obtain a unified location-specific view of risk.

Our project aims to transform environmental information into actionable landslide-risk intelligence.

---

## Proposed Solution

The system combines:

- Satellite rainfall data
- Official NER geographic boundaries
- Location-level rainfall feature extraction
- Machine-learning-based risk classification
- Interactive geospatial visualization
- Early-warning decision logic
- Location-based registered-user alert identification

The long-term goal is to provide authorities with location-specific information that can support monitoring and disaster-response decisions.

---

## Current System Workflow

```text
NASA GPM IMERG Rainfall
          ↓
NER Geographic Boundaries
          ↓
Rainfall Data Processing
          ↓
Location-Level Feature Extraction
          ↓
Mean / Maximum / Minimum Rainfall
          ↓
Random Forest Baseline
          ↓
LOW / MEDIUM / HIGH Risk
          ↓
Early-Warning Logic
          ↓
Interactive Streamlit Dashboard