<div align="center">

# 🫁 NeuroScan AI — Automated COVID-19 Chest X-Ray Diagnostic System

**An Enterprise-Grade Deep Learning Medical Imaging Platform with Real-Time Radiological Inference & Automated Clinical PDF Report Generation.**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow_CPU-2.15.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.11.9-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Render](https://img.shields.io/badge/Deployed_on-Render-46E3B7?style=for-the-badge&logo=render&logoColor=black)](https://covid19-xray-classification.onrender.com)

[Explore Features](#-key-features) • [Model Architecture](#-deep-learning--model-training) • [System Workflow](#-system-architecture--inference-pipeline) • [Deployment](#-getting-started--deployment) • [API Reference](#-api-reference)

</div>

---

## 📖 Overview

**NeuroScan AI** is an end-to-end medical diagnostic web application designed for rapid preliminary screening of Chest Radiographs (X-Rays). Powered by a Deep Convolutional Neural Network (**MobileNetV2** transfer learning architecture) served via an asynchronous **FastAPI** backend, the system classifies chest X-rays into three distinct categories: **COVID-19**, **Normal**, and **Viral Pneumonia**[cite: 22, 25, 26].

In addition to sub-second radiological inference, NeuroScan AI features a modern **Glassmorphism Diagnostic Dashboard** and an automated **Clinical PDF Report Generator** (`FPDF2` + `Matplotlib`) that compiles patient metadata, uploaded radiographs, probability distribution charts, and clinical recommendations into downloadable medical reports[cite: 24, 26, 27].

---

## ✨ Key Features

### 🧠 Deep Learning & Enterprise Backend
* **3-Class Radiological Classification:** Accurately distinguishes between `Covid` (`0`), `normal` (`1`), and `viral` pneumonia (`2`) from chest X-ray images[cite: 22, 25].
* **Standardized Image Preprocessing Pipeline:** Automatically validates file types and size limits (up to 5 MB), converts images to RGB, resizes to `224x224`, and normalizes pixel values to `[-1.0, 1.0]` via `(img_array / 127.5) - 1.0`.
* **Automated Clinical PDF Report Engine:** Dynamically generates downloadable PDF diagnostic reports containing patient details (Name, Age, Gender, Referring Doctor), Risk Level badges (`HIGH RISK` / `LOW RISK`), the original X-ray scan, a `Matplotlib` horizontal bar chart of class probabilities, and actionable clinical recommendations.
* **Non-Blocking Telemetry & Analytics:** Uses FastAPI `BackgroundTasks` to compute real-time rolling averages for system confidence, inference latency (`ms`), and per-class prediction counts without slowing down API responses.
* **Custom Middleware & Lifespan Management:** Loads the TensorFlow model (`keras_model.h5`) via FastAPI's `@asynccontextmanager` lifespan handler and injects an `X-Process-Time` header into every HTTP response.

### 💻 Interactive Diagnostic Dashboard (Frontend)
* **Drag-and-Drop X-Ray Workspace:** Supports instant image previewing with client-side format (`JPEG`/`PNG`) and size (`5 MB`) validation, paired with an animated CSS laser scanning effect during inference.
* **Visual Diagnostic Breakdown:** Displays dynamic risk badges, an animated SVG circular confidence gauge, per-class linear probability bars, and color-coded clinical observation cards[cite: 24].
* **Live System Health & Metrics Polling:** Continuously monitors backend readiness (`/health`) with live status indicators (`System Ready` / `Offline`) and real-time KPI cards tracking Total Scans, Average Confidence, and System Latency[cite: 24].
* **Adaptive Dark / Light Theme:** Built with CSS custom variables for seamless switching between high-contrast dark radiological viewing and light clinical themes[cite: 24].

---

## 🔬 Deep Learning & Model Training

The classification engine was trained in Google Colab using **Transfer Learning** on a curated 3-class Chest X-Ray dataset (`251` training images and `66` testing images).

### Model Architecture (`Sequential`)
```text
MobileNetV2 (imagenet weights, include_top=False, input_shape=(224, 224, 3))
   │  (Output: None, 7, 7, 1280 | 2,257,984 params)
   ▼
GlobalAveragePooling2D ()
   │  (Output: None, 1280)
   ▼
Dense (256, activation='relu') ──► Dropout (0.5)
   │  (327,936 params)
   ▼
Dense (128, activation='relu') ──► Dropout (0.3)
   │  (32,896 params)
   ▼
Dense (3, activation='softmax')
      (Output: 3 Classes | 387 params)
