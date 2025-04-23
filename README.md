# 2N2D - Neural Network Development Dashboard

**2N2D** is an interactive platform to load, visualize, optimize, and export neural network architectures using ONNX and PyTorch, with a web-based dashboard built in Next.js.

---

## Features

- Upload and inspect ONNX models
- Load and analyze CSV datasets
- Automatically optimize model architectures
- Export optimized models back to ONNX
- Beautiful, responsive UI powered by Next.js

## Project Structure

2n2d/ ├── backend/ # FastAPI + PyTorch + ONNX │
├── 2n2d.py # Core logic (model loading, optimization)
│
├── 2n2denp.py # FastAPI server with API endpoints
│ └── requirements.txt
├── frontend/ # Next.js frontend interface
│ └── (standard Next.js structure)
└── README.md

## API Overview

### Endpoint Method Description

`/test` - GET - Test server connection
`/upload-model` - POST - Upload ONNX model (binary file)
`/upload-csv` - POST - Upload CSV data (binary file)
`/optimize` - POST - Run optimization (JSON body)
`/download-optimized` - GET Download best model (base64 ONNX)

## Project Setup

### 1. Backend Setup (Python)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
pip install -r requirements.txt
```

## 2. Frontend Setup (Next.js)

```bash
Copy
Edit
cd frontend
npm install
npm run dev
```
