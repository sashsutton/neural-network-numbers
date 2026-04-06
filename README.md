# Neural Network — Handwritten Digit Recognition

An interactive educational tool that visualises a deep neural network recognising hand-drawn digits in real time. The network is built from scratch with NumPy and trained on the MNIST dataset.

![Python](https://img.shields.io/badge/python-3.11-lightgrey.svg)
![React](https://img.shields.io/badge/react-19-lightgrey.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

**Live demo:** [neural-network-numbers.vercel.app](https://neural-network-numbers.vercel.app/)

---

## What it does

Draw any digit (0–9) on the canvas. The signal propagates through four layers of neurons, and you can watch the activations light up in a live 3D scene. If the network guesses wrong, you can correct it — it will backpropagate and update its weights immediately.

- Neural network built with NumPy only (no PyTorch, no Keras for inference)
- Trained on 60,000 MNIST examples + synthetic "Not a Number" samples
- Forward pass, backpropagation, and weight persistence all implemented by hand
- 3D activation visualiser built with Three.js / React Three Fiber
- Online learning: corrections from the user update the model in real time

---

## Architecture

### Current: 784 → 512 → 256 → 128 → 11

| # | Layer | Neurons | Activation | Role |
|---|-------|---------|------------|------|
| 00 | Input | 784 | — | One neuron per pixel of the 28×28 image, normalised to [0, 1] |
| 01 | Hidden 1 | 512 | ReLU | Detects low-level features: edges, curves, stroke orientations |
| 02 | Hidden 2 | 256 | ReLU | Combines features into patterns: loops, serifs, verticals |
| 03 | Hidden 3 | 128 | ReLU | Compresses into the most discriminative features per digit class |
| 04 | Output | 11 | Softmax | Probability distribution over 10 digits + "Not a Number" |

**~600K parameters.** Trained with He initialisation, batch SGD (batch size 32), learning rate 0.003, L2 regularisation (λ=0.001), for 25 epochs.

### Why four layers?

The architecture arrived at empirically through iteration:

**v1 — One hidden layer (784 → 128 → 11)**
A single hidden layer can only learn a flat mapping from pixels to classes. It has no capacity for hierarchical reasoning: it tries to go directly from raw pixel values to digit identity without any intermediate feature extraction. Results were poor — the network struggled to distinguish visually similar digits (3 vs 8, 4 vs 9, 1 vs 7).

**v2 — Two hidden layers (784 → 256 → 128 → 11)**
Adding a second layer introduced one level of abstraction, which helped. The first layer could learn edge-like features; the second could begin combining them. Performance improved noticeably but still fell short on ambiguous or messily-drawn digits.

**v3 — Four layers / three hidden (784 → 512 → 256 → 128 → 11)**
The current architecture. A third hidden layer gives the network the capacity to build a proper three-level hierarchy: low-level features (edges, curves) → mid-level patterns (loops, strokes, corners) → high-level digit representations. This is where performance stabilised.

### Is this architecture optimal?

For a fully-connected network on 28×28 inputs, three hidden layers is the practical ceiling. Going deeper adds parameters without proportional gain — the task is not complex enough to justify a fifth or sixth layer, and gradient flow becomes harder to manage.

If the goal were higher accuracy rather than interpretability, the improvements that would matter most are:
- **Dropout** (0.3–0.5 between hidden layers) — reduces overfitting on ambiguous samples
- **Batch normalisation** — stabilises training and allows a higher learning rate
- **Learning rate scheduling** — cosine annealing or step decay over 25 epochs

These are regularisation improvements, not architectural ones. The depth is right.

### A note on the "Not a Number" class

The 11th output neuron represents inputs that aren't digits — random noise, non-digit shapes, blank canvases. It is trained on synthetic data: random noise images and inverted MNIST samples.

This class does not benefit from additional network depth. More layers would give the whole network more capacity, but NaN recognition is not a depth problem — it is a **training data** problem. The network can only recognise "not a digit" patterns it has seen during training. A more robust approach would be to use a confidence threshold: if `max(softmax) < threshold`, output NaN, regardless of which neuron wins.

---

## Tech stack

### Backend
- **Python 3.11** — core inference and training logic
- **FastAPI** — prediction and feedback API endpoints
- **NumPy** — all matrix mathematics (forward pass, backpropagation, weight updates)
- **TensorFlow** — used only at training time to load the MNIST dataset

### Frontend
- **React 19 + TypeScript** — application framework
- **Vite** — build tool and dev server
- **Three.js / React Three Fiber** — 3D neural activation visualiser
- **Axios** — API communication

---

## Project structure

```
neural-network-numbers/
├── backend/
│   ├── brain.py            # NeuralNetwork class: forward pass, backpropagation
│   ├── main.py             # FastAPI server (/predict, /feedback endpoints)
│   ├── train.py            # Training script (MNIST + NaN data, He init, batch SGD)
│   ├── weights.npz         # Saved model weights (784×512×256×128×11)
│   └── requirements.txt
└── frontend/
    └── src/
        ├── components/
        │   ├── sections/
        │   │   ├── Hero.tsx          # Landing section with architecture preview
        │   │   ├── Architecture.tsx  # Layer breakdown and training specs
        │   │   ├── HowItWorks.tsx    # Step-by-step process explainer
        │   │   └── Playground.tsx    # Interactive canvas + 3D visualiser
        │   ├── Nav.tsx               # Navigation
        │   ├── DrawingCanvas.tsx     # Drawing input and image preprocessing
        │   └── NeuralScene.tsx       # Three.js activation scene
        ├── App.tsx
        └── App.css
```

---

## Setup

### Backend

**Requirements:** Python 3.11, pip

```bash
cd backend
python -m venv venv
source venv/bin/activate       # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

If `weights.npz` is not present, train the model first:

```bash
python train.py
```

Start the API server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Frontend

**Requirements:** Node.js v18+

```bash
cd frontend
npm install
```

Create a `.env` file in the `frontend/` directory:

```
VITE_API_URL=http://127.0.0.1:8000
```

Start the dev server:

```bash
npm run dev
```

Build for production:

```bash
npm run build
```

---

## How the forward pass works

Each layer applies: **a⁽ˡ⁾ = ReLU(W⁽ˡ⁾ · a⁽ˡ⁻¹⁾ + b⁽ˡ⁾)**

The output layer uses Softmax: **ŷ = Softmax(W⁽⁴⁾ · a⁽³⁾ + b⁽⁴⁾)**

The prediction is `argmax(ŷ)`. When a user submits a correction, the network runs a full forward pass with the correct label as the target, computes the cross-entropy loss, and backpropagates through all four layers via the chain rule. Updated weights are saved to `weights.npz`.
