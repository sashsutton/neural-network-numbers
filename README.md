# Neural Network — Handwritten Digit Recognition

An interactive educational tool built to make neural networks tangible. Draw a digit, watch the signal travel through every neuron in a live 3D scene, and correct the network when it gets something wrong — it learns from your feedback on the spot.

The neural network is written from scratch using only NumPy. No PyTorch, no TensorFlow for inference — just matrix multiplication and calculus, implemented by hand.

![Python](https://img.shields.io/badge/python-3.11-lightgrey.svg)
![React](https://img.shields.io/badge/react-19-lightgrey.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

**Live demo:** [neural-network-numbers.vercel.app](https://neural-network-numbers.vercel.app/)

---

## Table of Contents

1. [What This Project Is](#what-this-project-is)
2. [How It Works, End to End](#how-it-works-end-to-end)
3. [Neural Network Architecture](#neural-network-architecture)
4. [Training](#training)
5. [API Reference](#api-reference)
6. [Tech Stack](#tech-stack)
7. [Project Structure](#project-structure)
8. [Running Locally](#running-locally)
9. [Deployment](#deployment)
10. [Author](#author)

---

## What This Project Is

Most neural network tutorials show you the code and the accuracy score. This project shows you the network *thinking* — every neuron, every layer, every activation value — as you draw.

You draw a digit on a canvas. The app sends your drawing to a Python backend, which runs it through the neural network layer by layer. The frontend then renders all of those activation values as glowing spheres in a 3D scene. The brighter a neuron, the more strongly it fired. The connections between layers light up to show where the signal flowed.

If the network gets the answer wrong, you can tell it the correct digit. It will immediately run backpropagation, update its weights, and save the new model — all in real time.

The goal is to make the concepts of forward passes, activation functions, and backpropagation something you can *see*, not just read about.

---

## How It Works, End to End

### 1. Drawing and pre-processing

The drawing canvas is 280×280 pixels. When you hit "Run Prediction", the app:

- Finds the bounding box of what you drew (ignoring blank edges)
- Crops and centres the drawing
- Scales it down to exactly 28×28 pixels
- Converts each pixel to a value between 0 (black) and 1 (white)
- Flattens the 28×28 grid into a list of 784 numbers

This matches the format of the MNIST dataset the network was trained on.

### 2. The forward pass

Those 784 numbers are sent to the Python backend as a POST request. The network processes them layer by layer:

```
Input (784)  →  Hidden 1 (512)  →  Hidden 2 (256)  →  Hidden 3 (128)  →  Output (11)
```

At each hidden layer, every neuron computes:

```
z = (weights × previous activations) + bias
activation = ReLU(z)      →     max(0, z)
```

ReLU simply means "if the value is negative, set it to zero." This non-linearity is what allows the network to learn complex patterns rather than just fitting straight lines through data.

At the output layer, Softmax is used instead of ReLU:

```
output = Softmax(z)       →     converts raw scores into probabilities that sum to 1
```

The result is 11 probabilities — one for each digit (0–9) and one for "Not a Number". The highest probability is the prediction.

### 3. Visualising activations

The backend returns every layer's activation values alongside the prediction. The frontend uses these to set the brightness of each neuron sphere in the 3D scene. A neuron with an activation of 0 is dim. One with a high activation glows brightly. You can rotate and zoom the scene freely to explore what the network is doing.

### 4. Correcting the network

If the prediction is wrong, you select the correct digit. The app sends your drawing and the correct label back to the backend, which:

1. Runs a full forward pass
2. Computes the cross-entropy loss between what the network predicted and what you said it should have been
3. Runs backpropagation — tracing the error backwards through all four layers and computing how much each weight contributed to the mistake
4. Updates every weight and bias using gradient descent
5. Saves the updated weights to `weights.npz`

The next time you draw, the network has already incorporated your correction.

---

## Neural Network Architecture

### Current architecture: `784 → 512 → 256 → 128 → 11`

| # | Layer | Neurons | Activation | What it does |
|---|-------|---------|------------|--------------|
| 00 | Input | 784 | — | One neuron per pixel of the 28×28 image. Values are normalised to [0, 1]. |
| 01 | Hidden 1 | 512 | ReLU | Learns low-level features: edges, curves, stroke directions, and corners. |
| 02 | Hidden 2 | 256 | ReLU | Combines those features into structural patterns: loops, crossbars, vertical lines. |
| 03 | Hidden 3 | 128 | ReLU | Compresses the representation into the most distinctive features for each digit class. |
| 04 | Output | 11 | Softmax | Outputs a probability for each of the 10 digits plus a "Not a Number" class. |

**Total parameters:** ~600,000 (weights + biases across all layers)

The shape follows a classic **funnel pattern**: wide at the start to capture many possible features, narrowing through each layer to force the network to generalise and compress. This is not arbitrary — if the first hidden layer were too narrow, it would bottleneck feature detection before any abstraction had taken place.

---

### Why four layers? The evolution of this architecture

The architecture was not designed upfront — it was discovered through iteration and failure.

**Version 1 — One hidden layer: `784 → 128 → 11`**

The first version had a single hidden layer with 128 neurons. In theory, a single hidden layer with enough neurons can approximate any function (the Universal Approximation Theorem). In practice, it struggled badly.

The problem is that one layer has to do everything at once: detect edges, understand their arrangement, recognise digit-level structure, and produce a classification — all in a single step. It has no capacity to build up a hierarchical understanding. The result was poor accuracy, especially on digits that look similar: 3 vs 8, 4 vs 9, 1 vs 7.

**Version 2 — Two hidden layers: `784 → 256 → 128 → 11`**

Adding a second hidden layer introduced one stage of abstraction between raw pixels and classification. The first layer could learn edge-like filters; the second could begin combining them. Accuracy improved noticeably, but the network still made avoidable mistakes on messy or ambiguous handwriting — the kind of thing a human would get right without thinking.

**Version 3 — Three hidden layers: `784 → 512 → 256 → 128 → 11`** *(current)*

Adding a third hidden layer — and widening the first from 256 to 512 neurons — created the capacity for a proper three-level hierarchy:

- **Layer 1 (512):** detects low-level features — individual edges, pixel intensity gradients, stroke directions
- **Layer 2 (256):** assembles those into mid-level patterns — loops, curves, corners, line junctions
- **Layer 3 (128):** builds digit-level representations — "this looks like it has a loop at the top and a tail at the bottom, so probably a 9"

This is where performance stabilised. The network generalises well across different handwriting styles.

---

### Is this architecture optimal?

For a **fully-connected** (dense) network on 28×28 inputs, three hidden layers is the practical ceiling. Beyond this point, adding more layers brings diminishing returns — the problem is not complex enough to justify the additional depth, and training becomes harder as gradients have further to travel during backpropagation.

The improvements that would meaningfully raise performance from here are not architectural — they are **regularisation** techniques:

| Technique | What it does | Expected benefit |
|-----------|-------------|-----------------|
| **Dropout** (p=0.3–0.5) | Randomly disables neurons during training, forcing redundancy and preventing over-reliance on any one path | Reduces overfitting on ambiguous samples |
| **Batch Normalisation** | Normalises layer inputs during training to keep activations in a healthy range | Stabilises training, allows a higher learning rate |
| **Learning rate scheduling** | Reduces the learning rate gradually over epochs rather than holding it constant | Better convergence towards the end of training |

These are potential future improvements. The current architecture is intentionally kept simple for readability and educational value.

---

### A note on the "Not a Number" class

The 11th output neuron handles inputs that are not digits — blank canvases, random scribbles, non-digit shapes. It is trained on synthetic data: 10,000 samples of low-intensity random noise, and 10,000 inverted MNIST images.

Importantly, this class does **not** benefit from additional network depth. Adding a fifth layer would increase capacity across the whole network, but NaN detection is not a capacity problem — it is a **training data problem**. The network can only classify something as "not a digit" if it has encountered similar-looking non-digit inputs during training.

A more robust approach for future iterations would be a **confidence threshold**: if the highest Softmax output falls below some threshold (say, 0.6), output "Not a Number" — regardless of which neuron technically won. This would catch ambiguous inputs the network has never seen, rather than confidently misclassifying them as the closest-looking digit.

---

## Training

Training is handled by `backend/train.py` and only needs to be run once to generate `weights.npz`. The pre-trained weights file is already included in the repository.

### Dataset

- **MNIST digits:** 60,000 training images of handwritten digits (0–9), each 28×28 greyscale pixels
- **Synthetic NaN samples:** 10,000 low-intensity random noise images + 10,000 inverted MNIST images, all labelled as class 10 ("Not a Number")
- **Total training samples:** 80,000

### Hyperparameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| Epochs | 25 | Sufficient for convergence on this dataset size without overfitting |
| Batch size | 32 | Small enough to keep gradient updates noisy (regularising effect), large enough to be efficient |
| Learning rate | 0.003 | Slightly conservative for a 4-layer network — prevents oscillation in deeper layers |
| L2 regularisation | λ = 0.001 | Adds a small penalty for large weights, discouraging over-reliance on individual features |
| Weight initialisation | He initialisation | Designed specifically for ReLU networks. Initialises weights scaled by `sqrt(2 / fan_in)` to prevent vanishing or exploding activations at the start of training |

### What He initialisation does

Standard random initialisation often causes training to stall in deep networks. If weights start too small, activations shrink to near-zero by the time they reach the later layers (vanishing gradients). If they start too large, activations explode. He initialisation sets the starting scale specifically for ReLU networks, keeping variance consistent layer to layer:

```python
W = np.random.randn(layer_out, layer_in) * np.sqrt(2.0 / layer_in)
```

### Backpropagation

The training loop implements full backpropagation by hand using the chain rule. For the output layer (cross-entropy + Softmax):

```
dL/dz4 = a4 - y        (predicted probabilities minus one-hot true labels)
```

For each hidden layer (working backwards):

```
dL/dz = (W_next^T · dL/dz_next) * ReLU'(z)
```

Where `ReLU'(z) = 1 if z > 0, else 0`. Weights and biases are then updated via gradient descent:

```
W -= learning_rate * (dL/dW + λ * W)
b -= learning_rate * dL/db
```

The L2 term `λ * W` is the regularisation penalty, added directly to the weight gradient.

---

## API Reference

The FastAPI backend exposes two endpoints. Both accept and return JSON.

### `POST /predict`

Runs a forward pass and returns all layer activations plus the prediction.

**Request body:**
```json
{
  "pixels": [0.0, 0.12, 0.95, "..."]
}
```
`pixels` is a flat array of 784 floats in the range [0, 1].

**Response:**
```json
{
  "input_layer":   ["..."],
  "hidden_layer1": ["..."],
  "hidden_layer2": ["..."],
  "hidden_layer3": ["..."],
  "output_layer":  ["..."],
  "prediction":    "7",
  "confidence":    0.9823
}
```
`prediction` is a string — either a digit `"0"`–`"9"` or `"Not a Number"`.  
`confidence` is the Softmax probability of the winning class, between 0.0 and 1.0.

---

### `POST /feedback`

Accepts a correction, runs backpropagation, and saves the updated weights.

**Request body:**
```json
{
  "pixels":        [0.0, 0.12, 0.95, "..."],
  "correct_label": 7
}
```
`correct_label` is an integer 0–9 for digits, or 10 for "Not a Number".

**Response:**
```json
{
  "status": "ok"
}
```

---

## Tech Stack

### Backend

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11 | Core language |
| FastAPI | 0.109 | REST API server |
| Uvicorn | 0.27 | ASGI server (runs FastAPI) |
| NumPy | 1.26 | All neural network mathematics |
| TensorFlow | 2.15 | Used **only** during training to download the MNIST dataset |
| Pydantic | 2.5 | Request validation |

TensorFlow is listed as a dependency solely because it provides a convenient MNIST loader. It plays no role in inference — the forward pass and backpropagation are implemented entirely in NumPy.

### Frontend

| Tool | Version | Purpose |
|------|---------|---------|
| React | 19 | UI framework |
| TypeScript | 5.9 | Type safety |
| Vite | 7 | Build tool and dev server |
| Three.js / React Three Fiber | latest | 3D neural activation visualiser |
| Drei | latest | Three.js helpers (OrbitControls, Stars, etc.) |
| Axios | 1.13 | HTTP client for API calls |

---

## Project Structure

```
neural-network-numbers/
│
├── backend/
│   ├── brain.py            # NeuralNetwork class — forward pass and backpropagation
│   ├── main.py             # FastAPI app — /predict and /feedback endpoints
│   ├── train.py            # Training script — run once to produce weights.npz
│   ├── weights.npz         # Pre-trained model weights (included, ~4.5 MB)
│   └── requirements.txt    # Python dependencies
│
└── frontend/
    ├── index.html
    ├── vite.config.ts
    ├── package.json
    └── src/
        ├── App.tsx                         # Root component — composes all sections
        ├── App.css                         # All styles
        ├── index.css                       # Global reset and CSS variables
        ├── main.tsx                        # React entry point
        └── components/
            ├── Nav.tsx                     # Sticky navigation bar
            ├── DrawingCanvas.tsx           # Canvas drawing + image pre-processing
            ├── NeuralScene.tsx             # Three.js 3D activation visualiser
            └── sections/
                ├── Hero.tsx                # Landing section with architecture diagram
                ├── Architecture.tsx        # Layer breakdown and training specs
                ├── HowItWorks.tsx          # Step-by-step process explainer
                └── Playground.tsx          # Interactive canvas + prediction + 3D scene
```

### Key files explained

**`backend/brain.py`** — The neural network class. Loads weights from `weights.npz`, exposes a `forward_pass` method that returns all layer activations, and handles the online learning feedback called from `main.py`.

**`backend/train.py`** — Standalone training script. Downloads MNIST, generates NaN samples, initialises weights with He initialisation, and runs batch SGD with backpropagation for 25 epochs. Only needs to be run if you want to retrain the model from scratch.

**`frontend/src/components/DrawingCanvas.tsx`** — Manages the 280×280 drawing canvas. When a prediction is requested, it finds the bounding box of the drawing, crops, centres, scales to 28×28, normalises pixel values, and sends the flattened array to the backend.

**`frontend/src/components/NeuralScene.tsx`** — Three.js scene. Renders each layer's neurons as spheres arranged in a grid, with brightness proportional to activation strength. Connection lines are drawn between the most active neurons across adjacent layers (top-K filtering to avoid visual clutter). Supports free orbit, zoom, and pan.

---

## Running Locally

You will need two terminals — one for the backend, one for the frontend.

### Prerequisites

- Python 3.11 with conda (recommended) or pip
- Node.js v18 or higher

---

### Backend setup

**Option A — Conda (recommended)**

```bash
conda create -n py-ai-3.11 python=3.11
conda activate py-ai-3.11
cd backend
pip install -r requirements.txt
```

**Option B — pip venv**

```bash
cd backend
python3.11 -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

**Start the API server:**

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. You can visit `http://localhost:8000/docs` for the auto-generated Swagger UI.

**If `weights.npz` is missing** (e.g. after a fresh clone where the file was not committed), retrain the model:

```bash
python train.py
# Takes approximately 5–10 minutes depending on hardware
```

---

### Frontend setup

In a second terminal:

```bash
cd frontend
npm install
```

Create a `.env` file inside the `frontend/` directory:

```
VITE_API_URL=http://127.0.0.1:8000
```

Start the dev server:

```bash
npm run dev
```

Open `http://localhost:5173` in your browser.

---

### How to use it

1. Draw a digit (0–9) on the black canvas using your mouse
2. Click **Run Prediction**
3. Watch the 3D network light up — drag to rotate, scroll to zoom
4. The prediction and confidence score appear in the left panel
5. If the network is wrong, click **Wrong? Correct it**, select the correct digit, and the network updates immediately
6. Clear the canvas and draw again to see the effect of the updated weights

---

## Deployment

The project is deployed as two separate services:

- **Frontend** → [Vercel](https://vercel.com) — deployed from the `frontend/` directory. Set `VITE_API_URL` as an environment variable in the Vercel project settings, pointing to your backend URL.

- **Backend** → [Render](https://render.com) (or any platform that supports Python). The start command is:
  ```bash
  uvicorn main:app --host 0.0.0.0 --port 8000
  ```
  Ensure `weights.npz` is committed to the repository — the model loads from this file on startup.

---

## Author

**Sasha Sutton**

Built as a personal deep-dive into how neural networks actually work — starting from a single hidden layer that couldn't tell a 3 from an 8, iterating through architectures, implementing backpropagation by hand, and building the tooling to see the network's internal state in real time.
