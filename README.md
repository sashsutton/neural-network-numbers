# Neural Vision 3D
An interactive 3D visualisation of a Neural Network built from scratch, capable of recognising hand-drawn digits using the MNIST dataset.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11.9-green.svg)
![React](https://img.shields.io/badge/react-19-blue.svg)

---
### 🌐 Live Demo
* **Dashboard:** [LIVE WEBSITE](https://neural-network-numbers.vercel.app/)
---

## Overview
Neural Vision 3D is an educational tool designed to demystify how artificial neurons "think." Users can draw numbers on a digital pad, and in real-time, see how the signals propagate through 784 input neurons, 64 hidden neurons, and 10 output neurons in a fully interactive 3D environment.

### Key Features
- **Neural Network from Scratch**: Built using NumPy (no high-level libraries like PyTorch/Keras for the inference logic).
- **3D Interactive Scene**: Rendered with Three.js (React Three Fiber), allowing users to rotate, zoom, and inspect neural activations.
- **Feedback Loop**: Users can correct the network's predictions, allowing it to learn from mistakes in real-time.
- **Stylized Notifications**: Modern, glassmorphism-styled notifications for user feedback.
- **Confidence Metre**: Visual feedback showing the probability of each prediction.
- **Responsive Dashboard**: A modern "Dark Lab" UI built with React and Vite.

### Architecture
- **Input Layer**: 784 Neurons (28x28 pixels)
- **Hidden Layer**: 128 Neurons (ReLU activation)
- **Output Layer**: 11 Neurons (10 Digits + "Not a Number")

---

## 🛠 Tech Stack

### Backend
- **Python 3.11.9**: Core logic.
- **FastAPI**: High-performance API for handling predictions and feedback.
- **NumPy**: Matrix mathematics for the forward and backward pass.
- **TensorFlow**: (Training only) Used to fetch the MNIST dataset.

### Frontend
- **React + TypeScript**: Application framework.
- **Three.js / React Three Fiber**: 3D rendering engine with optimized rendering.
- **React Three Drei**: Helpers for 3D lines, shapes, and environment.
- **Axios**: API communication.
- **CSS3**: Custom "Neon-Glassmorphism" styling with animations.

---

## 📂 Project Structure

```text
neural-network-numbers/
├── backend/
│   ├── brain.py        # The Neural Network class logic
│   ├── main.py         # FastAPI server & CORS config
│   ├── train.py        # Training script for generating weights
│   ├── weights.npz     # Saved model weights (784x64x10)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── DrawingCanvas.tsx # Drawing logic & image processing
│   │   │   └── NeuralScene.tsx   # 3D visualization logic
│   │   ├── App.tsx               # Main Dashboard layout
│   │   └── App.css               # Modern UI styling
│   └── .env                      # API URL configuration
└── README.md

```
---

## ⚡ Setup & Installation

### 1. Backend Setup (The Brain)
The backend is a FastAPI server that handles the matrix mathematics of the neural network.

**Prerequisites:**
* Python 3.11.9
* Pip (Python package manager)

**Installation Steps:**
1. **Navigate to the directory:**
   ```bash
   cd backend
    ```
2. **Create a Virtual Environment (It's recommended):**
    ```bash
   python -m venv venv
    # Activate on Windows:
    .\venv\Scripts\activate
    # Activate on Mac/Linux:
    source venv/bin/activate
   ```
3. **Install Dependecies:**
    ```bash
   pip install -r requirements.txt
   ```
4. **Prepare the Model Weights: If *weights.npz* is not present, you must train the model once to generate the neural connections:**
    ```bash
   python train.py
   ```
5. **Start the Production Server:**
To run the API in a production-ready state, use **Uvicorn**. This is the same command used by deployment platforms like Render:

    ```bash
    # From inside the /backend folder
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```
### 2. Frontend Setup (The Dashboard)
The frontend is a React application powered by Vite, using Three.js (React Three Fiber) for the neural visualisation.

**Prerequisites:**
* Node.js (v18 or higher)
* npm (Node Package Manager)

**Installation Steps:**
1. **Navigate to the directory:**
   ```bash
   cd frontend
   ```
2. **Install dependencies:**
    ```bash
   npm install
   ```
3. **Configure Environment Variables: Create a file named .env in the root of the frontend folder to point the dashboard to your API:**
    ```text
   VITE_API_URL=[http://127.0.0.1:8000](http://127.0.0.1:8000)
   ```
4. **Start the Development Server:**
    ```bash
   npm run dev
   ```
5. **Build for Production: When you are ready to deploy to Vercel or Netlify, run:**
    ```bash
   npm run build
   ```
---
## 🎮 How to Use
1. **Ensure both the Backend and Frontend are running.**

2. **Open your browser to the frontend URL.**

3. **Draw: Use your mouse to draw a single digit (0-9) on the black canvas.**

4. **Predict: Click the "Run Prediction" button.**

5. **Visualise:**

   - Watch the Input Layer (left) light up based on your pixels.

   - Observe the Hidden Layer (middle) process the features.

   - See the Output Layer (right) identify the digit with a neon highlight.

   - Use your mouse to Rotate and Zoom the 3D scene to inspect individual neurons.

6. **Correct: If the prediction is wrong, click "Wrong? Correct it" to teach the network the right answer!**

