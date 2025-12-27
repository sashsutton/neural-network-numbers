from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from brain import NeuralNetwork
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

brain = NeuralNetwork()
# Ensure this file exists in the same directory
brain.load_weights("weights.npz")

class DigitData(BaseModel):
    pixels: List[float]

@app.get("/")
def read_root():
    return {"Status": "Neural Network API is Live"}

@app.post("/predict")
async def predict(data: DigitData):
    result = brain.forward_pass(data.pixels)
    return result


@app.post("/feedback")
async def feedback(data: dict):
    pixels = np.array(data['pixels']).reshape(784, 1)
    correct_label = int(data['correct_label'])
    brain.load_weights("weights.npz")

    target = np.zeros((11, 1))
    target[correct_label] = 1

    # Forward
    z1 = np.dot(brain.W1, pixels) + brain.b1
    a1 = np.maximum(0, z1)
    z2 = np.dot(brain.W2, a1) + brain.b2
    a2 = np.maximum(0, z2)
    z3 = np.dot(brain.W3, a2) + brain.b3
    a3 = brain.softmax(z3)

    # Backprop
    dz3 = a3 - target
    dW3 = np.dot(dz3, a2.T)
    dz2 = np.dot(brain.W3.T, dz3) * (z2 > 0)
    dW2 = np.dot(dz2, a1.T)
    dz1 = np.dot(brain.W2.T, dz2) * (z1 > 0)
    dW1 = np.dot(dz1, pixels.T)

    lr = 0.001  # Stable learning rate
    brain.W3 -= lr * dW3
    brain.W2 -= lr * dW2
    brain.W1 -= lr * dW1

    np.savez("weights.npz", W1=brain.W1, b1=brain.b1, W2=brain.W2, b2=brain.b2, W3=brain.W3, b3=brain.b3)
    return {"status": "Deep brain updated!"}



if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    uvicorn.run(app, host="127.0.0.1", port=8000)