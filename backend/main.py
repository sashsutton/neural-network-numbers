import os
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import List
from brain import NeuralNetwork
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ──────────────────────────────────────────────────────────────────────
# In production, restrict to your Vercel URL via the ALLOWED_ORIGIN env variable.
# Locally, both localhost ports are allowed by default.
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ── Model ─────────────────────────────────────────────────────────────────────
brain = NeuralNetwork()
brain.load_weights("weights.npz")

# ── Request schemas ───────────────────────────────────────────────────────────

class DigitData(BaseModel):
    pixels: List[float]

    @field_validator("pixels")
    @classmethod
    def validate_pixels(cls, v):
        if len(v) != 784:
            raise ValueError(f"Expected 784 pixel values, got {len(v)}.")
        if any(p < 0.0 or p > 1.0 for p in v):
            raise ValueError("All pixel values must be in the range [0, 1].")
        return v


class FeedbackData(BaseModel):
    pixels: List[float]
    correct_label: int

    @field_validator("pixels")
    @classmethod
    def validate_pixels(cls, v):
        if len(v) != 784:
            raise ValueError(f"Expected 784 pixel values, got {len(v)}.")
        if any(p < 0.0 or p > 1.0 for p in v):
            raise ValueError("All pixel values must be in the range [0, 1].")
        return v

    @field_validator("correct_label")
    @classmethod
    def validate_label(cls, v):
        if v < 0 or v > 10:
            raise ValueError("correct_label must be between 0 and 10.")
        return v


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {"status": "Neural Network API is live."}


@app.post("/predict")
@limiter.limit("30/minute")
async def predict(request: Request, data: DigitData):
    result = brain.forward_pass(data.pixels)
    return result


@app.post("/feedback")
@limiter.limit("10/minute")
async def feedback(request: Request, data: FeedbackData):
    pixels = np.array(data.pixels).reshape(784, 1)
    correct_label = data.correct_label

    # Reload to ensure we have the latest trained state
    brain.load_weights("weights.npz")

    target = np.zeros((11, 1))
    target[correct_label] = 1

    # Forward pass (4 layers)
    z1 = np.dot(brain.W1, pixels) + brain.b1
    a1 = np.maximum(0, z1)
    z2 = np.dot(brain.W2, a1) + brain.b2
    a2 = np.maximum(0, z2)
    z3 = np.dot(brain.W3, a2) + brain.b3
    a3 = np.maximum(0, z3)
    z4 = np.dot(brain.W4, a3) + brain.b4
    a4 = brain.softmax(z4)

    # Backpropagation (4 layers)
    dz4 = a4 - target
    dW4 = np.dot(dz4, a3.T)

    dz3 = np.dot(brain.W4.T, dz4) * (z3 > 0)
    dW3 = np.dot(dz3, a2.T)

    dz2 = np.dot(brain.W3.T, dz3) * (z2 > 0)
    dW2 = np.dot(dz2, a1.T)

    dz1 = np.dot(brain.W2.T, dz2) * (z1 > 0)
    dW1 = np.dot(dz1, pixels.T)

    # Weight updates
    lr = 0.001
    brain.W4 -= lr * dW4
    brain.W3 -= lr * dW3
    brain.W2 -= lr * dW2
    brain.W1 -= lr * dW1

    brain.b4 -= lr * dz4
    brain.b3 -= lr * dz3
    brain.b2 -= lr * dz2
    brain.b1 -= lr * dz1

    np.savez("weights.npz",
             W1=brain.W1, b1=brain.b1,
             W2=brain.W2, b2=brain.b2,
             W3=brain.W3, b3=brain.b3,
             W4=brain.W4, b4=brain.b4)

    return {"status": "Weights updated."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
