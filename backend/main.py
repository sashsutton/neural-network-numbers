from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from brain import NeuralNetwork

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

if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    uvicorn.run(app, host="127.0.0.1", port=8000)