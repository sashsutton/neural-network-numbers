from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from brain import NeuralNetwork  # <--- Import your class!

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],             # Allow all origins (Vite usually runs on 5173)
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"], # Allow all methods (GET, POST, OPTIONS)
    allow_headers=["*"],
)

# Initialise and load the brain once when the server starts
brain = NeuralNetwork()
brain.load_weights("weights.npz")

class DigitData(BaseModel):
    pixels: List[float]

@app.post("/predict")
async def predict(data: DigitData):
    result = brain.forward_pass(data.pixels)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)