import { useState } from 'react';
import DrawingCanvas from './components/DrawingCanvas';
import NeuralScene from './components/NeuralScene';
import axios from 'axios';
import './App.css';

interface NetworkResponse {
    input_layer: number[];
    hidden_layer1: number[];
    hidden_layer2: number[];
    hidden_layer3: number[];
    output_layer: number[];
    prediction: string | number;
    confidence: number;
}

function App() {
    const [networkData, setNetworkData] = useState<NetworkResponse | null>(null);
    const [lastPixels, setLastPixels] = useState<number[] | null>(null);
    const [isCorrecting, setIsCorrecting] = useState(false);
    const [notification, setNotification] = useState<{ message: string; type: 'success' | 'error' } | null>(null);

    const showNotification = (message: string, type: 'success' | 'error' = 'success') => {
        setNotification({ message, type });
        setTimeout(() => setNotification(null), 3000);
    };

    const handleFeedback = async (correctLabel: number) => {
        if (!lastPixels) return;

        try {
            const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
            await axios.post(`${API_BASE_URL}/feedback`, {
                pixels: lastPixels,
                correct_label: correctLabel
            });
            showNotification("Brain updated! Run prediction again to see the change.", 'success');
            setIsCorrecting(false);
        } catch (err) {
            console.error("Feedback failed", err);
            showNotification("Failed to update brain. Try again.", 'error');
        }
    };

    return (
        <div className="App">
            {notification && (
                <div className={`notification ${notification.type}`}>
                    {notification.message}
                </div>
            )}
            <div className="dashboard">
                <div className="input-section">
                    <h1>Neural Vision 3D</h1>
                    <DrawingCanvas
                        onPrediction={(data: NetworkResponse, pixels: number[]) => {
                            setNetworkData(data);
                            setLastPixels(pixels);
                        }}
                        onClear={() => {
                            setNetworkData(null);
                            setLastPixels(null);
                            setIsCorrecting(false);
                        }}
                    />

                    {networkData && (
                        <div className="prediction-box">
                            <h2 className="guess-text">
                                {networkData.prediction === "Not a Number"
                                    ? "Not a Number"
                                    : `Guess: ${networkData.prediction}`}
                            </h2>

                            <div className="confidence-container">
                                <div className="confidence-label">
                                    Confidence: {(networkData.confidence * 100).toFixed(1)}%
                                </div>
                                <div className="meter-bg">
                                    <div
                                        className="meter-fill"
                                        style={{
                                            width: `${networkData.confidence * 100}%`,
                                            backgroundColor: networkData.prediction === "Not a Number" ? "#ff4444" : "#4facfe"
                                        }}
                                    ></div>
                                </div>
                            </div>

                            <div className="feedback-section">
                                {!isCorrecting ? (
                                    <button className="wrong-btn" onClick={() => setIsCorrecting(true)}>
                                        Wrong? Correct it
                                    </button>
                                ) : (
                                    <div className="correction-ui">
                                        <p>What was it?</p>
                                        <div className="digit-grid">
                                            {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((num) => (
                                                <button
                                                    key={num}
                                                    onClick={() => handleFeedback(num)}
                                                    className="digit-btn"
                                                >
                                                    {num === 10 ? "NaN" : num}
                                                </button>
                                            ))}
                                        </div>
                                        <button className="cancel-btn" onClick={() => setIsCorrecting(false)}>
                                            Cancel
                                        </button>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                </div>
                <div className="visualizer-section">
                    <NeuralScene networkData={networkData} />
                </div>
            </div>
        </div>
    );
}

export default App;