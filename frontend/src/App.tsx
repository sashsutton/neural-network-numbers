import { useState } from 'react';
import DrawingCanvas from './components/DrawingCanvas';
import NeuralScene from './components/NeuralScene';
import './App.css';

interface NetworkResponse {
    input_layer: number[];
    hidden_layer: number[];
    output_layer: number[];
    prediction: number;
    confidence: number;
}

function App() {
    const [networkData, setNetworkData] = useState<NetworkResponse | null>(null);

    return (
        <div className="App">
            <div className="dashboard">
                <div className="input-section">
                    <h1>Neural Vision</h1>
                    <DrawingCanvas
                        onPrediction={(data: NetworkResponse) => setNetworkData(data)}
                        onClear={() => setNetworkData(null)}
                    />

                    {networkData && (
                        <div className="prediction-box">
                            <h2 className="guess-text">Guess: {networkData.prediction}</h2>

                            {/* Confidence Meter */}
                            <div className="confidence-container">
                                <div className="confidence-label">
                                    Confidence: {(networkData.confidence * 100).toFixed(1)}%
                                </div>
                                <div className="meter-bg">
                                    <div
                                        className="meter-fill"
                                        style={{ width: `${networkData.confidence * 100}%` }}
                                    ></div>
                                </div>
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