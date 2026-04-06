import { useState } from 'react';
import DrawingCanvas from '../DrawingCanvas';
import NeuralScene from '../NeuralScene';
import axios from 'axios';

interface NetworkResponse {
    input_layer: number[];
    hidden_layer1: number[];
    hidden_layer2: number[];
    hidden_layer3: number[];
    output_layer: number[];
    prediction: string | number;
    confidence: number;
}

const Playground = () => {
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
                correct_label: correctLabel,
            });
            showNotification('Weights updated. Draw again to see the change.', 'success');
            setIsCorrecting(false);
        } catch {
            showNotification('Failed to update weights. Try again.', 'error');
        }
    };

    const isNaN = networkData?.prediction === 'Not a Number';
    const confidence = networkData ? Math.round(networkData.confidence * 100) : 0;

    return (
        <section className="section" id="playground">
            {notification && (
                <div className={`toast toast-${notification.type}`}>
                    {notification.message}
                </div>
            )}
            <div className="section-inner">
                <div className="section-header">
                    <span className="section-tag">Interactive</span>
                    <h2 className="section-title">Playground</h2>
                    <p className="section-desc">
                        Draw a digit and watch activations propagate through the network in real time.
                        Rotate the 3D view to explore every layer.
                    </p>
                </div>

                <div className="playground-grid">
                    {/* Left panel */}
                    <div className="playground-left">
                        <div className="panel">
                            <div className="panel-header">
                                <span className="panel-title">Canvas</span>
                                <span className="panel-hint">Draw a digit 0–9</span>
                            </div>
                            <DrawingCanvas
                                onPrediction={(data: NetworkResponse, pixels: number[]) => {
                                    setNetworkData(data);
                                    setLastPixels(pixels);
                                    setIsCorrecting(false);
                                }}
                                onClear={() => {
                                    setNetworkData(null);
                                    setLastPixels(null);
                                    setIsCorrecting(false);
                                }}
                            />
                        </div>

                        {networkData && (
                            <div className="result-panel">
                                <div className="result-top">
                                    <div className="result-prediction">
                                        <span className="result-label">Prediction</span>
                                        <span className={`result-value ${isNaN ? 'result-nan' : ''}`}>
                                            {isNaN ? 'Not a Number' : networkData.prediction}
                                        </span>
                                    </div>
                                    <div className="result-confidence">
                                        <span className="result-label">Confidence</span>
                                        <span className="result-pct">{confidence}%</span>
                                    </div>
                                </div>
                                <div className="conf-bar-bg">
                                    <div
                                        className="conf-bar-fill"
                                        style={{
                                            width: `${confidence}%`,
                                            background: isNaN ? 'var(--red)' : 'var(--accent)',
                                        }}
                                    />
                                </div>

                                <div className="feedback-row">
                                    {!isCorrecting ? (
                                        <button className="btn-ghost-sm" onClick={() => setIsCorrecting(true)}>
                                            Wrong prediction? Correct it
                                        </button>
                                    ) : (
                                        <div className="correction-panel">
                                            <span className="correction-label">Select the correct digit</span>
                                            <div className="digit-grid">
                                                {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((n) => (
                                                    <button key={n} className="digit-btn" onClick={() => handleFeedback(n)}>
                                                        {n}
                                                    </button>
                                                ))}
                                                <button className="digit-btn digit-nan" onClick={() => handleFeedback(10)}>
                                                    NaN
                                                </button>
                                            </div>
                                            <button className="btn-ghost-sm cancel" onClick={() => setIsCorrecting(false)}>
                                                Cancel
                                            </button>
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}

                        {!networkData && (
                            <div className="empty-state">
                                <span>Draw something, then run a prediction</span>
                            </div>
                        )}
                    </div>

                    {/* Right panel — 3D scene */}
                    <div className="playground-right">
                        <div className="panel scene-panel">
                            <div className="panel-header">
                                <span className="panel-title">Network Activations</span>
                                <span className="panel-hint">Drag to rotate · Scroll to zoom</span>
                            </div>
                            <div className="scene-container">
                                <NeuralScene networkData={networkData} />
                            </div>
                        </div>

                        <div className="legend">
                            {[
                                { color: '#E0E0E0', label: 'Input (784)' },
                                { color: '#00CFCF', label: 'Hidden 1 (512)' },
                                { color: '#3B82F6', label: 'Hidden 2 (256)' },
                                { color: '#5B8DEF', label: 'Hidden 3 (128)' },
                                { color: '#A855F7', label: 'Output (11)' },
                            ].map((item) => (
                                <div key={item.label} className="legend-item">
                                    <span className="legend-dot" style={{ background: item.color }} />
                                    <span>{item.label}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
};

export default Playground;
