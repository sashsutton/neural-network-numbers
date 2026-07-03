import { useState } from 'react';
import DrawingCanvas from './DrawingCanvas';
import NeuralScene from './NeuralScene';
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

const LEGEND = [
    { color: '#e8e1cd', label: 'input, 784' },
    { color: '#ffc46b', label: 'hidden 1, 512' },
    { color: '#f0973a', label: 'hidden 2, 256' },
    { color: '#d97728', label: 'hidden 3, 128' },
    { color: '#ff6a3d', label: 'output, 11' },
];

const Playground = () => {
    const [networkData, setNetworkData] = useState<NetworkResponse | null>(null);
    const [lastPixels, setLastPixels] = useState<number[] | null>(null);
    const [isCorrecting, setIsCorrecting] = useState(false);
    const [notification, setNotification] = useState<{ message: string; type: 'success' | 'error' } | null>(null);

    const showNotification = (message: string, type: 'success' | 'error' = 'success') => {
        setNotification({ message, type });
        setTimeout(() => setNotification(null), 3500);
    };

    const handleFeedback = async (correctLabel: number) => {
        if (!lastPixels) return;
        try {
            const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
            await axios.post(`${API_BASE_URL}/feedback`, {
                pixels: lastPixels,
                correct_label: correctLabel,
            });
            showNotification('Noted. The weights have been rewritten; draw it again and see.');
            setIsCorrecting(false);
        } catch {
            showNotification('The correction never arrived. Try once more.', 'error');
        }
    };

    const isNaN = networkData?.prediction === 'Not a Number';
    const confidence = networkData ? Math.round(networkData.confidence * 100) : 0;

    return (
        <section className="bench" id="bench">
            {notification && (
                <div className={`toast toast-${notification.type}`}>
                    {notification.message}
                </div>
            )}

            <div className="bench-left">
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
                <p className="figure-caption">
                    <b>fig. 1</b> — draw one digit, 0 through 9, roughly centred.
                </p>

                {networkData ? (
                    <div className="verdict">
                        <span className="verdict-label">the network says</span>
                        <div className="verdict-row">
                            <span className={`verdict-digit ${isNaN ? 'is-nan' : ''}`}>
                                {isNaN ? 'not a number' : networkData.prediction}
                            </span>
                            <span className="verdict-conf">{confidence}% sure</span>
                        </div>
                        <div className="conf-track">
                            <div
                                className={`conf-fill ${isNaN ? 'is-nan' : ''}`}
                                style={{ width: `${confidence}%` }}
                            />
                        </div>
                        <div className="dispute">
                            {!isCorrecting ? (
                                <button className="dispute-link" onClick={() => setIsCorrecting(true)}>
                                    wrong? tell it what you meant
                                </button>
                            ) : (
                                <div>
                                    <span className="dispute-label">what it should have said:</span>
                                    <div className="digit-grid">
                                        {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((n) => (
                                            <button key={n} className="digit-btn" onClick={() => handleFeedback(n)}>
                                                {n}
                                            </button>
                                        ))}
                                        <button className="digit-btn digit-nan" onClick={() => handleFeedback(10)}>
                                            n/a
                                        </button>
                                    </div>
                                    <button className="dispute-link cancel" onClick={() => setIsCorrecting(false)}>
                                        never mind
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>
                ) : (
                    <p className="idle-note">
                        The network is idle. It is waiting for you to draw something.
                    </p>
                )}
            </div>

            <div className="bench-right">
                <div className="screen">
                    <NeuralScene networkData={networkData} />
                </div>
                <p className="figure-caption">
                    <b>fig. 2</b> — all 1,691 neurons; brightness is the real activation
                    value from the forward pass. Drag to orbit, scroll to zoom.
                </p>
                <div className="legend">
                    {LEGEND.map((item) => (
                        <div key={item.label} className="legend-item">
                            <span className="legend-dot" style={{ background: item.color }} />
                            <span>{item.label}</span>
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
};

export default Playground;
