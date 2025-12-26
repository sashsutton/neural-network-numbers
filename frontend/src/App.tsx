import { useState } from 'react';
import DrawingCanvas from './components/DrawingCanvas';
import NeuralScene from './components/NeuralScene';
import './App.css';

function App() {
    const [networkData, setNetworkData] = useState(null);

    return (
        <div className="App">
            <div className="dashboard">
                <div className="input-section">
                    <h1>Neural Vision</h1>
                    {/* Ensure onClear resets the networkData state */}
                    <DrawingCanvas
                        onPrediction={(data: any) => setNetworkData(data)}
                        onClear={() => setNetworkData(null)}
                    />
                    {networkData && (
                        <div className="prediction-box">
                            <h2 style={{ color: '#4facfe' }}>Guess: {(networkData as any).prediction}</h2>
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