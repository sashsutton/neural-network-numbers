import { useRef, useState, useEffect } from 'react';
import axios from 'axios';

const DrawingCanvas = ({ onPrediction, onClear }: any) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isDrawing, setIsDrawing] = useState(false);

    useEffect(() => {
        clearCanvas(); // Initial clear to set background to black
    }, []);

    const clearCanvas = () => {
        const canvas = canvasRef.current;
        if (canvas) {
            const ctx = canvas.getContext('2d');
            if (ctx) {
                ctx.fillStyle = 'black';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                onClear(); // Reset data in parent
            }
        }
    };

    const startDrawing = (e: any) => {
        const ctx = canvasRef.current?.getContext('2d');
        if (ctx) {
            ctx.lineWidth = 18;
            ctx.lineCap = 'round';
            ctx.strokeStyle = 'white';
            ctx.beginPath();
            ctx.moveTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
            setIsDrawing(true);
        }
    };

    const draw = (e: any) => {
        if (!isDrawing) return;
        const ctx = canvasRef.current?.getContext('2d');
        if (ctx) {
            ctx.lineTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
            ctx.stroke();
        }
    };

    const handlePredict = async () => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        // Create 28x28 version
        const tinyCanvas = document.createElement('canvas');
        tinyCanvas.width = 28;
        tinyCanvas.height = 28;
        const tinyCtx = tinyCanvas.getContext('2d');
        tinyCtx?.drawImage(canvas, 0, 0, 28, 28);

        const imageData = tinyCtx?.getImageData(0, 0, 28, 28);
        const pixels = [];
        if (imageData) {
            for (let i = 0; i < imageData.data.length; i += 4) {
                pixels.push(imageData.data[i] / 255.0);
            }
        }

        try {
            const res = await axios.post('http://localhost:8000/predict', { pixels });
            onPrediction(res.data);
        } catch (err) {
            console.error("Prediction failed", err);
        }
    };

    return (
        <div className="canvas-container">
            <canvas
                ref={canvasRef}
                width={280}
                height={280}
                onMouseDown={startDrawing}
                onMouseMove={draw}
                onMouseUp={() => setIsDrawing(false)}
                onMouseLeave={() => setIsDrawing(false)}
                style={{ border: '2px solid #4facfe', borderRadius: '8px' }}
            />
            <div className="button-group">
                <button onClick={handlePredict} className="predict-btn">
                    Run Prediction
                </button>
                <button onClick={clearCanvas} className="clear-btn">
                    Clear Pad
                </button>
            </div>
        </div>
    );
};

export default DrawingCanvas;