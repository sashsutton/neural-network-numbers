import { useRef, useState, useEffect } from 'react';
import axios from 'axios';

const DrawingCanvas = ({ onPrediction, onClear }: any) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isDrawing, setIsDrawing] = useState(false);

    useEffect(() => {
        clearCanvas();
    }, []);

    const clearCanvas = () => {
        const canvas = canvasRef.current;
        if (canvas) {
            const ctx = canvas.getContext('2d');
            if (ctx) {
                ctx.fillStyle = 'black';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                onClear();
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

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const fullImageData = ctx.getImageData(0, 0, 280, 280);
        let minX = 280, minY = 280, maxX = 0, maxY = 0;
        let foundPixels = false;

        for (let y = 0; y < 280; y++) {
            for (let x = 0; x < 280; x++) {
                const i = (y * 280 + x) * 4;
                if (fullImageData.data[i] > 50) {
                    minX = Math.min(minX, x);
                    minY = Math.min(minY, y);
                    maxX = Math.max(maxX, x);
                    maxY = Math.max(maxY, y);
                    foundPixels = true;
                }
            }
        }

        if (!foundPixels) return;

        const tinyCanvas = document.createElement('canvas');
        tinyCanvas.width = 28;
        tinyCanvas.height = 28;
        const tinyCtx = tinyCanvas.getContext('2d');
        if (!tinyCtx) return;

        tinyCtx.fillStyle = 'black';
        tinyCtx.fillRect(0, 0, 28, 28);

        const width = maxX - minX;
        const height = maxY - minY;
        const scale = 20 / Math.max(width, height);
        const scaledW = width * scale;
        const scaledH = height * scale;

        tinyCtx.drawImage(
            canvas,
            minX, minY, width, height,
            14 - scaledW / 2, 14 - scaledH / 2,
            scaledW, scaledH
        );

        const imageData = tinyCtx.getImageData(0, 0, 28, 28);
        const pixels: number[] = [];
        for (let i = 0; i < imageData.data.length; i += 4) {
            pixels.push(imageData.data[i] / 255.0);
        }

        try {
            const API_BASE_URL = (import.meta as any).env.VITE_API_URL || 'http://localhost:8000';
            const res = await axios.post(`${API_BASE_URL}/predict`, { pixels });
            onPrediction(res.data, pixels);
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