import React, { useRef, useState } from 'react';
import axios from 'axios';

const DrawingCanvas = ({ onPrediction }: { onPrediction: (data: any) => void }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isDrawing, setIsDrawing] = useState(false);

    const startDrawing = (e: React.MouseEvent) => {
        const ctx = canvasRef.current?.getContext('2d');
        if (ctx) {
            ctx.lineWidth = 20;
            ctx.lineCap = 'round';
            ctx.strokeStyle = 'white';
            ctx.beginPath();
            ctx.moveTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
            setIsDrawing(true);
        }
    };

    const draw = (e: React.MouseEvent) => {
        if (!isDrawing) return;
        const ctx = canvasRef.current?.getContext('2d');
        if (ctx) {
            ctx.lineTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
            ctx.stroke();
        }
    };

    const processImage = async () => {
        setIsDrawing(false);
        const canvas = canvasRef.current;
        if (!canvas) return;

        // Create a 28x28 helper canvas to downsample the image
        const tinyCanvas = document.createElement('canvas');
        tinyCanvas.width = 28;
        tinyCanvas.height = 28;
        const tinyCtx = tinyCanvas.getContext('2d');
        tinyCtx?.drawImage(canvas, 0, 0, 28, 28);

        const imageData = tinyCtx?.getImageData(0, 0, 28, 28);
        if (!imageData) return;

        const pixels: number[] = [];
        for (let i = 0; i < imageData.data.length; i += 4) {
            pixels.push(imageData.data[i] / 255.0); // Normalize pixels to [0, 1]
        }

        try {
            const response = await axios.post('http://localhost:8000/predict', { pixels });
            onPrediction(response.data);
        } catch (error) {
            console.error("Error communicating with backend:", error);
        }
    };

    const clear = () => {
        const ctx = canvasRef.current?.getContext('2d');
        if (ctx && canvasRef.current) {
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        }
    };

    return (
        <div style={{ padding: '20px' }}>
            <canvas
                ref={canvasRef}
                width={280}
                height={280}
                style={{ background: 'black', border: '2px solid white', cursor: 'crosshair' }}
                onMouseDown={startDrawing}
                onMouseMove={draw}
                onMouseUp={processImage}
            />
            <br />
            <button onClick={clear}>Clear</button>
        </div>
    );
};

export default DrawingCanvas;