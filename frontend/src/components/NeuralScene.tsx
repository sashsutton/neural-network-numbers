import { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Line, Sphere, Stars } from '@react-three/drei';
import * as THREE from 'three';


const Neuron = ({ position, activation, color = "#00ffff" }: any) => (
    <Sphere position={position} args={[0.15, 16, 16]}>
        <meshStandardMaterial
            emissive={color}
            emissiveIntensity={activation * 6}
            color={activation > 0.1 ? color : "#2a2a2a"}
            toneMapped={false}
        />
    </Sphere>
);

const ConnectionLines = ({ fromPos, toPos, activations }: any) => {
    const lines = useMemo(() => {
        const list: any[] = [];
        fromPos.forEach((start: any, i: number) => {
            // Only render connections for active neurons for visual clarity and performance
            if (activations[i] > 0.05) {
                toPos.forEach((end: any, j: number) => {
                    list.push(
                        <Line
                            key={`line-${i}-${j}`}
                            points={[new THREE.Vector3(...start), new THREE.Vector3(...end)]}
                            color="#4facfe"
                            lineWidth={0.5}
                            transparent
                            opacity={activations[i] * 0.15}
                        />
                    );
                });
            }
        });
        return list;
    }, [fromPos, toPos, activations]);

    return <>{lines}</>;
};

const NeuralScene = ({ networkData }: any) => {
    // 1. Layer Position Definitions
    const inputPositions = useMemo(() => {
        const pts = [];
        const spacing = 0.22;
        const offset = (28 * spacing) / 2;

        for (let i = 0; i < 784; i++) {
            const x = (i % 28) * spacing - offset;
            const y_index = Math.floor(i / 28);
            const y = (27 - y_index) * spacing - offset;
            pts.push([-7, y, x]);
        }
        return pts;
    }, []);

    const hiddenPositions = useMemo(() => {
        const pts = [];
        for (let i = 0; i < 64; i++) {
            const x = (i % 8) * 0.7 - 2.45;
            const y = Math.floor(i / 8) * 0.7 - 2.45;
            pts.push([0, y, x]); // 8x8 Grid for the 64 neurons
        }
        return pts;
    }, []);

    const outputPositions = useMemo(() =>
        Array.from({ length: 10 }, (_, i) => [7, i * 1.1 - 4.95, 0]), []
    );

    return (
        <div style={{ height: '100%', width: '100%', background: '#0f0f1a' }}>
            <Canvas camera={{ position: [14, 6, 14], fov: 45 }}>
                {/* ATMOSPHERIC BACKGROUND: Deep midnight blue instead of pitch black */}
                <color attach="background" args={['#0f0f1a']} />

                {/* FOG: Makes distant connections and neurons fade softly for better depth perception */}
                <fog attach="fog" args={['#0f0f1a', 12, 30]} />

                {/* HIGH-TECH LIGHTING SETUP */}
                <ambientLight intensity={0.8} />
                <pointLight position={[10, 10, 10]} intensity={2.5} color="#4facfe" />
                <pointLight position={[-10, -10, -10]} intensity={1} color="#ff00ff" />

                <Stars radius={100} depth={50} count={3000} factor={4} saturation={0} fade speed={0.5} />

                {/* Input Layer (The Live Drawing) */}
                {networkData && inputPositions.map((pos: any, i: number) => (
                    networkData.input_layer[i] > 0.1 && (
                        <Neuron key={`in-${i}`} position={pos} activation={networkData.input_layer[i]} color="white" />
                    )
                ))}

                {/* Hidden Layer (64 Neurons) */}
                {hiddenPositions.map((pos: any, i: number) => (
                    <Neuron
                        key={`h-${i}`}
                        position={pos}
                        activation={networkData?.hidden_layer[i] || 0}
                    />
                ))}

                {/* Output Layer (0-9 Digits) */}
                {outputPositions.map((pos: any, i: number) => (
                    <Neuron
                        key={`o-${i}`}
                        position={pos}
                        activation={networkData?.output_layer[i] || 0}
                        color="#ff00ff"
                    />
                ))}

                {/* VISUAL LINKS: Between Hidden (Column 2) and Output (Column 3) */}
                {networkData?.hidden_layer && (
                    <ConnectionLines
                        fromPos={hiddenPositions}
                        toPos={outputPositions}
                        activations={networkData.hidden_layer}
                    />
                )}

                <OrbitControls />
            </Canvas>
        </div>
    );
};

export default NeuralScene;