import { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Line, Sphere, Stars } from '@react-three/drei';
import * as THREE from 'three';

const Neuron = ({ position, activation, color = "#00ffff" }: any) => (
    <Sphere position={position} args={[0.15, 16, 16]}>
        <meshStandardMaterial
            emissive={color}
            emissiveIntensity={activation * 5}
            color={activation > 0.1 ? color : "#111"}
            toneMapped={false}
        />
    </Sphere>
);

const ConnectionLines = ({ fromPos, toPos, activations }: any) => {
    const lines = useMemo(() => {
        const list: any[] = [];
        fromPos.forEach((start: any, i: number) => {
            // Only draw connections for active neurons to boost performance
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
    // 1. Define Layer Positions
    const inputPositions = useMemo(() => {
        const pts = [];
        for (let i = 0; i < 784; i++) {
            const x = (i % 28) * 0.2 - 2.8;
            const y = Math.floor(i / 28) * 0.2 - 2.8;
            pts.push([-6, y, x]); // Located at x = -6
        }
        return pts;
    }, []);

    const hiddenPositions = useMemo(() =>
        Array.from({ length: 16 }, (_, i) => [0, i * 0.6 - 4.5, 0]), []
    );

    const outputPositions = useMemo(() =>
        Array.from({ length: 10 }, (_, i) => [6, i * 1.0 - 4.5, 0]), []
    );

    return (
        <div style={{ height: '100%', width: '100%', background: '#020202' }}>
            <Canvas camera={{ position: [12, 5, 12], fov: 45 }}>
                <color attach="background" args={['#020202']} />
                <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />

                <ambientLight intensity={0.5} />
                <pointLight position={[10, 10, 10]} intensity={1.5} />

                {/* Input Layer (The Drawing) */}
                {networkData && inputPositions.map((pos: any, i: number) => (
                    networkData.input_layer[i] > 0.1 && (
                        <Neuron key={`in-${i}`} position={pos} activation={networkData.input_layer[i]} color="white" />
                    )
                ))}

                {/* Hidden Layer */}
                {hiddenPositions.map((pos: any, i: number) => (
                    <Neuron
                        key={`h-${i}`}
                        position={pos}
                        activation={networkData?.hidden_layer[i] || 0}
                    />
                ))}

                {/* Output Layer */}
                {outputPositions.map((pos: any, i: number) => (
                    <Neuron
                        key={`o-${i}`}
                        position={pos}
                        activation={networkData?.output_layer[i] || 0}
                        color="#ff00ff"
                    />
                ))}

                {/* Connections between Hidden and Output */}
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