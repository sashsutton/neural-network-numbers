import { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Line, Sphere, Stars } from '@react-three/drei';
import * as THREE from 'three';

// Performance Fix 1: Reduce sphere segments from 16x16 to 8x8
const Neuron = ({ position, activation, color = "#00ffff" }: any) => (
    <Sphere position={position} args={[0.15, 8, 8]}>
        <meshStandardMaterial
            emissive={color}
            emissiveIntensity={activation * 6}
            color={activation > 0.1 ? color : "#2a2a2a"}
            toneMapped={false}
        />
    </Sphere>
);

// Performance Fix 2: Implement "Top-K" connection filtering and higher thresholds
const ConnectionLines = ({ fromPos, toPos, activations, targetActivations }: any) => {
    const lines = useMemo(() => {
        const list: any[] = [];

        // Higher threshold: Only render lines from neurons that are significantly "firing"
        const activeSourceThreshold = 0.4;

        fromPos.forEach((start: any, i: number) => {
            if (activations[i] > activeSourceThreshold) {
                // Limit Targets: Only connect to the top 5 most active neurons in the NEXT layer
                // This prevents the "spiderweb" lag while highlighting the prediction path
                const topTargets = targetActivations
                    .map((act: number, index: number) => ({ act, index }))
                    .sort((a: any, b: any) => b.act - a.act)
                    .slice(0, 5);

                topTargets.forEach((target: any) => {
                    const end = toPos[target.index];
                    list.push(
                        <Line
                            key={`line-${i}-${target.index}`}
                            points={[new THREE.Vector3(...start), new THREE.Vector3(...end)]}
                            color="#4facfe"
                            lineWidth={0.3}
                            transparent
                            opacity={activations[i] * 0.15}
                        />
                    );
                });
            }
        });
        return list;
    }, [fromPos, toPos, activations, targetActivations]);

    return <>{lines}</>;
};

const NeuralScene = ({ networkData }: any) => {
    const inputPositions = useMemo(() => {
        const pts = [];
        const spacing = 0.22;
        const offset = (28 * spacing) / 2;
        for (let i = 0; i < 784; i++) {
            const x = (i % 28) * spacing - offset;
            const y_index = Math.floor(i / 28);
            const y = (27 - y_index) * spacing - offset;
            pts.push([-9, y, x]);
        }
        return pts;
    }, []);

    const hidden1Positions = useMemo(() => {
        const pts = [];
        const cols = 16;
        const spacing = 0.45;
        for (let i = 0; i < 256; i++) {
            const x = (i % cols) * spacing - (15 * spacing) / 2;
            const y = Math.floor(i / cols) * spacing - (15 * spacing) / 2;
            pts.push([-3, y, x]);
        }
        return pts;
    }, []);

    const hidden2Positions = useMemo(() => {
        const pts = [];
        const cols = 8;
        const spacing = 0.5;
        for (let i = 0; i < 128; i++) {
            const x = (i % cols) * spacing - (7 * spacing) / 2;
            const y = Math.floor(i / cols) * spacing - (15 * spacing) / 2;
            pts.push([3, y, x]);
        }
        return pts;
    }, []);

    const outputPositions = useMemo(() =>
        Array.from({ length: 11 }, (_, i) => [9, i * 1.0 - 5.0, 0]), []
    );

    return (
        <div style={{ height: '100%', width: '100%', background: '#1e1e2e' }}>
            <Canvas camera={{ position: [16, 8, 16], fov: 45 }}>
                <color attach="background" args={['#1e1e2e']} />
                <fog attach="fog" args={['#1e1e2e', 12, 45]} />

                <ambientLight intensity={0.8} />
                <pointLight position={[10, 10, 10]} intensity={2.5} color="#4facfe" />

                <Stars radius={100} depth={50} count={2000} factor={4} saturation={0} fade speed={0.5} />

                {/* Layer 1: Input (Culling: Only render pixels that are drawn) */}
                {networkData && inputPositions.map((pos: any, i: number) => (
                    networkData.input_layer[i] > 0.1 && (
                        <Neuron key={`in-${i}`} position={pos} activation={networkData.input_layer[i]} color="white" />
                    )
                ))}

                {/* Layer 2: Hidden 1 (256 units) */}
                {hidden1Positions.map((pos: any, i: number) => (
                    <Neuron
                        key={`h1-${i}`}
                        position={pos}
                        activation={networkData?.hidden_layer1?.[i] || 0}
                        color="#00ffff"
                    />
                ))}

                {/* Layer 3: Hidden 2 (128 units) */}
                {hidden2Positions.map((pos: any, i: number) => (
                    <Neuron
                        key={`h2-${i}`}
                        position={pos}
                        activation={networkData?.hidden_layer2?.[i] || 0}
                        color="#0088ff"
                    />
                ))}

                {/* Layer 4: Output */}
                {outputPositions.map((pos: any, i: number) => (
                    <Neuron
                        key={`o-${i}`}
                        position={pos}
                        activation={networkData?.output_layer?.[i] || 0}
                        color={i === 10 ? "#ff4444" : "#ff00ff"}
                    />
                ))}

                {/* Optimized Connections between Layers */}
                {networkData?.hidden_layer1 && networkData?.hidden_layer2 && (
                    <ConnectionLines
                        fromPos={hidden1Positions}
                        toPos={hidden2Positions}
                        activations={networkData.hidden_layer1}
                        targetActivations={networkData.hidden_layer2}
                    />
                )}

                {networkData?.hidden_layer2 && networkData?.output_layer && (
                    <ConnectionLines
                        fromPos={hidden2Positions}
                        toPos={outputPositions}
                        activations={networkData.hidden_layer2}
                        targetActivations={networkData.output_layer}
                    />
                )}

                <OrbitControls />
            </Canvas>
        </div>
    );
};

export default NeuralScene;