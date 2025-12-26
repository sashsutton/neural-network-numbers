import { Canvas } from '@react-three/fiber';
import { OrbitControls, Sphere } from '@react-three/drei';

const Neuron = ({ position, activation }: { position: [number, number, number], activation: number }) => (
    <Sphere position={position} args={[0.2, 16, 16]}>
        <meshStandardMaterial
            emissive="cyan"
            emissiveIntensity={activation * 2}
            color={activation > 0.1 ? "cyan" : "#111"}
        />
    </Sphere>
);

const NeuralScene = ({ networkData }: { networkData: any }) => {
    return (
        <div style={{ height: '500px', width: '100%', background: '#111' }}>
            <Canvas camera={{ position: [20, 10, 20] }}>
                <ambientLight intensity={0.5} />
                <pointLight position={[10, 10, 10]} />

                {/* Visualizing Hidden Layer (16 neurons) */}
                {networkData?.hidden_layer?.map((val: number, i: number) => (
                    <Neuron key={`h-${i}`} position={[0, i - 8, 0]} activation={val} />
                ))}

                {/* Visualizing Output Layer (10 neurons) */}
                {networkData?.output_layer?.map((val: number, i: number) => (
                    <Neuron key={`o-${i}`} position={[5, i - 5, 0]} activation={val} />
                ))}

                <OrbitControls />
            </Canvas>
        </div>
    );
};

export default NeuralScene;