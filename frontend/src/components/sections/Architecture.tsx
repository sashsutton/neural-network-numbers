const LAYERS = [
    {
        index: '00',
        label: 'Input Layer',
        neurons: 784,
        shape: '28 × 28',
        activation: '—',
        role: 'Each pixel of the input image maps to one neuron. Values are normalized to [0, 1].',
        color: 'layer-input',
    },
    {
        index: '01',
        label: 'Hidden Layer 1',
        neurons: 512,
        shape: '512 × 1',
        activation: 'ReLU',
        role: 'Learns low-level features — edges, curves, and stroke orientations.',
        color: 'layer-h1',
    },
    {
        index: '02',
        label: 'Hidden Layer 2',
        neurons: 256,
        shape: '256 × 1',
        activation: 'ReLU',
        role: 'Combines low-level features into higher-order patterns like loops and lines.',
        color: 'layer-h2',
    },
    {
        index: '03',
        label: 'Hidden Layer 3',
        neurons: 128,
        shape: '128 × 1',
        activation: 'ReLU',
        role: 'Compresses representations into the most discriminative features per digit class.',
        color: 'layer-h3',
    },
    {
        index: '04',
        label: 'Output Layer',
        neurons: 11,
        shape: '11 × 1',
        activation: 'Softmax',
        role: 'Returns a probability distribution over 10 digits plus a "Not a Number" class.',
        color: 'layer-output',
    },
];

const Architecture = () => (
    <section className="section" id="architecture">
        <div className="section-inner">
            <div className="section-header">
                <span className="section-tag">Architecture</span>
                <h2 className="section-title">4-Layer Deep Network</h2>
                <p className="section-desc">
                    Each layer transforms the representation, progressively abstracting raw pixels
                    into digit identity. Trained on MNIST with He initialization and L2 regularization.
                </p>
            </div>

            <div className="arch-flow">
                {LAYERS.map((layer, i) => (
                    <div key={i} className="arch-flow-item">
                        <div className={`arch-card ${layer.color}`}>
                            <div className="arch-card-header">
                                <span className="arch-card-index">{layer.index}</span>
                                <span className="arch-card-label">{layer.label}</span>
                                <span className={`arch-card-activation ${layer.activation === '—' ? 'act-none' : ''}`}>
                                    {layer.activation}
                                </span>
                            </div>
                            <div className="arch-card-neurons">
                                <span className="arch-big-num">{layer.neurons.toLocaleString()}</span>
                                <span className="arch-sub">neurons</span>
                            </div>
                            <p className="arch-card-role">{layer.role}</p>
                            <div className="arch-card-shape">
                                <span className="arch-shape-label">shape</span>
                                <code>{layer.shape}</code>
                            </div>
                        </div>
                        {i < LAYERS.length - 1 && (
                            <div className="arch-connector">
                                <div className="arch-connector-line" />
                                <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                                    <path d="M1 1L5 5L9 1" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                                </svg>
                            </div>
                        )}
                    </div>
                ))}
            </div>

            <div className="arch-stats">
                <div className="arch-stat">
                    <span className="arch-stat-val">~600K</span>
                    <span className="arch-stat-label">Parameters</span>
                </div>
                <div className="arch-stat-div" />
                <div className="arch-stat">
                    <span className="arch-stat-val">60,000</span>
                    <span className="arch-stat-label">Training samples</span>
                </div>
                <div className="arch-stat-div" />
                <div className="arch-stat">
                    <span className="arch-stat-val">25</span>
                    <span className="arch-stat-label">Epochs</span>
                </div>
                <div className="arch-stat-div" />
                <div className="arch-stat">
                    <span className="arch-stat-val">0.003</span>
                    <span className="arch-stat-label">Learning rate</span>
                </div>
            </div>
        </div>
    </section>
);

export default Architecture;
