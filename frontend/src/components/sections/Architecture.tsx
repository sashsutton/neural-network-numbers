const LAYERS = [
    {
        index: '00',
        label: 'Input',
        neurons: 784,
        shape: '28 × 28',
        activation: '—',
        weights: '—',
    },
    {
        index: '01',
        label: 'Hidden 1',
        neurons: 512,
        shape: '512',
        activation: 'ReLU',
        weights: '401,408',
    },
    {
        index: '02',
        label: 'Hidden 2',
        neurons: 256,
        shape: '256',
        activation: 'ReLU',
        weights: '131,072',
    },
    {
        index: '03',
        label: 'Hidden 3',
        neurons: 128,
        shape: '128',
        activation: 'ReLU',
        weights: '32,768',
    },
    {
        index: '04',
        label: 'Output',
        neurons: 11,
        shape: '11',
        activation: 'Softmax',
        weights: '1,408',
    },
];

const STATS = [
    { val: '~600K', label: 'Parameters' },
    { val: '60,000', label: 'Training samples' },
    { val: '25', label: 'Epochs' },
    { val: '0.003', label: 'Learning rate' },
    { val: 'L2', label: 'Regularisation' },
];

const Architecture = () => (
    <section className="section" id="architecture">
        <div className="section-inner">
            <div className="section-header">
                <span className="section-tag">Architecture</span>
                <h2 className="section-title">4 layers, ~600K parameters</h2>
                <p className="section-desc">
                    A fully-connected feedforward network. Each layer applies a linear transformation
                    followed by a non-linearity, progressively abstracting pixels into digit identity.
                </p>
            </div>

            {/* Pipeline diagram */}
            <div className="pipeline">
                {LAYERS.map((layer, i) => (
                    <div key={i} className="pipeline-item">
                        <div className={`pipeline-node ${i === 0 ? 'node-edge' : i === LAYERS.length - 1 ? 'node-edge node-out' : 'node-hidden'}`}>
                            <span className="pipeline-count">{layer.neurons}</span>
                            <span className="pipeline-name">{layer.label}</span>
                            <span className="pipeline-act">{layer.activation}</span>
                        </div>
                        {i < LAYERS.length - 1 && (
                            <div className="pipeline-connector">
                                <div className="pipeline-line" />
                                <svg width="6" height="10" viewBox="0 0 6 10" fill="none">
                                    <path d="M1 1L5 5L1 9" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
                                </svg>
                            </div>
                        )}
                    </div>
                ))}
            </div>

            {/* Spec table */}
            <table className="spec-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Layer</th>
                        <th>Neurons</th>
                        <th>Activation</th>
                        <th>Weights</th>
                    </tr>
                </thead>
                <tbody>
                    {LAYERS.map((layer) => (
                        <tr key={layer.index}>
                            <td className="spec-idx">{layer.index}</td>
                            <td className="spec-name">{layer.label}</td>
                            <td className="spec-num">{layer.neurons.toLocaleString()}</td>
                            <td>
                                <code className={`spec-act ${layer.activation === 'ReLU' ? 'act-relu' : layer.activation === 'Softmax' ? 'act-softmax' : 'act-none'}`}>
                                    {layer.activation}
                                </code>
                            </td>
                            <td className="spec-role">{layer.weights}</td>
                        </tr>
                    ))}
                </tbody>
            </table>

            {/* Training stats */}
            <div className="training-stats">
                {STATS.map((s, i) => (
                    <div key={i} className="training-stat">
                        <span className="training-val">{s.val}</span>
                        <span className="training-label">{s.label}</span>
                    </div>
                ))}
            </div>
            <p className="arch-note">
                L2 regularisation adds a penalty proportional to the sum of squared weights to the loss function. This discourages any single weight from growing too large, which reduces overfitting and keeps the model from memorising the training data.
            </p>
        </div>
    </section>
);

export default Architecture;
