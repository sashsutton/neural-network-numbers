const STEPS = [
    {
        num: '01',
        title: 'Draw',
        body: 'Sketch any digit 0–9 on the canvas. The drawing pad is 280×280 pixels — just like handwriting on paper.',
    },
    {
        num: '02',
        title: 'Preprocess',
        body: 'The drawing is cropped to its bounding box, centred, and downscaled to 28×28. Pixel values are normalized to [0, 1].',
    },
    {
        num: '03',
        title: 'Forward Pass',
        body: 'The 784-value vector flows through 4 layers. Each neuron applies a weighted sum and an activation function (ReLU or Softmax).',
    },
    {
        num: '04',
        title: 'Predict',
        body: 'The output layer produces 11 confidence scores via Softmax. The highest score becomes the prediction.',
    },
    {
        num: '05',
        title: 'Learn',
        body: 'If the prediction is wrong, you can correct it. The network runs backpropagation and updates its weights immediately — online learning.',
    },
];

const HowItWorks = () => (
    <section className="section section-alt" id="how-it-works">
        <div className="section-inner">
            <div className="section-header">
                <span className="section-tag">Process</span>
                <h2 className="section-title">How It Works</h2>
                <p className="section-desc">
                    From a brush stroke to a prediction in milliseconds.
                </p>
            </div>

            <div className="steps">
                {STEPS.map((step, i) => (
                    <div key={i} className="step">
                        <div className="step-num">{step.num}</div>
                        <div className="step-body">
                            <h3 className="step-title">{step.title}</h3>
                            <p className="step-text">{step.body}</p>
                        </div>
                        {i < STEPS.length - 1 && <div className="step-line" />}
                    </div>
                ))}
            </div>

            <div className="math-note">
                <div className="math-note-header">
                    <span className="section-tag">Forward pass equation</span>
                </div>
                <div className="math-blocks">
                    <div className="math-block">
                        <code className="math-code">a<sup>(l)</sup> = ReLU(W<sup>(l)</sup> · a<sup>(l-1)</sup> + b<sup>(l)</sup>)</code>
                        <span className="math-label">Hidden layers 1–3</span>
                    </div>
                    <div className="math-sep">→</div>
                    <div className="math-block">
                        <code className="math-code">ŷ = Softmax(W<sup>(4)</sup> · a<sup>(3)</sup> + b<sup>(4)</sup>)</code>
                        <span className="math-label">Output layer</span>
                    </div>
                </div>
            </div>
        </div>
    </section>
);

export default HowItWorks;
