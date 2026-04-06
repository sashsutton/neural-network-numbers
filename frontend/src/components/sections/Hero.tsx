const LAYERS = [
    { label: 'Input', neurons: 784, sub: '28 × 28 px' },
    { label: 'Hidden 1', neurons: 512, sub: 'ReLU' },
    { label: 'Hidden 2', neurons: 256, sub: 'ReLU' },
    { label: 'Hidden 3', neurons: 128, sub: 'ReLU' },
    { label: 'Output', neurons: 11, sub: 'Softmax' },
];

const MAX_DOTS = 12;

const Hero = () => {
    const scrollToPlayground = () => {
        document.getElementById('playground')?.scrollIntoView({ behavior: 'smooth' });
    };

    return (
        <section className="hero">
            <div className="hero-content">
                <div className="hero-label">Deep Learning / MNIST</div>
                <h1 className="hero-title">
                    Handwritten Digit<br />Recognition
                </h1>
                <p className="hero-desc">
                    A 4-layer neural network trained on 60,000 handwritten digits.
                    Draw a number and watch activations propagate in real time.
                </p>
                <div className="hero-actions">
                    <button className="btn-primary" onClick={scrollToPlayground}>
                        Open Playground
                        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                            <path d="M7 2.5L7 11.5M7 11.5L3 7.5M7 11.5L11 7.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                    </button>
                    <button className="btn-ghost" onClick={() => document.getElementById('architecture')?.scrollIntoView({ behavior: 'smooth' })}>
                        View Architecture
                    </button>
                </div>
            </div>

            <div className="hero-diagram">
                <div className="arch-preview">
                    {LAYERS.map((layer, li) => {
                        const dotCount = Math.min(layer.neurons, MAX_DOTS);
                        const showEllipsis = layer.neurons > MAX_DOTS;
                        return (
                            <div key={li} className="arch-layer-preview">
                                <div className="arch-dots">
                                    {Array.from({ length: dotCount }).map((_, i) => (
                                        <span
                                            key={i}
                                            className={`arch-dot ${li === 0 ? 'dot-input' : li === LAYERS.length - 1 ? 'dot-output' : 'dot-hidden'}`}
                                            style={{ animationDelay: `${(li * MAX_DOTS + i) * 30}ms` }}
                                        />
                                    ))}
                                    {showEllipsis && <span className="arch-ellipsis">…</span>}
                                </div>
                                {li < LAYERS.length - 1 && (
                                    <div className="arch-arrow">
                                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                                            <path d="M5 12H19M19 12L13 6M19 12L13 18" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
                                        </svg>
                                    </div>
                                )}
                                <div className="arch-layer-info">
                                    <span className="arch-neuron-count">{layer.neurons}</span>
                                    <span className="arch-layer-label">{layer.label}</span>
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>
        </section>
    );
};

export default Hero;
