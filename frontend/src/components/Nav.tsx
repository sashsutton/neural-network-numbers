const Nav = () => {
    const scrollTo = (id: string) => {
        document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
    };

    return (
        <nav className="nav">
            <div className="nav-inner">
                <span className="nav-logo">
                    <span className="nav-logo-dot" />
                    Neural Network
                </span>
                <div className="nav-links">
                    <button onClick={() => scrollTo('architecture')}>Architecture</button>
                    <button onClick={() => scrollTo('how-it-works')}>How It Works</button>
                    <button onClick={() => scrollTo('playground')} className="nav-cta">
                        Playground
                    </button>
                </div>
            </div>
        </nav>
    );
};

export default Nav;
