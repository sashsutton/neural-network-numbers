import Nav from './components/Nav';
import Hero from './components/sections/Hero';
import Architecture from './components/sections/Architecture';
import HowItWorks from './components/sections/HowItWorks';
import Playground from './components/sections/Playground';
import './App.css';

function App() {
    return (
        <div className="app">
            <Nav />
            <main>
                <Hero />
                <Architecture />
                <HowItWorks />
                <Playground />
            </main>
            <footer className="footer">
                <div className="footer-inner">
                    <span>Neural Network · MNIST · 784→512→256→128→11</span>
                    <span className="footer-sep">·</span>
                    <span>Built with React, FastAPI, Three.js</span>
                </div>
            </footer>
        </div>
    );
}

export default App;
