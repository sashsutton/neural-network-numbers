import Masthead from './components/Masthead';
import Playground from './components/Playground';
import Notes from './components/Notes';
import './App.css';

function App() {
    return (
        <div className="page">
            <Masthead />
            <main>
                <Playground />
                <Notes />
            </main>
            <footer className="colophon">
                <span>Sasha Sutton, 2026</span>
                <a href="https://github.com/sashsutton/neural-network-numbers" target="_blank" rel="noreferrer">
                    source
                </a>
                <span>MIT licence</span>
                <span>react &middot; three.js &middot; fastapi &middot; numpy</span>
            </footer>
        </div>
    );
}

export default App;
