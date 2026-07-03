const Masthead = () => (
    <header className="masthead">
        <div className="masthead-rule">
            <span>S. Sutton / notebook no. 3</span>
            <span>784 &rarr; 512 &rarr; 256 &rarr; 128 &rarr; 11</span>
        </div>
        <h1 className="masthead-title">
            a neural network<br />you can <em>watch think</em>
        </h1>
        <div className="masthead-sub">
            <span>written from scratch in numpy</span>
            <span>&middot;</span>
            <span>~600,000 parameters</span>
            <span>&middot;</span>
            <span>trained on 60,000 handwritten digits</span>
        </div>
        <p className="intro">
            There is no PyTorch here and no TensorFlow. Just matrix multiplication,
            the chain rule, and a few thousand lines of stubbornness. The network reads
            handwritten digits, and the screen below shows every one of its neurons
            firing while it makes up its mind.
        </p>
        <p className="intro">
            Draw a number. If it gets it wrong, tell it so. It will run backpropagation
            on your correction and quietly rewrite its own weights, so the next person
            who draws that digit gets a slightly smarter network than you did.
        </p>
    </header>
);

export default Masthead;
