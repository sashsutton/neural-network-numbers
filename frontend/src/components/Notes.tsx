const LAYERS = [
    { label: 'input', neurons: '784', activation: 'none', weights: 'the drawing itself, 28 × 28 px' },
    { label: 'hidden 1', neurons: '512', activation: 'ReLU', weights: '401,408 weights' },
    { label: 'hidden 2', neurons: '256', activation: 'ReLU', weights: '131,072 weights' },
    { label: 'hidden 3', neurons: '128', activation: 'ReLU', weights: '32,768 weights' },
    { label: 'output', neurons: '11', activation: 'Softmax', weights: '1,408 weights' },
];

const Notes = () => (
    <div className="notes">
        <section className="note">
            <h2><span className="sec">&sect;1</span> what actually happens</h2>
            <p>
                When you press <i>read it</i>, your drawing is cropped to its bounding box,
                centred, and shrunk to 28 by 28 pixels, the same format as the MNIST
                dataset the network learned from. That gives 784 grey values between 0
                and 1, which are fed to the first layer. Each layer after that is nothing
                more exotic than a weighted sum and a decision to keep or discard:
            </p>
            <div className="equation">
                a = ReLU(W&middot;a&prime; + b)
                <span className="eq-note">
                    ReLU keeps positive values and zeroes the rest. The last layer uses
                    Softmax instead, which turns raw scores into probabilities.
                </span>
            </div>
            <p>
                The brightness of each sphere in the 3D view is that neuron's actual
                activation value, straight from the backend. Nothing is decorative.
                When the screen lights up, that is the computation itself.
            </p>
        </section>

        <section className="note">
            <h2><span className="sec">&sect;2</span> the shape of the thing</h2>
            <p>
                The first version of this network had one hidden layer and could not
                reliably tell a 3 from an 8. One layer has to do everything at once:
                find the edges, understand their arrangement, and commit to an answer.
                Adding a second layer helped. Adding a third, and widening the first to
                512 neurons, is where it settled. Each layer gets a job: edges, then
                loops and junctions, then whole digits.
            </p>
            <table className="arch-table">
                <thead>
                    <tr>
                        <th>layer</th>
                        <th>neurons</th>
                        <th>activation</th>
                        <th>&nbsp;</th>
                    </tr>
                </thead>
                <tbody>
                    {LAYERS.map((l) => (
                        <tr key={l.label}>
                            <td>{l.label}</td>
                            <td>{l.neurons}</td>
                            <td className={l.activation === 'Softmax' ? 'act-softmax' : ''}>{l.activation}</td>
                            <td>{l.weights}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
            <p>
                Training took 25 epochs of batch gradient descent over 80,000 samples:
                the 60,000 MNIST digits plus 20,000 synthetic non-digits, which is why
                there are eleven outputs rather than ten. The eleventh neuron exists
                because people kept drawing things that were not numbers, and the network
                needed a polite way to say so.
            </p>
        </section>

        <section className="note">
            <h2><span className="sec">&sect;3</span> teaching it</h2>
            <p>
                When you correct a wrong answer, the backend runs a full backpropagation
                pass on your drawing: it measures how wrong each of the 600,000 weights
                was about you specifically, nudges every one of them, and saves the
                result to disk. This is the same algorithm used in training, applied to
                a single example, live. It learns your handwriting a little at a time,
                which is either charming or unsettling depending on your mood.
            </p>
            <p>
                The maths, the training script, and the API are all short enough to read
                in one sitting. They live in <a
                href="https://github.com/sashsutton/neural-network-numbers"
                target="_blank" rel="noreferrer">the repository</a>, and the network
                itself is a single NumPy class.
            </p>
        </section>
    </div>
);

export default Notes;
