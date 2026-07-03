# a neural network you can watch think

**Live:** [neural-network-numbers.vercel.app](https://neural-network-numbers.vercel.app/)

I wanted to know what a neural network is actually doing when it recognises a digit, so I built one where you can see it. The network is written from scratch in NumPy (no PyTorch, no TensorFlow at inference time, just matrix multiplication and the chain rule). You draw a number on a canvas, and every one of its neurons shows up as a sphere in a 3D scene, glowing in proportion to its real activation value.

If it gets your digit wrong, you can correct it. It runs backpropagation on your drawing, rewrites its weights, and saves them to disk. The network you leave behind is slightly different from the one you found.

## how it works

The drawing canvas is 280×280. On predict, the frontend crops your drawing to its bounding box, centres it, scales it to 28×28 and normalises the pixels to [0, 1]. That matches MNIST, which is what the network trained on. The 784 values go to a FastAPI backend, which runs the forward pass:

```
input (784) -> hidden 1 (512, ReLU) -> hidden 2 (256, ReLU) -> hidden 3 (128, ReLU) -> output (11, Softmax)
```

The response includes every layer's activations, not just the answer. The frontend maps those onto the spheres in the Three.js scene, so the brightness you see is the computation itself.

There are 11 outputs rather than 10 because people kept drawing things that were not numbers. The eleventh class ("Not a Number") was trained on 20,000 synthetic non-digits: low-intensity noise and inverted MNIST images.

## why this architecture

It was not designed upfront. The first version had a single hidden layer of 128 neurons and could not reliably tell a 3 from an 8; one layer has to find edges, understand their arrangement, and commit to an answer all at once. A second hidden layer helped. The third layer, plus widening the first to 512, is where accuracy settled and stopped improving with depth. Each layer ends up with a job: edges, then loops and junctions, then whole digits.

Around 600,000 parameters in total. If I wanted more accuracy from here I would reach for dropout or batch norm before more layers.

## training

`backend/train.py` trains from scratch and writes `weights.npz` (a pre-trained one is committed, so you only need this to retrain). 80,000 samples (60k MNIST + 20k synthetic NaN), 25 epochs, batch size 32, learning rate 0.003, L2 regularisation at λ=0.001, He initialisation. Backprop is implemented by hand; the output layer gradient is the usual softmax + cross-entropy shortcut `a - y`, then the chain rule walks backwards through the ReLU layers.

TensorFlow appears in `requirements-train.txt` only because it has a convenient MNIST loader. It plays no part in the forward pass.

## API

Two endpoints, both JSON, both rate-limited per IP.

`POST /predict` takes `{"pixels": [...784 floats in [0,1]...]}` and returns every layer's activations plus `prediction` (a digit string or `"Not a Number"`) and `confidence`.

`POST /feedback` takes `{"pixels": [...], "correct_label": 0-10}` (10 means Not a Number), runs one backprop pass, saves the weights, and returns `{"status": "ok"}`.

## running it

Backend, in one terminal:

```bash
cd backend
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --port 8000
```

Frontend, in another:

```bash
cd frontend
npm install
echo 'VITE_API_URL=http://127.0.0.1:8000' > .env
npm run dev
```

Then open http://localhost:5173, draw a digit, and press *read it*.

If `weights.npz` is ever missing, `pip install -r requirements-train.txt && python train.py` rebuilds it in about ten minutes.

## deploying

The frontend is on Vercel (set `VITE_API_URL` to your backend URL). The backend runs anywhere Python does; on Render the start command is `uvicorn main:app --host 0.0.0.0 --port 8000`. Set `ALLOWED_ORIGINS` to your frontend URL, and make sure `weights.npz` is in the repo since the model loads it at startup.

## licence

MIT. Built by Sasha Sutton.
