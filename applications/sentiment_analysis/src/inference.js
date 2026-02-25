const ort = require('onnxruntime-node');
const path = require('path');

const MODEL_PATH = path.join(__dirname, '../models/sentiment_model.onnx');

let session = null;

async function getSession() {
    if (!session) {
        session = await ort.InferenceSession.create(MODEL_PATH);
        console.log('Input names:', session.inputNames);
        console.log('Output names:', session.outputNames);
    }
    return session;
}

async function runInference(text) {
    const sess = await getSession();

    const tensor = new ort.Tensor('string', [text], [1, 1]); //model expects
    
    console.log('Running inference with text:', text);
    console.log('Tensor:', tensor);

    const results = await sess.run({ string_input: tensor });

    const label = results.label.data[0];
    const probabilities = Array.from(results.probabilities.data);
    const maxProbability = Math.max(...probabilities);

    console.log("Output label:", label);
    console.log("Output probability:", maxProbability);

    return { label, probability: maxProbability };
}

module.exports = { runInference };