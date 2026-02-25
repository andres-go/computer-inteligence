const express = require('express');
const {runInference} = require('./inference');

const app = express();
const port = 3000;


app.use(express.json());
app.use(express.static(__dirname + '/../public'));

app.get('/', (req, res) => {
  res.sendFile('index.html', { root: __dirname + '/../public' });
});

app.get('/main.js', (req, res) => {
  res.sendFile('main.js', { root: __dirname });
});

app.post('/predict', async (req, res) => {
    try {
        const {input} = req.body;
        if (!input) {
            return res.status(400).json({error: 'Input is required'});
        }
        const { label, probability } = await runInference(input);
        res.json({ prediction: label, probability });
    } catch (error) {
        res.status(500).json({error: error.message});
    }
});

app.listen(port, () => {
    console.log(`App is listening on port ${port}`);
});
