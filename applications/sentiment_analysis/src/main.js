document.getElementById('sentiment-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = document.getElementById('input-text').value;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ input: text })
        });

        const data = await response.json();

        if (data.error) {
            alert(`Error: ${data.error}`);
            return;
        }

        document.getElementById('sentiment-form').style.display = 'none';
        document.getElementById('result-gif').src = `/images/${data.prediction}.gif`;
        document.getElementById('result-label').innerText = data.prediction;
        document.getElementById('result-probability').innerText = `Confidence: ${(data.probability * 100).toFixed(1)}%`;
        document.getElementById('result').style.display = 'flex';

    } catch (error) {
        alert(`Error: ${error.message}`);
    }
});

document.getElementById('reset-btn').addEventListener('click', () => {
    document.getElementById('sentiment-form').reset();
    document.getElementById('sentiment-form').style.display = 'block';
    document.getElementById('result').style.display = 'none';
});

