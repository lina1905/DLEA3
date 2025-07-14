let model;
let wordIndex = {};
let indexWord = {};
const SEQUENCE_LENGTH = 8;

let autoMode = false;
let intervalId = null;

document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('inputText');
    const suggestionsDiv = document.getElementById('suggestions');
    const predictBtn = document.getElementById('predictBtn');
    const nextBtn = document.getElementById('nextBtn');
    const autoBtn = document.getElementById('autoBtn');
    const stopBtn = document.getElementById('stopBtn');
    const resetBtn = document.getElementById('resetBtn');

    predictBtn.onclick = async () => {
        const text = input.value.trim();
        if (!text) return;
        const suggestions = await predictNextWords(text);
        showPredictions(suggestions);
    };

    nextBtn.onclick = async () => {
        const text = input.value.trim();
        if (!text) return;
        const [top] = await predictNextWords(text);
        input.value += ' ' + top.word;
        const suggestions = await predictNextWords(input.value);
        showPredictions(suggestions);
    };

    autoBtn.onclick = async () => {
        autoMode = true;
        let counter = 0;
        intervalId = setInterval(async () => {
            if (!autoMode || counter >= 10) {
                clearInterval(intervalId);
                return;
            }
            const text = input.value.trim();
            const [top] = await predictNextWords(text);
            input.value += ' ' + top.word;
            const suggestions = await predictNextWords(input.value);
            showPredictions(suggestions);
            counter++;
        }, 600);
    };

    stopBtn.onclick = () => {
        autoMode = false;
        clearInterval(intervalId);
    };

    resetBtn.onclick = () => {
        input.value = '';
        suggestionsDiv.innerHTML = '';
        autoMode = false;
        clearInterval(intervalId);
    };

    loadModelAndTokenizer();
});

async function loadModelAndTokenizer() {
    model = await tf.loadLayersModel('model/model.json');
    console.log("✅ Modell geladen");

    const tokenizerRaw = await fetch('model/tokenizer.json');
    const tokenizer = await tokenizerRaw.json();

    if (!tokenizer.word_index) {
        console.error("❌ tokenizer.json hat kein 'word_index'!");
        return;
    }

    wordIndex = tokenizer.word_index;

    for (const [word, index] of Object.entries(wordIndex)) {
        indexWord[index] = word;
    }

    console.log(`✅ Tokenizer geladen mit ${Object.keys(wordIndex).length} Wörtern`);
}

function tokenizeInput(text) {
    const tokens = text
        .toLowerCase()
        .split(/\s+/)
        .map(word => wordIndex[word])
        .filter(t => t !== undefined);

    const inputTokens = tokens.slice(-SEQUENCE_LENGTH);
    while (inputTokens.length < SEQUENCE_LENGTH) {
        inputTokens.unshift(0);
    }

    return tf.tensor([inputTokens], [1, SEQUENCE_LENGTH]);
}

async function predictNextWords(inputText) {
    const inputTensor = tokenizeInput(inputText);
    const prediction = model.predict(inputTensor);
    const predictionData = await prediction.data();

    const topIndices = Array.from(predictionData)
        .map((p, i) => [i, p])
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5);

    const topWords = topIndices.map(([i, prob]) => ({
        word: indexWord[i] || `[${i}]`,
        probability: prob.toFixed(4)
    }));

    inputTensor.dispose();
    prediction.dispose();

    return topWords;
}

function showPredictions(words) {
    const suggestionsDiv = document.getElementById('suggestions');
    suggestionsDiv.innerHTML = '<b>Vorschläge:</b><br>' + words.map(w => {
        return `<button onclick="addWord('${w.word}')">${w.word} (${w.probability})</button>`;
    }).join(' ');
}

function addWord(word) {
    const input = document.getElementById('inputText');
    input.value += ' ' + word;
    predictNextWords(input.value).then(showPredictions);
}