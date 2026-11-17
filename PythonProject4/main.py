"""
main.py

Student Starter Template
RNN vs LSTM - Next Word Prediction

Instructions:
- Complete the TODO sections
- Train both a SimpleRNN model and an LSTM model
- Compare the predictions from both models
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

from data import TEXT


# -----------------------------
# SETTINGS
# -----------------------------
SEQ_LENGTH = 3
EMBED_DIM = 10
HIDDEN_UNITS = 32
EPOCHS = 150
VERBOSE = 0


# -----------------------------
# DATA PREPARATION
# -----------------------------
def prepare_data(text, seq_length=3):
    """
    Convert raw text into training sequences.

    Example:
    input words:  the dog ran
    target word:  across
    """
    # TODO: Create a tokenizer
    tokenizer = Tokenizer()

    # TODO: Fit the tokenizer on the text
    tokenizer.fit_on_texts([text])

    # TODO: Compute the vocabulary size
    total_words = len(tokenizer.word_index) + 1

    # TODO: Convert text into encoded integers
    encoded = tokenizer.texts_to_sequences([text])[0]

    X = []
    y = []

    # TODO: Build input/output sequences
    # Each X should contain seq_length words
    # Each y should contain the next word after that sequence
    for i in range(seq_length, len(encoded)):
        X.append(encoded[i - seq_length:i])
        y.append(encoded[i])

    # TODO: Convert X and y into NumPy arrays
    X = np.array(X)
    y = np.array(y)

    # TODO: One-hot encode y
    y = to_categorical(y, num_classes=total_words)

    return tokenizer, total_words, X, y


# -----------------------------
# MODEL BUILDERS
# -----------------------------
def build_rnn_model(vocab_size, seq_length, embed_dim=10, hidden_units=32):
    """
    Build a SimpleRNN next-word prediction model.
    """
    model = Sequential([
        # TODO: Add embedding layer
        Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=seq_length),

        # TODO: Add SimpleRNN layer
        SimpleRNN(hidden_units),

        # TODO: Add Dense output layer with softmax
        Dense(vocab_size, activation="softmax")
    ])

    # TODO: Compile the model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def build_lstm_model(vocab_size, seq_length, embed_dim=10, hidden_units=32):
    """
    Build an LSTM next-word prediction model.
    """
    model = Sequential([
        # TODO: Add embedding layer
        Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=seq_length),

        # TODO: Replace the SimpleRNN with an LSTM layer
        LSTM(hidden_units),

        # TODO: Add Dense output layer with softmax
        Dense(vocab_size, activation="softmax")
    ])

    # TODO: Compile the model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


# -----------------------------
# PREDICTION HELPERS
# -----------------------------
def predict_next_word(model, tokenizer, seed_text, seq_length=3):
    """
    Predict the next word from a prompt.
    """
    # TODO: Convert the seed text into token ids
    encoded = tokenizer.texts_to_sequences([seed_text])[0]

    # For this starter version, the prompt must contain exactly seq_length words
    if len(encoded) != seq_length:
        return "[ERROR: prompt must contain exactly 3 words]"

    # TODO: Reshape for the model
    encoded = np.array(encoded).reshape(1, seq_length)

    # TODO: Predict probabilities
    predicted_probs = model.predict(encoded, verbose=0)

    # TODO: Get the index of the highest-probability word
    predicted_index = np.argmax(predicted_probs, axis=1)[0]

    # TODO: Convert the predicted index back into a word
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word

    return "[UNKNOWN]"


def test_model_predictions(model, tokenizer, prompts, seq_length=3):
    """
    Run a list of prompts through the model.
    """
    results = []

    for prompt in prompts:
        # TODO: Predict the next word for each prompt
        next_word = predict_next_word(model, tokenizer, prompt, seq_length)
        results.append((prompt, next_word))

    return results


def print_results(title, results):
    """
    Print results in a readable way.
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    for prompt, prediction in results:
        print(f"Prompt: {prompt:<20} -> Prediction: {prediction}")


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("Preparing data...")
    tokenizer, vocab_size, X, y = prepare_data(TEXT, seq_length=SEQ_LENGTH)

    print(f"Vocabulary size: {vocab_size}")
    print(f"Training examples: {len(X)}")

    print("\nBuilding models...")
    rnn_model = build_rnn_model(vocab_size, SEQ_LENGTH, EMBED_DIM, HIDDEN_UNITS)
    lstm_model = build_lstm_model(vocab_size, SEQ_LENGTH, EMBED_DIM, HIDDEN_UNITS)

    print("\nTraining SimpleRNN model...")
    # TODO: Train the RNN model
    rnn_model.fit(X, y, epochs=EPOCHS, verbose=VERBOSE)

    print("Training LSTM model...")
    # TODO: Train the LSTM model
    lstm_model.fit(X, y, epochs=EPOCHS, verbose=VERBOSE)

    # TODO: Evaluate both models
    rnn_loss, rnn_acc = rnn_model.evaluate(X, y, verbose=0)
    lstm_loss, lstm_acc = lstm_model.evaluate(X, y, verbose=0)

    print("\nTraining Performance")
    print("-" * 60)
    print(f"SimpleRNN Accuracy: {rnn_acc:.4f}")
    print(f"LSTM Accuracy     : {lstm_acc:.4f}")

    # Prompts must contain exactly 3 words in this starter version
    prompts = [
        "the dog ran",
        "the cat sat",
        "the dog chased",
        "the boy threw",
        "the girl ran",
        "the bird flew"
    ]

    # TODO: Test both models
    rnn_results = test_model_predictions(rnn_model, tokenizer, prompts, SEQ_LENGTH)
    lstm_results = test_model_predictions(lstm_model, tokenizer, prompts, SEQ_LENGTH)

    # TODO: Print results
    print_results("SimpleRNN Predictions", rnn_results)
    print_results("LSTM Predictions", lstm_results)

    print("\nRequired reflection questions:")
    print("1. Which model had better training accuracy?")
    print("2. Which model gave more sensible next-word predictions?")
    print("3. Where did the SimpleRNN fail?")
    print("4. Why might the LSTM perform better on sequence tasks?")


if __name__ == "__main__":
    main()