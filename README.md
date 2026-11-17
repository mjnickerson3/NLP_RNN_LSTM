# NLP_RNN_LSTM

## Overview

This project compares a SimpleRNN model and an LSTM model for next-word prediction using TensorFlow. Both models were trained on the same dataset and then tested using the same prompts to compare their performance.

## Dataset

The dataset contains simple sentences about animals and people performing actions.

Examples:

- the dog ran across the street
- the cat sat on the mat
- the dog chased the cat
- the boy threw the ball
- the girl ran across the yard

## Requirements

```bash
pip install tensorflow numpy
```

## Run the Project

```bash
python main.py
```

## Model Architectures

### SimpleRNN Model

- Embedding Layer
- SimpleRNN Layer
- Dense Output Layer (Softmax)

### LSTM Model

- Embedding Layer
- LSTM Layer
- Dense Output Layer (Softmax)

---

## Screenshots

### Data Preparation

<img width="316" height="106" alt="image" src="https://github.com/user-attachments/assets/ddd1dc50-39cd-4be4-82ba-b0537ddba9aa" />


### Training Performance

<img width="766" height="177" alt="image" src="https://github.com/user-attachments/assets/da4b14bd-417d-4b77-93f8-611253c2a8f9" />


### SimpleRNN Predictions

<img width="786" height="338" alt="image" src="https://github.com/user-attachments/assets/63760e0c-c1ac-454e-abef-744dcf88db30" />


### LSTM Predictions

<img width="805" height="350" alt="image" src="https://github.com/user-attachments/assets/828ec4e1-8734-4670-a31a-b177dac36a06" />


---

## Results

### SimpleRNN Predictions

| Prompt | Prediction |
|----------|----------|
| the dog ran | to |
| the cat sat | on |
| the dog chased | the |
| the boy threw | the |
| the girl ran | across |
| the bird flew | in |

### LSTM Predictions

| Prompt | Prediction |
|----------|----------|
| the dog ran | across |
| the cat sat | across |
| the dog chased | the |
| the boy threw | the |
| the girl ran | the |
| the bird flew | the |

---

## Reflection Questions

### 1. Which model had better training accuracy?

The SimpleRNN model had better training accuracy. It had an accuracy of 0.8585 compared to 0.5283 for the LSTM model. Since the dataset was pretty small, the SimpleRNN was able to learn the patterns in the training data more effectively.

### 2. Which model gave more sensible next-word predictions?

The SimpleRNN generally gave more sensible predictions. For example, it predicted "on" after "the cat sat" and "across" after "the girl ran," which matched the training data. The LSTM tended to predict common words like "the" more often, even when another word would have made more sense.

### 3. Where did the SimpleRNN fail?

The SimpleRNN can have trouble with longer sequences because it does not remember earlier words as well as an LSTM. In a larger dataset, it could lose important context and make less accurate predictions. However, with this small dataset, I did not notice any major issues.

### 4. Why might the LSTM perform better on sequence tasks?

LSTMs are designed to keep important information in memory while filtering out information that is not needed. This helps them handle longer sequences and maintain context more effectively. Even though the LSTM did not perform better in this project, it would likely do better on larger and more complex datasets.

