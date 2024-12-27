# micropgpt -> GPT-like Language Model

This repository contains a GPT-style Language Model built using PyTorch. The architecture is inspired by the "Attention Is All You Need" paper and focuses on implementing key components such as self-attention and positional encoding.

## Features

- Multi-Head Attention mechanism
- Transformer blocks for sequence modeling
- Trained on the Tiny Shakespeare dataset
- Implements text generation functionality

## Table of Contents
- [Usage](#usage)
- [Architecture](#architecture)
- [Training](#training)
- [Generation](#generation)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)

## Usage

### Training

Run the `gpt_like_llm.ipynb` Jupyter Notebook or execute the code directly to train the model. 

### Text Generation

After training, use the `generate` method of the `GPTLanguageModel` to create text based on a prompt:
```python
index = torch.tensor([[model_start_token]])  # Input a starting token
generated_text = model.generate(index, max_new_tokens=100)
print(decode(generated_text.tolist()[0]))
```

## Architecture

- **Embedding Layer**: Token and positional embeddings.
- **Transformer Blocks**: Consists of self-attention and feed-forward layers.
- **Output Head**: Maps the final embedding to the vocabulary space.

Key Hyperparameters:
- `n_embd`: 384 (embedding dimension)
- `n_head`: 4 (number of attention heads)
- `n_layer`: 4 (number of Transformer layers)
- `block_size`: 128 (context window size)

## Training

- Optimizer: AdamW
- Loss Function: CrossEntropyLoss
- Default Parameters:
  - Learning Rate: 3e-4
  - Batch Size: 32
  - Maximum Iterations: 3000
- Training and Validation Splits:
  - Train: 80%
  - Validation: 20%

## Generation

The `generate` method allows the model to predict the next tokens based on a given input sequence. It uses a sampling technique to introduce variability in outputs.

## Future Work

- Implement a larger dataset for training.
- Fine-tune hyperparameters for improved performance.
- Add support for multilingual text.

## Acknowledgements

- Inspired by the "Attention Is All You Need" paper by Vaswani et al.
- Tiny Shakespeare dataset from [karpathy/char-rnn](https://github.com/karpathy/char-rnn).
---
## License

This project is licensed under the MIT License.
