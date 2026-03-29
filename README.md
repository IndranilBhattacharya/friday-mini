# slm-marvel

A PyTorch skeleton for a Small Language Model (SLM), focused on pure math and educational clarity.

## Folder Structure

- `configs/`: All hyperparameters live here as YAML files.
- `data/`: Raw and processed text data.
- `src/`: Core source code.
  - `core/`: Pure math implementations (attention, embeddings, transformer blocks).
  - `models/`: Assembled language models.
  - `training/`: Training loop, loss functions, optimizers.
  - `inference/`: Text generation capabilities.
  - `tokenization/`: Tokenizer.
  - `utils/`: Logging, debugging helpers, and config loader.
- `notebooks/`: One notebook per concept for visual exploration.
- `checkpoints/`: Saved model weights.
- `tests/`: One test file per src module.

## Rules
- Pure PyTorch (No HuggingFace abstractions).
- Every function has a docstring explaining tensor shapes going in and out.
