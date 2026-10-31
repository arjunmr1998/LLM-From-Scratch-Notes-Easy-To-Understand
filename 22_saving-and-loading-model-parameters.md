# 14. Saving and Loading Model Parameters in PyTorch

After training or fine-tuning a model, its learned parameters can be saved so that training does not need to be repeated every time the model is used.

## 14.1 Saving a state dictionary

PyTorch models expose their parameters through `state_dict()`.

```python
torch.save(model.state_dict(), "model.pth")
```

This saves the model's learned parameters and registered buffers.

## 14.2 Loading a state dictionary

First, initialize the model with the same architecture. Then load the saved state dictionary:

```python
model.load_state_dict(
    torch.load("model.pth", map_location="cpu")
)
```

The architecture must match the saved state dictionary. For example, if a classification head was added during fine-tuning, the model must be initialized with that same classification head before loading its weights.

## 14.3 Important distinction

A `state_dict` stores parameter values, not the complete Python model definition. Keep the model architecture code and configuration alongside the saved weights.

For reproducible training, you may also save the optimizer state, epoch, configuration and other checkpoint information.

> **Key takeaway:** Save learned parameters with `torch.save(model.state_dict(), path)` and restore them into a matching model architecture using `load_state_dict()`.
