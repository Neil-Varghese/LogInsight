# Research models

Store reproducible detector checkpoints and model-derived artifacts here.

- `checkpoints/`: versioned LSTM and second-detector model files.
- `attributions/`: Integrated Gradients and verification-only attribution outputs.
- `vocabularies/`: train-only event vocabulary manifests. These reserve
  `__PAD__` at ID 0 and a learned `__UNKNOWN_EVENT__` row for unseen events.
