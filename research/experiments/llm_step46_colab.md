# Step 46 Colab Qwen smoke test

The local runner is `research/scripts/test_llm_models.py`. To test the
publicly downloadable Qwen model in Colab, use a GPU runtime and run:

```python
!pip install -q "torch" "transformers>=4.45" "accelerate" "bitsandbytes" "huggingface_hub"
```

No gated-model approval or Hugging Face token is required for this public
checkpoint. A token may be supplied through a Colab Secret only if the
Hugging Face environment requests it; do not paste tokens into a notebook
cell or commit them.

Then run:

```python
%run research/scripts/test_llm_models.py --model qwen \
  --qwen-revision main \
  --output research/results/tables/step46_qwen_smoke_test.json
```

For the final Step 46 record, replace `main` with the exact Hugging Face
checkpoint revision and retain the runtime metadata printed by the script.

Completed capture: revision
`a09a35458c702b33eeacc393d103063234e8bc28`, Colab Tesla T4, Python 3.13.15,
PyTorch 2.11.0+cu130, and a passed synthetic-only JSON smoke test. The saved
report is `research/results/tables/step46_qwen_smoke_test.json`.
