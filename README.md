## Run the frontend

The frontend lists every non-hidden file in `test sets/` automatically. Start
the included local server from the project root:

```bash
python3 backend/server.py
```

Open http://localhost:8000/frontend/ and refresh the page after adding or
deleting files from `test sets/`.

Select a test set and click **Run** to parse it and score its BlockId sequences
with `models/lstm_model(v2).keras`. Parsing and prediction progress is printed
in the terminal that runs `backend/server.py`; the final normal/anomalous totals appear
in the frontend.

---

log-anomaly-explainer/
│
├── backend/
│   │
│   ├── app/
│   │   ├── main.py                 # FastAPI entry point
│   │   ├── config.py
│   │   │
│   │   ├── api/
│   │   │     predict.py
│   │   │
│   │   ├── preprocessing/
│   │   │     parser.py
│   │   │     sessionizer.py
│   │   │     encoder.py
│   │   │     pipeline.py
│   │   │
│   │   ├── models/
│   │   │     lstm_loader.py
│   │   │     predictor.py
│   │   │
│   │   ├── llm/
│   │   │     prompt_builder.py
│   │   │     explainer.py
│   │   │
│   │   ├── schemas/
│   │   │     request.py
│   │   │     response.py
│   │   │
│   │   └── utils/
│   │
│   ├── saved_models/
│   │     lstm.keras
│   │
│   ├── artifacts/
│   │     tokenizer.pkl
│   │     vocab.pkl
│   │     label_encoder.pkl
│   │
│   ├── sample_logs/
│   │
│   └── requirements.txt
│
├── frontend/
│
├── research/
│     preprocessing_pipeline.ipynb
│     LSTM.ipynb
│
└── README.md
