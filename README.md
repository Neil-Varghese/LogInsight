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