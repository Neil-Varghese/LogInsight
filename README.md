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

## LLM explanations

The React frontend can request an explanation for an anomalous block. Configure
the backend before starting it:

Create a `.env` file in the project root:

```dotenv
# Gemini is used automatically when this is present.
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.5-flash
```

If `GEMINI_API_KEY` is not configured, the backend can still use an
OpenAI-compatible provider such as Groq with these settings:

```dotenv
GROQ_API_KEY=your-groq-api-key
LOGINSIGHT_LLM_BASE_URL=https://api.groq.com/openai/v1/responses
LOGINSIGHT_LLM_MODEL=openai/gpt-oss-20b
```

Then start the backend normally:

```bash
python3 backend/server.py
```

Explanations are generated only for anomalous blocks, returned as structured
JSON, and cached in the run database. The selected block's raw logs are sent
to the configured LLM provider, so review privacy requirements before using
this with sensitive logs.

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
