"""Parse a raw HDFS log and classify each BlockId sequence with the LSTM."""

from contextlib import redirect_stdout
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import pickle
import re
import tempfile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "lstm_model(v2).keras"
EVENT_MAPPING_PATH = PROJECT_ROOT / "research" / "event2idx.pkl"
MAX_SEQUENCE_LENGTH = 50
PREDICTION_THRESHOLD = 0.5
LOG_FORMAT = "<Date> <Time> <Pid> <Level> <Component>: <Content>"
PERCENTAGE_PATTERN = re.compile(r"Processed\s+(\d+(?:\.\d+)?)%\s+of\s+log\s+lines")


MASTER_TEMPLATES_PATH = PROJECT_ROOT / "research" / "output" / "HDFS.log_templates.csv"


@lru_cache(maxsize=1)
def _load_master_template_patterns():
    """Load and compile master templates from training dataset for pattern matching."""
    if not MASTER_TEMPLATES_PATH.exists():
        return []
    df = pd.read_csv(MASTER_TEMPLATES_PATH)
    patterns = []
    for _, row in df.iterrows():
        evt_id = row["EventId"]
        template_str = str(row["EventTemplate"]).strip()
        escaped = re.escape(template_str)
        replaced = escaped.replace("<\\*>", ".*")
        pattern = re.compile(f"^{replaced}$")
        patterns.append((pattern, evt_id))
    return patterns


def _resolve_event_id(evt_id, evt_template, event2idx):
    """Return valid event2idx key directly or by matching generated template against master patterns."""
    if evt_id in event2idx:
        return evt_id
    if evt_template:
        for pattern, master_id in _load_master_template_patterns():
            if master_id in event2idx and pattern.match(str(evt_template).strip()):
                return master_id
    return None


@lru_cache(maxsize=1)
def _load_model_and_event_mapping():
    """Load these once; model startup is expensive."""
    from tensorflow.keras.models import load_model

    with EVENT_MAPPING_PATH.open("rb") as mapping_file:
        event2idx = pickle.load(mapping_file)
    return load_model(MODEL_PATH, compile=False), event2idx


def _extract_block_id(text):
    match = re.search(r"(blk_-?\d+)", str(text))
    return match.group(1) if match else None


def _timestamped(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


class DrainProgressOutput:
    """Consume Drain's printed progress and publish it without printing it."""

    def __init__(self, progress_callback):
        self.progress_callback = progress_callback
        self.buffer = ""
        self.last_progress = None

    def write(self, text):
        self.buffer += text
        for match in PERCENTAGE_PATTERN.finditer(self.buffer):
            progress = float(match.group(1))
            if progress != self.last_progress:
                self.progress_callback("preprocessing", progress)
                self.last_progress = progress
        # Keep a short suffix in case the next write completes a split message.
        self.buffer = self.buffer[-100:]
        return len(text)

    def flush(self):
        pass


def _build_block_sequences(log_file_path, progress_callback):
    """Apply the training notebook's Drain parsing and BlockId grouping."""
    from logparser.Drain import LogParser

    log_file_path = Path(log_file_path).resolve()
    with tempfile.TemporaryDirectory() as output_dir:
        parser = LogParser(
            log_format=LOG_FORMAT,
            indir=str(log_file_path.parent),
            outdir=output_dir,
            depth=4,
            st=0.5,
        )
        # Drain prints its actual percentage to stdout. Capture it and expose it
        # to the frontend instead of leaving thousands of lines in the terminal.
        try:
            with redirect_stdout(DrainProgressOutput(progress_callback)):
                parser.parse(log_file_path.name)
        except ValueError as error:
            raise ValueError(
                "The selected file is not in the expected HDFS log format "
                "(<Date> <Time> <Pid> <Level> <Component>: <Content>)."
            ) from error
        structured_path = Path(output_dir) / f"{log_file_path.name}_structured.csv"
        dataframe = pd.read_csv(structured_path)

    dataframe["BlockId"] = dataframe["Content"].apply(_extract_block_id)
    dataframe = dataframe[dataframe["BlockId"].notnull()].sort_values(by="LineId")
    grouped = dataframe.groupby("BlockId")[["EventId", "EventTemplate"]].apply(
        lambda g: list(zip(g["EventId"], g["EventTemplate"]))
    )
    return grouped.index.tolist(), grouped.tolist()


def predict_log_file(log_file_path, progress_callback=None, classification_callback=None):
    """Return normal and anomalous BlockId-sequence counts for one raw log."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    _timestamped(f"Preprocessing started: {Path(log_file_path).resolve()}")
    if progress_callback:
        progress_callback("preprocessing", 0)
    preprocessing_started = datetime.now()
    block_ids, raw_sequences = _build_block_sequences(log_file_path, progress_callback or (lambda *_: None))
    _timestamped(f"Preprocessing finished in {datetime.now() - preprocessing_started}")
    if progress_callback:
        progress_callback("preprocessing", 100)
    if not raw_sequences:
        return {"normal_logs": 0, "anomalous_logs": 0}

    _timestamped("Classification started")
    if progress_callback:
        progress_callback("classification", 0)
    classification_started = datetime.now()
    model, event2idx = _load_model_and_event_mapping()
    
    encoded_sequences = []
    clean_sequences = []
    for raw_seq in raw_sequences:
        encoded_seq = []
        clean_seq = []
        for event_id, event_template in raw_seq:
            resolved_id = _resolve_event_id(event_id, event_template, event2idx)
            if resolved_id is not None:
                encoded_seq.append(event2idx[resolved_id])
                clean_seq.append(resolved_id)
            else:
                _timestamped(f"Warning: Unrecognized event template variant '{event_id}' skipped.")
        encoded_sequences.append(encoded_seq)
        clean_sequences.append(clean_seq)

    model_input = pad_sequences(encoded_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post")
    from tensorflow.keras.callbacks import Callback

    total_batches = (len(model_input) + 255) // 256

    class ProgressCallback(Callback):
        def on_predict_batch_end(self, batch, logs=None):
            if progress_callback:
                progress_callback("classification", min(99, int((batch + 1) / total_batches * 100)))

    prediction_probabilities = model.predict(
        model_input, batch_size=256, verbose=0, callbacks=[ProgressCallback()]
    )
    predictions = (prediction_probabilities > PREDICTION_THRESHOLD).astype(int)
    if progress_callback:
        progress_callback("classification", 100)
    for block_id, sequence, prediction, probability in zip(
        block_ids, clean_sequences, predictions.reshape(-1), prediction_probabilities.reshape(-1)
    ):
        label = "anomalous" if prediction else "normal"
        _timestamped(f"Block {block_id}: {label} (anomaly confidence: {float(probability) * 100:.1f}%)")
        if classification_callback:
            classification_callback(
                block_id=block_id,
                anomaly_score=float(probability),
                is_anomalous=bool(prediction),
                event_ids=sequence,
            )
    _timestamped(f"Classification finished in {datetime.now() - classification_started}")
    return {
        "normal_logs": int((predictions == 0).sum()),
        "anomalous_logs": int((predictions == 1).sum()),
    }
