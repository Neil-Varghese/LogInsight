import re
from pathlib import Path

import pandas as pd


class Sessionizer:
    """Convert structured logs into BlockId-based event sequences."""

    BLOCK_PATTERN = re.compile(r"(blk_-?\d+)")

    @staticmethod
    def extract_block_id(text):
        if pd.isna(text):
            return None

        match = Sessionizer.BLOCK_PATTERN.search(str(text))
        return match.group(1) if match else None

    def process(self, structured_csv, label_csv, output_csv):

        df = pd.read_csv(structured_csv)

        # Extract BlockId from log content
        df["BlockId"] = df["Content"].apply(self.extract_block_id)

        # Remove logs that don't belong to a block
        df = df.dropna(subset=["BlockId"])

        # Preserve event order
        df = df.sort_values("LineId")

        # Create one event sequence per block
        sequences = (
            df.groupby("BlockId")["EventId"]
            .apply(list)
            .reset_index(name="EventSequence")
        )

        # Read anomaly labels
        labels = pd.read_csv(label_csv)

        # Rename for merge
        labels.rename(
            columns={
                "BlockId": "BlockId",
                "Label": "Label",
            },
            inplace=True,
        )

        # Merge labels
        sequences = sequences.merge(labels, on="BlockId", how="left")

        sequences.to_csv(output_csv, index=False)

        return sequences