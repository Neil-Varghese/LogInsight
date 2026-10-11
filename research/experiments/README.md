# Research experiments

Store experiment configurations, run metadata, and pilot/main experiment
outputs here. Do not overwrite a completed run; create a new run directory.

`experiment_config.json` is the single source of truth for the current
protocol. Values marked `TO_BE_*`, `null`, or with a provisional status must be
frozen before the pilot and copied into the preregistration.

`experiment_ledger_schema.json` defines one record for one block/window and
one explanation condition. Keep the exact prompt, raw response, attribution
visibility, perturbation outputs, and separate outcome fields for every record.

- `pilot/`: minimal HDFS go/no-go pilot.
- `main/`: registered main experiments.
