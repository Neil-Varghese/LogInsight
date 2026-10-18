# Research prompts

Store versioned prompt templates and rendered prompt records here. Prompt
conditions must remain separate so blinded, unblinded, shuffled-IG, random,
and no-detector-context runs cannot be confused.

All research prompt consumers must call
`research/scripts/prompt_attribution_boundary.py` before rendering prompt
text. The boundary loads the registered condition from
`research/experiments/experiment_config.json`, permits only that condition's
allowlisted fields, and rejects occlusion, leave-one-event-out, and
verification-only metadata anywhere in the supplied context. Integrated
Gradients is the only attribution signal allowed in the unblinded condition;
verification scores must remain in the research ledger and never enter an LLM
prompt. Step 49 adds a complete condition-matrix leakage test in
`research/scripts/test_prompt_leakage.py`; it also rejects condition fields
nested under the wrong visible root field.
