PY := .venv/bin/python

.PHONY: data rules judge validate app demo verify holdout coach stability

data:
	$(PY) -m gen.generate

rules:
	$(PY) -m qa.rules --validate

judge:
	$(PY) -m qa.pipeline

validate:
	$(PY) -m qa.validate

app:
	.venv/bin/streamlit run app.py

demo: data rules judge validate
	@echo "Launch the app with: make app"

verify:
	$(PY) -m gen.verify_labels

# held-out set: different seed, disjoint calls, judged once - never used for prompt tuning
HOLD := SYN_DIR=data/holdout OUT_DIR=out/holdout
holdout:
	$(HOLD) $(PY) -m gen.generate --seed 7 --exclude data/synthetic/calls_sample.csv
	$(HOLD) $(PY) -m qa.pipeline
	$(HOLD) $(PY) -m qa.validate

coach:
	$(PY) -m qa.coach

stability:
	$(PY) -m qa.validate --stability 15 --runs 3
