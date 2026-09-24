PY := .venv/bin/python

.PHONY: data rules judge validate app demo coach stability

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

coach:
	$(PY) -m qa.coach

stability:
	$(PY) -m qa.validate --stability 15 --runs 3
