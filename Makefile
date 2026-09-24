PY := .venv/bin/python

.PHONY: data rules judge validate app demo

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
