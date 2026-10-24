.PHONY: docs docs-figures docs-serve docs-clean
# The docs theme needs Python >= 3.13, so docs use their own environment (the library supports 3.12+).
DOCS = UV_PROJECT_ENVIRONMENT=.venv-docs uv run --python 3.13 --group docs

docs-figures:
	rm -rf docs/assets/generated
	$(DOCS) python -c "import sys; sys.path.insert(0, 'tools'); import docs_hooks; docs_hooks.on_pre_build(None)"

docs:
	$(DOCS) mkdocs build

docs-serve:
	$(DOCS) mkdocs serve

docs-clean:
	rm -rf site docs/assets/generated
