.PHONY: setup fmt test
setup:
	python -m pip install -r ../requirements.txt
fmt:
	black . && ruff check --fix .
test:
	pytest -q