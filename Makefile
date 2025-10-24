compose-up:
	docker compose up -d --build

compose-down:
	docker compose down

compose-logs:
	docker compose logs -f api

compose-smoke:
	OPENRAG_API_URL=http://localhost:8000 python scripts/compose_smoke.py

lint:
	ruff check .

typecheck:
	mypy src tests

test:
	pytest -q

run-api:
	uvicorn openrag.api.app:app --reload

run-ui:
	python -m openrag.ui.app
