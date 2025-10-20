compose-up:
	docker compose up -d --build

compose-down:
	docker compose down

compose-logs:
	docker compose logs -f api

compose-smoke:
	OPENRAG_API_URL=http://localhost:8000 python scripts/compose_smoke.py
