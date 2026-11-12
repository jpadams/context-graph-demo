.PHONY: help install install-backend install-frontend dev dev-backend dev-frontend \
        generate-data clean clean-data test lint format \
        docker-up docker-down docker-logs health stats \
        demo-setup load-data plant-scenarios embeddings wait-neo4j eval eval-fast eval-judgments

# Default target
help:
	@echo "Context Graph Demo - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install all dependencies (backend + frontend)"
	@echo "  make install-backend  Install backend dependencies with uv"
	@echo "  make install-frontend Install frontend dependencies with npm"
	@echo ""
	@echo "Development:"
	@echo "  make dev              Start both backend and frontend"
	@echo "  make dev-backend      Start backend server only"
	@echo "  make dev-frontend     Start frontend server only"
	@echo ""
	@echo "Demo (run in this order):"
	@echo "  make demo-setup       Neo4j up + generate data + plant labeled cases + embeddings"
	@echo "  make dev              Start backend (:8000) and frontend (:3000)"
	@echo "  make eval-fast        Routing eval, no Claude (~1 min)"
	@echo "  make eval             Full labeled eval, all arms incl. Claude (~15 min)"
	@echo "  make eval-judgments   Jev calibration on graph items + held-out hard items"
	@echo ""
	@echo "Data:"
	@echo "  make generate-data    Generate sample data in Neo4j"
	@echo "  make clean-data       Clear all data from Neo4j"
	@echo ""
	@echo "Docker (Local Neo4j):"
	@echo "  make docker-up        Start Neo4j with Docker Compose"
	@echo "  make docker-down      Stop Neo4j Docker container"
	@echo "  make docker-logs      View Neo4j Docker logs"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  make test             Run backend tests"
	@echo "  make lint             Run linters"
	@echo "  make format           Format code"
	@echo ""
	@echo "Utilities:"
	@echo "  make health           Check backend health"
	@echo "  make stats            Show Neo4j graph statistics"
	@echo "  make clean            Clean build artifacts"

# Python with .env loaded. CLAUDE_MODEL defaults to a current model (the SDK default is retired).
# Set CLAUDE_LOGIN=1 to ignore ANTHROPIC_API_KEY from .env and use the Claude CLI login instead.
CLAUDE_MODEL ?= claude-sonnet-5-5
ENV = set -a && . ../.env && set +a && export CLAUDE_MODEL=$${CLAUDE_MODEL:-$(CLAUDE_MODEL)} && \
	if [ -n "$(CLAUDE_LOGIN)" ]; then export ANTHROPIC_API_KEY=; fi
PY = cd backend && $(ENV) && .venv/bin/python

# =============================================================================
# Setup
# =============================================================================

install: install-backend install-frontend
	@echo "All dependencies installed!"

install-backend:
	cd backend && uv venv && uv pip install -e .

install-frontend:
	cd frontend && npm install

# =============================================================================
# Development
# =============================================================================

dev:
	@echo "Starting backend and frontend..."
	@make dev-backend &
	@sleep 3
	@make dev-frontend

dev-backend:
	@echo "Starting backend on http://localhost:8000..."
	cd backend && $(ENV) && .venv/bin/uvicorn app.main:app --reload --port 8000

dev-frontend:
	@echo "Starting frontend on http://localhost:3000..."
	cd frontend && npm run dev

# =============================================================================
# Data Management
# =============================================================================

generate-data:
	@echo "Generating sample data in Neo4j (clears the database first)..."
	$(PY) scripts/generate_sample_data.py

# The 14 labeled customers the evals and the Tanya Trujillo demo depend on (idempotent).
plant-scenarios:
	$(PY) scripts/plant_decision_scenarios.py

# Decision/policy reasoning embeddings; precedent search returns nothing without them.
embeddings:
	$(PY) -c "from app.vector_client import vector_client as v; \
		d = v.batch_update_decision_embeddings(5000); p = v.batch_update_policy_embeddings(500); \
		print(f'embedded {d} decisions, {p} policies')"

load-data: generate-data plant-scenarios embeddings
	@echo "Data loaded."

# One command from nothing to a demo-ready database.
demo-setup: docker-up load-data
	@echo "Ready. Next: make dev"

# =============================================================================
# Evals (results in backend/evals/results/)
# =============================================================================

eval-fast:
	$(PY) scripts/eval_decision_paths.py --labeled --arms fast_answer,decomposed --no-claude

eval:
	$(PY) scripts/eval_decision_paths.py --labeled \
		--arms original,evidence_claude,evidence_claude_policy,fast_answer,decomposed

eval-judgments:
	$(PY) scripts/eval_judgments.py --repeats 2
	$(PY) scripts/eval_judgments.py --hard --repeats 3

clean-data:
	@echo "Clearing all data from Neo4j..."
	$(PY) -c "from app.context_graph_client import context_graph_client; \
			context_graph_client.driver.session().run('MATCH (n) DETACH DELETE n'); \
			print('All data cleared.')"

# =============================================================================
# Docker (Local Neo4j)
# =============================================================================

docker-up:
	docker compose up -d
	@$(MAKE) --no-print-directory wait-neo4j

wait-neo4j:
	@echo "Waiting for Neo4j (first start downloads GDS, can take a minute)..."
	@for i in $$(seq 1 60); do \
		docker exec context-graph-neo4j cypher-shell -u neo4j -p password 'RETURN 1' >/dev/null 2>&1 \
			&& echo "Neo4j ready at http://localhost:7474" && exit 0; sleep 3; done; \
		echo "Neo4j did not come up; see: make docker-logs"; exit 1

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f neo4j

# =============================================================================
# Testing & Quality
# =============================================================================

test:
	cd backend && .venv/bin/pytest

lint:
	cd backend && source .venv/bin/activate && ruff check .
	cd frontend && npm run lint

format:
	cd backend && source .venv/bin/activate && ruff format .
	cd frontend && npm run lint -- --fix

# =============================================================================
# Utilities
# =============================================================================

health:
	@curl -s http://localhost:8000/health | python3 -m json.tool

stats:
	@curl -s http://localhost:8000/api/graph/statistics | python3 -m json.tool

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf backend/.venv 2>/dev/null || true
	rm -rf frontend/node_modules 2>/dev/null || true
	rm -rf frontend/.next 2>/dev/null || true
	@echo "Cleaned build artifacts."
