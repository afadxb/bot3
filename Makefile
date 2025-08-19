.PHONY: format lint test up down

format:
	black .

lint:
	flake8 barchart_swing_bot tests

test:
	pytest

up:
	docker-compose up -d

down:
	docker-compose down
