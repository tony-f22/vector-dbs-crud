## ATTENTION! uv will automatically manage the virtual environment

## install packages, install pre-commit
install-dev:
	uv sync --all-extras
# 	uv run pre-commit install

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	find . -name '.pytest_cache' -exec rm -fr {} +

clean: clean-pyc

## uninstall all dev packages
uninstall-dev:
	rm -rf .venv uv.lock

## Run linting checks
check:
	uv run ruff check src
	uv run ruff format --check src
	uv run mypy src

## reformat the files using the formatters
format:
	uv run ruff check --fix src
	uv run ruff format src

## down build docker image
drop-image:
	docker compose -f docker-compose.yaml down -v --rmi all

## build docker image
build-image:
	docker compose -f docker-compose.yaml build

## run docker image
run-image:
	docker compose -f docker-compose.yaml up -d

## drop containers
drop-containers:
	docker compose -f docker-compose.yaml down --volumes --remove-orphans
