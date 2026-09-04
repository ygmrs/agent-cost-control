.PHONY: install test bench sweep explain clean help

install:   ## install with dev extras
	pip install -e ".[dev]"

test:      ## run the test suite
	pytest -q

bench:     ## reproduce the ablation table
	agent-cost-control bench

sweep:     ## cost against solve rate as the ceiling moves
	agent-cost-control sweep

explain:   ## trace one task's decisions (T=task_0150)
	agent-cost-control explain $(or $(T),task_0150)

clean:
	rm -rf .pytest_cache *.egg-info results.json
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  %-10s %s\n", $$1, $$2}'
