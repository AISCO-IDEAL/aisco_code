# AI Context & Instructions: AISCO Codebase

## 1. Environment & Paths
* **System:** Euler (Ubuntu HPC)
* **Virtual Environment:** Always use `/cluster/home/elihuber/AISCO_code/aisco_env/bin/activate`
* **Data & Results:** Located in `/cluster/work/fuge/AISCO/`
* **Slurm Logs:** Located in `/cluster/scratch/elihuber/` (primarily in `logs/` or `scrath_logs/`)

## 2. Tech Stack & Dependencies
* **Core:** Python, PyTorch.
* **Tracking:** All dependencies are strictly managed in `pyproject.toml`.
* **CRITICAL RULE:** Do NOT install new libraries or update existing ones without explicit approval from project maintainers. If an update seems necessary, verify compatibility and ASK FOR PERMISSION first.

## 3. Code Quality & Linting
* **Commands:** Always run `ruff check .`, `mypy .`, and `ruff format .` to check for style and type issues. Expect these to run automatically via pre-commit hooks.
* **Overrides:** Be extremely careful with `noqa` and `type: ignore`. Only use them when absolutely necessary, and always include an inline comment explaining why they are required.
* **Refactoring Boundary:** Too many branches or statements in existing code is acceptable. If a necessary refactor would be exceptionally large, leave the code as-is and add a comment explaining why it is not being refactored.

## 4. Testing Protocol
* **General:** Always test generated or modified code.
* **Unit Tests:** Write unit tests for all new features and modifications. Run these new tests, alongside existing tests, to guarantee zero regressions.
* **Reporting:** When instructed to run a test, execute it and report the exact results. If a test fails, analyze the root cause and report your findings. If you are unsure how to fix the issue, state so explicitly.

## 5. Git Workflow & Safety
* **CRITICAL RULE:** NEVER use Git commands unless explicitly told to do so.
* **CRITICAL RULE:** NEVER EVER push code to `main`.
* **Commits:** Write clear and very concise commit messages. Break up large architectural shifts into multiple, atomic commits with distinct messages.

## 6. Communication Protocol
* **Message Efficiency:** Optimize interactions to save prompt turns. If you need clarification on how to run a test, how to commit, or missing context, ask ALL of your questions in a single response rather than waiting for individual replies.
