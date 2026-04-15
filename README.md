# RMIT Advanced Programming for Data Science

Course materials and assignments. Dependencies are declared in `pyproject.toml` (Python **≥ 3.9**; `.python-version` pins **3.9** for local tooling).

## Install dependencies

### Option A: uv (recommended)

[uv](https://docs.astral.sh/uv/) installs the locked set from `uv.lock` and manages a virtual environment.

1. Install uv (see [Installing uv](https://docs.astral.sh/uv/getting-started/installation/)), for example on macOS/Linux:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. From the repository root:

   ```bash
   uv sync
   ```

   This creates `.venv` and installs the project in editable mode with pinned versions.

3. Run tools with uv’s environment, for example:

   ```bash
   uv run jupyter lab
   # or
   uv run python -m pytest
   ```

   Alternatively, activate the venv: `source .venv/bin/activate` (Unix) or `.venv\Scripts\activate` (Windows).

**Without using the lockfile** (resolves from `pyproject.toml` only):

```bash
uv pip install -e .
```

### Option B: conda

Use conda (or [Mamba](https://mamba.readthedocs.io/)) when you prefer a conda environment for notebooks and scientific stacks.

1. Create and activate an environment with a compatible Python:

   ```bash
   conda create -n rmit-apds python=3.9
   conda activate rmit-apds
   ```

2. Install this package and its dependencies from `pyproject.toml`:

   ```bash
   pip install -e .
   ```

   Use the same `pip` tied to that conda env so packages land in `rmit-apds`.

3. Open Jupyter or run scripts as usual inside the activated environment.

---

**Note:** Core dependencies include Jupyter, NumPy, Pandas, Matplotlib, Seaborn, and related libraries listed in `pyproject.toml`. Adjust the Python version in the conda example if you standardize on 3.10+.
