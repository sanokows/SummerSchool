# Installation

## 1. Install uv

Linux/macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## 2. Clone and install the environment

```bash
git clone https://github.com/sanokows/SummerSchool.git
cd SummerSchool
uv sync --locked
```

This installs Python 3.12 and all dependencies in `.venv`.

## 3. Start JupyterLab

```bash
uv run jupyter lab
```
