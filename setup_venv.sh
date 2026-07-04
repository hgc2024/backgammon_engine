#!/usr/bin/env bash

set -euo pipefail

VENV_DIR=".venv"

echo "Creating virtual environment '$VENV_DIR'..."

if command -v python3.10 >/dev/null 2>&1; then
	PYTHON_CMD="python3.10"
elif command -v python3 >/dev/null 2>&1; then
	PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
	PYTHON_CMD="python"
else
	echo "Error: Python is not installed or not available in PATH."
	exit 1
fi

"$PYTHON_CMD" -m venv "$VENV_DIR"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
	echo "Error: Failed to create virtual environment in '$VENV_DIR'."
	exit 1
fi

echo "Installing dependencies from requirements.txt..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/pip" install -r requirements.txt

echo ""
echo "Setup complete!"
echo "To activate the environment in the future, run:"
echo "source $VENV_DIR/bin/activate"