#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [ "$(id -u)" -eq 0 ]; then
    echo "Applio does not require root permissions and should be run as a regular user."
    exit 1
fi

if [ ! -d env ]; then
    echo "Please run './run-install.sh' first to set up the environment."
    exit 1
fi

env/bin/python app.py --open
