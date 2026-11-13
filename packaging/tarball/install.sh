#!/usr/bin/env sh
# Installs lastengine from this release folder.
# Usage: ./install.sh [PREFIX]   (default PREFIX: /usr/local; use ~/.local for no root)
set -eu
PREFIX="${1:-/usr/local}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$PREFIX/bin"
install -m 755 "$HERE/lastengine" "$PREFIX/bin/lastengine"
"$PREFIX/bin/lastengine" --version
echo "Installed to $PREFIX/bin/lastengine. Needs an NVIDIA driver with CUDA; run 'lastengine probe' to check."
