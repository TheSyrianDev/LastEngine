#!/usr/bin/env bash
# Tests a LastEngine release tarball in a clean container (no GPU): checksums, install, version, probe.
# Usage: container-test.sh <image> <tarball>
set -euo pipefail
IMAGE="$1"; TAR="$(readlink -f "$2")"
docker run --rm -v "$TAR:/pkg/le.tar.gz:ro" "$IMAGE" sh -c '
set -eu
if command -v pacman >/dev/null; then pacman -Syu --noconfirm >/dev/null 2>&1 && pacman -S --noconfirm tar ca-certificates >/dev/null 2>&1; fi
if command -v apt-get >/dev/null; then apt-get update >/dev/null 2>&1 && apt-get install -y -qq ca-certificates >/dev/null 2>&1; fi
cd /tmp && tar xzf /pkg/le.tar.gz && cd lastengine-*-linux-x86_64
sha256sum -c SHA256SUMS.txt >/dev/null && echo CHECKSUMS OK
./install.sh /root/.local
/root/.local/bin/lastengine --version
/root/.local/bin/lastengine probe | head -c 300; echo
'
