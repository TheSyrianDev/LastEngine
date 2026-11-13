#!/usr/bin/env bash
# Builds the Linux release tarball: lastengine binary, install script, licences, checksums.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/Cargo.toml" | head -1)"
OUT="$ROOT/dist/lastengine-$VERSION-linux-x86_64"
cargo build --release --locked --manifest-path "$ROOT/Cargo.toml"
rm -rf "$OUT" && mkdir -p "$OUT"
install -m 755 "$ROOT/target/release/lastengine" "$OUT/lastengine"
install -m 755 "$ROOT/packaging/tarball/install.sh" "$OUT/install.sh"
install -m 644 "$ROOT/packaging/tarball/README.txt" "$OUT/README.txt"
install -m 644 "$ROOT/LICENSE" "$OUT/LICENSE"
install -m 644 "$ROOT/NOTICE" "$OUT/NOTICE"
(cd "$OUT" && sha256sum lastengine install.sh README.txt LICENSE NOTICE > SHA256SUMS.txt)
tar -C "$ROOT/dist" -czf "$OUT.tar.gz" "$(basename "$OUT")"
sha256sum "$OUT.tar.gz"
