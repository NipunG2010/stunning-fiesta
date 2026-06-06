#!/usr/bin/env bash
# Download one month of the Lichess open standard database.
# Usage: ./scripts/download_lichess.sh YYYY-MM [dest_dir]
#   ./scripts/download_lichess.sh 2024-01 data/raw
set -euo pipefail

MONTH="${1:?usage: $0 YYYY-MM [dest_dir]}"
DEST="${2:-data/raw}"
URL="https://database.lichess.org/standard/lichess_db_standard_rated_${MONTH}.pgn.zst"
FILE="${DEST}/lichess_db_standard_rated_${MONTH}.pgn.zst"

mkdir -p "$DEST"

if [[ -f "$FILE" ]]; then
    echo "Already downloaded: $FILE"
    exit 0
fi

echo "Downloading $URL"
echo "Destination: $FILE"
curl --fail --location --progress-bar --output "$FILE" "$URL"

echo "Done. File size: $(du -h "$FILE" | cut -f1)"
