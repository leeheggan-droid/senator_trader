#!/usr/bin/env bash
# Download congressional trading disclosure data from S3.
# Run once: bash download_data.sh
set -e

mkdir -p data/cache

SENATE_URLS=(
  "https://senate-stock-watcher-data.s3.us-west-2.amazonaws.com/aggregate/all_transactions.json"
  "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json"
)
HOUSE_URLS=(
  "https://house-stock-watcher-data.s3.us-west-2.amazonaws.com/data/all_transactions.json"
  "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"
)

download_first() {
  local dest=$1; shift
  for url in "$@"; do
    echo "Trying $url ..."
    if curl -sf -L --max-time 30 -o "$dest" "$url"; then
      size=$(wc -c < "$dest")
      if [ "$size" -gt 10000 ]; then
        echo "  OK — ${size} bytes saved to $dest"
        return 0
      else
        echo "  Too small (${size} bytes) — trying next"
      fi
    else
      echo "  Failed — trying next"
    fi
  done
  echo "ERROR: all URLs failed for $dest"
  return 1
}

download_first data/cache/senate_all.json "${SENATE_URLS[@]}"
download_first data/cache/house_all.json  "${HOUSE_URLS[@]}"

echo ""
echo "Done. Run: venv/bin/python backtest.py --save"
