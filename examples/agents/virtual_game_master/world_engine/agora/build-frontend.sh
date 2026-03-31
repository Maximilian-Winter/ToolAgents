#!/bin/bash
set -e
echo "=== Building Agora Frontend ==="
cd "$(dirname "$0")/frontend"
npm install
npm run build
echo "=== Build complete. Restart the server to serve the new frontend. ==="
