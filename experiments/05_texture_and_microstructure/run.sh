#!/usr/bin/env bash
# Detail texture, 3D pore structure, microstructure statistics and geological-state control (Sections 4.1-4.2).
set -e
cd "$(dirname "$0")"
python tex59.py; python micro59.py; python vis3d59.py; python geo_ctrl59.py
