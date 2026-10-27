#!/usr/bin/env bash
# Westminster_graph from blocks.py's result to Build (LOG 2026-10-21 #W1). Run from Knoxmap/:
#   bash ../Building_types/src/area/westminster_chain.sh
set -eo pipefail
PY="$(pwd)/.venv/Scripts/python.exe"; S="../Building_types/src"; A="../Building_types/data/westminster_core"
M="output/Westminster_graph"; export PYTHONIOENCODING=utf-8
step() { echo "=== $(date +%H:%M:%S) $*"; }
step lidar_outlines; "$PY" $S/buildings/lidar_outlines.py $M $A | tail -1
step split_units;    "$PY" $S/buildings/split_units.py $M $A | tail -1
step shops;          "$PY" $S/buildings/shops.py $M $A --epc ../Building_types/data/central_london/epc_nondomestic.csv | tail -1
step rails;          "$PY" $S/streets/rails.py $M 2>&1 | grep -v Warn | tail -1
step plots2;         "$PY" $S/ground/plots2.py $M $A --inspire-attrs $A/inspire_town.geojson | tail -2
step trees;          "$PY" $S/ground/trees.py $M $A | tail -1
step sports;         "$PY" $S/ground/sports.py $M | tail -1
step rails_apply;    "$PY" $S/streets/rails.py $M --apply 2>&1 | grep -v Warn | tail -1
step kerbs;          "$PY" $S/streets/kerbs.py $M | tail -1
step regional;       "$PY" $S/regional/export_regional.py $A ../Building_types/palettes/english.json $M | tail -1
step roof_plan;      "$PY" $S/regional/roof_plan.py $M $A | tail -1
step listed;         "$PY" ../../asset_creator/listed_building_maker/apply.py $M $A | tail -1
step build;          "$PY" -m knoxbuild $M | grep -E "buildings generated|wrote .*tbx|Traceback|Error"
step validate;       "$PY" tools/validate_tbx.py $M/buildings | tail -1
step done
