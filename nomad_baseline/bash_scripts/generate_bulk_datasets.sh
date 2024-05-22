#!/bin/bash 

# Directory containing the .nwb files
# DATA_DIR="/home/bkarpo2/bin/stability-benchmark/data/h1/held_in_calib"
DATA_DIR="/snel/home/bkarpo2/bin/falcon-challenge/data/m1/sub-MonkeyL-held-in-calib"

# Loop over all .nwb files in the directory
for FILE in $DATA_DIR/*.nwb
do
  RUN_FLAG=$(basename "$FILE" .nwb)
  RUN_FLAG=${RUN_FLAG%_calib}
  # Run make_lfads_run.py with the current file
  python make_lfads_run.py $FILE M1 "$RUN_FLAG"
done