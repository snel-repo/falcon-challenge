#!/bin/bash

# Parent directory containing the directories
PARENT_DIR="/snel/share/runs/falcon"
DATA_DIR="/home/bkarpo2/bin/stability-benchmark/data/h1/held_in_calib/"

# Loop over all directories in the parent directory
for DIR in $PARENT_DIR/*/
do
  # Check if the directory does not contain a PDF file
  if [[ -d "$DIR/pbt_run" && -z $(find "$DIR" -name "*.pdf") ]]
  then
    # Run eval_lfads.py with the current directory
    echo "$DIR"
    RUN_FLAG=$(basename "$DIR")
    DS_FLAG=${RUN_FLAG:3:-1}_${RUN_FLAG: -1}_calib.nwb
    CLOSEST_FILE=$(ls $DATA_DIR | grep "$DS_FLAG")
    python /home/bkarpo2/bin/stability-benchmark/nomad_baseline/eval_lfads.py "$DATA_DIR/$CLOSEST_FILE" H1 ${RUN_FLAG:3}
  fi
done
