#!/bin/bash

# Parent directory containing the directories
PARENT_DIR="/snel/share/runs/falcon"

# Loop over all directories in the parent directory
for DIR in $PARENT_DIR/*/
do
  # Check if 'H1' is in the directory name and 'pbt_run' is not a subdirectory
  if [[ $DIR == *"coinspkrem"* && ! -d "$DIR/pbt_run" ]]
  then
    # Run pbt.py with the current directory
    echo "$DIR"
    python pbt.py "$DIR"
  fi
done