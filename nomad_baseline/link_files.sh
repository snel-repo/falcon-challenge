#!/bin/bash
while IFS= read -r path; do
    mkdir -p "/home/bkarpo2/bin/falcon-challenge/nomad_baseline/submittable_models$(dirname "$path")"
    cp -R "$path" "/home/bkarpo2/bin/falcon-challenge/nomad_baseline/submittable_models$(dirname "$path")"

    # Get the directory two levels up
    two_levels_up=$(dirname "$(dirname "$path")")
    # Check if "input_data" is a child directory of the directory two levels up
    if [[ -d "$two_levels_up/input_data" ]]; then
        # Copy the "input_data" directory
        cp -R "$two_levels_up/input_data" "/home/bkarpo2/bin/falcon-challenge/nomad_baseline/submittable_models/$two_levels_up/input_data"
    fi
done < paths.txt