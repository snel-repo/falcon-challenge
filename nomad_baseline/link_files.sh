#!/bin/bash
while IFS= read -r path; do
    mkdir -p "/home/bkarpo2/bin/falcon-challenge/nomad_baseline/submittable_models$(dirname "$path")"
    cp -R "$path" "/home/bkarpo2/bin/falcon-challenge/nomad_baseline/submittable_models$(dirname "$path")"
done < paths.txt