#!/bin/bash
while IFS= read -r path; do
    cp -r "../../../..$path" .
done < paths.txt