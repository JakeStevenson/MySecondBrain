#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
  echo "Please provide an argument."
  exit 1
fi

# Store the argument in a variable
arg=$1

# Generate the JSON file
json_file="payload.json"
echo "{\"url\":\"$arg\"}" > "$json_file"

# Send the JSON file using http POST
http POST http://localhost:5173/read_article < "$json_file"

# Cleanup the JSON file
rm "$json_file"