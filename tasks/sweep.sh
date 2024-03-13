#!/bin/bash

. .venv/bin/activate

set -a
. .env
set +a

if [[ $WANDB_API_KEY == 0* ]]; then
    echo "WANDB_API_KEY starts with 0."
else
    echo "WANDB_API_KEY does not start with 0."
fi

echo $WANDB_API_KEY

output=$(wandb sweep --project rotten-tomatoes $1 2>&1)
sweep_id=$(echo "$output" | grep -oP '(?<=wandb: Creating sweep with ID: )\w+')

if [ -n "$sweep_id" ]; then
    echo "Extracted sweep ID: $sweep_id"
else
    echo "Failed to extract sweep ID."
fi

wandb agent szafrixxx/rotten-tomatoes/$sweep_id