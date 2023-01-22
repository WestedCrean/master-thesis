#!/bin/bash

# Remove old container
docker rm -f master-thesis-workspace

# Build ML Workspace container
docker build -t tf-workspace .

# -u $(id -u):$(id -g)
# Run ML Workspace container
docker run -p 8080:8080 --name master-thesis-workspace --gpus all -v $(pwd):/workspace -it tf-workspace:latest