#!/bin/bash
# run_in_docker.sh - Execute quick_compile.sh inside Docker container
# Run this from your host machine (outside Docker)

set -e

# Try to detect running container
CONTAINER_ID=$(docker ps -q --filter "ancestor=*gem5*" | head -1)

# If no gem5 container found, try any running container with /work mount
if [ -z "$CONTAINER_ID" ]; then
    CONTAINER_ID=$(docker ps -q | head -1)
fi

# If still no container, try to find stopped containers
if [ -z "$CONTAINER_ID" ]; then
    echo "No running container found. Trying stopped containers..."
    CONTAINER_ID=$(docker ps -aq | head -1)
    if [ -n "$CONTAINER_ID" ]; then
        echo "Starting container $CONTAINER_ID..."
        docker start "$CONTAINER_ID"
    fi
fi

# Verify we have a container
if [ -z "$CONTAINER_ID" ]; then
    echo "Error: No Docker container found!"
    echo "Start your gem5 Docker container first, or specify container ID:"
    echo "  docker ps -a    # list all containers"
    echo "Then edit this script or run manually:"
    echo "  docker exec -it <CONTAINER_ID> /work/src/softmax/quick_compile.sh"
    exit 1
fi

echo "Using container: $CONTAINER_ID"
echo "Running compilation in Docker..."
echo "=========================================="
echo ""

# Execute the compilation script inside Docker with login shell to get full PATH
docker exec -it "$CONTAINER_ID" bash -l -c "/work/src/softmax/quick_compile.sh"

echo ""
echo "=========================================="
echo "Compilation complete!"
echo "Check results in: src/softmax/bin/"
