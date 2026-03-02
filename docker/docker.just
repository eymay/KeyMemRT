# KeyMemRT Docker-based Build System
# Usage: just --justfile docker/docker.just <recipe>
# Runs all build commands inside Docker container with proper volume mounts

# Configuration
IMAGE_NAME := "keymemrt"
IMAGE_TAG := "latest"
DOCKERFILE_DIR := justfile_directory()

# Networks
NETWORKS := "mlp lola lenet alexnet vgg yolo resnet1 resnet8 resnet10 resnet18 resnet20 resnet32 resnet34 resnet44 resnet50 resnet56 resnet110"

# ============================================================================
# Docker Management
# ============================================================================

# Default recipe - show available commands
default:
    @just --justfile docker/docker.just --list

# Ensure Docker image exists, build if needed
_ensure-image:
    #!/usr/bin/env bash
    if ! docker image inspect {{IMAGE_NAME}}:{{IMAGE_TAG}} >/dev/null 2>&1; then
        echo "=== Docker image not found, building {{IMAGE_NAME}}:{{IMAGE_TAG}} ==="
        echo "Building with compiler from source (this may take a while...)"
        docker build -f {{DOCKERFILE_DIR}}/Dockerfile -t {{IMAGE_NAME}}:{{IMAGE_TAG}} -f {{DOCKERFILE_DIR}}/Dockerfile {{DOCKERFILE_DIR}}/..
        echo "✅ Docker image built successfully"
    fi

# Build Docker image
build-image compiler_version="":
    @echo "=== Building Docker image ==="
    @if [ -n "{{compiler_version}}" ]; then \
        echo "Building with compiler version: {{compiler_version}}"; \
        docker build -f {{DOCKERFILE_DIR}}/Dockerfile --build-arg COMPILER_VERSION={{compiler_version}} -t {{IMAGE_NAME}}:{{IMAGE_TAG}} .; \
    else \
        echo "Building with compiler from source (this may take a while...)"; \
        docker build -f {{DOCKERFILE_DIR}}/Dockerfile -t {{IMAGE_NAME}}:{{IMAGE_TAG}} .; \
    fi
    @echo "✅ Docker image built: {{IMAGE_NAME}}:{{IMAGE_TAG}}"

# Build image with specific compiler version from GitHub releases
build-image-release version:
    @echo "=== Building with KeyMemRT-Compiler {{version}} ==="
    docker build -f {{DOCKERFILE_DIR}}/Dockerfile --build-arg COMPILER_VERSION={{version}} -t {{IMAGE_NAME}}:{{IMAGE_TAG}} .

# Rebuild image from scratch (no cache - use for troubleshooting)
rebuild-image:
    @echo "=== Rebuilding Docker image (no cache) ==="
    @echo "⚠️  This will rebuild everything from scratch (slow)"
    docker build -f {{DOCKERFILE_DIR}}/Dockerfile --no-cache -t {{IMAGE_NAME}}:{{IMAGE_TAG}} .

# Remove Docker image
remove-image:
    @echo "=== Removing Docker image ==="
    docker rmi {{IMAGE_NAME}}:{{IMAGE_TAG}}

# Show Docker configuration
config:
    @echo "Docker Configuration:"
    @echo "  Image: {{IMAGE_NAME}}:{{IMAGE_TAG}}"
    @echo "  Host KEYMEMRT_COMPILER: ${KEYMEMRT_COMPILER:-not set}"
    @echo "  Volumes:"
    @echo "    - ./build → /workspace/build"
    @echo "    - /tmp/keymemrt → /tmp/keymemrt"
    @if [ -n "$KEYMEMRT_COMPILER" ]; then echo "    - $KEYMEMRT_COMPILER → /compiler (read-only)"; fi
    @echo "  Environment:"
    @echo "    - OMP_NUM_THREADS=4"
    @echo "    - CC=clang, CXX=clang++"

# Shell into container for debugging
shell: _ensure-image setup
    #!/usr/bin/env bash
    echo "=== Starting interactive shell in container ==="
    set -euo pipefail
    VOLUMES="-v $(pwd):/workspace -v $(pwd)/build:/workspace/build -v /tmp/keymemrt:/tmp/keymemrt"
    if [ -n "${KEYMEMRT_COMPILER:-}" ]; then
        VOLUMES="$VOLUMES -v $KEYMEMRT_COMPILER:/compiler:ro"
    fi
    docker run --rm -it $VOLUMES \
        -e OMP_NUM_THREADS=4 \
        -e CC=clang -e CXX=clang++ \
        -e KEYMEMRT_COMPILER=/compiler \
        {{IMAGE_NAME}}:{{IMAGE_TAG}} /bin/bash

# Helper to run docker with proper mounts
_docker-run +args: _ensure-image setup
    #!/usr/bin/env bash
    set -euo pipefail
    VOLUMES="-v $(pwd):/workspace -v $(pwd)/build:/workspace/build -v /tmp/keymemrt:/tmp/keymemrt"
    # Note: We use the compiler built into the Docker image at /compiler
    # Host KEYMEMRT_COMPILER is ignored for Docker builds
    docker run --rm $VOLUMES \
        -e OMP_NUM_THREADS=4 \
        -e CC=clang -e CXX=clang++ \
        -e KEYMEMRT_COMPILER=/compiler \
        {{IMAGE_NAME}}:{{IMAGE_TAG}} {{args}}

# ============================================================================
# Build Commands (mirrored from Justfile)
# ============================================================================

# Setup build directory
setup:
    @mkdir -p build/mlir build/cpp build/bin build/profiles build/results build/logs
    @mkdir -p /tmp/keymemrt
    @# Regenerate CMake config if needed (for mounted source code)
    @if [ ! -f build/build.config ] || [ CMakeLists.txt -nt build/build.config ]; then \
        cmake -B build -S . > /dev/null 2>&1 || true; \
    fi
    @echo "✅ Directory structure ready"

# Generate MLIR from Python
gen-mlir network: _ensure-image setup
    @echo "=== Generating MLIR for {{network}} in Docker ==="
    @just --justfile docker/docker.just _docker-run make {{network}}-mlir

# Full optimization pipeline for a network
optimize network: _ensure-image setup
    @echo "=== Running optimization pipeline for {{network}} in Docker ==="
    @just --justfile docker/docker.just _docker-run make {{network}}-prefetch-opt

# Build executables for a network
build-executables network: _ensure-image setup
    @echo "=== Building executables for {{network}} in Docker ==="
    @just --justfile docker/docker.just _docker-run make {{network}}-baseline-exe {{network}}-opt-exe {{network}}-kc-exe {{network}}-prefetch-exe -j 20

# Build server executables for a network
build-executables-server network: _ensure-image setup
    @echo "=== Building server executables for {{network}} in Docker ==="
    @just --justfile docker/docker.just _docker-run make {{network}}-generate-keys-baseline {{network}}-generate-keys-prefetch {{network}}-generate-keys-fhelipe build/bin/{{network}}_baseline_server_exe build/bin/{{network}}_prefetch_server_exe build/bin/{{network}}_fhelipe_server_exe -j 31

# Build server executables for specific config
build-executables-server-config config: _ensure-image setup
    @echo "=== Building server executables for config {{config}} in Docker ==="
    @just --justfile docker/docker.just _docker-run make mlp-generate-keys-{{config}} lola-generate-keys-{{config}} lenet-generate-keys-{{config}} resnet-generate-keys-{{config}} build/bin/mlp_{{config}}_server_exe build/bin/lola_{{config}}_server_exe build/bin/lenet_{{config}}_server_exe build/bin/resnet_{{config}}_server_exe -j 31

# Build server executables for multiple networks
build-executables-servers +networks: _ensure-image setup
    @echo "=== Building server executables for networks: {{networks}} in Docker ==="
    @just --justfile docker/docker.just _docker-run bash -c 'set -euo pipefail; key_types=("baseline" "prefetch" "fhelipe"); networks=({{networks}}); targets=(); for network in "$${networks[@]}"; do for key_type in "$${key_types[@]}"; do targets+=("$${network}-generate-keys-$${key_type}"); done; done; for network in "$${networks[@]}"; do for key_type in "$${key_types[@]}"; do targets+=("build/bin/$${network}_$${key_type}_server_exe"); done; done; make "$${targets[@]}" -i -j 20'

# ============================================================================
# Key Generation (mirrored from Justfile)
# ============================================================================

gen-keys-baseline network: _ensure-image setup
    @echo "🔑 Generating {{network}} baseline keys in Docker"
    @just --justfile docker/docker.just _docker-run bash -c 'mkdir -p /tmp/keymemrt/baseline/{{network}} && SERIALIZED_DATA_DIR=/tmp/keymemrt/baseline/{{network}} ./build/bin/{{network}}_baseline_client_exe --key-mode ignore'

gen-keys-opt network: _ensure-image setup
    @echo "🔑 Generating {{network}} optimized keys in Docker"
    @just --justfile docker/docker.just _docker-run bash -c 'mkdir -p /tmp/keymemrt/opt/{{network}} && SERIALIZED_DATA_DIR=/tmp/keymemrt/opt/{{network}} ./build/bin/{{network}}_opt_client_exe --key-mode imperative'

gen-keys-kc network: _ensure-image setup
    @echo "🔑 Generating {{network}} key-compressed keys in Docker"
    @just --justfile docker/docker.just _docker-run bash -c 'mkdir -p /tmp/keymemrt/kc/{{network}} && SERIALIZED_DATA_DIR=/tmp/keymemrt/kc/{{network}} ./build/bin/{{network}}_kc_client_exe --key-mode imperative'

gen-keys-prefetch network: _ensure-image setup
    @echo "🔑 Generating {{network}} prefetch keys in Docker"
    @just --justfile docker/docker.just _docker-run bash -c 'mkdir -p /tmp/keymemrt/prefetch/{{network}} && SERIALIZED_DATA_DIR=/tmp/keymemrt/prefetch/{{network}} ./build/bin/{{network}}_prefetch_client_exe --key-mode prefetching'

gen-keys-all network: _ensure-image setup
    @echo "🔑 Generating ALL {{network}} keys in Docker"
    just --justfile docker/docker.just gen-keys-baseline {{network}}
    just --justfile docker/docker.just gen-keys-opt {{network}}
    just --justfile docker/docker.just gen-keys-kc {{network}}
    just --justfile docker/docker.just gen-keys-prefetch {{network}}

# ============================================================================
# Server Execution (mirrored from Justfile)
# ============================================================================

run-baseline-server network logsuffix="": _ensure-image setup
    @echo "--- Running baseline server for {{network}} in Docker ---"
    @just --justfile docker/docker.just _docker-run bash -c 'mkdir -p /tmp/keymemrt/baseline/{{network}} build/results/baseline/ build/logs && make {{network}}-generate-keys-baseline build/bin/{{network}}_baseline_server_exe -j 5 && OMP_NUM_THREADS=4 SERIALIZED_DATA_DIR=/tmp/keymemrt/baseline/{{network}} ./build/bin/{{network}}_baseline_server_exe --key-mode ignore --result-dir build/results/baseline/ > build/logs/{{network}}_baseline_server_logs{{logsuffix}}.txt'

run-opt-server network verbose="": _ensure-image setup
    @echo "--- Running opt server for {{network}} in Docker ---"
    @just --justfile docker/docker.just _docker-run bash -c 'mkdir -p /tmp/keymemrt/opt/{{network}} build/results/opt/ build/logs && make {{network}}-generate-keys-opt build/bin/{{network}}_opt_server_exe && OMP_NUM_THREADS=4 SERIALIZED_DATA_DIR=/tmp/keymemrt/opt/{{network}} ./build/bin/{{network}}_opt_server_exe --key-mode imperative --result-dir build/results/opt/ {{verbose}} > build/logs/{{network}}_opt_server_logs.txt'

run-kc-server network: _ensure-image setup
    @echo "--- Running kc server for {{network}} in Docker ---"
    @just --justfile docker/docker.just _docker-run bash -c 'mkdir -p /tmp/keymemrt/kc/{{network}} build/results/kc/ build/logs && make {{network}}-generate-keys-kc build/bin/{{network}}_kc_server_exe -j 5 && OMP_NUM_THREADS=4 SERIALIZED_DATA_DIR=/tmp/keymemrt/kc/{{network}} ./build/bin/{{network}}_kc_server_exe --key-mode imperative --result-dir build/results/kc/ > build/logs/{{network}}_kc_server_logs.txt'

run-prefetch-server network verbose="": _ensure-image setup
    @echo "--- Running prefetch server for {{network}} in Docker ---"
    @just --justfile docker/docker.just _docker-run bash -c 'set -euo pipefail; mkdir -p /tmp/keymemrt/prefetch/{{network}} build/results/prefetch/ build/logs; make {{network}}-generate-keys-prefetch build/bin/{{network}}_prefetch_server_exe -j 5; if [[ "{{network}}" == "mlp" || "{{network}}" == "lola" ]]; then PREFETCH_SAT=600; elif [[ "{{network}}" == "resnet1" || "{{network}}" == "lenet" ]]; then PREFETCH_SAT=2000; elif [[ "{{network}}" == "resnet8" || "{{network}}" == "resnet10" || "{{network}}" == "resnet20" || "{{network}}" == "resnet32" || "{{network}}" == "resnet44" || "{{network}}" == "resnet56" ]]; then PREFETCH_SAT=2500; elif [[ "{{network}}" == "alexnet" || "{{network}}" == "resnet18" || "{{network}}" == "resnet34" ]]; then PREFETCH_SAT=3500; elif [[ "{{network}}" == "resnet50" || "{{network}}" == "wresnet50-2" || "{{network}}" == "vgg" || "{{network}}" == "yolo" ]]; then PREFETCH_SAT=4500; else PREFETCH_SAT=3500; fi; OMP_NUM_THREADS=4 SERIALIZED_DATA_DIR=/tmp/keymemrt/prefetch/{{network}} ./build/bin/{{network}}_prefetch_server_exe --key-mode prefetching --prefetch-sat $$PREFETCH_SAT {{verbose}} --result-dir build/results/prefetch/ > build/logs/{{network}}_prefetch_server_logs.txt'

run-fhelipe-server network logsuffix="": _ensure-image setup
    @echo "--- Running fhelipe server for {{network}} in Docker ---"
    @just --justfile docker/docker.just _docker-run bash -c 'mkdir -p /tmp/keymemrt/fhelipe/{{network}} build/results/fhelipe/ build/logs && make {{network}}-generate-keys-fhelipe build/bin/{{network}}_fhelipe_server_exe -j 5 && OMP_NUM_THREADS=4 SERIALIZED_DATA_DIR=/tmp/keymemrt/fhelipe/{{network}} ./build/bin/{{network}}_fhelipe_server_exe --key-mode ignore --result-dir build/results/fhelipe/ > build/logs/{{network}}_fhelipe_server_logs{{logsuffix}}.txt'

# Run all server variants for a network
run-servers network: _ensure-image setup
    @echo "=== Running all benchmarks for {{network}} in Docker ==="
    just --justfile docker/docker.just build-executables-server {{network}}
    just --justfile docker/docker.just run-baseline-server {{network}}
    just --justfile docker/docker.just run-opt-server {{network}}
    just --justfile docker/docker.just run-fhelipe-server {{network}}
    just --justfile docker/docker.just run-prefetch-server {{network}}

# ============================================================================
# Batch Operations
# ============================================================================

# Generate MLIR for all networks
gen-all: _ensure-image setup
    @echo "=== Generating MLIR for all networks in Docker ==="
    @just --justfile docker/docker.just _docker-run make all-mlir

# Optimize all networks
optimize-all: _ensure-image setup
    @echo "=== Optimizing all networks in Docker ==="
    @just --justfile docker/docker.just _docker-run make all-optimized

# Build all executables
build-all: _ensure-image setup
    @echo "=== Building all executables in Docker ==="
    @just --justfile docker/docker.just _docker-run make all-executables

# Run all benchmarks
benchmark network: _ensure-image setup
    @echo "=== Running all benchmarks for {{network}} in Docker ==="
    just --justfile docker/docker.just build-executables {{network}}
    @just --justfile docker/docker.just _docker-run bash -c 'just _run-baseline {{network}}'
    @just --justfile docker/docker.just _docker-run bash -c 'just _run-opt {{network}}'
    @just --justfile docker/docker.just _docker-run bash -c 'just _run-kc {{network}}'
    @just --justfile docker/docker.just _docker-run bash -c 'just _run-prefetch {{network}}'

# ============================================================================
# Utility Commands
# ============================================================================

# Clean build artifacts (on host and in container)
clean:
    @echo "=== Cleaning build artifacts ==="
    @rm -rf build/mlir/* build/cpp/* build/bin/* build/profiles/* build/results/* build/logs/*
    @echo "✅ Build artifacts cleaned"

# List available networks
list-networks:
    @echo "Available networks:"
    @echo "  {{NETWORKS}}"

# Show build status for a network
status network: _ensure-image setup
    @echo "=== Checking status for {{network}} in Docker ==="
    @just --justfile docker/docker.just _docker-run make status-{{network}}

# Run tests
test: _ensure-image setup
    @echo "=== Running tests in Docker ==="
    @just --justfile docker/docker.just _docker-run bash -c 'cd build && cmake -B . -S .. -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && ninja keymemrt_tests && ./keymemrt_tests'

# ============================================================================
# Examples and Quick Start
# ============================================================================

# Quick start: Build image and run MLP example
quickstart:
    @echo "=== KeyMemRT Docker Quick Start ==="
    @echo "1. Building Docker image..."
    just --justfile docker/docker.just build-image
    @echo "2. Setting up directories..."
    just --justfile docker/docker.just setup
    @echo "3. Running MLP servers..."
    just --justfile docker/docker.just run-servers mlp
    @echo "✅ Quick start complete! Check build/results/ for outputs"

# Example: Run with host compiler
example-host-compiler:
    @echo "=== Example: Using host compiler ==="
    @echo "Make sure KEYMEMRT_COMPILER is set:"
    @echo "  export KEYMEMRT_COMPILER=/path/to/KeyMemRT-Compiler"
    @echo "Then run:"
    @echo "  just --justfile docker/docker.just build-image"
    @echo "  just --justfile docker/docker.just run-servers mlp"

# Example: Run with downloaded compiler release
example-release-compiler:
    @echo "=== Example: Using release compiler ==="
    @echo "Build with a specific release version:"
    @echo "  just --justfile docker/docker.just build-image-release v1.0.0"
    @echo "Then run:"
    @echo "  just --justfile docker/docker.just run-servers mlp"
