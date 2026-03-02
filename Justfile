# KeyMem-RT Research Pipeline
# Usage: just <recipe>
# Heavy lifting (compilation, optimization) delegated to Makefile

# Default recipe - show available commands
default:
    @just --list

# Configuration
KEYMEMRT_COMPILER := env_var_or_default('KEYMEMRT_COMPILER', 'UNDEFINED')
NETWORKS_DIR := "benchmarks/networks"
BUILD_DIR := "build"
BIN_DIR := BUILD_DIR + "/bin"
LOGS_DIR := BUILD_DIR + "/logs"
RESULTS_DIR := justfile_directory() + "/" + BUILD_DIR + "/results/"

# Neural network benchmarks
NETWORKS := "mlp lola lenet alexnet vgg yolo resnet1 resnet8 resnet10 resnet18 resnet20 resnet32 resnet34 resnet44 resnet50 resnet56 resnet110"

SERIALIZED_BASE_DIR := env_var_or_default('SERIALIZED_DATA_DIR', '/tmp/keymemrt')

# ============================================================================
# Setup and Build
# ============================================================================

# Setup build directory and create network directories
setup:
    mkdir -p {{BUILD_DIR}} data {{RESULTS_DIR}} {{NETWORKS_DIR}} {{LOGS_DIR}}
    mkdir -p drivers include
    @echo "✅ Directory structure ready"

# Configure build with CMake (discovers dependencies, generates build.config)
configure:
    @echo "🔧 Running CMake to discover dependencies..."
    cmake -B {{BUILD_DIR}} -S .
    @echo "✅ Generated {{BUILD_DIR}}/build.config"
    @echo "   Makefile will now use auto-discovered include paths"
    @echo "Build directory: {{BUILD_DIR}}/"

# Build tests only with CMake
build-tests:
    cd build && cmake -B . -S .. -GNinja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    cd build && ninja keymemrt_tests

# Run tests
test: build-tests
    cd build && ./keymemrt_tests

# ============================================================================
# Neural Network Pipeline (delegates to Makefile)
# ============================================================================

# Generate MLIR from Python
gen-mlir network:
    @echo "=== Generating MLIR for {{network}} (via Makefile) ==="
    make {{network}}-mlir

# Full optimization pipeline for a network
optimize network:
    @echo "=== Running optimization pipeline for {{network}} (via Makefile) ==="
    make {{network}}-prefetch-opt

# Build executables for a network
build-executables network:
    @echo "=== Building executables for {{network}} (via Makefile) ==="
    make {{network}}-baseline-exe {{network}}-opt-exe {{network}}-kc-exe {{network}}-prefetch-exe -j 20

# Run all benchmarks for a network
benchmark network:
    @echo "=== Running all benchmarks for {{network}} ==="
    # Ensure executables are built
    just build-executables {{network}}
    # Run benchmarks (quick operations, no file tracking needed)
    just _run-baseline {{network}}
    just _run-opt {{network}}
    just _run-kc {{network}}
    just _run-prefetch {{network}}

# ============================================================================
# Benchmark Execution (Quick operations - no file tracking needed)
# ============================================================================

# Generate keys for ResNet variants
gen-keys-baseline network: 
    @echo "🔑 Generating {{network}} baseline keys"
    @mkdir -p {{SERIALIZED_BASE_DIR}}/baseline/{{network}}
    SERIALIZED_DATA_DIR={{SERIALIZED_BASE_DIR}}/baseline/{{network}} ./{{BIN_DIR}}/{{network}}_baseline_client_exe --key-mode ignore

gen-keys-opt network:
    @echo "🔑 Generating {{network}} optimized keys" 
    @mkdir -p {{SERIALIZED_BASE_DIR}}/opt/{{network}}
    SERIALIZED_DATA_DIR={{SERIALIZED_BASE_DIR}}/opt/{{network}} ./{{BIN_DIR}}/{{network}}_opt_client_exe --key-mode imperative

gen-keys-kc network:
    @echo "🔑 Generating {{network}} key-compressed keys"
    @mkdir -p {{SERIALIZED_BASE_DIR}}/kc/{{network}}
    SERIALIZED_DATA_DIR={{SERIALIZED_BASE_DIR}}/kc/{{network}} ./{{BIN_DIR}}/{{network}}_kc_client_exe --key-mode imperative

gen-keys-prefetch network:
    @echo "🔑 Generating {{network}} prefetch keys"
    @mkdir -p {{SERIALIZED_BASE_DIR}}/prefetch/{{network}}
    SERIALIZED_DATA_DIR={{SERIALIZED_BASE_DIR}}/prefetch/{{network}} ./{{BIN_DIR}}/{{network}}_prefetch_client_exe --key-mode prefetching

gen-keys-all network:
    @echo "🔑 Generating ALL {{network}} keys"
    just gen-keys-baseline {{network}}
    just gen-keys-opt {{network}}
    just gen-keys-kc {{network}}
    just gen-keys-prefetch {{network}}

    

# Run baseline benchmark
_run-baseline network:
    @echo "--- Running baseline benchmark for {{network}} ---"
    @mkdir -p /tmp/keymemrt/baseline {{RESULTS_DIR}} {{LOGS_DIR}}
    SERIALIZED_DATA_DIR=/tmp/keymemrt/baseline make {{network}}-generate-keys-baseline
    OMP_NUM_THREADS=4 SERIALIZED_DATA_DIR=/tmp/keymemrt/baseline \
    ./{{BIN_DIR}}/{{network}}_baseline_exe --key-mode ignore \
    --result-dir {{RESULTS_DIR}} > {{LOGS_DIR}}/{{network}}_baseline_logs.txt

# Run optimized benchmark
_run-opt network:
    @echo "--- Running opt benchmark for {{network}} ---"
    @mkdir -p /tmp/keymemrt/opt {{RESULTS_DIR}} {{LOGS_DIR}}
    OMP_NUM_THREADS=4 SERIALIZED_DATA_DIR=/tmp/keymemrt/opt \
    ./{{BIN_DIR}}/{{network}}_opt_exe --key-mode imperative \
    --result-dir {{RESULTS_DIR}} > {{LOGS_DIR}}/{{network}}_opt_logs.txt

# Run key-compressed benchmark
_run-kc network:
    @echo "--- Running key-compressed benchmark for {{network}} ---"
    @mkdir -p /tmp/keymemrt/kc {{RESULTS_DIR}} {{LOGS_DIR}}
    OMP_NUM_THREADS=4 SERIALIZED_DATA_DIR=/tmp/keymemrt/kc \
    ./{{BIN_DIR}}/{{network}}_kc_exe --key-mode imperative \
    --result-dir {{RESULTS_DIR}} > {{LOGS_DIR}}/{{network}}_kc_logs.txt

# Run prefetching benchmark
_run-prefetch network:
    @echo "--- Running prefetching benchmark for {{network}} ---"
    @mkdir -p /tmp/keymemrt/prefetch {{RESULTS_DIR}} {{LOGS_DIR}}
    OMP_NUM_THREADS=4 SERIALIZED_DATA_DIR=/tmp/keymemrt/prefetch \
    ./{{BIN_DIR}}/{{network}}_prefetch_exe --key-mode prefetching --prefetch-sat 1700 \
    --result-dir {{RESULTS_DIR}} > {{LOGS_DIR}}/{{network}}_prefetch_logs.txt

build-executables-server network:
    make {{network}}-generate-keys-baseline \
    {{network}}-generate-keys-prefetch \
    {{network}}-generate-keys-fhelipe \
    {{BIN_DIR}}/{{network}}_baseline_server_exe \
    {{BIN_DIR}}/{{network}}_prefetch_server_exe \
    {{BIN_DIR}}/{{network}}_fhelipe_server_exe \
    -j 31

build-executables-server-config config:
    make mlp-generate-keys-{{config}} \
    lola-generate-keys-{{config}} \
    lenet-generate-keys-{{config}} \
    resnet-generate-keys-{{config}} \
    {{BIN_DIR}}/mlp_{{config}}_server_exe \
    {{BIN_DIR}}/lola_{{config}}_server_exe \
    {{BIN_DIR}}/lenet_{{config}}_server_exe \
    {{BIN_DIR}}/resnet_{{config}}_server_exe \
    -j 31

build-executables-servers +networks:
    #!/usr/bin/env bash
    set -euo pipefail
    
    key_types=("baseline" "prefetch" "fhelipe")
    networks=({{networks}})
    
    targets=()
    
    # Generate key targets
    for network in "${networks[@]}"; do
        for key_type in "${key_types[@]}"; do
            targets+=("${network}-generate-keys-${key_type}")
        done
    done
    
    # Generate exe targets  
    for network in "${networks[@]}"; do
        for key_type in "${key_types[@]}"; do
            targets+=("{{BIN_DIR}}/${network}_${key_type}_server_exe")
        done
    done
    
    make "${targets[@]}" -i -j 20

run-baseline-server network logsuffix="":
    @echo "--- Running baseline server for {{network}} ---"
    @mkdir -p {{SERIALIZED_BASE_DIR}}/baseline/{{network}} {{RESULTS_DIR}}baseline/ {{LOGS_DIR}}
    make {{network}}-generate-keys-baseline {{BIN_DIR}}/{{network}}_baseline_server_exe -j 5
    OMP_NUM_THREADS=4 SERIALIZED_DATA_DIR={{SERIALIZED_BASE_DIR}}/baseline/{{network}} \
    ./{{BIN_DIR}}/{{network}}_baseline_server_exe --key-mode ignore \
    --result-dir {{RESULTS_DIR}}baseline/ > {{LOGS_DIR}}/{{network}}_baseline_server_logs{{logsuffix}}.txt

run-opt-server network verbose="":
    @echo "--- Running opt server for {{network}} ---"
    @mkdir -p {{SERIALIZED_BASE_DIR}}/opt/{{network}} {{RESULTS_DIR}}opt/ {{LOGS_DIR}}
    make {{network}}-generate-keys-opt {{BIN_DIR}}/{{network}}_opt_server_exe
    OMP_NUM_THREADS=4 SERIALIZED_DATA_DIR={{SERIALIZED_BASE_DIR}}/opt/{{network}} \
    ./{{BIN_DIR}}/{{network}}_opt_server_exe --key-mode imperative \
    --result-dir {{RESULTS_DIR}}opt/ {{verbose}} > {{LOGS_DIR}}/{{network}}_opt_server_logs.txt

run-kc-server network:
    @echo "--- Running kc server for {{network}} ---"
    @mkdir -p {{SERIALIZED_BASE_DIR}}/kc/{{network}} {{RESULTS_DIR}}kc/ {{LOGS_DIR}}
    make {{network}}-generate-keys-kc {{BIN_DIR}}/{{network}}_kc_server_exe -j 5
    OMP_NUM_THREADS=4 SERIALIZED_DATA_DIR={{SERIALIZED_BASE_DIR}}/kc/{{network}} \
    ./{{BIN_DIR}}/{{network}}_kc_server_exe --key-mode imperative \
    --result-dir {{RESULTS_DIR}}kc/ > {{LOGS_DIR}}/{{network}}_kc_server_logs.txt

run-prefetch-server network verbose="":
    #!/usr/bin/env bash
    set -euo pipefail
    echo "--- Running prefetch server for {{network}} ---"
    mkdir -p {{SERIALIZED_BASE_DIR}}/prefetch/{{network}} {{RESULTS_DIR}}prefetch/ {{LOGS_DIR}}
    make {{network}}-generate-keys-prefetch
    make  {{BIN_DIR}}/{{network}}_prefetch_server_exe -j 5
    # Set prefetch-sat based on network type
    if [[ "{{network}}" == "mlp" || "{{network}}" == "lola" ]]; then
        PREFETCH_SAT=600
    elif [["{{network}}" == "resnet1" || "{{network}}" == "lenet" ]]; then
        PREFETCH_SAT=2000
    elif [[ "{{network}}" == "resnet8" || "{{network}}" == "resnet10" || "{{network}}" == "resnet20" || "{{network}}" == "resnet32" || "{{network}}" == "resnet44" || "{{network}}" == "resnet56" ]]; then
        PREFETCH_SAT=2500
    elif [[ "{{network}}" == "alexnet" || "{{network}}" == "resnet18" || "{{network}}" == "resnet34" ]]; then
        PREFETCH_SAT=3500
    elif [[ "{{network}}" == "resnet50" || "{{network}}" == "wresnet50-2" || "{{network}}" == "vgg" || "{{network}}" == "yolo"  ]]; then
        PREFETCH_SAT=4500
    else
        PREFETCH_SAT=3500
    fi
    OMP_NUM_THREADS=4 SERIALIZED_DATA_DIR={{SERIALIZED_BASE_DIR}}/prefetch/{{network}} \
    ./{{BIN_DIR}}/{{network}}_prefetch_server_exe --key-mode prefetching \
    --prefetch-sat $PREFETCH_SAT {{verbose}} \
    --result-dir {{RESULTS_DIR}}prefetch/ > {{LOGS_DIR}}/{{network}}_prefetch_server_logs.txt


run-fhelipe-server network logsuffix="":
    @echo "--- Running prefetch server for {{network}} ---"
    @mkdir -p {{SERIALIZED_BASE_DIR}}/fhelipe/{{network}} {{RESULTS_DIR}}fhelipe/ {{LOGS_DIR}}
    make {{network}}-generate-keys-fhelipe {{BIN_DIR}}/{{network}}_fhelipe_server_exe -j 5
    OMP_NUM_THREADS=4 SERIALIZED_DATA_DIR={{SERIALIZED_BASE_DIR}}/fhelipe/{{network}} \
    ./{{BIN_DIR}}/{{network}}_fhelipe_server_exe --key-mode ignore \
    --result-dir {{RESULTS_DIR}}fhelipe/ > {{LOGS_DIR}}/{{network}}_fhelipe_server_logs{{logsuffix}}.txt

run-servers network:
    @echo "=== Running all benchmarks for {{network}} ==="
    # Ensure executables are built
    just build-executables-server {{network}}
    just run-baseline-server {{network}}
    just run-opt-server {{network}}
    just run-fhelipe-server {{network}}
    just run-prefetch-server {{network}}

# Helper: run command with email notification on completion/failure
_notify name command:
    #!/usr/bin/env bash
    if {{command}}; then
        echo "✅ {{name}} succeeded" | mail -s "✅ {{name}}" ${NOTIFY_EMAIL:-/dev/null} 2>/dev/null || true
    else
        echo "❌ {{name}} failed (exit $?)" | mail -s "❌ {{name}} FAILED" ${NOTIFY_EMAIL:-/dev/null} 2>/dev/null || true
        exit 1
    fi

# Wrap any just command with email notification (usage: just notify run-prefetch-server alexnet)
notify +args:
    #!/usr/bin/env bash
    name="{{args}}"
    echo "📧 Running with notifications: $name"
    echo "Email: ${NOTIFY_EMAIL:-not set}"
    just _notify "$name" "just {{args}}"

# Run all servers with email notifications (set NOTIFY_EMAIL env var)
run-servers-notify network:
    @echo "=== Running all benchmarks for {{network}} with notifications ==="
    @echo "Email: ${NOTIFY_EMAIL:-not set}"
    just build-executables-server {{network}}
    just _notify "{{network}}-baseline" "just run-baseline-server {{network}}" || true
    just _notify "{{network}}-opt" "just run-opt-server {{network}}" || true
    just _notify "{{network}}-fhelipe" "just run-fhelipe-server {{network}}" || true
    just _notify "{{network}}-prefetch" "just run-prefetch-server {{network}}" || true
    @echo "All complete" | mail -s "🏁 {{network}} complete" ${NOTIFY_EMAIL:-/dev/null} 2>/dev/null || true

# ============================================================================
# Batch Operations
# ============================================================================


# Parallel server commands for each mode
run-parallel-baseline-server network processes="8" iterations="3" threads="":
    @echo "🚀 Running {{processes}} parallel baseline servers for {{network}}"
    {{BIN_DIR}}/parallel_launcher \
        --executable ./{{BIN_DIR}}/{{network}}_baseline_server_exe \
        --processes {{processes}} \
        --iterations {{iterations}} \
        {{ if threads != "" { "--max-threads " + threads } else { "" } }} \
        --exe-args "--key-mode ignore"

run-parallel-opt-server network processes="8" iterations="3" threads="":
    @echo "🚀 Running {{processes}} parallel opt servers for {{network}}"
    {{BIN_DIR}}/parallel_launcher \
        --executable ./{{BIN_DIR}}/{{network}}_opt_server_exe \
        --processes {{processes}} \
        --iterations {{iterations}} \
        {{ if threads != "" { "--max-threads " + threads } else { "" } }} \
        --exe-args "--key-mode imperative"

run-parallel-kc-server network processes="8" iterations="3" threads="":
    @echo "🚀 Running {{processes}} parallel kc servers for {{network}}"
    {{BIN_DIR}}/parallel_launcher \
        --executable ./{{BIN_DIR}}/{{network}}_kc_server_exe \
        --processes {{processes}} \
        --iterations {{iterations}} \
        {{ if threads != "" { "--max-threads " + threads } else { "" } }} \
        --exe-args "--key-mode imperative"

run-parallel-prefetch-server network processes="8" iterations="3" threads="":
    @echo "🚀 Running {{processes}} parallel prefetch servers for {{network}}"
    {{BIN_DIR}}/parallel_launcher \
        --executable ./{{BIN_DIR}}/{{network}}_prefetch_server_exe \
        --processes {{processes}} \
        --iterations {{iterations}} \
        {{ if threads != "" { "--max-threads " + threads } else { "" } }} \
        --exe-args "--key-mode prefetching --prefetch-sat 1700"

run-parallel-fhelipe-server network processes="8" iterations="3" threads="":
    @echo "🚀 Running {{processes}} parallel fhelipe servers for {{network}}"
    {{BIN_DIR}}/parallel_launcher \
        --executable ./{{BIN_DIR}}/{{network}}_fhelipe_server_exe \
        --processes {{processes}} \
        --iterations {{iterations}} \
        {{ if threads != "" { "--max-threads " + threads } else { "" } }} \
        --exe-args "--key-mode ignore"

# Generate MLIR for all networks
gen-all:
    @echo "=== Generating MLIR for all networks ==="
    make all-mlir

# Optimize all networks
optimize-all:
    @echo "=== Optimizing all networks ==="
    make all-optimized

# Build all executables
build-all:
    @echo "=== Building all executables ==="
    make all-executables

benchmark-all: build-all
    @echo "=== Benchmarking all networks ==="
    just benchmark mlp || echo "Failed to benchmark mlp, continuing..."
    just benchmark resnet || echo "Failed to benchmark resnet, continuing..."
    just benchmark alexnet || echo "Failed to benchmark alexnet, continuing..."
    just benchmark lolanet || echo "Failed to benchmark lolanet, continuing..."


# Full pipeline for all networks
pipeline-all: optimize-all build-all benchmark-all
    @echo "=== Full pipeline completed for all networks ==="

# ============================================================================
# Analysis and Reporting (Quick operations)
# ============================================================================

# Analyze results from a specific network
analyze network:
    @echo "=== Analyzing results for {{network}} ==="
    python3 scripts/memory_analysis.py \
        --basic {{RESULTS_DIR}}/{{network}}_baseline_results.txt \
        --events {{RESULTS_DIR}}/{{network}}_opt_results.txt \
        --output {{RESULTS_DIR}}/{{network}}_analysis.txt

# Generate plots from results
plot network plot_type="all":
    @echo "=== Generating plots for {{network}} ==="
    python3 scripts/resilient_plotter.py \
        --plot-type {{plot_type}} \
        --data-dir {{RESULTS_DIR}} \
        --output {{RESULTS_DIR}}/{{network}}_{{plot_type}}.png

plot-latest-time-memory network:
    @echo "=== Finding latest results for {{network}} ==="
    python3 scripts/plot_latest.py --network {{network}} --run


# ============================================================================
# Quick Commands
# ============================================================================

# List available networks
list-networks:
    @echo "Available networks in {{NETWORKS_DIR}}:"
    @ls {{NETWORKS_DIR}}/*.py 2>/dev/null | xargs -n1 basename -s .py | sed 's/^/  - /' || echo "  No network files found"

# Show status of files for a network (delegates to Makefile)
status network:
    make status-{{network}}

# Debug a specific step
debug-step network step:
    @echo "=== Debug {{step}} for {{network}} ==="
    @case {{step}} in \
        "mlir") make {{network}}-mlir ;; \
        "base") make {{network}}-base-opt ;; \
        "profile") make {{network}}-profile ;; \
        "kc") make {{network}}-kc-opt ;; \
        "prefetch") make {{network}}-prefetch-opt ;; \
        "baseline-exe") make {{network}}-baseline-exe ;; \
        "opt-exe") make {{network}}-opt-exe ;; \
        "kc-exe") make {{network}}-kc-exe ;; \
        "prefetch-exe") make {{network}}-prefetch-exe ;; \
        "baseline") just _run-baseline {{network}} ;; \
        "opt") just _run-opt {{network}} ;; \
        "kc-run") just _run-kc {{network}} ;; \
        "prefetch-run") just _run-prefetch {{network}} ;; \
        *) echo "Unknown step: {{step}}" ;; \
    esac

# ============================================================================
# Container Management
# ============================================================================

# Build and ship Apptainer container to invershin
ship-container:
    @echo "=== Building Apptainer container ==="
    apptainer build keymemrt-build.sif keymemrt-build.def
    @echo "=== Shipping container to invershin ==="
    scp keymemrt-build.sif invershin:/disk/fast2/eunay/
    @echo "✅ Container shipped to invershin:/disk/fast2/eunay/keymemrt-build.sif"

# Build Apptainer container only
build-container:
    @echo "=== Building Apptainer container ==="
    apptainer build keymemrt-build.sif keymemrt-build.def
    @echo "✅ Container built: keymemrt-build.sif"

# Ship existing container to invershin
upload-container:
    @echo "=== Shipping container to invershin ==="
    scp keymemrt-build.sif invershin:/disk/fast2/eunay/
    @echo "✅ Container shipped to invershin:/disk/fast2/eunay/keymemrt-build.sif"

# ============================================================================
# Development Commands
# ============================================================================

# Clean generated files (delegates to Makefile)
clean:
    make clean
    rm -rf {{RESULTS_DIR}}/*

# Clean only intermediate files
clean-intermediate:
    make clean-cpp

# Clean only executables
clean-executables:
    make clean-exe

# Show current configuration
config:
    @echo "Configuration:"
    @echo "  KEYMEMRT_COMPILER: {{KEYMEMRT_COMPILER}}"
    @echo "  NETWORKS_DIR: {{NETWORKS_DIR}}"
    @echo "  RESULTS_DIR: {{RESULTS_DIR}}"
    @echo "  BUILD_DIR: {{BUILD_DIR}}"
    @echo "  Available Networks: {{NETWORKS}}"
    @echo ""
    @echo "Directory structure:"
    @echo "  MLIR files:     {{BUILD_DIR}}/mlir/"
    @echo "  C++ files:      {{BUILD_DIR}}/cpp/"
    @echo "  Executables:    {{BUILD_DIR}}/bin/"
    @echo "  Profiles:       {{BUILD_DIR}}/profiles/"
    @echo "  Results:        {{RESULTS_DIR}}/"
    @echo ""
    @echo "File tracking: Makefile handles expensive operations"
    @echo "Quick operations: Justfile handles analysis, plotting, benchmarking"
