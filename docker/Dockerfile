# Multi-stage Dockerfile for KeyMemRT
# Optimized for fast rebuilds with layer caching

# Stage 1: Base dependencies and OpenFHE
FROM ubuntu:22.04 AS base-deps

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies in a single layer
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    clang \
    ninja-build \
    git \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    curl \
    libssl-dev \
    libomp-dev \
    lld \
    zlib1g-dev \
    libcereal-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Build and install OpenFHE (rarely changes, good for caching)
WORKDIR /tmp/openfhe-build
RUN git clone --depth 1 https://github.com/eymay/openfhe-development.git . && \
    mkdir build && cd build && \
    CC=clang CXX=clang++ cmake -GNinja \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED=ON \
    .. && \
    ninja && \
    ninja install && \
    cd / && rm -rf /tmp/openfhe-build

# Stage 2: Compiler setup (flexible sourcing)
FROM base-deps AS compiler-stage

ARG COMPILER_VERSION=""
ARG KEYMEMRT_COMPILER=""

WORKDIR /compiler

# Option 1: Download pre-built compiler binary if COMPILER_VERSION is set
# Option 2: Use mounted compiler from host (checked at runtime)
# Option 3: Build from source if neither is available
RUN if [ -n "$COMPILER_VERSION" ]; then \
        echo "Downloading KeyMemRT-Compiler $COMPILER_VERSION..."; \
        wget -O keymemrt-compiler.tar.gz \
            "https://github.com/eymay/KeyMemRT-Compiler/releases/download/${COMPILER_VERSION}/keymemrt-compiler.tar.gz" && \
        tar -xzf keymemrt-compiler.tar.gz && \
        rm keymemrt-compiler.tar.gz; \
    elif [ -z "$KEYMEMRT_COMPILER" ]; then \
        echo "Building KeyMemRT-Compiler from source..."; \
        apt-get update && apt-get install -y openjdk-11-jdk && \
        git clone --depth 1 https://github.com/eymay/KeyMemRT-Compiler.git . && \
        wget https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64 && \
        chmod +x bazelisk-linux-amd64 && \
        mv bazelisk-linux-amd64 /usr/local/bin/bazel && \
        bazel build //tools:keymemrt-opt //tools:keymemrt-translate && \
        mkdir -p bin && \
        cp bazel-bin/tools/keymemrt-opt bin/ && \
        cp bazel-bin/tools/keymemrt-translate bin/ && \
        cp bazel-bin/tools/keymemrt-opt . && \
        cp bazel-bin/tools/keymemrt-translate . && \
        rm -rf /var/lib/apt/lists/*; \
    else \
        echo "Will use compiler from host at $KEYMEMRT_COMPILER"; \
    fi

# Stage 3: Build environment with Python/Orion
FROM base-deps AS build-env

# Copy compiler from previous stage (if built/downloaded)
COPY --from=compiler-stage /compiler /compiler

# Set up Python virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch numpy

# Install Orion and Orion-HEIR-Translator
WORKDIR /tmp/orion-build
RUN git clone --depth 1 https://github.com/eymay/orion_heir_translator.git . && \
    pip install --no-cache-dir .[orion] && \
    cd / && rm -rf /tmp/orion-build

# Install Just command runner
RUN wget https://github.com/casey/just/releases/download/1.14.0/just-1.14.0-x86_64-unknown-linux-musl.tar.gz && \
    tar -xzf just-1.14.0-x86_64-unknown-linux-musl.tar.gz && \
    mv just /usr/local/bin/ && \
    rm just-1.14.0-x86_64-unknown-linux-musl.tar.gz

# Set up workspace
WORKDIR /workspace

# Set environment variables
# KEYMEMRT_COMPILER will be set at runtime based on mounted compiler or built compiler
ENV HEIR_PATH=/compiler
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV CC=clang
ENV CXX=clang++

# Create build directories
RUN mkdir -p build/mlir build/cpp build/bin build/profiles build/results build/logs

# Copy only CMakeLists.txt and run CMake to discover dependencies
# Source code will be mounted at runtime via docker.just, so we only need CMakeLists.txt
COPY CMakeLists.txt ./
RUN cmake -B build -S . && \
    echo "✓ CMake configuration generated at build/build.config"

# Default command
CMD ["/bin/bash"]
