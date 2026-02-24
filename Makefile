# KeyMem-RT Makefile - File dependency tracking for expensive operations
# Handles: Python->MLIR, HEIR optimization, C++ compilation

# Configuration
KEYMEMRT_COMPILER ?= $(shell echo $$KEYMEMRT_COMPILER)
# Check if HEIR tools are already directly in KEYMEMRT_COMPILER (container setup)
# or if we need to append bazel-bin/tools (local development setup)
ifeq ($(wildcard $(KEYMEMRT_COMPILER)/keymemrt-opt),$(KEYMEMRT_COMPILER)/keymemrt-opt)
    KEYMEMRT_OPT = $(KEYMEMRT_COMPILER)/keymemrt-opt
    KEYMEMRT_TRANSLATE = $(KEYMEMRT_COMPILER)/keymemrt-translate
else
    KEYMEMRT_OPT = $(KEYMEMRT_COMPILER)/bazel-bin/tools/keymemrt-opt
    KEYMEMRT_TRANSLATE = $(KEYMEMRT_COMPILER)/bazel-bin/tools/keymemrt-translate
endif


ROOT_DIR = $(shell pwd)
# Directory structure
NETWORKS_DIR = benchmarks/networks
BUILD_DIR = build
RESULTS_DIR = $(ROOT_DIR)/$(BUILD_DIR)/results
LOGS_DIR = $(ROOT_DIR)/$(BUILD_DIR)/logs
MLIR_DIR = $(BUILD_DIR)/mlir
CPP_DIR = $(BUILD_DIR)/cpp
BIN_DIR = $(BUILD_DIR)/bin
PROFILE_DIR = $(BUILD_DIR)/profiles

# Create build directories
$(shell mkdir -p $(RESULTS_DIR) $(MLIR_DIR) $(CPP_DIR) $(BIN_DIR) $(PROFILE_DIR))

SERIALIZED_BASE_DIR ?= /tmp/keymemrt

# Compiler settings
CXX = clang++
CFLAGS = -std=c++17 -O2 -w -g0 -fPIC
INCLUDES = -I../../include -I/usr/local/include/ -I/usr/local/include/openfhe \
           -I/usr/local/include/openfhe/pke -I/usr/local/include/openfhe/core/ \
           -I/usr/local/include/openfhe/binfhe -I/usr/local/include/cereal \
           -I$(shell pwd)/include \
           -I$(shell pwd)/drivers \
           -I/usr/local
LIBS = -L/usr/local/lib -lOPENFHEcore -lOPENFHEpke -lOPENFHEbinfhe

# Standalone networks
STANDALONE_NETWORKS = mlp lola lenet alexnet vgg
# vgg - Disabled due to MLIR scheme attribute error
# yolo - Disabled due to memory requirements

# ResNet variants (CIFAR and ImageNet)
RESNET_VARIANTS = resnet1 resnet8 resnet10 resnet20 resnet18 resnet34 resnet50 resnet32 resnet44 resnet56 resnet110 wresnet50-2 

# All networks
NETWORKS = $(STANDALONE_NETWORKS) $(RESNET_VARIANTS)

# Default target
.PHONY: help
help:
	@echo "KeyMem-RT Makefile - Expensive operations with file tracking"
	@echo "Build directory: $(BUILD_DIR)/"
	@echo ""
	@echo "Per-network targets:"
	@echo "  make <network>-mlir                    # Generate MLIR from Python"
	@echo "  make <network>-base-opt                # Base optimization"
	@echo "  make <network>-profile                 # Profiling"
	@echo "  make <network>-kc-opt                  # Key compression"
	@echo "  make <network>-prefetch-opt            # Prefetching"
	@echo "  make <network>-baseline-exe            # Baseline executable"
	@echo "  make <network>-opt-exe                 # Optimized executable" 
	@echo "  make <network>-kc-exe                  # Key-compressed executable"
	@echo "  make <network>-prefetch-exe            # Prefetching executable"
	@echo ""
	@echo "Batch targets:"
	@echo "  make all-mlir                          # Generate all MLIR files"
	@echo "  make all-optimized                     # Optimize all networks"
	@echo ""
	@echo "Build only:"
	@echo "  make all-test-executables              # Build test executables (baseline/opt/prefetch/fhelipe)"
	@echo "  make all-server-executables            # Build all server executables"
	@echo "  make all-client-executables            # Build all client executables"
	@echo "  make all-executables                   # Build all executables (server + client)"
	@echo ""
	@echo "Keys only:"
	@echo "  make all-test-keys                     # Generate test keys (baseline/opt/prefetch/fhelipe)"
	@echo "  make all-keys                          # Generate all keys (all variations)"
	@echo ""
	@echo "⚡ COMBINED (Build + Keys in parallel):"
	@echo "  make all-test-ready                    # Build test executables AND generate test keys (RECOMMENDED!)"
	@echo "  make all-ready                         # Build ALL executables AND generate ALL keys"
	@echo ""
	@echo "File locations:"
	@echo "  MLIR files:     $(MLIR_DIR)/"
	@echo "  C++ files:      $(CPP_DIR)/"
	@echo "  Executables:    $(BIN_DIR)/"
	@echo "  Profiles:       $(PROFILE_DIR)/"
	@echo ""
	@echo "Example workflow:"
	@echo "  make resnet-mlir                       # Generate MLIR"
	@echo "  make resnet-prefetch-opt               # Full optimization pipeline"
	@echo "  make resnet-prefetch-exe               # Build final executable"

# ============================================================================
# MLIR Generation (Python -> MLIR)
# ============================================================================



# ============================================================================
# HEIR Optimization Pipeline
# ============================================================================

# Base optimization
$(MLIR_DIR)/%_openfhe_staticopt.mlir: $(MLIR_DIR)/%.mlir
	@echo "=== Static optimizations for $* ==="
	$(KEYMEMRT_OPT) \
		--ckks-to-lwe --lwe-to-openfhe \
		--lower-linear-transform \
		--symbolic-bsgs-decomposition --kmrt-merge-rotation-keys \
		--annotate-module="backend=openfhe scheme=ckks" \
		--openfhe-configure-crypto-context --openfhe-fast-rotation-precompute \
		$< > $@
	@echo "✅ Generated $@"

$(MLIR_DIR)/%_openfhe_pgoopt.mlir: $(MLIR_DIR)/%_openfhe_staticopt.mlir $(PROFILE_DIR)/%_staticopt_profile.txt
	@echo "=== PGO1 optimizations for $* ==="
	$(KEYMEMRT_OPT) \
		--profile-annotator="profile-file=$(shell pwd)/$(PROFILE_DIR)/$*_staticopt_profile.txt" \
		--remove-unnecessary-bootstraps --bootstrap-rotation-analysis --kmrt-merge-rotation-keys \
		--cse --openfhe-insert-clear-ops  \
		$< > $@
	@echo "✅ Generated $@"

$(MLIR_DIR)/%_openfhe_pgoopt_kc.mlir: $(MLIR_DIR)/%_openfhe_pgoopt.mlir
	@echo "=== Key compression optimization for $* ==="
	$(KEYMEMRT_OPT) \
		--key-compression \
		$< > $@
	@echo "✅ Generated $@"


define prefetch_network
$(MLIR_DIR)/$1_openfhe_pgoopt_pf15.mlir: $(MLIR_DIR)/$1_openfhe_pgoopt.mlir
	@echo "=== Prefetching optimization for $$* ==="
	$(KEYMEMRT_OPT) \
		--kmrt-key-prefetching="prefetch-threshold=15 runtime-delegated=1" --lower-affine \
		$$< > $$@
	@echo "✅ Generated $$@"
endef

$(foreach net,$(NETWORKS),$(eval $(call prefetch_network,$(net))))


# ============================================================================
# Other Pipelines for Evaluation
# ============================================================================

# Powers-of-Two decomposition for Fhelipe
# $(MLIR_DIR)/%_po2_staticopt.mlir: $(MLIR_DIR)/%.mlir
# 	@echo "=== Static optimizations for $* ==="
# 	$(KEYMEMRT_OPT) \
# 		--ckks-to-lwe --lwe-to-openfhe \
# 		--lower-linear-transform \
# 		--annotate-module="backend=openfhe scheme=ckks" --openfhe-configure-crypto-context \
# 		$< > $@
# 	@echo "✅ Generated $@"

$(MLIR_DIR)/%_fhelipe_pgoopt.mlir: $(MLIR_DIR)/%_openfhe_staticopt.mlir $(PROFILE_DIR)/%_staticopt_profile.txt
	@echo "=== PGO1 optimizations for $* ==="
	$(KEYMEMRT_OPT) \
		--profile-annotator="profile-file=$(shell pwd)/$(PROFILE_DIR)/$*_staticopt_profile.txt" \
		--remove-unnecessary-bootstraps --openfhe-insert-clear-ops \
		--add-rotation-keys="overwrite=1 rotation-indices=1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,32767,32766,32764,32760,32752,32736,32704,32640,32512,32256,31744,30720,28672,24576" \
		$< > $@
	@echo "✅ Generated $@"

.PRECIOUS: $(MLIR_DIR)/%openfhe_staticopt.mlir

.PRECIOUS: $(MLIR_DIR)/%_openfhe_pgoopt.mlir
.PRECIOUS: $(MLIR_DIR)/%.mlir

$(PROFILE_DIR)/%_staticopt_profile.txt: $(BIN_DIR)/%_staticopt_exe
	@echo "=== Profiling $* staticopt ==="
	@mkdir -p /tmp/keymemrt/profile
	OMP_NUM_THREADS=4 SERIALIZED_DATA_DIR=/tmp/keymemrt/profile $< --key-mode ignore > $@
	@echo "✅ Generated profile $@"

# Missing profile generation for po2_staticopt variant  
$(PROFILE_DIR)/%_po2_staticopt_profile.txt: $(BIN_DIR)/%_po2_staticopt_exe
	@echo "=== Profiling $* po2_staticopt ==="
	@mkdir -p /tmp/keymemrt/profile
	OMP_NUM_THREADS=4 SERIALIZED_DATA_DIR=/tmp/keymemrt/profile $< --key-mode ignore > $@
	@echo "✅ Generated profile $@"

.PRECIOUS: $(PROFILE_DIR)/%_staticopt_profile.txt
.PRECIOUS: $(PROFILE_DIR)/%_po2_staticopt_profile.txt

# ============================================================================
# C++ Code Generation
# ============================================================================

# Generate C++ from MLIR
$(CPP_DIR)/%_openfhe_pgoopt.cpp $(CPP_DIR)/%_openfhe_pgoopt.h: $(MLIR_DIR)/%_openfhe_pgoopt.mlir
	@echo "=== Generating C++ for $*_openfhe_pgoopt ==="
	$(KEYMEMRT_TRANSLATE) \
		--emit-openfhe-pke \
		--weights-file=$(CPP_DIR)/$*_openfhe_pgoopt.bin \
		--openfhe-include-type=install-relative \
		$< > $(CPP_DIR)/$*_openfhe_pgoopt.cpp
	$(KEYMEMRT_TRANSLATE) \
		--emit-openfhe-pke-header \
		--weights-file=$(CPP_DIR)/$*_openfhe_pgoopt.bin \
		--openfhe-include-type=install-relative \
		$< > $(CPP_DIR)/$*_openfhe_pgoopt.h
	@echo "✅ Generated C++ files for $*"

$(CPP_DIR)/%_baseline.cpp $(CPP_DIR)/%_baseline.h: $(MLIR_DIR)/%_openfhe_pgoopt.mlir
	@echo "=== Generating C++ for $*_baseline ==="
	$(KEYMEMRT_TRANSLATE) \
		--emit-openfhe-pke \
		--weights-file=$(CPP_DIR)/$*_baseline.bin \
		--openfhe-include-type=install-relative \
		$< > $(CPP_DIR)/$*_baseline.cpp
	$(KEYMEMRT_TRANSLATE) \
		--emit-openfhe-pke-header \
		--weights-file=$(CPP_DIR)/$*_baseline.bin \
		--openfhe-include-type=install-relative \
		$< > $(CPP_DIR)/$*_baseline.h
	@echo "✅ Generated C++ files for $*_baseline"

$(CPP_DIR)/%_kc.cpp $(CPP_DIR)/%_kc.h: $(MLIR_DIR)/%_openfhe_pgoopt_kc.mlir
	@echo "=== Generating C++ for $*_kc ==="
	$(KEYMEMRT_TRANSLATE) \
		--emit-openfhe-pke \
		--weights-file=$(CPP_DIR)/$*_kc.bin \
		--openfhe-include-type=install-relative \
		$< > $(CPP_DIR)/$*_kc.cpp
	$(KEYMEMRT_TRANSLATE) \
		--emit-openfhe-pke-header \
		--weights-file=$(CPP_DIR)/$*_kc.bin \
		--openfhe-include-type=install-relative \
		$< > $(CPP_DIR)/$*_kc.h
	@echo "✅ Generated C++ files for $*_kc"

$(CPP_DIR)/%_prefetch.cpp $(CPP_DIR)/%_prefetch.h: $(MLIR_DIR)/%_openfhe_pgoopt_kc_pf5.mlir
	@echo "=== Generating C++ for $*_prefetch ==="
	$(KEYMEMRT_TRANSLATE) \
		--emit-openfhe-pke \
		--weights-file=$(CPP_DIR)/$*_prefetch.bin \
		--openfhe-include-type=install-relative \
		$< > $(CPP_DIR)/$*_prefetch.cpp
	$(KEYMEMRT_TRANSLATE) \
		--emit-openfhe-pke-header \
		--weights-file=$(CPP_DIR)/$*_prefetch.bin \
		--openfhe-include-type=install-relative \
		$< > $(CPP_DIR)/$*_prefetch.h
	@echo "✅ Generated C++ files for $*_prefetch"


$(CPP_DIR)/%_prefetch.cpp $(CPP_DIR)/%_prefetch.h: $(MLIR_DIR)/%_openfhe_pgoopt_pf15.mlir
	@echo "=== Generating C++ for $*_prefetch ==="
	$(KEYMEMRT_TRANSLATE) \
		--emit-openfhe-pke \
		--weights-file=$(CPP_DIR)/$*_prefetch.bin \
		--openfhe-include-type=install-relative \
		$< > $(CPP_DIR)/$*_prefetch.cpp
	$(KEYMEMRT_TRANSLATE) \
		--emit-openfhe-pke-header \
		--weights-file=$(CPP_DIR)/$*_prefetch.bin \
		--openfhe-include-type=install-relative \
		$< > $(CPP_DIR)/$*_prefetch.h
	@echo "✅ Generated C++ files for $*_prefetch"


# Translate directly from affine version (keymemrt-translate now supports affine, kmrt, lwe)
$(CPP_DIR)/%_staticopt.cpp $(CPP_DIR)/%_staticopt.h: $(MLIR_DIR)/%_openfhe_staticopt.mlir
	@echo "=== Generating C++ for $*_staticopt ==="
	$(KEYMEMRT_TRANSLATE) \
		--emit-openfhe-pke \
		--weights-file=$(CPP_DIR)/$*_staticopt.bin \
		--openfhe-include-type=install-relative \
		$< > $(CPP_DIR)/$*_staticopt.cpp
	$(KEYMEMRT_TRANSLATE) \
		--emit-openfhe-pke-header \
		--weights-file=$(CPP_DIR)/$*_staticopt.bin \
		--openfhe-include-type=install-relative \
		$< > $(CPP_DIR)/$*_staticopt.h
	@echo "✅ Generated C++ files for $*_staticopt"

# C++ generation for po2_staticopt
$(CPP_DIR)/%_po2_staticopt.cpp $(CPP_DIR)/%_po2_staticopt.h: $(MLIR_DIR)/%_po2_staticopt.mlir
	@echo "=== Generating C++ for $*_po2_staticopt ==="
	$(KEYMEMRT_TRANSLATE) \
		--emit-openfhe-pke \
		--weights-file=$(CPP_DIR)/$*_po2_staticopt.bin \
		--openfhe-include-type=install-relative \
		$< > $(CPP_DIR)/$*_po2_staticopt.cpp
	$(KEYMEMRT_TRANSLATE) \
		--emit-openfhe-pke-header \
		--weights-file=$(CPP_DIR)/$*_po2_staticopt.bin \
		--openfhe-include-type=install-relative \
		$< > $(CPP_DIR)/$*_po2_staticopt.h
	@echo "✅ Generated C++ files for $*_po2_staticopt"

# C++ generation for fhelipe_pgoopt
$(CPP_DIR)/%_fhelipe.cpp $(CPP_DIR)/%_fhelipe.h: $(MLIR_DIR)/%_fhelipe_pgoopt.mlir
	@echo "=== Generating C++ for $*_fhelipe ==="
	$(KEYMEMRT_TRANSLATE) \
		--emit-openfhe-pke \
		--weights-file=$(CPP_DIR)/$*_fhelipe.bin \
		--openfhe-include-type=install-relative \
		$< > $(CPP_DIR)/$*_fhelipe.cpp
	sed -i 's/cc->EvalRotate(/keymem_rt.rotateBinary(/g' $(CPP_DIR)/$*_fhelipe.cpp
	sed -i 's/cc->EvalFastRotation(/keymem_rt.fastRotateBinary(/g' $(CPP_DIR)/$*_fhelipe.cpp
	$(KEYMEMRT_TRANSLATE) \
		--emit-openfhe-pke-header \
		--weights-file=$(CPP_DIR)/$*_fhelipe.bin \
		--openfhe-include-type=install-relative \
		$< > $(CPP_DIR)/$*_fhelipe.h
	@echo "✅ Generated C++ files for $*_fhelipe"

# ============================================================================
# C++ Compilation
# ============================================================================

# Generic compilation rule for different network types

define compile_server_network
$(BIN_DIR)/$1_$2_server_exe: $(CPP_DIR)/$1_$3.cpp $(CPP_DIR)/$1_$3.h
	@echo "=== Compiling $$@ for $1 ==="
	$(CXX) $(CFLAGS) -DNETWORK_$(shell echo $1 | tr -d '0-9' | tr '[:lower:]' '[:upper:]') \
		-DNETWORK_VARIANT_$(shell echo $1 | tr '[:lower:]' '[:upper:]') \
		drivers/server_driver.cpp drivers/server_runner.cpp $(CPP_DIR)/$1_$3.cpp -fPIC \
		$(INCLUDES) $(LIBS) -O2 -o $$@
	@echo "✅ Built $$@"

# Convenience target
$1-$2-server-exe: $(BIN_DIR)/$1_$2_server_exe
	@echo "✅ Executable ready: $<"
endef

$(foreach net,$(NETWORKS),$(eval $(call compile_server_network,$(net),baseline,baseline)))
$(foreach net,$(NETWORKS),$(eval $(call compile_server_network,$(net),opt,openfhe_pgoopt)))
$(foreach net,$(NETWORKS),$(eval $(call compile_server_network,$(net),kc,kc)))
$(foreach net,$(NETWORKS),$(eval $(call compile_server_network,$(net),prefetch,prefetch)))
$(foreach net,$(NETWORKS),$(eval $(call compile_server_network,$(net),po2_staticopt,po2_staticopt)))
$(foreach net,$(NETWORKS),$(eval $(call compile_server_network,$(net),staticopt,staticopt)))
$(foreach net,$(NETWORKS),$(eval $(call compile_server_network,$(net),fhelipe,fhelipe)))

define compile_client_network
$(BIN_DIR)/$1_$2_client_exe: $(CPP_DIR)/$1_$3.cpp $(CPP_DIR)/$1_$3.h
	@echo "=== Compiling $$@ for $1 ==="
	$(CXX) $(CFLAGS) -DNETWORK_$(shell echo $1 | tr -d '0-9' | tr '[:lower:]' '[:upper:]') \
		-DNETWORK_VARIANT_$(shell echo $1 | tr '[:lower:]' '[:upper:]') \
		drivers/client_driver.cpp $(CPP_DIR)/$1_$3.cpp -fPIC \
		$(INCLUDES) $(LIBS) -O2 -o $$@
	@echo "✅ Built $$@"

# Convenience target
$1-$2-client-exe: $(BIN_DIR)/$1_$2_client_exe
	@echo "✅ Executable ready: $<"
endef

$(foreach net,$(NETWORKS),$(eval $(call compile_client_network,$(net),baseline,baseline)))
$(foreach net,$(NETWORKS),$(eval $(call compile_client_network,$(net),opt,openfhe_pgoopt)))
$(foreach net,$(NETWORKS),$(eval $(call compile_client_network,$(net),kc,kc)))
$(foreach net,$(NETWORKS),$(eval $(call compile_client_network,$(net),prefetch,prefetch)))
$(foreach net,$(NETWORKS),$(eval $(call compile_client_network,$(net),po2_staticopt,po2_staticopt)))
$(foreach net,$(NETWORKS),$(eval $(call compile_client_network,$(net),staticopt,staticopt)))
$(foreach net,$(NETWORKS),$(eval $(call compile_client_network,$(net),fhelipe,fhelipe)))

define compile_network
$(BIN_DIR)/$1_$2_exe: $(CPP_DIR)/$1_$3.cpp $(CPP_DIR)/$1_$3.h
	@echo "=== Compiling $$@ for $1 ==="
	$(CXX) $(CFLAGS) -DNETWORK_$(shell echo $1 | tr -d '0-9' | tr '[:lower:]' '[:upper:]') \
		-DNETWORK_VARIANT_$(shell echo $1 | tr '[:lower:]' '[:upper:]') \
		drivers/main_driver.cpp $(CPP_DIR)/$1_$3.cpp -fPIC \
		$(INCLUDES) $(LIBS) -O2 -o $$@
	@echo "✅ Built $$@"

# Convenience target
$1-$2-exe: $(BIN_DIR)/$1_$2_exe
	@echo "✅ Executable ready: $<"
endef

$(foreach net,$(NETWORKS),$(eval $(call compile_network,$(net),baseline,baseline)))
$(foreach net,$(NETWORKS),$(eval $(call compile_network,$(net),opt,openfhe_pgoopt)))
$(foreach net,$(NETWORKS),$(eval $(call compile_network,$(net),kc,kc)))
$(foreach net,$(NETWORKS),$(eval $(call compile_network,$(net),prefetch,prefetch)))
$(foreach net,$(NETWORKS),$(eval $(call compile_network,$(net),po2_staticopt,po2_staticopt)))
$(foreach net,$(NETWORKS),$(eval $(call compile_network,$(net),staticopt,staticopt)))
$(foreach net,$(NETWORKS),$(eval $(call compile_network,$(net),fhelipe,fhelipe)))

# ============================================================================
# Profiling (Special case - requires execution)
# ============================================================================



# Schedule Launcher
$(BIN_DIR)/multi_run_launcher: drivers/multi_run_launcher.cpp
	@echo "=== Compiling schedule launcher ==="
	$(CXX) $(CFLAGS) drivers/multi_run_launcher.cpp \
		$(INCLUDES) -o $@
	@echo "✅ Built multi run launcher launcher"

$(BIN_DIR)/parallel_launcher: drivers/parallel_launcher.cpp
	@echo "=== Compiling parallel launcher ==="
	$(CXX) $(CFLAGS) drivers/parallel_launcher.cpp \
		$(INCLUDES) -o $@
	@echo "✅ Built parallel launcher"

# ============================================================================
# Clients
# ============================================================================

# Network-specific key generation rules - keys stored per network
$(SERIALIZED_BASE_DIR)/baseline/%/cryptocontext.txt: $(BIN_DIR)/%_baseline_client_exe
	@echo "=== Generating baseline keys for $* ==="
	@mkdir -p $(SERIALIZED_BASE_DIR)/baseline/$*
	@mkdir -p $(LOGS_DIR)
	SERIALIZED_DATA_DIR=$(SERIALIZED_BASE_DIR)/baseline/$* ./$(BIN_DIR)/$*_baseline_client_exe --key-mode ignore > $(LOGS_DIR)/$*_baseline_client_logs.txt
	@echo "✅ Keys generated for $* baseline at $(SERIALIZED_BASE_DIR)/baseline/$*"

$(SERIALIZED_BASE_DIR)/opt/%/cryptocontext.txt: $(BIN_DIR)/%_opt_client_exe
	@echo "=== Generating opt keys for $* ==="
	@mkdir -p $(SERIALIZED_BASE_DIR)/opt/$*
	@mkdir -p $(LOGS_DIR)
	SERIALIZED_DATA_DIR=$(SERIALIZED_BASE_DIR)/opt/$* ./$(BIN_DIR)/$*_opt_client_exe --key-mode imperative > $(LOGS_DIR)/$*_opt_client_logs.txt
	@echo "✅ Keys generated for $* opt at $(SERIALIZED_BASE_DIR)/opt/$*"

$(SERIALIZED_BASE_DIR)/kc/%/cryptocontext.txt: $(BIN_DIR)/%_kc_client_exe
	@echo "=== Generating kc keys for $* ==="
	@mkdir -p $(SERIALIZED_BASE_DIR)/kc/$*
	@mkdir -p $(LOGS_DIR)
	SERIALIZED_DATA_DIR=$(SERIALIZED_BASE_DIR)/kc/$* ./$(BIN_DIR)/$*_kc_client_exe --key-mode imperative  > $(LOGS_DIR)/$*_kc_client_logs.txt
	@echo "✅ Keys generated for $* kc at $(SERIALIZED_BASE_DIR)/kc/$*"

# Prefetch uses the same keys as opt (different runtime behavior only)
# Create a symlink to opt directory instead of generating duplicate keys
$(SERIALIZED_BASE_DIR)/prefetch/%/cryptocontext.txt: $(SERIALIZED_BASE_DIR)/opt/%/cryptocontext.txt
	@echo "=== Setting up prefetch keys for $* (using opt keys via symlink) ==="
	@rm -rf $(SERIALIZED_BASE_DIR)/prefetch/$*
	@mkdir -p $(SERIALIZED_BASE_DIR)/prefetch
	@ln -sf $(abspath $(SERIALIZED_BASE_DIR)/opt/$*) $(SERIALIZED_BASE_DIR)/prefetch/$*
	@echo "✅ Prefetch keys ready for $* (symlinked to opt at $(SERIALIZED_BASE_DIR)/opt/$*)"

$(SERIALIZED_BASE_DIR)/fhelipe/%/cryptocontext.txt: $(BIN_DIR)/%_fhelipe_client_exe
	@echo "=== Generating fhelipe keys for $* ==="
	@mkdir -p $(SERIALIZED_BASE_DIR)/fhelipe/$*
	@mkdir -p $(LOGS_DIR)
	SERIALIZED_DATA_DIR=$(SERIALIZED_BASE_DIR)/fhelipe/$* ./$(BIN_DIR)/$*_fhelipe_client_exe --key-mode ignore > $(LOGS_DIR)/$*_fhelipe_client_logs.txt
	@echo "✅ Keys generated for $* fhelipe at $(SERIALIZED_BASE_DIR)/fhelipe/$*"


# Key generation convenience targets
%-generate-keys-baseline: $(SERIALIZED_BASE_DIR)/baseline/%/cryptocontext.txt
	@echo "✅ Baseline keys ready for $*"

%-generate-keys-opt: $(SERIALIZED_BASE_DIR)/opt/%/cryptocontext.txt  
	@echo "✅ Opt keys ready for $*"

%-generate-keys-kc: $(SERIALIZED_BASE_DIR)/kc/%/cryptocontext.txt
	@echo "✅ KC keys ready for $*"

%-generate-keys-prefetch: $(SERIALIZED_BASE_DIR)/prefetch/%/cryptocontext.txt
	@echo "✅ Prefetch keys ready for $* (symlinked to opt)"

%-generate-keys-fhelipe: $(SERIALIZED_BASE_DIR)/fhelipe/%/cryptocontext.txt
	@echo "✅ Fhelipe keys ready for $*"

.PRECIOUS: $(SERIALIZED_BASE_DIR)/baseline/%/cryptocontext.txt
.PRECIOUS: $(SERIALIZED_BASE_DIR)/opt/%/cryptocontext.txt
.PRECIOUS: $(SERIALIZED_BASE_DIR)/kc/%/cryptocontext.txt
# Note: prefetch is a symlink to opt, not a precious file
.PRECIOUS: $(SERIALIZED_BASE_DIR)/fhelipe/%/cryptocontext.txt

# ============================================================================
# Base MLIR Generation - Explicit rules for all known networks
# ============================================================================

# Template for ResNet variants (use resnet.py with --variant)
define generate_resnet_base_mlir
$(MLIR_DIR)/$1.mlir:
	@echo "=== Generating MLIR for $1 ==="
	cd $(NETWORKS_DIR) && python3 resnet.py --variant $1 --output $(shell pwd)/$$@
	@echo "✅ Generated $$@"
endef

# Template for standard networks (run network.py directly)
define generate_standard_base_mlir
$(MLIR_DIR)/$1.mlir:
	@echo "=== Generating MLIR for $1 ==="
	cd $(NETWORKS_DIR) && python3 $1.py
	@if [ -f "$(NETWORKS_DIR)/$1.mlir" ]; then \
		mv "$(NETWORKS_DIR)/$1.mlir" $$@; \
	elif [ -f "$(NETWORKS_DIR)/$1_fhe.mlir" ]; then \
		mv "$(NETWORKS_DIR)/$1_fhe.mlir" $$@; \
	else \
		echo "❌ No MLIR file generated for $1!"; \
		exit 1; \
	fi
	@echo "✅ Generated $$@"
endef

# Auto-generate rules for all networks
$(foreach net,$(NETWORKS),\
	$(if $(findstring resnet,$(net)),\
		$(eval $(call generate_resnet_base_mlir,$(net))),\
		$(eval $(call generate_standard_base_mlir,$(net)))))

# Convenience targets (network-mlir instead of build/mlir/network.mlir)
%-mlir: $(MLIR_DIR)/%.mlir
	@echo "✅ MLIR ready: $<"

# ============================================================================
# Batch Build Targets
# ============================================================================

# Build all server executables for testing (baseline, opt, prefetch, fhelipe)
.PHONY: all-test-executables
all-test-executables:
	@echo "=== Building all test server executables (baseline, opt, prefetch, fhelipe) ==="
	@$(MAKE) -j $(shell nproc) \
		$(foreach net,$(NETWORKS),$(BIN_DIR)/$(net)_baseline_server_exe) \
		$(foreach net,$(NETWORKS),$(BIN_DIR)/$(net)_opt_server_exe) \
		$(foreach net,$(NETWORKS),$(BIN_DIR)/$(net)_prefetch_server_exe) \
		$(foreach net,$(NETWORKS),$(BIN_DIR)/$(net)_fhelipe_server_exe)
	@echo "✅ All test server executables built successfully!"

# Build all server executables (all variations)
.PHONY: all-server-executables
all-server-executables:
	@echo "=== Building all server executables ==="
	@$(MAKE) -j $(shell nproc) \
		$(foreach net,$(NETWORKS),$(BIN_DIR)/$(net)_baseline_server_exe) \
		$(foreach net,$(NETWORKS),$(BIN_DIR)/$(net)_opt_server_exe) \
		$(foreach net,$(NETWORKS),$(BIN_DIR)/$(net)_prefetch_server_exe) \
		$(foreach net,$(NETWORKS),$(BIN_DIR)/$(net)_fhelipe_server_exe) \
		$(foreach net,$(NETWORKS),$(BIN_DIR)/$(net)_kc_server_exe)
	@echo "✅ All server executables built successfully!"

# Build all client executables
.PHONY: all-client-executables
all-client-executables:
	@echo "=== Building all client executables ==="
	@$(MAKE) -j $(shell nproc) \
		$(foreach net,$(NETWORKS),$(BIN_DIR)/$(net)_baseline_client_exe) \
		$(foreach net,$(NETWORKS),$(BIN_DIR)/$(net)_opt_client_exe) \
		$(foreach net,$(NETWORKS),$(BIN_DIR)/$(net)_prefetch_client_exe) \
		$(foreach net,$(NETWORKS),$(BIN_DIR)/$(net)_fhelipe_client_exe)
	@echo "✅ All client executables built successfully!"

# Build everything (all server and client executables)
.PHONY: all-executables
all-executables: all-server-executables all-client-executables
	@echo "✅ All executables built successfully!"

# Generate all keys for testing (baseline, opt, prefetch, fhelipe)
.PHONY: all-test-keys
all-test-keys:
	@echo "=== Generating all test keys (baseline, opt, prefetch, fhelipe) ==="
	@$(MAKE) -j $(shell nproc) \
		$(foreach net,$(NETWORKS),$(net)-generate-keys-baseline) \
		$(foreach net,$(NETWORKS),$(net)-generate-keys-opt) \
		$(foreach net,$(NETWORKS),$(net)-generate-keys-prefetch) \
		$(foreach net,$(NETWORKS),$(net)-generate-keys-fhelipe)
	@echo "✅ All test keys generated successfully!"

# Generate all keys (all variations)
.PHONY: all-keys
all-keys:
	@echo "=== Generating all keys ==="
	@$(MAKE) -j $(shell nproc) \
		$(foreach net,$(NETWORKS),$(net)-generate-keys-baseline) \
		$(foreach net,$(NETWORKS),$(net)-generate-keys-opt) \
		$(foreach net,$(NETWORKS),$(net)-generate-keys-prefetch) \
		$(foreach net,$(NETWORKS),$(net)-generate-keys-fhelipe) \
		$(foreach net,$(NETWORKS),$(net)-generate-keys-kc)
	@echo "✅ All keys generated successfully!"

# Build ALL test executables AND generate ALL test keys in one parallel operation
.PHONY: all-test-ready
all-test-ready:
	@echo "=== Building all test executables AND generating keys in parallel ==="
	@$(MAKE)  \
		$(foreach net,$(NETWORKS),$(BIN_DIR)/$(net)_baseline_server_exe) \
		$(foreach net,$(NETWORKS),$(BIN_DIR)/$(net)_opt_server_exe) \
		$(foreach net,$(NETWORKS),$(BIN_DIR)/$(net)_prefetch_server_exe) \
		$(foreach net,$(NETWORKS),$(BIN_DIR)/$(net)_fhelipe_server_exe) \
		$(foreach net,$(NETWORKS),$(BIN_DIR)/$(net)_baseline_client_exe) \
		$(foreach net,$(NETWORKS),$(BIN_DIR)/$(net)_opt_client_exe) \
		$(foreach net,$(NETWORKS),$(BIN_DIR)/$(net)_prefetch_client_exe) \
		$(foreach net,$(NETWORKS),$(BIN_DIR)/$(net)_fhelipe_client_exe) \
		$(foreach net,$(NETWORKS),$(net)-generate-keys-baseline) \
		$(foreach net,$(NETWORKS),$(net)-generate-keys-opt) \
		$(foreach net,$(NETWORKS),$(net)-generate-keys-prefetch) \
		$(foreach net,$(NETWORKS),$(net)-generate-keys-fhelipe)
	@echo "✅ Everything ready for testing! All executables built and keys generated!"

# Build ALL executables AND generate ALL keys in one parallel operation
.PHONY: all-ready
all-ready:
	@echo "=== Building ALL executables AND generating ALL keys in parallel ==="
	@$(MAKE) -j $(shell nproc) \
		$(foreach net,$(NETWORKS),$(BIN_DIR)/$(net)_baseline_server_exe) \
		$(foreach net,$(NETWORKS),$(BIN_DIR)/$(net)_opt_server_exe) \
		$(foreach net,$(NETWORKS),$(BIN_DIR)/$(net)_prefetch_server_exe) \
		$(foreach net,$(NETWORKS),$(BIN_DIR)/$(net)_fhelipe_server_exe) \
		$(foreach net,$(NETWORKS),$(BIN_DIR)/$(net)_kc_server_exe) \
		$(foreach net,$(NETWORKS),$(BIN_DIR)/$(net)_baseline_client_exe) \
		$(foreach net,$(NETWORKS),$(BIN_DIR)/$(net)_opt_client_exe) \
		$(foreach net,$(NETWORKS),$(BIN_DIR)/$(net)_prefetch_client_exe) \
		$(foreach net,$(NETWORKS),$(BIN_DIR)/$(net)_fhelipe_client_exe) \
		$(foreach net,$(NETWORKS),$(net)-generate-keys-baseline) \
		$(foreach net,$(NETWORKS),$(net)-generate-keys-opt) \
		$(foreach net,$(NETWORKS),$(net)-generate-keys-prefetch) \
		$(foreach net,$(NETWORKS),$(net)-generate-keys-fhelipe) \
		$(foreach net,$(NETWORKS),$(net)-generate-keys-kc)
	@echo "✅ Everything ready! All executables built and all keys generated!"

# ============================================================================
# Cleanup
# ============================================================================

.PHONY: clean clean-mlir clean-cpp clean-exe clean-build

clean: clean-build
	rm -rf /tmp/keymemrt/

clean-build:
	rm -rf $(BUILD_DIR)

clean-mlir:
	rm -rf $(MLIR_DIR)

clean-cpp:
	rm -rf $(CPP_DIR)

clean-exe:
	rm -rf $(BIN_DIR)

