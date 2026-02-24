#pragma once

#include "KeyMemRT.hpp"
#include "ResourceMonitor.hpp"
#include "generic_header.h"
#include "network.h"
#include "openfhe.h"

#include <atomic>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <streambuf>
#include <string>

using namespace lbcrypto;

// Progress callback for Run 0 operation tracking
using ProgressCallback = std::function<void(int operation_id)>;

// Configuration structure for inference runs
struct InferenceConfig {
  std::string key_mode = "ignore";
  int depth = DEFAULT_DEPTH;
  int ring_dim = DEFAULT_RING_DIM;
  int prefetch_sat = 5;
  std::string result_dir = "./results";
  bool verbose = false;
  SecurityLevel security_level = HEStd_128_classic; // Default to HEStd_128_classic
};

// Core inference function used by both single and multi drivers
int run_single_inference(int run_id,
                         const InferenceConfig &config = InferenceConfig{},
                         ProgressCallback progress_callback = nullptr);

// Helper functions
KeyMemMode parseKeyMemMode(const std::string &mode_str);
std::vector<double> generateTestInput();
