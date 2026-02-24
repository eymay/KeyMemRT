#include "KeyMemRT.hpp"
#include "ResourceMonitor.hpp"
#include "network.h"
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

KeyMemRT keymem_rt;
std::unique_ptr<ResourceMonitor> monitor;

using namespace lbcrypto;

// Command line argument parsing
struct Config {
  std::string key_mode = "ignore";
  std::string result_dir = "./build/results/";
  bool verbose = false;
  int depth = DEFAULT_DEPTH;
  int ring_dim = DEFAULT_RING_DIM;
  int prefetch_sat = 5;

  void parse_args(int argc, char *argv[]) {
    for (int i = 1; i < argc; i++) {
      if (strcmp(argv[i], "--key-mode") == 0 && i + 1 < argc) {
        key_mode = argv[++i];
      } else if (strcmp(argv[i], "--result-dir") == 0) {
        result_dir = argv[++i];
      } else if (strcmp(argv[i], "--verbose") == 0) {
        verbose = true;
      } else if (strcmp(argv[i], "--depth") == 0 && i + 1 < argc) {
        depth = std::stoi(argv[++i]);
      } else if (strcmp(argv[i], "--ring-dim") == 0 && i + 1 < argc) {
        ring_dim = std::stoi(argv[++i]);
      } else if (strcmp(argv[i], "--prefetch-sat") == 0 && i + 1 < argc) {
        prefetch_sat = std::stoi(argv[++i]);
      } else if (strcmp(argv[i], "--help") == 0) {
        print_help();
        exit(0);
      }
    }

    // Validate key_mode
    if (key_mode != "ignore" && key_mode != "imperative" &&
        key_mode != "prefetching") {
      std::cerr
          << "Error: key-mode must be one of: ignore, imperative, prefetching"
          << std::endl;
      exit(1);
    }
  }

  void print_help() {
    std::cout << "Usage: " << NETWORK_NAME << " [options]\n"
              << "Options:\n"
              << "  --key-mode <mode>      Key management mode: ignore, "
                 "imperative, prefetching (default: ignore)\n"
              << "  --verbose              Enable verbose output\n"
              << "  --depth <n>            Multiplicative depth (default: "
              << DEFAULT_DEPTH << ")\n"
              << "  --ring-dim <n>         Ring dimension (default: "
              << DEFAULT_RING_DIM << ")\n"
              << "  --prefetch-sat <n>     Prefetch saturation (default: 5)\n"
              << "  --help                 Show this help message\n";
  }
};

KeyMemMode parseKeyMemMode(const std::string &mode_str) {
  if (mode_str == "ignore")
    return KeyMemMode::IGNORE;
  if (mode_str == "imperative")
    return KeyMemMode::IMPERATIVE;
  if (mode_str == "prefetching")
    return KeyMemMode::PREFETCH;
  throw std::invalid_argument("Invalid key mode: " + mode_str);
}

std::vector<double> generateTestInput() {
  std::vector<double> input(INPUT_SIZE);

// Generate network-appropriate test data
#ifdef NETWORK_MLP
  // MNIST-like data: normalized pixel values
  for (size_t i = 0; i < INPUT_SIZE; ++i) {
    input[i] = (i % 256) / 255.0; // Simulate pixel values
  }
#elif defined(NETWORK_RESNET) || defined(NETWORK_ALEXNET)
  // Image data: small uniform values with some variation
  std::fill(input.begin(), input.end(), 0.1);
  for (size_t i = 0; i < std::min(INPUT_SIZE, 100); ++i) {
    input[i] = 0.5; // Some higher values for variation
  }
#elif defined(NETWORK_LOLANET)
  // LolaNet-specific input pattern
  for (size_t i = 0; i < INPUT_SIZE; ++i) {
    input[i] = 0.01 * (i % 100); // Small values with pattern
  }
#endif

  return input;
}

int main(int argc, char *argv[]) {
  std::cout << "=== " << NETWORK_NAME << " FHE Inference Test ===" << std::endl;

  Config config;
  config.parse_args(argc, argv);

  if (config.verbose) {
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Network: " << NETWORK_NAME << std::endl;
    std::cout << "  Key mode: " << config.key_mode << std::endl;
    std::cout << "  Depth: " << config.depth << std::endl;
    std::cout << "  Ring dimension: " << config.ring_dim << std::endl;
    std::cout << "  Input size: " << INPUT_SIZE << std::endl;
  }

  // Setup resource monitoring
  monitor = std::make_unique<ResourceMonitor>(true);

  // Setup crypto context with network-appropriate parameters
  CCParamsT params;
  params.SetMultiplicativeDepth(config.depth);
  params.SetRingDim(config.ring_dim);
  params.SetScalingModSize(59);
  params.SetFirstModSize(60);
  params.SetSecurityLevel(HEStd_NotSet);

  std::cout << "Setting up crypto context..." << std::endl;
  std::cout << "  Multiplicative depth: " << config.depth << std::endl;
  std::cout << "  Ring dimension: " << config.ring_dim << std::endl;

  auto cryptoContext = GenCryptoContext(params);
  cryptoContext->Enable(PKE);
  cryptoContext->Enable(KEYSWITCH);
  cryptoContext->Enable(LEVELEDSHE);
  cryptoContext->Enable(ADVANCEDSHE);
  if (BOOTSTRAP_ENABLED)
    cryptoContext->Enable(FHE);

  // std::cout << "Auto selected Ring Dimension: "
  //           << cryptoContext->GetRingDimension() << "\n";
  std::cout << "Generating keys..." << std::endl;
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;

  // Setup KeyMemRT
  auto mode = parseKeyMemMode(config.key_mode);
  keymem_rt.setKeyMemMode(mode);
  keymem_rt.setPlatform(Platform::CLIENT);
  keymem_rt.setKeyTag(secretKey->GetKeyTag());
  keymem_rt.setCryptoContext(cryptoContext);
  keymem_rt.setMultDepth(config.depth);
  keymem_rt.setPrefetchSaturation(config.prefetch_sat);

  // Configure context for the specific network (uses UNIVERSAL_CONFIGURE macro)
  cryptoContext = UNIVERSAL_CONFIGURE(cryptoContext, secretKey);
  keymem_rt.setPlatform(Platform::SERVER);

  // Setup resource monitoring with timestamp
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
  std::string timestamp = ss.str();

  std::string filename = config.result_dir + "resource_usage_" +
                         std::string(NETWORK_NAME) + "_" + config.key_mode +
                         "_depth" + std::to_string(config.depth) + "_" +
                         timestamp + ".csv";
  monitor->start(filename);

  // Generate test input
  auto inputVector = generateTestInput();

  if (config.verbose) {
    std::cout << "Input size: " << inputVector.size() << " elements"
              << std::endl;
    std::cout << "Sample values: [" << inputVector[0] << ", " << inputVector[1]
              << ", " << inputVector[2] << ", ...]" << std::endl;
  }

  // Encrypt input
  auto inputPlaintext = cryptoContext->MakeCKKSPackedPlaintext(inputVector);
  auto inputEncrypted = cryptoContext->Encrypt(publicKey, inputPlaintext);

  // Run inference with timing (uses UNIVERSAL_INFERENCE macro)
  std::cout << "\nRunning " << NETWORK_NAME << " inference..." << std::endl;

  auto start_chrono = std::chrono::high_resolution_clock::now();
  std::clock_t c_start = std::clock();

  auto outputEncrypted = UNIVERSAL_INFERENCE(cryptoContext, inputEncrypted);

  std::clock_t c_end = std::clock();
  auto end_chrono = std::chrono::high_resolution_clock::now();

  auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_chrono - start_chrono);
  double time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;

  std::cout << "✅ Inference completed!" << std::endl;
  std::cout << "Time (chrono): " << duration_ms.count() << " ms" << std::endl;
  std::cout << "Time (clock):  " << time_elapsed_ms << " ms" << std::endl;

  // Stop resource monitoring
  monitor->stop();
  monitor->save_to_file(filename);
  std::cout << "Resource usage saved to: " << filename << std::endl;

  // Optional: Print KeyMemRT statistics
  if (config.key_mode != "ignore") {
    // keymem_rt.printStats();  // If this method exists
  }

  std::cout << "\n🎉 " << NETWORK_NAME
            << " FHE inference completed successfully!" << std::endl;

  return 0;
}
