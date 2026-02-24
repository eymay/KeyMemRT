#include "server_runner.h"
#include <iostream>
#include <string>

// Global KeyMemRT instance (required by the inference runner)
KeyMemRT keymem_rt;

// Configuration structure for command line parsing
struct Config {
  int run_id = 0; // Default to Run 0 for backward compatibility
  std::string key_mode = "ignore";
  int depth = DEFAULT_DEPTH;
  int ring_dim = DEFAULT_RING_DIM;
  int prefetch_sat = 5;
  std::string result_dir = "./results";
  bool verbose = false;
  SecurityLevel security_level = HEStd_NotSet; // Default to HEStd_128_classic

  void parse_args(int argc, char *argv[]) {
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];

      if (arg == "--help") {
        print_help();
        exit(0);
      } else if (arg == "--run-id" && i + 1 < argc) {
        run_id = std::stoi(argv[++i]);
      } else if (arg == "--key-mode" && i + 1 < argc) {
        key_mode = argv[++i];
      } else if (arg == "--depth" && i + 1 < argc) {
        depth = std::stoi(argv[++i]);
      } else if (arg == "--ring-dim" && i + 1 < argc) {
        ring_dim = std::stoi(argv[++i]);
      } else if (arg == "--prefetch-sat" && i + 1 < argc) {
        prefetch_sat = std::stoi(argv[++i]);
      } else if (arg == "--result-dir" && i + 1 < argc) {
        result_dir = argv[++i];
      } else if (arg == "--security-level" && i + 1 < argc) {
        std::string level = argv[++i];
        if (level == "HEStd_128_classic")
          security_level = HEStd_128_classic;
        else if (level == "HEStd_192_classic")
          security_level = HEStd_192_classic;
        else if (level == "HEStd_256_classic")
          security_level = HEStd_256_classic;
        else if (level == "HEStd_128_quantum")
          security_level = HEStd_128_quantum;
        else if (level == "HEStd_192_quantum")
          security_level = HEStd_192_quantum;
        else if (level == "HEStd_256_quantum")
          security_level = HEStd_256_quantum;
        else if (level == "HEStd_NotSet")
          security_level = HEStd_NotSet;
        else {
          std::cerr << "Unknown security level: " << level << std::endl;
          print_help();
          exit(1);
        }
      } else if (arg == "--verbose") {
        verbose = true;
      } else {
        std::cerr << "Unknown argument: " << arg << std::endl;
        print_help();
        exit(1);
      }
    }
  }

  void print_help() {
    std::cout
        << "Usage: " << NETWORK_NAME << "_server [options]\n"
        << "Options:\n"
        << "  --key-mode <mode>      Key memory mode (default: ignore)\n"
        << "                         Options: ignore, imperative, prefetching, speculative\n"
        << "  --verbose              Enable verbose output\n"
        << "  --depth <n>            Multiplicative depth (default: "
        << DEFAULT_DEPTH << ")\n"
        << "  --ring-dim <n>         Ring dimension (default: "
        << DEFAULT_RING_DIM << ")\n"
        << "  --prefetch-sat <n>     Prefetch saturation (default: 5)\n"
        << "  --result-dir <path>    Results directory (default: ./results)\n"
        << "  --security-level <lvl> Security level (default: "
           "HEStd_128_classic)\n"
        << "                         Options: HEStd_128_classic, "
           "HEStd_192_classic,\n"
        << "                                  HEStd_256_classic, "
           "HEStd_128_quantum,\n"
        << "                                  HEStd_192_quantum, "
           "HEStd_256_quantum,\n"
        << "                                  HEStd_NotSet (for minimal "
           "overhead)\n"
        << "  --help                 Show this help message\n";
  }
};

int main(int argc, char *argv[]) {
  std::cout << "=== " << NETWORK_NAME
            << " FHE Inference Server ===" << std::endl;

  // Parse command line arguments
  Config config;
  config.parse_args(argc, argv);

  // Convert to InferenceConfig
  InferenceConfig inference_config;
  inference_config.key_mode = config.key_mode;
  inference_config.depth = config.depth;
  inference_config.ring_dim = config.ring_dim;
  inference_config.prefetch_sat = config.prefetch_sat;
  inference_config.result_dir = config.result_dir;
  inference_config.verbose = config.verbose;
  inference_config.security_level = config.security_level;

  // Create results directory if it doesn't exist
  // std::filesystem::create_directories(config.result_dir);

  // Run single inference (Run ID = 0, no progress callback needed)
  int result = run_single_inference(config.run_id, inference_config);

  if (result == 0) {
    std::cout << "\n🎉 " << NETWORK_NAME << " server completed successfully!"
              << std::endl;
  } else {
    std::cerr << "\n❌ " << NETWORK_NAME
              << " server failed with error code: " << result << std::endl;
  }

  return result;
}
