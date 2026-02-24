#include "KeyMemRT.hpp"
#include "ResourceMonitor.hpp"
#include "network.h"
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"

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
  SecurityLevel security_level = HEStd_NotSet; // Default to HEStd_128_classic

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
      } else if (strcmp(argv[i], "--security-level") == 0 && i + 1 < argc) {
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

KeyMemMode parseKeyMemMode(const std::string &mode_str) {
  if (mode_str == "ignore")
    return KeyMemMode::IGNORE;
  if (mode_str == "imperative")
    return KeyMemMode::IMPERATIVE;
  if (mode_str == "prefetching")
    return KeyMemMode::PREFETCH;
  throw std::invalid_argument("Invalid key mode: " + mode_str);
}

const char *securityLevelToString(SecurityLevel level) {
  switch (level) {
  case HEStd_128_classic:
    return "HEStd_128_classic";
  case HEStd_192_classic:
    return "HEStd_192_classic";
  case HEStd_256_classic:
    return "HEStd_256_classic";
  case HEStd_128_quantum:
    return "HEStd_128_quantum";
  case HEStd_192_quantum:
    return "HEStd_192_quantum";
  case HEStd_256_quantum:
    return "HEStd_256_quantum";
  case HEStd_NotSet:
    return "HEStd_NotSet";
  default:
    return "Unknown";
  }
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

  const std::string DATAFOLDER = getSerializedDataFolder();
  std::string ccLocation = DATAFOLDER + "/cryptocontext.txt";
  std::string pubKeyLocation = DATAFOLDER + "/key_pub.txt";
  std::string secKeyLocation = DATAFOLDER + "/key_sec.txt";
  std::string cipherInputLocation = DATAFOLDER + "/ciphertext_input.txt";
  std::string multKeyLocation = DATAFOLDER + "/mult_key.txt";
  std::string configLocation = DATAFOLDER + "/config.txt";

  if (config.verbose) {
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Network: " << NETWORK_NAME << std::endl;
    std::cout << "  Key mode: " << config.key_mode << std::endl;
    std::cout << "  Depth: " << config.depth << std::endl;
    std::cout << "  Ring dimension: " << config.ring_dim << std::endl;
    std::cout << "  Security level: "
              << securityLevelToString(config.security_level) << std::endl;
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
  params.SetSecurityLevel(config.security_level);

  std::cout << "Setting up crypto context..." << std::endl;
  std::cout << "  Multiplicative depth: " << config.depth << std::endl;
  std::cout << "  Ring dimension: " << config.ring_dim << std::endl;
  std::cout << "  Security level: "
            << securityLevelToString(config.security_level) << std::endl;

  auto cryptoContext = GenCryptoContext(params);
  cryptoContext->Enable(PKE);
  cryptoContext->Enable(KEYSWITCH);
  cryptoContext->Enable(LEVELEDSHE);
  cryptoContext->Enable(ADVANCEDSHE);
  if (BOOTSTRAP_ENABLED)
    cryptoContext->Enable(FHE);

  std::cout << "Generating keys..." << std::endl;
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;

  cryptoContext->EvalMultKeyGen(secretKey);

  // Setup KeyMemRT
  auto mode = parseKeyMemMode(config.key_mode);
  keymem_rt.setKeyMemMode(mode);
  keymem_rt.setPlatform(Platform::CLIENT);
  keymem_rt.setKeyTag(secretKey->GetKeyTag());
  keymem_rt.setCryptoContext(cryptoContext);
  keymem_rt.setMultDepth(config.depth);
  keymem_rt.setPrefetchSaturation(config.prefetch_sat);
  if (mode == KeyMemMode::IGNORE)
    BenchmarkCLI::setSerSingleFile(true);
  BenchmarkCLI::setInputDir(DATAFOLDER);
  // Setup resource monitoring with timestamp
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
  std::string timestamp = ss.str();

  std::string filename = config.result_dir + "resource_usage_client_" +
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

  // Serialize cryptocontext
  monitor->mark_event_start("serialize_contexts");
  std::cout << "Serializing cryptocontext...\n";
  if (!Serial::SerializeToFile(ccLocation, cryptoContext, SerType::BINARY)) {
    std::cerr << "Error writing serialization of the crypto context to "
              << ccLocation << std::endl;
    return 1;
  }
  std::cout << "Cryptocontext serialized to " << ccLocation << std::endl;

  // Serialize public key
  std::cout << "Serializing public key...\n";
  if (!Serial::SerializeToFile(pubKeyLocation, publicKey, SerType::BINARY)) {
    std::cerr << "Error writing serialization of the public key to "
              << pubKeyLocation << std::endl;
    return 1;
  }
  std::cout << "Public key serialized to " << pubKeyLocation << std::endl;

  // Serialize secret key
  std::cout << "Serializing secret key...\n";
  if (!Serial::SerializeToFile(secKeyLocation, secretKey, SerType::BINARY)) {
    std::cerr << "Error writing serialization of the secret key to "
              << secKeyLocation << std::endl;
    return 1;
  }
  std::cout << "Secret key serialized to " << secKeyLocation << std::endl;

  // Note: Rotation keys have already been serialized by
  // matmul__configure_crypto_context via keymem_rt
  std::cout << "Rotation keys have been serialized through KeyMemRT\n";

  std::cout << "Serializing encrypted input...\n";
  if (!Serial::SerializeToFile(cipherInputLocation, inputEncrypted,
                               SerType::BINARY)) {
    std::cerr << "Error writing serialization of the encrypted input to "
              << cipherInputLocation << std::endl;
    return 1;
  }
  std::cout << "Encrypted input serialized to " << cipherInputLocation
            << std::endl;

  std::ofstream multKeyFile(multKeyLocation, std::ios::out | std::ios::binary);
  if (multKeyFile.is_open()) {
    if (!cryptoContext->SerializeEvalMultKey(multKeyFile, SerType::BINARY)) {
      std::cerr << "Error writing eval mult keys" << std::endl;
      return 1;
    }
    std::cout << "EvalMult/ relinearization keys have been serialized"
              << std::endl;
    multKeyFile.close();
  } else {
    std::cerr << "Error serializing EvalMult keys" << std::endl;
    std::exit(1);
  }
  monitor->mark_event_end("serialize_context");

  auto start_chrono = std::chrono::high_resolution_clock::now();
  std::clock_t c_start = std::clock();

  monitor->mark_event_start("serialize_rotation_keys");
  // Configure context for the specific network (uses UNIVERSAL_CONFIGURE macro)
  keymem_rt.setPlatform(Platform::CLIENT);
  cryptoContext = UNIVERSAL_CONFIGURE(cryptoContext, secretKey);
  monitor->mark_event_end("serialize_rotation_keys");

  std::clock_t c_end = std::clock();
  auto end_chrono = std::chrono::high_resolution_clock::now();

  auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_chrono - start_chrono);
  double time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;

  std::cout << "✅ Key Generation completed!" << std::endl;
  std::cout << "Time (chrono): " << duration_ms.count() << " ms" << std::endl;
  std::cout << "Time (clock):  " << time_elapsed_ms << " ms" << std::endl;

  // Stop resource monitoring
  monitor->stop();
  monitor->save_to_file(filename);
  std::cout << "Resource usage saved to: " << filename << std::endl;
  return 0;
}
