#include "server_runner.h"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <regex>

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"

// Global KeyMemRT instance
extern KeyMemRT keymem_rt;
std::unique_ptr<ResourceMonitor> monitor;

// Helper function implementations (moved from server_driver.cpp)
KeyMemMode parseKeyMemMode(const std::string &mode_str) {
  if (mode_str == "ignore")
    return KeyMemMode::IGNORE;
  if (mode_str == "imperative")
    return KeyMemMode::IMPERATIVE;
  if (mode_str == "prefetching")
    return KeyMemMode::PREFETCH;
  if (mode_str == "speculative")
    return KeyMemMode::SPECULATIVE;
  throw std::invalid_argument("Invalid key mode: " + mode_str);
}

const char* securityLevelToString(SecurityLevel level) {
  switch(level) {
    case HEStd_128_classic: return "HEStd_128_classic";
    case HEStd_192_classic: return "HEStd_192_classic";
    case HEStd_256_classic: return "HEStd_256_classic";
    case HEStd_128_quantum: return "HEStd_128_quantum";
    case HEStd_192_quantum: return "HEStd_192_quantum";
    case HEStd_256_quantum: return "HEStd_256_quantum";
    case HEStd_NotSet: return "HEStd_NotSet";
    default: return "Unknown";
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

// Core inference function
int run_single_inference(int run_id, const InferenceConfig &config,
                         ProgressCallback progress_callback) {

  // Log run start
  std::cout << "=== " << NETWORK_NAME << " FHE Inference Run " << run_id
            << " ===" << std::endl;

  const std::string DATAFOLDER = getSerializedDataFolder();
  std::string ccLocation = DATAFOLDER + "/cryptocontext.txt";
  std::string pubKeyLocation = DATAFOLDER + "/key_pub.txt";
  std::string secKeyLocation = DATAFOLDER + "/key_sec.txt";
  std::string cipherInputLocation = DATAFOLDER + "/ciphertext_input.txt";
  std::string multKeyLocation = DATAFOLDER + "/mult_key.txt";
  std::string configLocation = DATAFOLDER + "/config.txt";

  if (config.verbose) {
    std::cout << "Run " << run_id << " Configuration:" << std::endl;
    std::cout << "  Network: " << NETWORK_NAME << std::endl;
    std::cout << "  Key mode: " << config.key_mode << std::endl;
    std::cout << "  Depth: " << config.depth << std::endl;
    std::cout << "  Ring dimension: " << config.ring_dim << std::endl;
    std::cout << "  Security level: " << securityLevelToString(config.security_level) << std::endl;
    std::cout << "  Input size: " << INPUT_SIZE << std::endl;
    keymem_rt.setLogLevel(LogLevel::DEBUG);
  }

  monitor = std::make_unique<ResourceMonitor>(true);

  // Setup resource monitoring with run-specific filename
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
  std::string timestamp = ss.str();

  std::string filename =
      config.result_dir + "resource_usage_server_run" + std::to_string(run_id) +
      "_" + std::string(NETWORK_NAME) + "_" + config.key_mode + "_depth" +
      std::to_string(config.depth) + "_" + timestamp + ".csv";
  monitor->start(filename);

  // Setup KeyMemRT
  auto mode = parseKeyMemMode(config.key_mode);
  keymem_rt.setKeyMemMode(mode);
  keymem_rt.setPlatform(Platform::SERVER);
  keymem_rt.setMultDepth(config.depth);
  keymem_rt.setPrefetchSaturation(config.prefetch_sat);
  if (mode == KeyMemMode::IGNORE)
    BenchmarkCLI::setSerSingleFile(true);
  BenchmarkCLI::setInputDir(DATAFOLDER);

  // monitor.mark_event_start("context_loads");

  // Load cryptocontext
  std::cout << "Run " << run_id << ": Loading cryptocontext...\n";
  CryptoContext<DCRTPoly> cryptoContext;
  cryptoContext->ClearEvalMultKeys();
  cryptoContext->ClearEvalAutomorphismKeys();
  lbcrypto::CryptoContextFactory<lbcrypto::DCRTPoly>::ReleaseAllContexts();

  if (!Serial::DeserializeFromFile(ccLocation, cryptoContext,
                                   SerType::BINARY)) {
    std::cerr << "Run " << run_id
              << ": Error reading serialized cryptocontext from " << ccLocation
              << std::endl;
    return 1;
  }
  std::cout << "Run " << run_id << ": Cryptocontext loaded successfully\n";

  // Load public key
  std::cout << "Run " << run_id << ": Loading public key...\n";
  PublicKey<DCRTPoly> publicKey;
  if (!Serial::DeserializeFromFile(pubKeyLocation, publicKey,
                                   SerType::BINARY)) {
    std::cerr << "Run " << run_id
              << ": Error reading serialized public key from " << pubKeyLocation
              << std::endl;
    return 1;
  }
  std::cout << "Run " << run_id << ": Public key loaded successfully\n";

  // Load secret key
  std::cout << "Run " << run_id << ": Loading secret key...\n";
  PrivateKey<DCRTPoly> secretKey;
  if (!Serial::DeserializeFromFile(secKeyLocation, secretKey,
                                   SerType::BINARY)) {
    std::cerr << "Run " << run_id
              << ": Error reading serialized secret key from " << secKeyLocation
              << std::endl;
    return 1;
  }
  std::cout << "Run " << run_id << ": Secret key loaded successfully\n";

  std::ifstream multKeyIStream(multKeyLocation,
                               std::ios::in | std::ios::binary);
  if (!multKeyIStream.is_open()) {
    std::cerr << "Run " << run_id << ": Cannot read serialization from "
              << multKeyLocation << std::endl;
    return 1;
  }
  if (!cryptoContext->DeserializeEvalMultKey(multKeyIStream, SerType::BINARY)) {
    std::cerr << "Run " << run_id
              << ": Could not deserialize eval mult key file" << std::endl;
    return 1;
  }

  // Set up KeyMemRT for loading rotation keys
  keymem_rt.setCryptoContext(cryptoContext);
  keymem_rt.setKeyTag(secretKey->GetKeyTag());
  keymem_rt.setPlatform(Platform::SERVER);
  cryptoContext = UNIVERSAL_CONFIGURE(cryptoContext, secretKey);

  // Load the encrypted input
  std::cout << "Run " << run_id << ": Loading encrypted input...\n";
  Ciphertext<DCRTPoly> input_encrypted;
  if (!Serial::DeserializeFromFile(cipherInputLocation, input_encrypted,
                                   SerType::BINARY)) {
    std::cerr << "Run " << run_id << ": Error reading serialized input from "
              << cipherInputLocation << std::endl;
    return 1;
  }

  // monitor.mark_event_end("context_loads");

  // Run inference with timing
  std::cout << "\nRun " << run_id << ": Running " << NETWORK_NAME
            << " inference..." << std::endl;

  auto start_chrono = std::chrono::high_resolution_clock::now();
  std::clock_t c_start = std::clock();

  // monitor.mark_event_start("run");
  if (mode == KeyMemMode::IGNORE)
    keymem_rt.deserializeAllKeys(BOOTSTRAP_ENABLED);

  auto outputEncrypted = UNIVERSAL_INFERENCE(cryptoContext, input_encrypted);
  // monitor.mark_event_end("run");

  std::clock_t c_end = std::clock();
  auto end_chrono = std::chrono::high_resolution_clock::now();

  auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_chrono - start_chrono);
  double time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;

  std::cout << "Run " << run_id << ": ✅ Inference completed!" << std::endl;
  std::cout << "Run " << run_id << ": Time (chrono): " << duration_ms.count()
            << " ms" << std::endl;
  std::cout << "Run " << run_id << ": Time (clock):  " << time_elapsed_ms
            << " ms" << std::endl;

  // Stop resource monitoring
  monitor->stop();
  monitor->save_to_file(filename);
  std::cout << "Run " << run_id << ": Resource usage saved to: " << filename
            << std::endl;

  std::cout << "\nRun " << run_id << ": 🎉 " << NETWORK_NAME
            << " FHE inference completed successfully!" << std::endl;

  return 0;
}
