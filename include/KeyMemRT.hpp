#ifndef KEY_MANAGER_H_
#define KEY_MANAGER_H_

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <string>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

#include "KeyCompression.hpp"
#include "Logger.hpp"
#include "cryptocontext.h"
#include "openfhe.h"

using namespace lbcrypto;

inline EvalKey<DCRTPoly>
KeyMemRTConjugateKeyGen(const PrivateKey<DCRTPoly> privateKey) {
  const auto cc = privateKey->GetCryptoContext();
  auto algo = cc->GetScheme();

  const DCRTPoly &s = privateKey->GetPrivateElement();
  usint N = s.GetRingDimension();

  PrivateKey<DCRTPoly> privateKeyPermuted =
      std::make_shared<PrivateKeyImpl<DCRTPoly>>(cc);

  usint index = 2 * N - 1;
  std::vector<usint> vec(N);
  PrecomputeAutoMap(N, index, &vec);

  DCRTPoly sPermuted = s.AutomorphismTransform(index, vec);

  privateKeyPermuted->SetPrivateElement(sPermuted);
  privateKeyPermuted->SetKeyTag(privateKey->GetKeyTag());

  return algo->KeySwitchGen(privateKey, privateKeyPermuted);
}

#include "openfhe.h"

using namespace lbcrypto;

using RotKey = int;

enum class KeyMemMode {
  IGNORE,     // Ignores all key management operations
  IMPERATIVE, // Performs operations as requested
  PREFETCH,   // Background deserialization with queue
  SPECULATIVE // Prefetch non-blocking, load blocking with wait for key arrival
};

enum class Platform { CLIENT, SERVER };

inline std::string getModeString(KeyMemMode mode) {
  switch (mode) {
  case KeyMemMode::IGNORE:
    return "ignore";
  case KeyMemMode::IMPERATIVE:
    return "imperative";
  case KeyMemMode::PREFETCH:
    return "prefetch";
  case KeyMemMode::SPECULATIVE:
    return "speculative";
  }
  return "unknown";
}

// Define the data folder for serialization - can be overridden by environment
inline std::string getSerializedDataFolder() {
  const char *env_dir = std::getenv("SERIALIZED_DATA_DIR");
  if (env_dir != nullptr) {
    return std::string(env_dir);
  }
  return "serializedData"; // Default fallback
}

// Simple CLI helper that can be reused across benchmarks
class BenchmarkCLI {
public:
  // Parse command line arguments
  static void parseArgs(int argc, char *argv[]) {
    // Default params
    ser_single_file = false;
    log_level = LogLevel::INFO;
    log_to_file = false;
    prefetchSat = 250;
    log_filename = "keymemrt.log";
    input_dir = getSerializedDataFolder(); // Env variable
    output_dir = "";                       // Default empty directory
    depth = 4;                             // Default depth
    verbose = false;                       // Default verbose setting

    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];

      if (arg == "--key-mode" && i + 1 < argc) {
        std::string mode = argv[++i];
        if (mode == "ignore") {
          keymem_mode = KeyMemMode::IGNORE;
          ser_single_file = true;
        } else if (mode == "imperative") {
          keymem_mode = KeyMemMode::IMPERATIVE;
          ser_single_file = false;
        } else if (mode == "prefetch") {
          keymem_mode = KeyMemMode::PREFETCH;
          ser_single_file = false;
        } else if (mode == "speculative") {
          keymem_mode = KeyMemMode::SPECULATIVE;
          ser_single_file = false;
        }
        std::cout << "KeyMemRT mode set to: " << getModeString(keymem_mode)
                  << std::endl;
      } else if (arg == "--output-base" && i + 1 < argc) {
        output_base = argv[++i];
      } else if (arg == "--input-dir" && i + 1 < argc) {
        input_dir = argv[++i];
        std::cout << "Input directory set to: " << input_dir << std::endl;
      } else if (arg == "--output-dir" && i + 1 < argc) {
        output_dir = argv[++i];
        std::cout << "Output directory set to: " << output_dir << std::endl;
      } else if (arg == "--depth" && i + 1 < argc) {
        depth = std::stoi(argv[++i]);
        std::cout << "Multiplicative depth set to: " << depth << std::endl;
      } else if (arg == "--prefetch-sat" && i + 1 < argc) {
        prefetchSat = std::stoi(argv[++i]);
        std::cout << "Prefetch saturation set to : " << prefetchSat
                  << std::endl;
      } else if (arg == "--verbose" || arg == "-v") {
        verbose = true;
        std::cout << "Verbose output enabled" << std::endl;
      } else if (arg == "--ser-single-file") {
        ser_single_file = true;
        std::cout << "Single file serialization enabled" << std::endl;
      } else if (arg == "--log-level" && i + 1 < argc) {
        std::string level = argv[++i];
        if (level == "debug") {
          log_level = LogLevel::DEBUG;
        } else if (level == "info") {
          log_level = LogLevel::INFO;
        } else if (level == "warning") {
          log_level = LogLevel::WARNING;
        } else if (level == "error") {
          log_level = LogLevel::ERROR;
        } else if (level == "off") {
          log_level = LogLevel::OFF;
        }
        std::cout << "Log level set to: " << level << std::endl;
      } else if (arg == "--log-file" && i + 1 < argc) {
        log_filename = argv[++i];
        log_to_file = true;
        std::cout << "Logging to file: " << log_filename << std::endl;
      } else if (arg == "--log-console-off") {
        log_to_console = false;
        std::cout << "Console logging disabled" << std::endl;
      } else if (arg == "--help" || arg == "-h") {
        printHelp();
      }
    }

    // Initialize logger based on parameters
    Logger::getInstance().setLogLevel(log_level);
    if (log_to_file) {
      Logger::getInstance().setLogToFile(true, log_filename);
    }
    Logger::getInstance().setLogToConsole(log_to_console);

    LOG_INFO("BenchmarkCLI initialized with mode: {}",
             getModeString(keymem_mode));
    if (ser_single_file)
      LOG_INFO("Single file serialization enabled");
  }

  static void printHelp() {
    std::cout
        << "Benchmark CLI Options:\n"
        << "  --key-mode <mode>    : Set key memory mode "
           "(ignore|imperative|prefetch|speculative)\n"
        << "  --output-base <name> : Base name for output files\n"
        << "  --output-dir <dir>   : Directory for output files\n"
        << "  --depth <n>          : Set multiplicative depth parameter\n"
        << "  --verbose, -v        : Enable verbose output\n"
        << "  --ser-force          : Force serialization even in ignore mode\n"
        << "  --ser-single-file    : Use single file for all keys\n"
        << "  --log-level <level>  : Set log level "
           "(debug|info|warning|error|off)\n"
        << "  --log-file <file>    : Enable logging to specified file\n"
        << "  --log-console-off    : Disable console logging\n"
        << "  --help, -h           : Display this help message\n"
        << std::endl;
  }

  // Generate output filename based on benchmark name and mode
  static std::string getOutputFilename(const std::string &benchmark_name) {
    std::string filename =
        output_base.empty() ? "resource_usage_" + benchmark_name : output_base;

    // Add mode suffix if not already present
    std::string mode_str = getModeString(keymem_mode);
    if (filename.find(mode_str) == std::string::npos) {
      filename += "_" + mode_str;
    }

    // Add directory prefix if specified
    if (!output_dir.empty()) {
      filename = output_dir + "/" + filename;
    }

    LOG_DEBUG("Generated output filename: {}", filename);
    return filename;
  }

  // Gets the key memory mode
  static KeyMemMode getKeyMemMode() { return keymem_mode; }
  static bool getSerSingleFile() { return ser_single_file; }
  static void setSerSingleFile(bool sf) { ser_single_file = sf; }
  static LogLevel getLogLevel() { return log_level; }
  static std::string getLogFilename() { return log_filename; }
  static std::string getInputDir() { return input_dir; }
  static void setInputDir(std::string input) { input_dir = input; }
  static std::string getOutputDir() { return output_dir; }
  static int getDepth() { return depth; }
  static int getPrefetchSat() { return prefetchSat; }
  static bool getVerbose() { return verbose; }

private:
  static inline KeyMemMode keymem_mode;
  static inline std::string output_base;
  static inline std::string input_dir;
  static inline std::string output_dir;
  static inline bool ser_single_file;
  static inline LogLevel log_level;
  static inline bool log_to_file;
  static inline bool log_to_console = true;
  static inline std::string log_filename;
  static inline int prefetchSat;
  static inline int depth = 4;
  static inline bool verbose = false;
};

class KeyMemRT {
public:
  KeyMemRT(CryptoContext<DCRTPoly> context,
           KeyMemMode mode = KeyMemMode::IMPERATIVE)
      : cc(std::move(context)), keyTag(""), operationMode(mode) {
    LOG_INFO("KeyMemRT initialized with mode: {}", getModeString(mode));
    if (mode == KeyMemMode::PREFETCH) {
      startPrefetchMode();
    }
  }

  KeyMemRT() : cc(nullptr), keyTag(""), operationMode(KeyMemMode::IMPERATIVE) {
    LOG_INFO("KeyMemRT default initialized with mode: {}",
             getModeString(operationMode));
  }

  ~KeyMemRT() {
    if (operationMode == KeyMemMode::PREFETCH)
      stopPrefetchMode();
  }
  struct Step {
    int8_t value; // -1, 0, or 1 for NAF; 0 or 1 for binary
    int stepSize; // The actual rotation amount (e.g., 32, 16, 8, etc.)

    Step(int8_t v, int s) : value(v), stepSize(s) {}
    Step(int s) : value(1), stepSize(s) {}
  };

  Ciphertext<DCRTPoly> rotateBinary(ConstCiphertext<DCRTPoly> input,
                                    int rotation) {
    if (rotation % input->GetSlots() == 0) {
      return input->Clone();
    }

    std::cout << "Steps for " << rotation << ": ";
    std::vector<Step> steps;
    for (int i = 31; i >= 0; --i) {
      auto stepSize = (1 << i);
      if (rotation & stepSize) {
        steps.emplace_back(1, stepSize);
        std::cout << stepSize << ", ";
      }
    }
    std::cout << "\n";
    Ciphertext<DCRTPoly> result = input->Clone();
    for (auto step : steps)
      result = cc->EvalRotate(result, step.stepSize);
    return result;
  }

  Ciphertext<DCRTPoly>
  fastRotateBinary(ConstCiphertext<DCRTPoly> input, int rotation,
                   const uint32_t M,
                   const std::shared_ptr<std::vector<DCRTPoly>> digit_decomp) {
    if (rotation % input->GetSlots() == 0) {
      return input->Clone();
    }

    std::cout << "Steps for " << rotation << ": ";
    std::vector<Step> steps;
    for (int i = 31; i >= 0; --i) {
      auto stepSize = (1 << i);
      if (rotation & stepSize) {
        steps.emplace_back(1, stepSize);
        std::cout << stepSize << ", ";
      }
    }
    std::cout << "\n";

    Ciphertext<DCRTPoly> result;

    // First step: use EvalFastRotation
    if (!steps.empty()) {
      result = cc->EvalFastRotation(input, steps[0].stepSize,
                                    cc->GetRingDimension() * 2, digit_decomp);

      // Remaining steps: use normal EvalRotate
      for (size_t i = 1; i < steps.size(); ++i) {
        result = cc->EvalRotate(result, steps[i].stepSize);
      }
    } else {
      result = input->Clone();
    }

    return result;
  }

  // Initialize from CLI arguments
  void initFromArgs(int argc, char *argv[]) {
    LOG_INFO("Initializing KeyMemRT from command line arguments");
    // Use the CLI helper to parse settings
    BenchmarkCLI::parseArgs(argc, argv);
    operationMode = BenchmarkCLI::getKeyMemMode();
    prefetchTowerLimit = BenchmarkCLI::getPrefetchSat();
    LOG_INFO("KeyMemRT mode set to: {}", getModeString(operationMode));
  }

  void setCryptoContext(CryptoContext<DCRTPoly> &context) {
    cc = context;
    LOG_INFO("Crypto context set");
  }

  void setKeyTag(const std::string &tag) {
    keyTag = tag;
    LOG_INFO("Key tag set to: {}", tag);
  }

  void setKeyMemMode(KeyMemMode mode) {
    LOG_INFO("Changing mode from {} to {}", getModeString(operationMode),
             getModeString(mode));
    if (operationMode == KeyMemMode::PREFETCH && mode != KeyMemMode::PREFETCH) {
      stopPrefetchMode();
    }
    operationMode = mode;
    if (mode == KeyMemMode::PREFETCH) {
      startPrefetchMode();
    }
  }

  void startPrefetchMode() {
    if (operationMode != KeyMemMode::PREFETCH) {
      return;
    }

    shouldStop.store(false);
    deserializationThread = std::thread(&KeyMemRT::deserializationWorker, this);
    LOG_INFO("PREFETCH mode started with background deserialization thread");
  }

  void stopPrefetchMode() {
    shouldStop.store(true);
    queueCondition.notify_all();

    if (deserializationThread.joinable()) {
      deserializationThread.join();
    }

    // Clear queues and status
    {
      std::lock_guard<std::mutex> queueLock(queueMutex);
      std::queue<std::pair<int, int>> emptyMain;
      std::queue<WaitQueueItem> emptyWait;
      deserializationQueue.swap(emptyMain);
      waitQueue.swap(emptyWait);
    }
    {
      std::lock_guard<std::mutex> statusLock(keyStatusMutex);
      keysBeingDeserialized.clear();
      keysReadyForUse.clear();
    }

    LOG_INFO("PREFETCH mode stopped");
  }

  KeyMemMode getOperationMode() const { return operationMode; }
  Platform getPlatform() const { return platform; }
  void setPlatform(Platform pl) { platform = pl; }
  void setMultDepth(int depth) { m_multDepth = depth; }
  void setPrefetchSaturation(int towerLimit) {
    prefetchTowerLimit = towerLimit;
    LOG_INFO("PREFETCH: Tower limit set to {}", prefetchTowerLimit);
  }

  int getPrefetchSaturation() const { return prefetchTowerLimit; }

  int calculateTowerCount(int depth) { return 32; }

  void setRotIndices(const std::vector<int32_t> &indices) {
    rotIndices = indices;
    LOG_INFO("Rotation indices set: {} indices", indices.size());
    if (Logger::getInstance().isDebugEnabled()) {
      std::stringstream ss;
      for (size_t i = 0; i < std::min(indices.size(), size_t(10)); i++) {
        ss << indices[i];
        if (i < std::min(indices.size(), size_t(10)) - 1)
          ss << ", ";
      }
      if (indices.size() > 10)
        ss << "...";
      LOG_DEBUG("Sample indices: {}", ss.str());
    }
  }
  void addRotIndices(const std::vector<int32_t> &newIndices) {

    // Create a set with existing indices for efficient merging
    std::set<int32_t> mergedIndicesSet(rotIndices.begin(), rotIndices.end());

    // Insert new indices (duplicates automatically handled by set)
    mergedIndicesSet.insert(newIndices.begin(), newIndices.end());

    // Convert back to vector and update internal storage
    rotIndices.assign(mergedIndicesSet.begin(), mergedIndicesSet.end());
  }

  /**
   * Serialize keys at specific level by compressing and serializing each key
   * individually Keys are compressed in place since they will be cleared
   * anyway
   */
  bool serializeKeysAtLevel(const std::vector<int32_t> &indices, int level) {
    if (operationMode == KeyMemMode::IGNORE) {
      LOG_DEBUG("IGNORE mode: Skipping serialize keys at level {}", level);
      return true;
    }

    LOG_INFO("Serializing {} keys at level {}", indices.size(), level);

    // Important: Get keys BEFORE attempting to compress them
    auto &allKeys = cc->GetEvalAutomorphismKeyMap(keyTag);

    // If no keys are found, we can't proceed
    if (allKeys.empty()) {
      LOG_ERROR("No evaluation keys found for key tag [{}]", keyTag);
      return false;
    }

    LOG_DEBUG("Found {} total keys in crypto context", allKeys.size());

    bool success = true;

    // Now compress and serialize each key individually
    for (int32_t rotIndex : indices) {
      auto automorphismIndex = cc->FindAutomorphismIndex(rotIndex);

      // Verify the key exists before trying to compress/serialize it
      if (allKeys.find(automorphismIndex) == allKeys.end()) {
        LOG_ERROR("Key not found for rotation index {} (automorphism {})",
                  rotIndex, automorphismIndex);
        success = false;
        continue;
      }

      auto &evalKey = allKeys[automorphismIndex];
      if (!evalKey) {
        LOG_ERROR("Eval key is null for rotation index {}", rotIndex);
        success = false;
        continue;
      }

      if (level > 0) {
        // Log key info before compression
        auto aVector = evalKey->GetAVector();
        if (!aVector.empty()) {
          size_t currentTowers = aVector[0].GetParams()->GetParams().size();
          LOG_DEBUG("Key {} has {} towers before compression", rotIndex,
                    currentTowers);
        }

        // Compress this key directly using our compressKeysToLevel method
        if (!compressKeyToLevel(evalKey, level)) {
          LOG_ERROR("Failed to compress key for rotation index {} to level {}",
                    rotIndex, level);
          success = false;
          continue;
        }

        aVector = evalKey->GetAVector();
        // Log compression result
        if (!aVector.empty()) {
          size_t compressedTowers = aVector[0].GetParams()->GetParams().size();
          LOG_DEBUG("Key {} compressed to {} towers", rotIndex,
                    compressedTowers);
        }
      }

      // Now serialize the compressed key
      std::string filename = getKeyFilename(rotIndex, level);
      LOG_INFO("Serializing key for rotation index {} (automorphism {}) at "
               "level {} to {}",
               rotIndex, automorphismIndex, level, filename);

      std::ofstream keyFile(filename, std::ios::binary);
      if (!keyFile) {
        LOG_ERROR("Failed to open file for writing: {}", filename);
        success = false;
        continue;
      }

      bool result = cc->SerializeEvalAutomorphismKey(
          keyFile, SerType::BINARY, keyTag, {automorphismIndex});

      if (result) {
        LOG_INFO(
            "Successfully serialized key for rotation index {} at level {}",
            rotIndex, level);
      } else {
        LOG_ERROR("Failed to serialize key for rotation index {} at level {}",
                  rotIndex, level);
        success = false;
      }
    }

    if (success) {
      LOG_INFO("Successfully serialized all {} keys at level {}",
               indices.size(), level);
    } else {
      LOG_ERROR("Some keys failed to serialize at level {}", level);
    }

    return success;
  }
  RotKey deserializeKey(int rotationIndex, int keyDepth = 0) {
    if (operationMode == KeyMemMode::IGNORE) {
      LOG_DEBUG(
          "IGNORE mode: Skipping deserialize for rotation index {} at depth {}",
          rotationIndex, keyDepth);
      return rotationIndex;
    }

    if (!rotationIndex) {
      LOG_DEBUG("deserializeKey called with 0 index, ignoring");
      return 0;
    }

    if (operationMode == KeyMemMode::PREFETCH) {

      debugPrintQueues();
      auto keyInfo = std::make_pair(rotationIndex, keyDepth);

      // In PREFETCH mode, check if key is ready, otherwise wait
      {
        std::lock_guard<std::mutex> statusLock(keyStatusMutex);
        if (keysReadyForUse.count(keyInfo)) {
          LOG_DEBUG("PREFETCH: Key {} at depth {} already ready for use",
                    rotationIndex, keyDepth);
          return rotationIndex;
        }
      }

      // Wait for deserialization to complete
      LOG_DEBUG("PREFETCH: Waiting for key {} at depth {} to be deserialized",
                rotationIndex, keyDepth);
      auto waitStart = std::chrono::high_resolution_clock::now();
      static const int DEADLOCK_CHECK_INTERVAL_SEC =
          10; // Check for deadlock every 10 seconds
      static const int DEADLOCK_TIMEOUT_SEC =
          120; // 2 minutes before declaring deadlock

      while (true) {
        bool shouldEnqueue = false;
        {
          std::lock_guard<std::mutex> statusLock(keyStatusMutex);
          if (keysReadyForUse.count(keyInfo)) {
            auto waitEnd = std::chrono::high_resolution_clock::now();
            auto waitDuration =
                std::chrono::duration_cast<std::chrono::microseconds>(waitEnd -
                                                                      waitStart)
                    .count();
            LOG_INFO("PREFETCH: Key {} at depth {} ready after waiting {} us",
                     rotationIndex, keyDepth, waitDuration);
            return rotationIndex;
          }

          // If not being deserialized and not ready, enqueue it
          if (!keysBeingDeserialized.count(keyInfo)) {
            shouldEnqueue = true;
          }
        }
        if (shouldEnqueue) {
          LOG_DEBUG("PREFETCH: Key {} at depth {} not in queue, enqueuing now",
                    rotationIndex, keyDepth);
          enqueueKey(rotationIndex, keyDepth);
        }

        std::unique_lock<std::mutex> lock(keyReadyMutex);
        bool keyReady = keyReadyCondition.wait_for(
            lock, std::chrono::seconds(DEADLOCK_CHECK_INTERVAL_SEC),
            [this, keyInfo]() {
              std::lock_guard<std::mutex> statusLock(keyStatusMutex);
              return keysReadyForUse.count(keyInfo) > 0;
            });

        if (keyReady) {
          // Key is ready
          std::lock_guard<std::mutex> statusLock(keyStatusMutex);
          auto waitEnd = std::chrono::high_resolution_clock::now();
          auto waitDuration =
              std::chrono::duration_cast<std::chrono::microseconds>(waitEnd -
                                                                    waitStart)
                  .count();
          LOG_INFO("PREFETCH: Key {} at depth {} ready after waiting {} us",
                   rotationIndex, keyDepth, waitDuration);
          return rotationIndex;
        }

        // Timeout - check for deadlock
        auto elapsed =
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::high_resolution_clock::now() - waitStart)
                .count();

        {
          std::lock_guard<std::mutex> statusLock(keyStatusMutex);
          std::lock_guard<std::mutex> queueLock(queueMutex);

          // Check if we're in a deadlock: tower limit reached and key not being
          // processed
          bool atCapacity = (currentTowerCount >= prefetchTowerLimit);
          bool keyNotReady = (keysReadyForUse.count(keyInfo) == 0);
          bool keyNotBeingProcessed =
              (keysBeingDeserialized.count(keyInfo) == 0);

          if (atCapacity && keyNotReady && keyNotBeingProcessed &&
              elapsed >= DEADLOCK_TIMEOUT_SEC) {
            LOG_ERROR("========================================================"
                      "============");
            LOG_ERROR("PREFETCH DEADLOCK DETECTED - ABORTING");
            LOG_ERROR("========================================================"
                      "============");
            LOG_ERROR("Key {} at depth {} not available after {} seconds",
                      rotationIndex, keyDepth, elapsed);
            LOG_ERROR("Tower capacity reached: {}/{} towers", currentTowerCount,
                      prefetchTowerLimit);
            LOG_ERROR(
                "This indicates --prefetch-sat is too low for this workload");
            LOG_ERROR(
                "SOLUTION: Increase --prefetch-sat parameter (current: {})",
                prefetchTowerLimit);
            LOG_ERROR("========================================================"
                      "============");

            std::cerr
                << "\n"
                << "==========================================================="
                   "=========\n"
                << "PREFETCH DEADLOCK DETECTED - ABORTING\n"
                << "==========================================================="
                   "=========\n"
                << "Key " << rotationIndex << " at depth " << keyDepth
                << " not available after " << elapsed << " seconds\n"
                << "Tower capacity reached: " << currentTowerCount << "/"
                << prefetchTowerLimit << " towers\n"
                << "This indicates --prefetch-sat is too low for this "
                   "workload\n"
                << "SOLUTION: Increase --prefetch-sat parameter (current: "
                << prefetchTowerLimit << ")\n"
                << "==========================================================="
                   "=========\n"
                << std::endl;

            std::exit(1); // Abort with error code
          } else if (elapsed % 30 == 0 && elapsed > 0) {
            // Log periodic warning every 30 seconds
            LOG_WARNING("PREFETCH: Still waiting for key {} at depth {} ({} "
                        "seconds elapsed)",
                        rotationIndex, keyDepth, elapsed);
          }
        }
      }
    }

    // IMPERATIVE mode - immediate deserialization
    return _deserializeKey(rotationIndex, keyDepth);
  }

  RotKey _deserializeKeyWithWait(int rotationIndex, int keyDepth) {
    /**
     * SPECULATIVE mode: Wait for key file to arrive before deserializing.
     * This simulates network transfer delays in cold-start scenarios.
     *
     * Behavior:
     * - Poll for key file existence with configurable interval
     * - Block until file appears or timeout occurs
     * - Log wait times for performance analysis
     */

    static const int KEY_POLL_INTERVAL_MS = 50;  // Poll every 50ms
    static const int KEY_WAIT_TIMEOUT_SEC = 120; // 2 minute timeout

    std::string filename = getKeyFilename(rotationIndex, keyDepth);
    auto automorphismIndex = getAutomorphismIndex(rotationIndex);

    LOG_INFO("SPECULATIVE: Waiting for key {} at depth {} (file: {})",
             rotationIndex, keyDepth, filename);

    auto startTime = std::chrono::steady_clock::now();

    // Poll for file existence
    while (true) {
      // Check if file exists and is readable
      std::ifstream testFile(filename);
      if (testFile.good()) {
        testFile.close();

        // File exists - proceed with deserialization
        auto waitEnd = std::chrono::steady_clock::now();
        auto waitDuration =
            std::chrono::duration_cast<std::chrono::milliseconds>(waitEnd -
                                                                  startTime)
                .count();

        LOG_INFO("SPECULATIVE: Key {} at depth {} arrived after {} ms, "
                 "deserializing...",
                 rotationIndex, keyDepth, waitDuration);

        // Now deserialize the key
        std::ifstream keyFile(filename, std::ios::binary);
        if (!keyFile) {
          LOG_ERROR("SPECULATIVE: Failed to open file for reading: {}",
                    filename);
          return rotationIndex;
        }

        bool success = cc->DeserializeEvalAutomorphismKey(
            keyFile, SerType::BINARY, keyTag, {automorphismIndex});

        if (success) {
          LOG_INFO("SPECULATIVE: Successfully deserialized key {} at depth {} "
                   "(waited {} ms)",
                   rotationIndex, keyDepth, waitDuration);

          // Handle key compression restoration if needed
          if (keyDepth > 0) {
            auto &keyMap = cc->GetEvalAutomorphismKeyMap(keyTag);
            auto keyIterator = keyMap.find(automorphismIndex);
            if (keyIterator != keyMap.end() && keyIterator->second) {
              auto cryptoParams = cc->GetCryptoParameters();
              RNSKeyCompressor::RestoreDynamicQSize(keyIterator->second,
                                                    cryptoParams);
            }
          }

          loadedKeys.insert(rotationIndex);
        } else {
          LOG_ERROR("SPECULATIVE: Failed to deserialize key {} at depth {}",
                    rotationIndex, keyDepth);
        }

        return rotationIndex;
      }

      // Check timeout
      auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::steady_clock::now() - startTime)
                         .count();

      if (elapsed > KEY_WAIT_TIMEOUT_SEC) {
        LOG_ERROR(
            "SPECULATIVE: Timeout ({} seconds) waiting for key {} at depth {}",
            KEY_WAIT_TIMEOUT_SEC, rotationIndex, keyDepth);
        return rotationIndex;
      }

      // Wait before next poll
      std::this_thread::sleep_for(
          std::chrono::milliseconds(KEY_POLL_INTERVAL_MS));

      // Log periodic status updates (every 5 seconds)
      if (elapsed > 0 && elapsed % 5 == 0) {
        LOG_DEBUG("SPECULATIVE: Still waiting for key {} at depth {} ({} "
                  "seconds elapsed)",
                  rotationIndex, keyDepth, elapsed);
      }
    }
  }

  RotKey _deserializeKey(int rotationIndex, int keyDepth) {
    if (operationMode == KeyMemMode::IGNORE) {
      LOG_DEBUG("IGNORE mode: Skipping deserialize for rotation index {}",
                rotationIndex);
      return rotationIndex;
    }

    // SPECULATIVE mode: wait for key file to arrive
    if (operationMode == KeyMemMode::SPECULATIVE) {
      return _deserializeKeyWithWait(rotationIndex, keyDepth);
    }

    auto automorphismIndex = getAutomorphismIndex(rotationIndex);

    std::string filename;

    filename = getKeyFilename(rotationIndex, keyDepth);
    std::ifstream testFile(filename);
    if (!testFile.good()) {
      LOG_ERROR("Could not find any depth file for rotation index {}",
                rotationIndex);
      return rotationIndex;
    }

    LOG_INFO("Deserializing key for rotation index {} at depth {} from {}",
             rotationIndex, keyDepth, filename);

    std::ifstream keyFile(filename, std::ios::binary);
    if (!keyFile) {
      LOG_ERROR("Failed to open file for reading: {}", filename);
      return rotationIndex;
    }

    bool success = cc->DeserializeEvalAutomorphismKey(
        keyFile, SerType::BINARY, keyTag, {automorphismIndex});

    if (success) {
      LOG_INFO(
          "Successfully deserialized key for rotation index {} at depth {}",
          rotationIndex, keyDepth);
      if (keyDepth == 0)
        return rotationIndex;

      // CRITICAL FIX: Restore the DynamicQSize after deserialization
      auto &keyMap = cc->GetEvalAutomorphismKeyMap(keyTag);
      auto keyIterator = keyMap.find(automorphismIndex);
      if (keyIterator != keyMap.end() && keyIterator->second) {
        auto cryptoParams = cc->GetCryptoParameters();
        if (RNSKeyCompressor::RestoreDynamicQSize(keyIterator->second,
                                                  cryptoParams))
          LOG_DEBUG("Restored DynamicQSize for key {} at level {}",
                    rotationIndex, keyDepth);
        else
          LOG_DEBUG("[FAIL] Not Restored DynamicQSize for key "
                    "{} at level {}",
                    rotationIndex, keyDepth);
      }
      loadedKeys.insert(rotationIndex);
    } else {
      LOG_ERROR("Failed to deserialize key for rotation index {} at depth {}",
                rotationIndex, keyDepth);
    }

    return rotationIndex;
  }

  // Add overload for backward compatibility
  bool serializeKey(int rotationIndex) {
    return serializeKeysAtLevel({rotationIndex}, 0);
  }

  // Helper to get depth-specific filename
  std::string getKeyFilename(int rotationIndex, int depth) const {
    if (depth == 0) {
      return BenchmarkCLI::getInputDir() + "/rotation_key_" +
             std::to_string(rotationIndex) + ".bin";
    } else {
      return BenchmarkCLI::getInputDir() + "/rotation_key_" +
             std::to_string(rotationIndex) + "_l" + std::to_string(depth) +
             ".bin";
    }
  }

  bool clearKey(RotKey rotationIndex) {
    if (operationMode == KeyMemMode::IGNORE) {
      LOG_DEBUG("IGNORE mode: Skipping clear for rotation index {}",
                rotationIndex);
      return true;
    }
    if (!rotationIndex) {
      LOG_DEBUG("clearKey called with 0 index, ignoring");
      return false;
    }

    auto automorphismIndex = getAutomorphismIndex(rotationIndex);
    LOG_INFO("Clearing key for rotation index {} (automorphism {})",
             rotationIndex, automorphismIndex);

    try {
      using ElementType = DCRTPoly;

      // Get a reference to the key map using the public static method
      auto &keyMap =
          lbcrypto::CryptoContextImpl<ElementType>::GetEvalAutomorphismKeyMap(
              keyTag);

      // Find the specific key by automorphism index
      auto keyIt = keyMap.find(automorphismIndex);
      if (keyIt == keyMap.end()) {
        LOG_WARNING("Key for rotation index {} (automorphism {}) not found",
                    rotationIndex, automorphismIndex);
        return false;
      }

      // Remove just this specific key
      keyMap.erase(keyIt);

      // Update our tracking
      loadedKeys.erase(rotationIndex);
      if (operationMode == KeyMemMode::PREFETCH) {
        {
          std::lock_guard<std::mutex> statusLock(keyStatusMutex);
          // Remove all entries for this rotation index regardless of depth
          auto it = keysReadyForUse.begin();
          int removedCount = 0;
          while (it != keysReadyForUse.end()) {
            if (it->first == rotationIndex) {
              int towers = calculateTowerCount(it->second);
              currentTowerCount -= towers;
              LOG_DEBUG("PREFETCH: Removing key {} at depth {} from ready set "
                        "(towers: {}, remaining: {}/{})",
                        it->first, it->second, towers, currentTowerCount,
                        prefetchTowerLimit);

              it = keysReadyForUse.erase(it);
              removedCount++;
            } else {
              ++it;
            }
          }
          if (removedCount > 1) {
            LOG_WARNING(
                "PREFETCH: WARNING - Found {} entries for rotation index "
                "{} in ready set! "
                "This should not happen with proper conflict detection.",
                removedCount, rotationIndex);
          } else if (removedCount == 0) {
            LOG_WARNING(
                "PREFETCH: WARNING - No entries found for rotation index "
                "{} in ready set! "
                "Key may have been cleared already.",
                rotationIndex);
          }
        }

        // Wait queue disabled - just notify worker if tower count allows more
        // work
        {
          std::lock_guard<std::mutex> queueLock(queueMutex);
          if (currentTowerCount < prefetchTowerLimit) {
            queueCondition.notify_one();
          }
        }

        LOG_DEBUG("PREFETCH: Notified worker after clearing key {}",
                  rotationIndex);
      }
      LOG_INFO("Successfully cleared key for rotation index {}", rotationIndex);
      return true;
    } catch (const std::exception &e) {
      LOG_ERROR("Exception while clearing key: {}", e.what());
      return false;
    }
  }

  /**
   * Compress a single evaluation key to a specific level
   * Calculates target towers as: multDepth + 1 - level
   */
  bool compressKeyToLevel(EvalKey<DCRTPoly> &evalKey, size_t level) {
    if (operationMode == KeyMemMode::IGNORE) {
      LOG_DEBUG(
          "IGNORE mode: Skipping compression for rotation index to depth {}",
          level);
      return true;
    }

    if (!evalKey) {
      LOG_DEBUG("Key null");
      return false;
    }

    // Get multiplicative depth from crypto parameters
    auto cryptoParams = cc->GetCryptoParameters();
    auto rnsCryptoParams =
        std::dynamic_pointer_cast<CryptoParametersRNS>(cryptoParams);
    if (!rnsCryptoParams) {
      LOG_DEBUG("Params not found");
      return false;
    }

    // Calculate target towers: multDepth + 1 - level
    if (level > m_multDepth + 1) {
      LOG_DEBUG("Invalid level");
      return false; // Invalid level
    }

    size_t targetQTowers = m_multDepth + 2 - level;
    LOG_DEBUG("Target Q Towers for key {}", targetQTowers);

    return RNSKeyCompressor::CompressKeyToTargetLevel(evalKey, targetQTowers,
                                                      cryptoParams);
  }

  bool compressKey(int rotationIndex, int level) {
    if (operationMode == KeyMemMode::IGNORE) {
      LOG_DEBUG("IGNORE mode: Skipping compression for rotation index {}",
                rotationIndex);
      return true;
    }

    auto &keyMap = cc->GetEvalAutomorphismKeyMap(keyTag);
    auto automorphismIndex = cc->FindAutomorphismIndex(rotationIndex);
    auto it = keyMap.find(automorphismIndex);
    LOG_INFO("Compressing key for rotation index {} to depth {}", rotationIndex,
             level);

    if (it != keyMap.end() && it->second) {
      return compressKeyToLevel(it->second, level);
    }

    return false;
  }

  bool compressKeyToCt(int rotationIndex,
                       ConstCiphertext<DCRTPoly> ciphertext) {
    if (operationMode == KeyMemMode::IGNORE) {
      LOG_DEBUG("IGNORE mode: Skipping compression for rotation index {}",
                rotationIndex);
      return true;
    }

    auto &keyMap = cc->GetEvalAutomorphismKeyMap(keyTag);
    auto automorphismIndex = cc->FindAutomorphismIndex(rotationIndex);
    auto it = keyMap.find(automorphismIndex);
    LOG_INFO("Compressing key for rotation index {}", rotationIndex);

    if (it != keyMap.end() && it->second) {
      return RNSKeyCompressor::CompressKeyToLevel(it->second, ciphertext);
    }

    return false;
  }

  bool serializeAllKeysToSingleFile(const std::string &filename,
                                    bool bootstrapEnabled = false) {
    LOG_INFO("Serializing all keys to single file: {}", filename);

    // Create vector of automorphism indices
    std::vector<usint> automorphismIndices;
    for (const auto &rotIndex : rotIndices) {
      automorphismIndices.push_back(getAutomorphismIndex(rotIndex));
    }

    if (bootstrapEnabled)
      automorphismIndices.push_back(cc->GetCyclotomicOrder() - 1);

    // Open file for writing
    std::ofstream keyFile(filename, std::ios::binary);
    if (!keyFile) {
      LOG_ERROR("Could not open file for writing: {}", filename);
      return false;
    }

    // Serialize all keys at once
    bool success = cc->SerializeEvalAutomorphismKey(
        keyFile, SerType::BINARY, keyTag, automorphismIndices);

    if (success) {
      LOG_INFO("Successfully serialized {} keys to {}",
               automorphismIndices.size(), filename);
    } else {
      LOG_ERROR("Failed to serialize keys to {}", filename);
    }

    return success;
  }

  bool deserializeAllKeysFromSingleFile(const std::string &filename,
                                        bool bootstrapEnabled = false) {
    LOG_INFO("Deserializing all keys from single file: {}", filename);

    // Create vector of automorphism indices
    std::vector<usint> automorphismIndices;
    for (const auto &rotIndex : rotIndices) {
      automorphismIndices.push_back(getAutomorphismIndex(rotIndex));
    }
    if (bootstrapEnabled)
      automorphismIndices.push_back(cc->GetCyclotomicOrder() - 1);

    // Open file for reading
    std::ifstream keyFile(filename, std::ios::binary);
    if (!keyFile) {
      LOG_ERROR("Could not open file for reading: {}", filename);
      return false;
    }

    // Deserialize all keys at once
    bool success = cc->DeserializeEvalAutomorphismKey(
        keyFile, SerType::BINARY, keyTag, automorphismIndices);

    if (success) {
      LOG_INFO("Successfully deserialized {} keys from {}",
               automorphismIndices.size(), filename);
      // Mark all keys as loaded
      for (const auto &rotIndex : rotIndices) {
        loadedKeys.insert(rotIndex);
      }
    } else {
      LOG_ERROR("Failed to deserialize keys from {}", filename);
    }

    return success;
  }

  bool serializeAllKeys(bool bootstrapEnabled = false) {
    bool singleFile = BenchmarkCLI::getSerSingleFile();
    if (singleFile)
      return serializeAllKeysToSingleFile(BenchmarkCLI::getInputDir() +
                                              "/rotation_key_all.bin",
                                          bootstrapEnabled);

    if (operationMode == KeyMemMode::IGNORE) {
      LOG_DEBUG("IGNORE mode: Skipping serialize all keys");
      return true;
    }

    LOG_INFO("Serializing all keys individually ({} rotation indices)",
             rotIndices.size());
    bool success = true;
    for (const auto &rotIndex : rotIndices) {
      success &= serializeKey(rotIndex);
    }

    if (success) {
      LOG_INFO("Successfully serialized all {} keys", rotIndices.size());
    } else {
      LOG_ERROR("Failed to serialize some keys");
    }

    return success;
  }

  bool deserializeAllKeys(bool bootstrapEnabled = false) {
    bool singleFile = BenchmarkCLI::getSerSingleFile();

    if (singleFile)
      return deserializeAllKeysFromSingleFile(BenchmarkCLI::getInputDir() +
                                                  "/rotation_key_all.bin",
                                              bootstrapEnabled);

    if (operationMode == KeyMemMode::IGNORE) {
      LOG_DEBUG("IGNORE mode: Skipping deserialize all keys");
      return true;
    }

    LOG_INFO("Deserializing all keys individually ({} rotation indices)",
             rotIndices.size());
    bool success = true;
    for (const auto &rotIndex : rotIndices) {
      success &= _deserializeKey(rotIndex, 0);
    }

    if (success) {
      LOG_INFO("Successfully deserialized all {} keys", rotIndices.size());
    } else {
      LOG_ERROR("Failed to deserialize some keys");
    }

    return success;
  }

  bool clearAllKeys() {
    if (operationMode == KeyMemMode::IGNORE) {
      LOG_DEBUG("IGNORE mode: Skipping clear all keys");
      return true;
    }

    LOG_INFO("Clearing all keys for tag: {}", keyTag);
    cc->ClearEvalAutomorphismKeys(keyTag);
    auto numCleared = loadedKeys.size();
    loadedKeys.clear();
    LOG_INFO("Successfully cleared {} keys", numCleared);
    if (operationMode == KeyMemMode::PREFETCH) {
      {
        std::lock_guard<std::mutex> statusLock(keyStatusMutex);
        keysReadyForUse.clear();
        currentTowerCount = 0; // Reset tower count since all keys are cleared
        LOG_DEBUG("PREFETCH: Reset tower count to 0 after clearing all keys");
      }

      // Wait queue disabled - just notify worker since all keys are cleared
      {
        std::lock_guard<std::mutex> queueLock(queueMutex);
        queueCondition.notify_all();
        LOG_DEBUG("PREFETCH: Notified all workers after clearing all keys");
      }
    }
    return true;
  }

  std::string getKeyFilename(int rotationIndex) const {
    return BenchmarkCLI::getInputDir() + "/rotation_key_" +
           std::to_string(rotationIndex) + ".bin";
  }

  bool checkKeyExists(int rotationIndex) const {
    std::string filename = getKeyFilename(rotationIndex);
    std::ifstream keyFile(filename);
    bool exists = keyFile.good();
    if (exists) {
      LOG_DEBUG("Key file for rotation index {} exists: {}", rotationIndex,
                filename);
    } else {
      LOG_DEBUG("Key file for rotation index {} does not exist: {}",
                rotationIndex, filename);
    }
    return exists;
  }

  bool generateConjugateKey(const PrivateKey<DCRTPoly> &privateKey) {
    try {
      // Get the cyclotomic order
      uint32_t M = cc->GetCyclotomicOrder();

      // The conjugate automorphism index is always M-1
      uint32_t conjAutomorphismIndex = M - 1;

      LOG_INFO("Generating conjugate key (automorphism index: {})",
               conjAutomorphismIndex);

      // Generate the conjugate key using the CKKS scheme's ConjugateKeyGen
      auto conjKey = KeyMemRTConjugateKeyGen(privateKey);

      auto &evalAutomorphismKeyMap =
          lbcrypto::CryptoContextImpl<DCRTPoly>::GetEvalAutomorphismKeyMap(
              privateKey->GetKeyTag());

      // Insert the conjugate key directly into the existing map
      evalAutomorphismKeyMap[conjAutomorphismIndex] = conjKey;

      LOG_INFO("Successfully generated and inserted conjugate key");
      return true;

    } catch (const std::exception &e) {
      LOG_ERROR("Failed to generate conjugate key: {}", e.what());
      return false;
    }
  }

  void debugPrintQueues(const std::string &context = "") {
    // if (!LOG_DEBUG_ENABLED) return; // Only if debug logging is on

    std::lock_guard<std::mutex> queueLock(queueMutex);
    std::lock_guard<std::mutex> statusLock(keyStatusMutex);

    // Build main queue string
    std::string mainQueueStr = "[";
    std::queue<std::pair<int, int>> tempMain = deserializationQueue;
    bool first = true;
    while (!tempMain.empty()) {
      if (!first)
        mainQueueStr += ",";
      mainQueueStr += std::to_string(tempMain.front().first) + ":" +
                      std::to_string(tempMain.front().second);
      tempMain.pop();
      first = false;
    }
    mainQueueStr += "]";

    // Build wait queue string
    std::string waitQueueStr = "[";
    std::queue<WaitQueueItem> tempWait = waitQueue;
    first = true;
    while (!tempWait.empty()) {
      if (!first)
        waitQueueStr += ",";
      waitQueueStr += std::to_string(tempWait.front().rotationIndex) + ":" +
                      std::to_string(tempWait.front().depth) +
                      (tempWait.front().ready ? "R" : "W");
      tempWait.pop();
      first = false;
    }
    waitQueueStr += "]";

    // Build ready keys string
    std::string readyStr = "[";
    first = true;
    for (const auto &key : keysReadyForUse) {
      if (!first)
        readyStr += ",";
      readyStr += std::to_string(key.first) + ":" + std::to_string(key.second);
      first = false;
    }
    readyStr += "]";

    // Build processing keys string
    std::string processingStr = "[";
    first = true;
    for (const auto &key : keysBeingDeserialized) {
      if (!first)
        processingStr += ",";
      processingStr +=
          std::to_string(key.first) + ":" + std::to_string(key.second);
      first = false;
    }
    processingStr += "]";

    LOG_DEBUG("QUEUE_STATE {}: Main{} Wait{} Ready{} Processing{}", context,
              mainQueueStr, waitQueueStr, readyStr, processingStr);
  }

  bool enqueueKey(int rotationIndex, int keyDepth = 0) {
    if (operationMode != KeyMemMode::PREFETCH) {
      LOG_DEBUG("enqueue called in non-PREFETCH mode, ignoring");
      return false;
    }
    if (!rotationIndex) {
      LOG_DEBUG("enqueue called with 0 index, ignoring");
      return false;
    }
    if (std::find(rotIndices.begin(), rotIndices.end(), rotationIndex) ==
        rotIndices.end()) {
      LOG_DEBUG("Enqueue called with unsupported index {}, ignoring",
                rotationIndex);
      return false;
    }

    auto keyInfo = std::make_pair(rotationIndex, keyDepth);
    debugPrintQueues();

    // Directly enqueue to main queue - wait queue disabled
    {
      std::lock_guard<std::mutex> queueLock(queueMutex);
      deserializationQueue.push(keyInfo);
      LOG_DEBUG("PREFETCH: Enqueued key {} at depth {} to main queue",
                rotationIndex, keyDepth);
      queueCondition.notify_one();
    }

    return true;
  }

  void deserializationWorker() {
    while (!shouldStop.load()) {
      std::unique_lock<std::mutex> lock(queueMutex);

      // Wait for items in queue or stop signal (with timeout to prevent
      // indefinite blocking)
      queueCondition.wait_for(lock, std::chrono::milliseconds(100), [this] {
        // Wait queue disabled - only check main deserialization queue
        bool hasWork = false;
        if (!deserializationQueue.empty()) {
          std::lock_guard<std::mutex> statusLock(keyStatusMutex);

          // Check if next item is already in ready set
          auto nextItem = deserializationQueue.front();
          if (keysReadyForUse.count(nextItem) > 0) {
            LOG_DEBUG(
                "PREFETCH: Next key {} at depth {} already loaded, worker "
                "blocking until cleared",
                nextItem.first, nextItem.second);
            return false; // Keep waiting until clearKey() notifies us
          }

          // Calculate tower count for next item
          int requiredTowers = calculateTowerCount(nextItem.second);

          if (currentTowerCount + requiredTowers <= prefetchTowerLimit) {
            hasWork = true;
          } else {
            LOG_DEBUG("PREFETCH: Capacity limit reached ({}/{} towers), "
                      "waiting for keys to be cleared",
                      currentTowerCount, prefetchTowerLimit);
          }
        }

        return hasWork || shouldStop.load();
      });

      if (shouldStop.load()) {
        break;
      }

      std::pair<int, int> keyInfo;
      bool foundItem = false;

      // Wait queue disabled - only process main deserialization queue
      if (!deserializationQueue.empty()) {
        std::lock_guard<std::mutex> statusLock(keyStatusMutex);
        auto nextItem = deserializationQueue.front();

        // Check if key is already loaded - if so, don't process
        if (keysReadyForUse.count(nextItem) > 0) {
          LOG_DEBUG("PREFETCH: Key {} at depth {} already loaded, skipping "
                    "processing and continuing to wait",
                    nextItem.first, nextItem.second);
          // Don't pop, don't process, just continue waiting
          continue;
        }

        int requiredTowers = calculateTowerCount(nextItem.second);
        if (currentTowerCount + requiredTowers <= prefetchTowerLimit) {
          keyInfo = deserializationQueue.front();
          deserializationQueue.pop();
          foundItem = true;
          LOG_DEBUG("PREFETCH: Processing item from main queue: key {} at "
                    "depth {} (towers: {}, total: {}/{})",
                    keyInfo.first, keyInfo.second, requiredTowers,
                    currentTowerCount + requiredTowers, prefetchTowerLimit);
        }
      }

      // If no item found, continue waiting
      if (!foundItem) {
        continue;
      }

      lock.unlock();

      int rotationIndex = keyInfo.first;
      int keyDepth = keyInfo.second;
      int keyTowers = calculateTowerCount(keyDepth);

      // Mark as being deserialized
      {
        std::lock_guard<std::mutex> statusLock(keyStatusMutex);
        keysBeingDeserialized.insert(keyInfo);
        // Only increment tower count if key doesn't already exist (avoid
        // double-counting)
        if (keysReadyForUse.count(keyInfo) == 0) {
          currentTowerCount += keyTowers;
        }
      }

      LOG_DEBUG("PREFETCH: Starting background deserialization for key {} at "
                "depth {} (towers: {}, total: {}/{})",
                rotationIndex, keyDepth, keyTowers, currentTowerCount,
                prefetchTowerLimit);

      // Perform actual deserialization
      bool success = _deserializeKey(rotationIndex, keyDepth);

      // Update status
      {
        std::lock_guard<std::mutex> statusLock(keyStatusMutex);
        keysBeingDeserialized.erase(keyInfo);
        if (success) {
          keysReadyForUse.insert(keyInfo);
          LOG_DEBUG("PREFETCH: Key {} at depth {} ready for use (towers: {}, "
                    "total: {}/{})",
                    rotationIndex, keyDepth, keyTowers, currentTowerCount,
                    prefetchTowerLimit);

        } else {
          currentTowerCount -= keyTowers;
          LOG_ERROR("PREFETCH: Failed to deserialize key {} at depth {}, "
                    "reducing tower count to {}",
                    rotationIndex, keyDepth, currentTowerCount);
        }
      }

      {
        std::lock_guard<std::mutex> readyLock(keyReadyMutex);
        keyReadyCondition.notify_all();
      }
      queueCondition.notify_one();
    }
  }
  // Print stats about key operations
  void printKeyStats() const {
    LOG_INFO("KeyMemRT Stats:");
    LOG_INFO("  Mode: {}", getModeString(operationMode));
    LOG_INFO("  Total rotation indices: {}", rotIndices.size());
    LOG_INFO("  Keys loaded: {}", loadedKeys.size());

    std::cout << "KeyMemRT Stats:\n"
              << "  Mode: " << getModeString(operationMode) << "\n"
              << "  Total rotation indices: " << rotIndices.size() << "\n"
              << "  Keys loaded: " << loadedKeys.size() << "\n";
  }

  // Enable/disable logging to file
  void enableFileLogging(const std::string &filename = "keymemrt.log") {
    Logger::getInstance().setLogToFile(true, filename);
    LOG_INFO("File logging enabled: {}", filename);
  }

  // Set log level
  void setLogLevel(LogLevel level) {
    Logger::getInstance().setLogLevel(level);
    LOG_INFO("Log level set to: {}",
             Logger::getInstance().levelToString(level));
  }

  usint getAutomorphismIndex(usint rotationIndex) const {
    return cc->FindAutomorphismIndex(rotationIndex);
  }

  int getActualTowerCount(int level) {
    // Use the formula: multDepth - level + 2
    // where level = depth for compressed keys
    return m_multDepth - level + 2;
  }

private:
  CryptoContext<DCRTPoly> cc;
  int m_multDepth;
  std::string keyTag;
  std::vector<int32_t> rotIndices;
  KeyMemMode operationMode;
  std::set<int32_t> loadedKeys;
  Platform platform;
  std::condition_variable keyReadyCondition; // Signal when keys become ready
  mutable std::mutex keyReadyMutex;

  struct WaitQueueItem {
    int rotationIndex;
    int depth;
    bool ready;

    WaitQueueItem(int idx, int d)
        : rotationIndex(idx), depth(d), ready(false) {}
  };

  int prefetchTowerLimit = 250; // Default tower limit
  int currentTowerCount = 0;    // Current towers in memory

  std::queue<std::pair<int, int>>
      deserializationQueue;            // pair of (rotationIndex, depth)
  std::queue<WaitQueueItem> waitQueue; // wait queue with ready flags
  std::mutex queueMutex;
  std::condition_variable queueCondition;
  std::thread deserializationThread;
  std::atomic<bool> shouldStop{false};
  std::mutex keyStatusMutex;
  std::set<std::pair<int, int>>
      keysBeingDeserialized; // pair of (rotationIndex, depth)
  std::set<std::pair<int, int>>
      keysReadyForUse; // pair of (rotationIndex, depth)
};
#endif // KEY_MANAGER_H_
