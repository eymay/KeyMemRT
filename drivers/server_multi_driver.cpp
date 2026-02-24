#include "server_runner.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <map>
#include <mutex>
#include <omp.h>
#include <string>
#include <thread>
#include <vector>

// Global KeyMemRT instance (required by the server_runner)
// KeyMemRT keymem_rt;

// Global progress tracking for Run 0
std::atomic<int> run0_progress{0};
std::condition_variable progress_cv;
std::mutex progress_mutex;

// Run schedule structure
struct RunSchedule {
  int group_id;
  int start_op_run0; // When Run 0 reaches this operation, trigger this run
};

// Example schedule - modify this as needed
const std::map<int, RunSchedule> RUN_SCHEDULE = {
    // Group 0: Initial batch - starts when Run 0 starts (operation 0)
    {0, {0, 0}},  // Run 0: reference run
    {1, {0, 0}},  // Run 1: starts when Run 0 starts
    {2, {0, 0}},  // Run 2: starts when Run 0 starts
    {3, {0, 0}},  // Run 3: starts when Run 0 starts
    {4, {0, 0}},  // Run 4: starts when Run 0 starts
    {5, {0, 0}},  // Run 5: starts when Run 0 starts
    {6, {0, 0}},  // Run 6: starts when Run 0 starts
    {7, {0, 0}},  // Run 7: starts when Run 0 starts
    {8, {0, 0}},  // Run 8: starts when Run 0 starts
    {9, {0, 0}},  // Run 9: starts when Run 0 starts
    {10, {0, 0}}, // Run 10: starts when Run 0 starts
    {11, {0, 0}}, // Run 11: starts when Run 0 starts
    {12, {0, 0}}, // Run 12: starts when Run 0 starts

    // Group 1: Mid-execution batch - starts when Run 0 hits operation 2500
    // {13, {1, 2500}}, // Run 13: starts when Run 0 hits op 2500
    // {14, {1, 2500}}, // Run 14: starts when Run 0 hits op 2500
    // {15, {1, 2500}}, // Run 15: starts when Run 0 hits op 2500
    // {16, {1, 2500}}, // Run 16: starts when Run 0 hits op 2500
    // {17, {1, 2500}}, // Run 17: starts when Run 0 hits op 2500
    // {18, {1, 2500}}, // Run 18: starts when Run 0 hits op 2500
    // {19, {1, 2500}}, // Run 19: starts when Run 0 hits op 2500
    // {20, {1, 2500}}, // Run 20: starts when Run 0 hits op 2500
    // {21, {1, 2500}}, // Run 21: starts when Run 0 hits op 2500
    // {22, {1, 2500}}, // Run 22: starts when Run 0 hits op 2500
    // {23, {1, 2500}}, // Run 23: starts when Run 0 hits op 2500
    // {24, {1, 2500}}, // Run 24: starts when Run 0 hits op 2500
};

// Configuration structure for command line parsing
struct MultiConfig {
  std::string key_mode = "ignore";
  int depth = DEFAULT_DEPTH;
  int ring_dim = DEFAULT_RING_DIM;
  int prefetch_sat = 5;
  std::string result_dir = "./results";
  bool verbose = false;
  int max_threads = 72;

  void parse_args(int argc, char *argv[]) {
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];

      if (arg == "--help") {
        print_help();
        exit(0);
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
      } else if (arg == "--max-threads" && i + 1 < argc) {
        max_threads = std::stoi(argv[++i]);
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
        << "Usage: " << NETWORK_NAME << "_multi_server [options]\n"
        << "Options:\n"
        << "  --key-mode <mode>      Key memory mode (default: ignore)\n"
        << "                         Options: ignore, imperative, prefetching\n"
        << "  --verbose              Enable verbose output\n"
        << "  --depth <n>            Multiplicative depth (default: "
        << DEFAULT_DEPTH << ")\n"
        << "  --ring-dim <n>         Ring dimension (default: "
        << DEFAULT_RING_DIM << ")\n"
        << "  --prefetch-sat <n>     Prefetch saturation (default: 5)\n"
        << "  --result-dir <path>    Results directory (default: ./results)\n"
        << "  --max-threads <n>      Maximum OpenMP threads (default: 72)\n"
        << "  --help                 Show this help message\n";
  }
};

// Progress callback for Run 0
void updateRun0Progress(int operation_id) {
  int old_progress = run0_progress.exchange(operation_id);
  if (operation_id > old_progress) {
    std::cout << "🔄 Run 0 reached operation: " << operation_id << std::endl;
    progress_cv.notify_all();
  }
}

// Wait for Run 0 to reach a specific operation
void waitForRun0Operation(int target_op) {
  std::unique_lock<std::mutex> lock(progress_mutex);
  progress_cv.wait(lock,
                   [target_op]() { return run0_progress.load() >= target_op; });
  std::cout << "✅ Run 0 reached trigger operation " << target_op
            << ", launching next group..." << std::endl;
}

// Get all runs for a specific group
std::vector<int> getRunsForGroup(int group_id) {
  std::vector<int> runs;
  for (const auto &[run_id, schedule] : RUN_SCHEDULE) {
    if (schedule.group_id == group_id) {
      runs.push_back(run_id);
    }
  }
  return runs;
}

// Get unique trigger operations (sorted)
std::vector<int> getTriggerOperations() {
  std::set<int> unique_triggers;
  for (const auto &[run_id, schedule] : RUN_SCHEDULE) {
    unique_triggers.insert(schedule.start_op_run0);
  }
  return std::vector<int>(unique_triggers.begin(), unique_triggers.end());
}

// Execute a group of runs in parallel
void executeRunGroup(const std::vector<int> &runs,
                     const InferenceConfig &config) {
  std::cout << "🚀 Starting group with " << runs.size() << " runs: [";
  for (size_t i = 0; i < runs.size(); ++i) {
    std::cout << runs[i];
    if (i < runs.size() - 1)
      std::cout << ", ";
  }
  std::cout << "]" << std::endl;

#pragma omp parallel for
  for (size_t i = 0; i < runs.size(); ++i) {
    int run_id = runs[i];

    // Only Run 0 gets progress tracking
    ProgressCallback callback = (run_id == 0) ? updateRun0Progress : nullptr;

    std::cout << "🏃 Thread " << omp_get_thread_num() << " starting Run "
              << run_id << std::endl;

    int result = run_single_inference(run_id, config, callback);

    if (result == 0) {
      std::cout << "✅ Run " << run_id << " completed successfully on thread "
                << omp_get_thread_num() << std::endl;
    } else {
      std::cerr << "❌ Run " << run_id << " failed with error " << result
                << " on thread " << omp_get_thread_num() << std::endl;
    }
  }
}

int main(int argc, char *argv[]) {
  std::cout << "=== Multi-Run " << NETWORK_NAME
            << " FHE Scheduler ===" << std::endl;

  // Parse command line arguments
  MultiConfig config;
  config.parse_args(argc, argv);

  // Set OpenMP thread limit
  omp_set_num_threads(config.max_threads);
  std::cout << "🧵 OpenMP configured for max " << config.max_threads
            << " threads" << std::endl;

  // Convert to InferenceConfig
  InferenceConfig inference_config;
  inference_config.key_mode = config.key_mode;
  inference_config.depth = config.depth;
  inference_config.ring_dim = config.ring_dim;
  inference_config.prefetch_sat = config.prefetch_sat;
  inference_config.result_dir = config.result_dir;
  inference_config.verbose = config.verbose;

  // Create results directory if it doesn't exist
  // std::filesystem::create_directories(config.result_dir);

  // Group runs by their trigger operations
  std::map<int, std::vector<int>> operation_groups;
  for (const auto &[run_id, schedule] : RUN_SCHEDULE) {
    operation_groups[schedule.start_op_run0].push_back(run_id);
  }

  std::cout << "📋 Schedule Summary:" << std::endl;
  for (const auto &[trigger_op, runs] : operation_groups) {
    std::cout << "  Operation " << trigger_op << ": " << runs.size()
              << " runs [";
    for (size_t i = 0; i < runs.size(); ++i) {
      std::cout << runs[i];
      if (i < runs.size() - 1)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;
  }

  auto start_time = std::chrono::high_resolution_clock::now();

// Execute the multi-run schedule
#pragma omp parallel sections
  {
#pragma omp section
    {
      // Execute Group 0 (starts immediately)
      auto group0_runs = operation_groups[0];
      if (!group0_runs.empty()) {
        executeRunGroup(group0_runs, inference_config);
      }
    }

#pragma omp section
    {
      // Monitor Run 0's progress and launch subsequent groups
      auto triggers = getTriggerOperations();

      for (int trigger_op : triggers) {
        if (trigger_op == 0)
          continue; // Skip Group 0 (already started)

        std::cout << "⏳ Waiting for Run 0 to reach operation " << trigger_op
                  << "..." << std::endl;

        // Wait for Run 0 to reach this operation
        waitForRun0Operation(trigger_op);

        // Launch all runs triggered by this operation
        auto triggered_runs = operation_groups[trigger_op];
        if (!triggered_runs.empty()) {
          executeRunGroup(triggered_runs, inference_config);
        }
      }
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto total_duration =
      std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

  std::cout << "\n🎉 All runs completed!" << std::endl;
  std::cout << "⏱️  Total execution time: " << total_duration.count()
            << " seconds" << std::endl;

  // Summary statistics
  std::cout << "\n📊 Execution Summary:" << std::endl;
  std::cout << "  Total runs: " << RUN_SCHEDULE.size() << std::endl;
  std::cout << "  Groups: " << operation_groups.size() << std::endl;
  std::cout << "  Max threads used: " << config.max_threads << std::endl;
  std::cout << "  Network: " << NETWORK_NAME << std::endl;
  std::cout << "  Key mode: " << config.key_mode << std::endl;

  return 0;
}
