#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <regex>
#include <signal.h>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

class OOMDetector {
private:
  std::atomic<bool> monitoring{true};
  std::atomic<bool> emergency_shutdown{false};
  std::thread monitor_thread;

  // Configurable thresholds - less noisy, more focused on actual OOM
  const double MEMORY_THRESHOLD_GB = 500.0;   // Primary shutdown threshold
  const double CRITICAL_THRESHOLD_GB = 490.0; // Start rapid monitoring
  const int CHECK_INTERVAL_MS = 2000; // Check every 2 seconds (less frequent)
  const int EMERGENCY_INTERVAL_MS = 250; // Emergency checks every 250ms

  std::function<void()> shutdown_callback;
  std::function<void(const std::string &, double)> event_callback;

public:
  OOMDetector(
      std::function<void()> callback,
      std::function<void(const std::string &, double)> event_cb = nullptr)
      : shutdown_callback(callback), event_callback(event_cb) {}

  ~OOMDetector() { stop(); }

  void start() {
    monitoring = true;
    monitor_thread = std::thread(&OOMDetector::monitorLoop, this);
    std::cout << "OOM Detector started - monitoring for emergency shutdown at "
              << MEMORY_THRESHOLD_GB << "GB" << std::endl;

    // Log start event
    if (event_callback) {
      event_callback("OOM_MONITORING_STARTED", 0.0);
    }
  }

  void stop() {
    monitoring = false;
    if (monitor_thread.joinable()) {
      monitor_thread.join();
    }

    // Log stop event
    if (event_callback) {
      event_callback("OOM_MONITORING_STOPPED", 0.0);
    }
  }

  bool isEmergencyShutdown() const { return emergency_shutdown.load(); }

private:
  struct MemoryInfo {
    double total_ram_gb = 0.0;
    double used_ram_gb = 0.0;
    double available_ram_gb = 0.0;
    double process_rss_gb = 0.0;
    double swap_used_gb = 0.0;
    bool swap_detected = false;
  };

  MemoryInfo getMemoryInfo() {
    MemoryInfo info;

    // Read /proc/meminfo
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    double mem_total_kb = 0, mem_available_kb = 0, swap_total_kb = 0,
           swap_free_kb = 0;

    while (std::getline(meminfo, line)) {
      std::istringstream ss(line);
      std::string key;
      unsigned long value;
      ss >> key >> value;

      if (key == "MemTotal:")
        mem_total_kb = value;
      else if (key == "MemAvailable:")
        mem_available_kb = value;
      else if (key == "SwapTotal:")
        swap_total_kb = value;
      else if (key == "SwapFree:")
        swap_free_kb = value;
    }

    info.total_ram_gb = mem_total_kb / 1024.0 / 1024.0;
    info.available_ram_gb = mem_available_kb / 1024.0 / 1024.0;
    info.used_ram_gb = info.total_ram_gb - info.available_ram_gb;
    info.swap_used_gb = (swap_total_kb - swap_free_kb) / 1024.0 / 1024.0;
    info.swap_detected = info.swap_used_gb > 0.1; // More than 100MB swap

    // Get process RSS
    std::ifstream status("/proc/self/status");
    while (std::getline(status, line)) {
      if (line.substr(0, 6) == "VmRSS:") {
        std::istringstream iss(line);
        std::string label, size, unit;
        iss >> label >> size >> unit;
        info.process_rss_gb = std::stoll(size) / 1024.0 / 1024.0; // KB to GB
        break;
      }
    }

    return info;
  }

  void logMemoryStatus(const MemoryInfo &info, const std::string &level) {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);

    std::cout << "[" << std::put_time(std::localtime(&time_t), "%H:%M:%S")
              << "] " << level << " - Memory Status:" << std::endl;
    std::cout << "  System RAM: " << info.used_ram_gb << "/"
              << info.total_ram_gb << " GB used ("
              << (info.used_ram_gb / info.total_ram_gb * 100) << "%)"
              << std::endl;
    std::cout << "  Available: " << info.available_ram_gb << " GB" << std::endl;
    std::cout << "  Process RSS: " << info.process_rss_gb << " GB" << std::endl;
    std::cout << "  Swap used: " << info.swap_used_gb << " GB"
              << (info.swap_detected ? " [SWAPPING DETECTED!]" : "")
              << std::endl;
  }

  void handleEmergencyShutdown(const MemoryInfo &info) {
    emergency_shutdown = true;

    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);

    std::cout << "\n🚨 [" << std::put_time(std::localtime(&time_t), "%H:%M:%S")
              << "] MEMORY THRESHOLD EXCEEDED - EMERGENCY SHUTDOWN 🚨"
              << std::endl;
    std::cout << "Memory usage: " << info.used_ram_gb
              << "GB >= " << MEMORY_THRESHOLD_GB << "GB threshold" << std::endl;

    // Log detailed emergency event
    if (event_callback) {
      event_callback("OOM_EMERGENCY_SHUTDOWN", info.used_ram_gb);
    }

    // Call the shutdown callback
    if (shutdown_callback) {
      shutdown_callback();
    }
  }

  void monitorLoop() {
    bool critical_mode = false;
    int check_interval = CHECK_INTERVAL_MS;

    while (monitoring) {
      MemoryInfo info = getMemoryInfo();

      // Check for emergency shutdown condition
      if (info.used_ram_gb >= MEMORY_THRESHOLD_GB) {
        handleEmergencyShutdown(info);
        break;
      }

      // Only switch to critical mode when very close to threshold
      if (info.used_ram_gb >= CRITICAL_THRESHOLD_GB && !critical_mode) {
        critical_mode = true;
        check_interval = EMERGENCY_INTERVAL_MS;

        std::cout << "Entering critical memory monitoring mode ("
                  << info.used_ram_gb << "GB used, checking every "
                  << EMERGENCY_INTERVAL_MS << "ms)" << std::endl;

        if (event_callback) {
          event_callback("OOM_CRITICAL_MODE", info.used_ram_gb);
        }
      }
      // Exit critical mode if memory drops significantly
      else if (info.used_ram_gb < CRITICAL_THRESHOLD_GB - 5.0 &&
               critical_mode) {
        critical_mode = false;
        check_interval = CHECK_INTERVAL_MS;

        std::cout << "Exiting critical memory monitoring mode ("
                  << info.used_ram_gb << "GB used)" << std::endl;

        if (event_callback) {
          event_callback("OOM_NORMAL_MODE", info.used_ram_gb);
        }
      }

      // Only alert on swap usage (this is always bad for performance)
      if (info.swap_detected) {
        static bool swap_warning_issued = false;
        if (!swap_warning_issued) {
          std::cout << "⚠️  SWAP DETECTED: " << info.swap_used_gb
                    << "GB - performance degradation expected" << std::endl;

          if (event_callback) {
            event_callback("OOM_SWAP_DETECTED", info.swap_used_gb);
          }
          swap_warning_issued = true;
        }
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(check_interval));
    }

    std::cout << "OOM monitor thread exiting" << std::endl;
  }
};

class ParallelLauncher {
private:
  std::unique_ptr<OOMDetector> oom_detector;

  void setupOOMDetection() {
    // Create shutdown callback that will terminate all processes
    auto shutdown_callback = [this]() {
      std::cout << "OOM Detector triggered shutdown - terminating all processes"
                << std::endl;
      this->shutdown_requested = true;
      this->terminateAllRuns();
    };

    oom_detector = std::make_unique<OOMDetector>(shutdown_callback);
    oom_detector->start();
  }

  struct RunInfo {
    pid_t pid = 0;
    int run_id = 0;
    std::string log_filename;
    std::atomic<bool> active{false};
    std::atomic<int> current_operation{0};
    std::atomic<int> iteration_count{0};
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
    int assigned_threads = 0; // Threads assigned to this process

    // Default constructor
    RunInfo() = default;

    RunInfo(int id) : run_id(id) {
      log_filename = "run_" + std::to_string(run_id) + "_output.log";
    }

    // Delete copy constructor and assignment operator (atomics can't be copied)
    RunInfo(const RunInfo &) = delete;
    RunInfo &operator=(const RunInfo &) = delete;

    // Add move constructor and assignment operator
    RunInfo(RunInfo &&other) noexcept
        : pid(other.pid), run_id(other.run_id),
          log_filename(std::move(other.log_filename)),
          active(other.active.load()),
          current_operation(other.current_operation.load()),
          iteration_count(other.iteration_count.load()),
          start_time(other.start_time), end_time(other.end_time),
          assigned_threads(other.assigned_threads) {}

    RunInfo &operator=(RunInfo &&other) noexcept {
      if (this != &other) {
        pid = other.pid;
        run_id = other.run_id;
        log_filename = std::move(other.log_filename);
        active.store(other.active.load());
        current_operation.store(other.current_operation.load());
        iteration_count.store(other.iteration_count.load());
        start_time = other.start_time;
        end_time = other.end_time;
        assigned_threads = other.assigned_threads;
      }
      return *this;
    }
  };

  struct LaunchEvent {
    int run_id;
    int iteration;
    int trigger_operation; // Operation that triggered this launch (-1 for
                           // initial)
    std::chrono::steady_clock::time_point timestamp;
    std::string reason;
  };

  struct CompletionEvent {
    int run_id;
    int iteration;
    int exit_code;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
    double duration_seconds;
  };

  int process_count;
  std::map<int, RunInfo> runs;
  std::atomic<bool> shutdown_requested{false};
  std::atomic<int> max_iterations{3}; // Default iteration count
  std::string executable_path;
  std::vector<std::string> exe_args;
  std::mutex launch_mutex;
  std::mutex triggered_operations_mutex;

  // Thread management
  int max_system_threads = 72;            // Default, will be detected
  std::map<int, int> thread_distribution; // run_id -> thread_count

  // Directory management
  std::string base_results_dir;

  // Global monitoring
  std::chrono::steady_clock::time_point global_start_time;
  std::vector<LaunchEvent> launch_events;
  std::vector<CompletionEvent> completion_events;
  std::mutex events_mutex;
  std::ofstream global_log;
  std::ofstream csv_log;

public:
  ParallelLauncher(const std::string &exe_path, int num_processes)
      : executable_path(exe_path), process_count(num_processes) {
    setupSignalHandlers();
    detectSystemThreads();
    createResultsDirectory();
    initializeLogging();
  }

  ~ParallelLauncher() {
    if (global_log.is_open()) {
      global_log.close();
    }
    if (csv_log.is_open()) {
      csv_log.close();
    }
    if (oom_detector) {
      oom_detector->stop();
    }
  }

  void setMaxIterations(int iterations) {
    max_iterations = iterations;
    std::cout << "Set max iterations to " << iterations << std::endl;
  }

  void setMaxThreads(int threads) {
    max_system_threads = threads;
    std::cout << "Set max system threads to " << threads << std::endl;
    calculateThreadDistribution();
  }

  void setExecutableArgs(const std::vector<std::string> &args) {
    exe_args = args;
  }

  void run() {
    global_start_time = std::chrono::steady_clock::now();

    std::cout << "Starting parallel launcher..." << std::endl;
    logGlobal("=== Parallel Launcher Started ===");
    logGlobal("Parallel processes: " + std::to_string(process_count));
    logGlobal("Max iterations per process: " +
              std::to_string(max_iterations.load()));
    logGlobal("Executable: " + executable_path);

    setupOOMDetection();
    // Initialize run tracking structures
    for (int i = 0; i < process_count; ++i) {
      runs[i] = RunInfo(i);
    }

    // Calculate thread distribution
    calculateThreadDistribution();

    // Launch all processes immediately
    for (int i = 0; i < process_count; ++i) {
      launchRun(i, -1, "Initial parallel launch");
      // Small delay to prevent resource contention during startup
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Start monitoring threads
    std::thread monitor_thread(&ParallelLauncher::monitorRuns, this);
    std::thread status_thread(&ParallelLauncher::periodicStatusReport, this);

    // Wait for user interrupt or completion
    int loop_count = 0;
    while (!shutdown_requested && hasActiveRuns()) {
      if (oom_detector->isEmergencyShutdown()) {
        std::cout << "Emergency shutdown detected in main loop" << std::endl;
        break;
      }
      std::this_thread::sleep_for(std::chrono::seconds(1));
      loop_count++;

      // Debug output every 10 seconds
      if (loop_count % 10 == 0) {
        int active = 0, completed = 0;
        for (const auto &[run_id, run_info] : runs) {
          if (run_info.active)
            active++;
          else if (run_info.iteration_count.load() >= max_iterations)
            completed++;
        }
        std::cout << "Main loop: active=" << active
                  << ", completed=" << completed << "/" << runs.size()
                  << ", iterations=" << max_iterations.load() << std::endl;
      }
    }

    std::cout << "Main loop exited: shutdown_requested=" << shutdown_requested
              << ", hasActiveRuns=" << hasActiveRuns() << std::endl;

    // Cleanup
    terminateAllRuns();

    if (monitor_thread.joinable())
      monitor_thread.join();
    if (status_thread.joinable())
      status_thread.join();

    generateExecutionSummary();
    std::cout << "Parallel execution completed!" << std::endl;
  }

private:
  void setupSignalHandlers() {
    signal(SIGINT, [](int) {
      std::cout << "\nReceived SIGINT, shutting down gracefully..."
                << std::endl;
      exit(0);
    });

    signal(SIGTERM, [](int) {
      std::cout << "\nReceived SIGTERM, shutting down gracefully..."
                << std::endl;
      exit(0);
    });
  }

  void detectSystemThreads() {
    // Try to read from /proc/cpuinfo first
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (cpuinfo.is_open()) {
      int core_count = 0;
      std::string line;
      while (std::getline(cpuinfo, line)) {
        if (line.find("processor") == 0) {
          core_count++;
        }
      }
      if (core_count > 0) {
        max_system_threads = core_count;
      }
    }

    // Fallback to std::thread
    if (max_system_threads <= 0) {
      max_system_threads = std::thread::hardware_concurrency();
    }

    // Final fallback
    if (max_system_threads <= 0) {
      max_system_threads = 8;
    }

    std::cout << "Detected " << max_system_threads << " system threads"
              << std::endl;
  }

  void calculateThreadDistribution() {
    int total_processes = process_count;
    int base_threads = max_system_threads / total_processes;
    if (base_threads < 1)
      base_threads = 1;

    if (total_processes * base_threads > max_system_threads) {
      std::cout << "Warning: Requested " << total_processes
                << " processes with " << base_threads
                << " threads each = " << (total_processes * base_threads)
                << " total threads. "
                << "System has " << max_system_threads << " threads. "
                << "Will oversubscribe." << std::endl;
    }

    // Distribute threads with smart algorithm
    int target_total = max_system_threads;
    int current_total = base_threads * total_processes;

    if (current_total < target_total) {
      // We're under target, increase threads for some processes
      int processes_to_upgrade = target_total - current_total;

      for (int i = 0; i < total_processes; ++i) {
        if (i < processes_to_upgrade) {
          thread_distribution[i] = base_threads + 1;
        } else {
          thread_distribution[i] = base_threads;
        }
      }
    } else if (current_total > target_total) {
      // We're over target, reduce threads for some processes
      int processes_to_downgrade = current_total - target_total;

      for (int i = 0; i < total_processes; ++i) {
        if (i < processes_to_downgrade) {
          thread_distribution[i] = std::max(1, base_threads - 1);
        } else {
          thread_distribution[i] = base_threads;
        }
      }
    } else {
      // Perfect fit
      for (int i = 0; i < total_processes; ++i) {
        thread_distribution[i] = base_threads;
      }
    }

    // Log thread distribution
    std::cout << "Thread distribution:" << std::endl;
    int actual_total = 0;
    for (int i = 0; i < total_processes; ++i) {
      std::cout << "  Run " << i << ": " << thread_distribution[i] << " threads"
                << std::endl;
      actual_total += thread_distribution[i];
    }
    std::cout << "Total threads used: " << actual_total << "/"
              << max_system_threads << std::endl;
  }

  void createResultsDirectory() {
    // Create timestamped results directory
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    std::string timestamp = ss.str();

    base_results_dir = "parallel_results_" + timestamp;

    // Create base directory
    if (mkdir(base_results_dir.c_str(), 0755) != 0) {
      throw std::runtime_error("Failed to create results directory: " +
                               base_results_dir);
    }

    // Create subdirectories for each iteration
    for (int iter = 1; iter <= max_iterations; ++iter) {
      std::string iter_dir =
          base_results_dir + "/iteration" + std::to_string(iter) + "/";
      if (mkdir(iter_dir.c_str(), 0755) != 0) {
        throw std::runtime_error("Failed to create iteration directory: " +
                                 iter_dir);
      }
    }

    std::cout << "Results directory: " << base_results_dir << std::endl;
  }

  void initializeLogging() {
    // Create global log file
    std::string log_file = base_results_dir + "/parallel_launcher.log";
    global_log.open(log_file);
    if (!global_log) {
      throw std::runtime_error("Failed to create log file: " + log_file);
    }

    // Create CSV log file
    std::string csv_file = base_results_dir + "/launcher_events.csv";
    csv_log.open(csv_file);
    if (!csv_log) {
      throw std::runtime_error("Failed to create CSV file: " + csv_file);
    }

    // Write CSV header
    csv_log << "timestamp_sec,event_type,run_id,iteration,trigger_operation,"
               "exit_code,duration_sec,reason"
            << std::endl;

    std::cout << "Logging to: " << log_file << std::endl;
    std::cout << "CSV events: " << csv_file << std::endl;
  }

  void logGlobal(const std::string &message) {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);

    std::lock_guard<std::mutex> lock(events_mutex);
    global_log << std::put_time(std::localtime(&time_t), "[%Y-%m-%d %H:%M:%S] ")
               << message << std::endl;
    global_log.flush();
  }

  void launchRun(int run_id, int trigger_operation, const std::string &reason) {
    std::lock_guard<std::mutex> lock(launch_mutex);

    RunInfo &run_info = runs[run_id];

    if (run_info.active) {
      logGlobal("Warning: Attempted to launch already active Run " +
                std::to_string(run_id));
      return;
    }

    int current_iteration = run_info.iteration_count.load();
    if (current_iteration >= max_iterations) {
      logGlobal("Run " + std::to_string(run_id) +
                " has completed all iterations");
      return;
    }

    current_iteration++; // Start from iteration 1
    run_info.iteration_count = current_iteration;

    // Create output files
    std::string stdout_file = base_results_dir + "/run_" +
                              std::to_string(run_id) + "_iter" +
                              std::to_string(current_iteration) + "_stdout.log";
    std::string stderr_file = base_results_dir + "/run_" +
                              std::to_string(run_id) + "_iter" +
                              std::to_string(current_iteration) + "_stderr.log";

    pid_t pid = fork();
    if (pid == 0) {
      // Child process
      freopen(stdout_file.c_str(), "w", stdout);
      freopen(stderr_file.c_str(), "w", stderr);

      // Set thread count environment variable
      std::string omp_threads = std::to_string(thread_distribution[run_id]);
      setenv("OMP_NUM_THREADS", omp_threads.c_str(), 1);

      // Prepare arguments
      std::vector<char *> args;
      args.push_back(const_cast<char *>(executable_path.c_str()));

      // Add run-id argument (this is what was missing!)
      std::string run_id_arg = "--run-id";
      std::string run_id_value = std::to_string(run_id);
      args.push_back(const_cast<char *>(run_id_arg.c_str()));
      args.push_back(const_cast<char *>(run_id_value.c_str()));

      // Add result-dir pointing to the iteration directory
      std::string result_dir_arg = "--result-dir";
      std::string result_dir_value = base_results_dir + "/iteration" +
                                     std::to_string(current_iteration) + "/";
      args.push_back(const_cast<char *>(result_dir_arg.c_str()));
      args.push_back(const_cast<char *>(result_dir_value.c_str()));

      for (const auto &arg : exe_args) {
        args.push_back(const_cast<char *>(arg.c_str()));
      }
      args.push_back(nullptr);

      execv(executable_path.c_str(), args.data());

      // If we get here, execv failed
      std::cerr << "Failed to execute: " << executable_path << std::endl;
      exit(1);
    } else if (pid > 0) {
      // Parent process
      run_info.pid = pid;
      run_info.active = true;
      run_info.start_time = std::chrono::steady_clock::now();
      run_info.assigned_threads = thread_distribution[run_id];

      logGlobal("Launched Run " + std::to_string(run_id) + " iteration " +
                std::to_string(current_iteration) + " (PID " +
                std::to_string(pid) + ", threads=" +
                std::to_string(run_info.assigned_threads) + ") - " + reason);

      // Record launch event
      {
        std::lock_guard<std::mutex> lock(events_mutex);
        launch_events.push_back({run_id, current_iteration, trigger_operation,
                                 std::chrono::steady_clock::now(), reason});

        auto time_sec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          global_start_time)
                .count();
        csv_log << time_sec << ",launch," << run_id << "," << current_iteration
                << "," << trigger_operation << ",0,0," << reason << std::endl;
        csv_log.flush();
      }
    } else {
      throw std::runtime_error("Failed to fork process for Run " +
                               std::to_string(run_id));
    }
  }

  void monitorRuns() {
    while (!shutdown_requested) {
      std::vector<int> completed_runs;
      bool has_any_active = false;

      for (auto &[run_id, run_info] : runs) {
        if (run_info.active && run_info.pid > 0) {
          has_any_active = true;
          int status;
          pid_t result = waitpid(run_info.pid, &status, WNOHANG);

          if (result == run_info.pid) {
            // Process completed
            run_info.end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration<double>(run_info.end_time -
                                                          run_info.start_time)
                                .count();

            int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
            std::cout << "✅ Run " << run_id << " iteration "
                      << run_info.iteration_count.load() << " completed in "
                      << duration << "s (exit=" << exit_code << ")"
                      << std::endl;

            logGlobal("Run " + std::to_string(run_id) + " iteration " +
                      std::to_string(run_info.iteration_count.load()) +
                      " completed (exit=" + std::to_string(exit_code) +
                      ", duration=" + std::to_string(duration) + "s)");

            // Record completion event
            {
              std::lock_guard<std::mutex> lock(events_mutex);
              completion_events.push_back(
                  {run_id, run_info.iteration_count.load(), exit_code,
                   run_info.start_time, run_info.end_time, duration});

              auto time_sec = std::chrono::duration<double>(run_info.end_time -
                                                            global_start_time)
                                  .count();
              csv_log << time_sec << ",completion," << run_id << ","
                      << run_info.iteration_count.load() << ",-1," << exit_code
                      << "," << duration << ",process_completed" << std::endl;
              csv_log.flush();
            }

            run_info.active = false;
            run_info.pid = 0;

            // Check if we should launch another iteration
            if (run_info.iteration_count < max_iterations) {
              std::cout << "🚀 Starting Run " << run_id << " iteration "
                        << (run_info.iteration_count.load() + 1) << std::endl;
              launchRun(run_id, -1, "Next iteration");
              has_any_active = true; // We just launched a new process
            } else {
              std::cout << "🏁 Run " << run_id << " completed all "
                        << max_iterations << " iterations" << std::endl;
              completed_runs.push_back(run_id);
            }
          }
        } else if (run_info.iteration_count.load() < max_iterations) {
          // Not active but still has iterations to run
          has_any_active = true;
        }
      }

      // Exit monitor thread if no active runs and all completed
      if (!has_any_active && !hasActiveRuns()) {
        std::cout << "🎯 Monitor thread exiting - all work completed"
                  << std::endl;
        break;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
  }

  void periodicStatusReport() {
    while (!shutdown_requested) {
      std::this_thread::sleep_for(std::chrono::seconds(30));

      if (shutdown_requested)
        break;

      // Check if we should exit (all work completed)
      if (!hasActiveRuns()) {
        std::cout << "📊 Status thread exiting - all work completed"
                  << std::endl;
        break;
      }

      int active_runs = 0;
      std::string status_msg = "Status: ";

      for (const auto &[run_id, run_info] : runs) {
        if (run_info.active) {
          active_runs++;
          status_msg += "Run" + std::to_string(run_id) + "(iter" +
                        std::to_string(run_info.iteration_count.load()) + ") ";
        }
      }

      if (active_runs > 0) {
        status_msg += "- " + std::to_string(active_runs) + " active";
        logGlobal(status_msg);
        std::cout << status_msg << std::endl;
      }
    }
  }

  bool hasActiveRuns() {
    int active_count = 0;
    int completed_count = 0;

    for (const auto &[run_id, run_info] : runs) {
      if (run_info.active) {
        active_count++;
      } else if (run_info.iteration_count.load() < max_iterations) {
        // Process is not active but still has iterations to run
        return true;
      } else if (run_info.iteration_count.load() >= max_iterations) {
        completed_count++;
      }
    }

    // Debug output
    if (active_count > 0 || completed_count < runs.size()) {
      std::cout << "hasActiveRuns: active=" << active_count
                << ", completed=" << completed_count << "/" << runs.size()
                << std::endl;
    }

    return active_count > 0 || completed_count < runs.size();
  }

  void terminateAllRuns() {
    logGlobal("Terminating all active runs...");
    for (auto &[run_id, run_info] : runs) {
      if (run_info.active && run_info.pid > 0) {
        logGlobal("Terminating Run " + std::to_string(run_id) + " (PID " +
                  std::to_string(run_info.pid) + ")");
        kill(run_info.pid, SIGTERM);

        // Wait a bit for graceful shutdown
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        // Force kill if still running
        int status;
        if (waitpid(run_info.pid, &status, WNOHANG) == 0) {
          logGlobal("Force killing Run " + std::to_string(run_id));
          kill(run_info.pid, SIGKILL);
          waitpid(run_info.pid, &status, 0);
        }

        run_info.active = false;
        run_info.pid = 0;
      }
    }
  }

  void generateExecutionSummary() {
    auto global_end_time = std::chrono::steady_clock::now();
    auto total_seconds =
        std::chrono::duration<double>(global_end_time - global_start_time)
            .count();

    std::string summary_file = base_results_dir + "/execution_summary.txt";
    std::ofstream summary(summary_file);

    if (summary.is_open()) {
      summary << "=============\n";
      summary << "PARALLEL EXECUTION SUMMARY\n";
      summary << "=============\n\n";
      summary << "Total execution time: " << total_seconds << " seconds\n";
      summary << "Parallel processes: " << process_count << "\n";
      summary << "Total launches: " << launch_events.size() << "\n";
      summary << "Total completions: " << completion_events.size() << "\n";
      summary << "Results directory: " << base_results_dir << "\n\n";

      summary << "Thread Distribution:\n";
      for (const auto &[run_id, threads] : thread_distribution) {
        summary << "  Run " << run_id << ": " << threads << " threads\n";
      }
      summary << "\n";

      summary << "Launch Events:\n";
      for (const auto &event : launch_events) {
        auto time_sec =
            std::chrono::duration<double>(event.timestamp - global_start_time)
                .count();
        summary << "  [" << time_sec << "s] Run " << event.run_id
                << " iteration " << event.iteration << " - " << event.reason
                << "\n";
      }

      summary << "\nCompletion Events:\n";
      for (const auto &event : completion_events) {
        auto start_sec =
            std::chrono::duration<double>(event.start_time - global_start_time)
                .count();
        auto end_sec =
            std::chrono::duration<double>(event.end_time - global_start_time)
                .count();
        summary << "  [" << start_sec << "s-" << end_sec << "s] Run "
                << event.run_id << " iteration " << event.iteration << " ("
                << event.duration_seconds << "s, exit=" << event.exit_code
                << ")\n";
      }

      summary.close();
      logGlobal("Detailed summary saved to: " + summary_file);
    }
  }
};

void printUsage(const char *program_name) {
  std::cout
      << "Usage: " << program_name << " [OPTIONS] --executable EXE_PATH\n"
      << "\nRequired:\n"
      << "  --executable PATH     Path to inference executable\n"
      << "\nOptional:\n"
      << "  --processes N         Number of parallel processes (default: 4)\n"
      << "  --iterations N        Max iterations per run (default: 3)\n"
      << "  --max-threads N       Max system threads (default: auto-detect)\n"
      << "  --exe-args \"ARGS\"     Arguments to pass to executable\n"
      << "  --help                Show this help\n"
      << "\nExample:\n"
      << "  " << program_name
      << " --executable ./bin/resnet_opt_server_exe \\\n"
      << "    --processes 8 --iterations 5 --max-threads 72 \\\n"
      << "    --exe-args \"--key-mode imperative --depth 8\"\n"
      << "\nOutput:\n"
      << "  Creates parallel_results_TIMESTAMP/ directory with:\n"
      << "    - parallel_launcher.log (human-readable log)\n"
      << "    - parallel_launcher_events.csv (structured events)\n"
      << "    - execution_summary.txt (final summary)\n"
      << "    - iteration1/, iteration2/, ... (results by iteration)\n"
      << "    - run_X_iterY_stdout.log (process outputs)\n"
      << std::endl;
}

std::vector<std::string> parseArgs(const std::string &args_str) {
  std::vector<std::string> args;
  std::istringstream iss(args_str);
  std::string arg;

  while (iss >> std::quoted(arg)) {
    args.push_back(arg);
  }

  return args;
}

int main(int argc, char *argv[]) {
  std::string executable_path;
  int processes = 4;
  int iterations = 3;
  int max_threads = 0;
  std::string exe_args_str;

  // Parse command line arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      return 0;
    } else if (arg == "--executable") {
      if (i + 1 < argc) {
        executable_path = argv[++i];
      } else {
        std::cerr << "Error: --executable requires a path" << std::endl;
        return 1;
      }
    } else if (arg == "--processes") {
      if (i + 1 < argc) {
        processes = std::stoi(argv[++i]);
      } else {
        std::cerr << "Error: --processes requires a number" << std::endl;
        return 1;
      }
    } else if (arg == "--iterations") {
      if (i + 1 < argc) {
        iterations = std::stoi(argv[++i]);
      } else {
        std::cerr << "Error: --iterations requires a number" << std::endl;
        return 1;
      }
    } else if (arg == "--max-threads") {
      if (i + 1 < argc) {
        max_threads = std::stoi(argv[++i]);
      } else {
        std::cerr << "Error: --max-threads requires a number" << std::endl;
        return 1;
      }
    } else if (arg == "--exe-args") {
      if (i + 1 < argc) {
        exe_args_str = argv[++i];
      } else {
        std::cerr << "Error: --exe-args requires argument string" << std::endl;
        return 1;
      }
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      printUsage(argv[0]);
      return 1;
    }
  }

  // Validate required arguments
  if (executable_path.empty()) {
    std::cerr << "Error: --executable is required" << std::endl;
    printUsage(argv[0]);
    return 1;
  }

  if (access(executable_path.c_str(), X_OK) != 0) {
    std::cerr << "Error: Cannot execute: " << executable_path << std::endl;
    return 1;
  }

  try {
    ParallelLauncher launcher(executable_path, processes);
    launcher.setMaxIterations(iterations);

    if (max_threads > 0) {
      launcher.setMaxThreads(max_threads);
    }

    if (!exe_args_str.empty()) {
      auto parsed_args = parseArgs(exe_args_str);
      launcher.setExecutableArgs(parsed_args);
    }

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Executable: " << executable_path << std::endl;
    std::cout << "  Processes: " << processes << std::endl;
    std::cout << "  Max iterations: " << iterations << std::endl;
    if (max_threads > 0) {
      std::cout << "  Max threads: " << max_threads << std::endl;
    }
    if (!exe_args_str.empty()) {
      std::cout << "  Executable args: " << exe_args_str << std::endl;
    }
    std::cout << std::endl;

    launcher.run();

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "Parallel launcher completed successfully" << std::endl;
  return 0;
}
