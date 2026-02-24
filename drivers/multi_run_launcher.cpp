#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fcntl.h>
#include <fstream>
#include <iomanip>#include <iostream>
#include <iomanip>
#include <iostream>
#include <json.hpp>
#include <map>
#include <mutex>
#include <regex>
#include <set>
#include <signal.h>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

using json = nlohmann::json;

class ScheduleLauncher {
private:
  std::atomic<int> run0_current_iteration{0};
  std::set<int>
      triggered_operations; // Track which operations have triggered launches
  std::mutex triggered_operations_mutex;

  void detectSystemThreads() {
    // Try to detect number of hardware threads
    int hw_threads = std::thread::hardware_concurrency();
    if (hw_threads > 0) {
      max_system_threads = hw_threads;
    }

    // Check OMP_NUM_THREADS environment variable
    const char *omp_threads = getenv("OMP_NUM_THREADS");
    if (omp_threads) {
      max_system_threads = std::stoi(omp_threads);
    }

    std::cout << "Detected " << max_system_threads << " system threads"
              << std::endl;
    calculateThreadDistribution();
  }

  void calculateThreadDistribution() {
    int total_processes = process_starts.size() + 1; // +1 for Run 0

    // Calculate base threads per process
    int base_threads = max_system_threads / total_processes;
    int extra_threads = max_system_threads % total_processes;

    // If base_threads is 0, give everyone 1 thread (will oversubscribe)
    if (base_threads == 0) {
      base_threads = 1;
      std::cout << "Warning: More processes (" << total_processes
                << ") than threads (" << max_system_threads
                << "). Will oversubscribe." << std::endl;
    }

    // Distribute threads with your smart algorithm
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

    base_results_dir = "launcher_results_" + timestamp;

    // Create base directory
    if (mkdir(base_results_dir.c_str(), 0755) != 0) {
      throw std::runtime_error("Failed to create results directory: " +
                               base_results_dir);
    }

    // Create subdirectories for each iteration
    for (int iter = 1; iter <= max_iterations; ++iter) {
      std::string iter_dir =
          base_results_dir + "/iteration" + std::to_string(iter);
      if (mkdir(iter_dir.c_str(), 0755) != 0) {
        throw std::runtime_error("Failed to create iteration directory: " +
                                 iter_dir);
      }
    }

    std::cout << "Results directory: " << base_results_dir << std::endl;
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

  std::vector<int> process_starts;
  std::map<int, RunInfo> runs;
  std::atomic<bool> shutdown_requested{false};
  std::atomic<int> max_iterations{3}; // Default iteration count
  std::string executable_path;
  std::vector<std::string> exe_args;
  std::mutex launch_mutex;

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
  ScheduleLauncher(const std::string &schedule_file,
                   const std::string &exe_path)
      : executable_path(exe_path) {
    loadSchedule(schedule_file);
    setupSignalHandlers();
    detectSystemThreads();
    createResultsDirectory();
    initializeLogging();
  }

  ~ScheduleLauncher() {
    if (global_log.is_open()) {
      global_log.close();
    }
    if (csv_log.is_open()) {
      csv_log.close();
    }
  }

  void loadSchedule(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open schedule file: " + filename);
    }

    json root;
    file >> root;

    if (root.contains("process_starts")) {

      const auto &starts = root["process_starts"];
      for (const auto &val : starts) {
        process_starts.push_back(val.get<int>());
      }

      std::cout << "Loaded schedule with " << process_starts.size()
                << " process start points: ";
      for (size_t i = 0; i < process_starts.size(); ++i) {
        std::cout << process_starts[i];
        if (i < process_starts.size() - 1)
          std::cout << ", ";
      }
      std::cout << std::endl;
    } else {
      throw std::runtime_error(
          "Invalid schedule format - missing process_starts");
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

    std::cout << "Starting schedule launcher..." << std::endl;
    logGlobal("=== Schedule Launcher Started ===");
    logGlobal("Schedule: " + std::to_string(process_starts.size()) +
              " process start points");
    logGlobal("Max iterations per process: " +
              std::to_string(max_iterations.load()));
    logGlobal("Executable: " + executable_path);

    // Initialize run tracking structures
    for (size_t i = 0; i <= process_starts.size(); ++i) {
      runs[i] = RunInfo(i);
    }

    // Launch Run 0 immediately
    launchRun(0, -1, "Initial launch");

    // Start monitoring threads
    std::thread monitor_thread(&ScheduleLauncher::monitorRuns, this);
    std::thread tracker_thread(&ScheduleLauncher::trackRun0Progress, this);
    std::thread status_thread(&ScheduleLauncher::periodicStatusReport, this);

    // Wait for shutdown
    while (!shutdown_requested) {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // Cleanup
    std::cout << "Shutting down launcher..." << std::endl;
    logGlobal("=== Shutdown Initiated ===");

    tracker_thread.join();
    monitor_thread.join();
    status_thread.join();
    terminateAllRuns();

    generateSummaryReport();
    logGlobal("=== Schedule Launcher Completed ===");
  }

private:
  void initializeLogging() {
    // Create timestamped log files in results directory
    std::string log_filename = base_results_dir + "/launcher.log";
    std::string csv_filename = base_results_dir + "/launcher_events.csv";

    global_log.open(log_filename);
    csv_log.open(csv_filename);

    if (global_log.is_open()) {
      std::cout << "Global log: " << log_filename << std::endl;
    }

    if (csv_log.is_open()) {
      // Write CSV header
      csv_log << "timestamp_sec,event_type,run_id,iteration,trigger_operation,"
                 "duration_sec,exit_code,threads,reason\n";
      std::cout << "CSV log: " << csv_filename << std::endl;
    }
  }

  void logGlobal(const std::string &message) {
    auto now = std::chrono::steady_clock::now();
    auto duration = now - global_start_time;
    auto seconds = std::chrono::duration<double>(duration).count();

    std::string log_line = "[" + std::to_string(seconds) + "s] " + message;

    std::cout << log_line << std::endl;

    if (global_log.is_open()) {
      global_log << log_line << std::endl;
      global_log.flush();
    }
  }

  void logEvent(const std::string &event_type, int run_id, int iteration = -1,
                int trigger_op = -1, double duration = -1, int exit_code = -1,
                const std::string &reason = "") {
    auto now = std::chrono::steady_clock::now();
    auto duration_from_start = now - global_start_time;
    auto seconds = std::chrono::duration<double>(duration_from_start).count();

    int threads =
        (run_id < thread_distribution.size()) ? thread_distribution[run_id] : 0;

    if (csv_log.is_open()) {
      csv_log << seconds << "," << event_type << "," << run_id << ","
              << iteration << "," << trigger_op << "," << duration << ","
              << exit_code << "," << threads << "," << reason << "\n";
      csv_log.flush();
    }
  }

  void setupSignalHandlers() {
    signal(SIGINT, [](int) {
      std::cout << "\nReceived SIGINT, shutting down..." << std::endl;
      exit(0);
    });
  }

  void launchRun(int run_id, int trigger_operation = -1,
                 const std::string &reason = "") {
    std::lock_guard<std::mutex> lock(launch_mutex);

    if (runs[run_id].active) {
      logGlobal("Run " + std::to_string(run_id) +
                " is already active, skipping launch");
      return;
    }

    if (runs[run_id].iteration_count >= max_iterations) {
      logGlobal("Run " + std::to_string(run_id) + " reached max iterations (" +
                std::to_string(max_iterations) + "), not launching");
      return;
    }

    int new_iteration = runs[run_id].iteration_count + 1;
    int assigned_threads = thread_distribution[run_id];

    logGlobal("Launching Run " + std::to_string(run_id) + " (iteration " +
              std::to_string(new_iteration) + ", " +
              std::to_string(assigned_threads) + " threads) - " + reason);

    if (trigger_operation >= 0) {
      logGlobal("  Triggered by Run 0 reaching operation " +
                std::to_string(trigger_operation));
    }

    pid_t pid = fork();
    if (pid == 0) {
      // Child process - redirect stdout/stderr to files
      std::string stdout_file = base_results_dir + "/run_" +
                                std::to_string(run_id) + "_iter" +
                                std::to_string(new_iteration) + "_stdout.log";
      std::string stderr_file = base_results_dir + "/run_" +
                                std::to_string(run_id) + "_iter" +
                                std::to_string(new_iteration) + "_stderr.log";

      // Redirect stdout
      freopen(stdout_file.c_str(), "w", stdout);
      // Redirect stderr
      freopen(stderr_file.c_str(), "w", stderr);

      executeInference(run_id, new_iteration, assigned_threads);
      exit(0);
    } else if (pid > 0) {
      // Parent process
      runs[run_id].pid = pid;
      runs[run_id].active = true;
      runs[run_id].current_operation = 0;
      runs[run_id].iteration_count++;
      runs[run_id].assigned_threads = assigned_threads;
      runs[run_id].start_time = std::chrono::steady_clock::now();

      // Log the launch event
      {
        std::lock_guard<std::mutex> events_lock(events_mutex);
        launch_events.push_back({run_id, new_iteration, trigger_operation,
                                 runs[run_id].start_time, reason});
      }

      logEvent("LAUNCH", run_id, new_iteration, trigger_operation, -1, -1,
               reason);
      logGlobal("  Started with PID " + std::to_string(pid));
    } else {
      logGlobal("ERROR: Failed to fork for Run " + std::to_string(run_id));
    }
  }

  void executeInference(int run_id, int iteration, int threads) {
    // Set environment variables for thread control
    setenv("OMP_NUM_THREADS", std::to_string(threads).c_str(), 1);

    // Redirect stdout and stderr to files
    std::string stdout_file = base_results_dir + "/run_" +
                              std::to_string(run_id) + "_iter" +
                              std::to_string(iteration) + "_stdout.log";
    std::string stderr_file = base_results_dir + "/run_" +
                              std::to_string(run_id) + "_iter" +
                              std::to_string(iteration) + "_stderr.log";

    // Close and reopen stdout/stderr
    int stdout_fd =
        open(stdout_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    int stderr_fd =
        open(stderr_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);

    if (stdout_fd == -1 || stderr_fd == -1) {
      perror("Failed to open log files");
      exit(1);
    }

    // Redirect file descriptors
    if (dup2(stdout_fd, STDOUT_FILENO) == -1) {
      perror("Failed to redirect stdout");
      exit(1);
    }

    if (dup2(stderr_fd, STDERR_FILENO) == -1) {
      perror("Failed to redirect stderr");
      exit(1);
    }

    close(stdout_fd);
    close(stderr_fd);

    // Prepare arguments for the executable
    std::vector<char *> argv;
    argv.push_back(const_cast<char *>(executable_path.c_str()));

    // Add run-id argument
    std::string run_id_str = std::to_string(run_id);
    argv.push_back(const_cast<char *>("--run-id"));
    argv.push_back(const_cast<char *>(run_id_str.c_str()));

    // Add result directory argument
    std::string result_dir =
        base_results_dir + "/iteration" + std::to_string(iteration) + "/";
    argv.push_back(const_cast<char *>("--result-dir"));
    argv.push_back(const_cast<char *>(result_dir.c_str()));

    // Add user-provided args
    for (const auto &arg : exe_args) {
      argv.push_back(const_cast<char *>(arg.c_str()));
    }

    argv.push_back(nullptr);

    // Print launch info to redirected stdout
    printf("=== Run %d Iteration %d ===\n", run_id, iteration);
    printf("Threads: %d\n", threads);
    printf("Result directory: %s\n", result_dir.c_str());
    printf("Command: %s", executable_path.c_str());
    for (size_t i = 1; i < argv.size() - 1; ++i) {
      printf(" %s", argv[i]);
    }
    printf("\n");
    printf("==========================================\n");
    fflush(stdout);

    // Execute the inference binary
    execv(executable_path.c_str(), argv.data());

    // If we reach here, execv failed
    fprintf(stderr, "Failed to execute %s for Run %d\n",
            executable_path.c_str(), run_id);
    exit(1);
  }

  void trackRun0Progress() {
    logGlobal("Starting Run 0 progress tracker...");

    while (!shutdown_requested) {
      if (!runs[0].active) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }

      int current_iteration = runs[0].iteration_count.load();

      if (current_iteration != run0_current_iteration.load()) {
        logGlobal("Run 0 started iteration " +
                  std::to_string(current_iteration) +
                  ", resetting trigger tracking");

        // Reset triggered operations for new iteration
        {
          std::lock_guard<std::mutex> lock(triggered_operations_mutex);
          triggered_operations.clear();
        }

        run0_current_iteration = current_iteration;
      }

      int current_op = parseLogFileForCurrentOperation(0);
      if (current_op >= 0) {
        runs[0].current_operation = current_op;
        checkScheduleTriggers(current_op, current_iteration);
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }

  int parseLogFileForCurrentOperation(int run_id, int iteration = -1) {

    // Read from the current iteration's stdout file
    if (iteration == -1) {
      iteration = runs[run_id].iteration_count.load();
    }
    std::string stdout_file = base_results_dir + "/run_" +
                              std::to_string(run_id) + "_iter" +
                              std::to_string(iteration) + "_stdout.log";

    std::ifstream file(stdout_file);
    if (file.is_open()) {
      int latest_operation = -1;
      std::string line;

      while (std::getline(file, line)) {
        int op = extractOperationFromLine(line);
        if (op >= 0) {
          latest_operation = std::max(latest_operation, op);
        }
      }

      return latest_operation;
    }

    return -1;
  }

  int extractOperationFromLine(const std::string &line) {
    // Parse new LOG_CT format: LOG_CT:ct2:TIME:4.25743:LINE:971
    if (line.find("LOG_CT:") != std::string::npos &&
        line.find(":TIME:") != std::string::npos) {

      // Extract operation number from ct<number>
      std::regex ct_regex(R"(LOG_CT:ct(\d+):)");
      std::smatch match;
      if (std::regex_search(line, match, ct_regex)) {
        return std::stoi(match[1].str());
      }
    }

    // Also handle old PROFILE format as fallback
    if (line.find("PROFILE:") != std::string::npos ||
        line.find("ROTATION_PROFILE:") != std::string::npos) {

      size_t line_pos = line.find(":LINE:");
      if (line_pos != std::string::npos) {
        try {
          std::string line_num_str = line.substr(line_pos + 6);
          line_num_str.erase(line_num_str.find_last_not_of(" \t\r\n") + 1);
          return std::stoi(line_num_str);
        } catch (const std::exception &e) {
          // Invalid format, skip
        }
      }
    }

    return -1;
  }

  void checkScheduleTriggers(int run0_operation, int run0_iteration) {
    std::lock_guard<std::mutex> lock(triggered_operations_mutex);

    for (size_t i = 0; i < process_starts.size(); ++i) {
      int trigger_op = process_starts[i];
      int target_run_id = i + 1; // Run 1, 2, 3, etc.

      // Check if this operation has already triggered a launch
      if (triggered_operations.count(trigger_op) > 0) {
        continue; // Already triggered this operation
      }

      // Check if Run 0 has reached the trigger operation
      if (run0_operation >= trigger_op) {
        // Check if target run needs to be launched
        bool should_launch = false;

        // Launch if the run is not active
        if (!runs[target_run_id].active) {
          should_launch = true;
        }
        // Or if the target run is behind Run 0's iteration
        else if (runs[target_run_id].iteration_count < run0_iteration) {
          should_launch = true;
        }

        if (should_launch) {
          std::string reason =
              "Run 0 iteration " + std::to_string(run0_iteration) +
              " reached operation " + std::to_string(run0_operation) +
              " (scheduled at op " + std::to_string(trigger_op) + ")";

          // Mark this operation as triggered
          triggered_operations.insert(trigger_op);

          logGlobal("Triggering Run " + std::to_string(target_run_id) +
                    " due to trigger_op " + std::to_string(trigger_op));

          launchRun(target_run_id, run0_operation, reason);
        }
      }
    }
  }

  void monitorRuns() {
    logGlobal("Starting run monitor...");

    while (!shutdown_requested) {
      for (auto &[run_id, run_info] : runs) {
        if (run_info.active && run_info.pid > 0) {
          int status;
          pid_t result = waitpid(run_info.pid, &status, WNOHANG);

          if (result == run_info.pid) {
            // Process finished
            run_info.end_time = std::chrono::steady_clock::now();
            auto duration = run_info.end_time - run_info.start_time;
            double duration_sec =
                std::chrono::duration<double>(duration).count();

            int exit_code = 0;
            std::string status_msg;
            if (WIFEXITED(status)) {
              exit_code = WEXITSTATUS(status);
              status_msg = "exit code " + std::to_string(exit_code);
            } else if (WIFSIGNALED(status)) {
              exit_code = -WTERMSIG(status);
              status_msg =
                  "killed by signal " + std::to_string(WTERMSIG(status));
            }

            logGlobal("Run " + std::to_string(run_id) + " completed (" +
                      status_msg + ") after " + std::to_string(duration_sec) +
                      "s");

            // Log completion event
            {
              std::lock_guard<std::mutex> events_lock(events_mutex);
              completion_events.push_back(
                  {run_id, run_info.iteration_count.load(), exit_code,
                   run_info.start_time, run_info.end_time, duration_sec});
            }

            logEvent("COMPLETE", run_id, run_info.iteration_count.load(), -1,
                     duration_sec, exit_code, status_msg);

            run_info.active = false;
            run_info.pid = 0;
            run_info.current_operation = 0;

            // Auto-restart if under iteration limit
            if (run_info.iteration_count < max_iterations) {
              logGlobal("Auto-restarting Run " + std::to_string(run_id) +
                        " (iteration " +
                        std::to_string(run_info.iteration_count + 1) + "/" +
                        std::to_string(max_iterations) + ")");

              // Brief delay to avoid rapid restarts
              std::this_thread::sleep_for(std::chrono::milliseconds(100));
              launchRun(run_id, -1, "Auto-restart");
            } else {
              logGlobal("Run " + std::to_string(run_id) +
                        " reached max iterations, not restarting");
            }
          } else if (result == -1) {
            // Process doesn't exist anymore
            logGlobal("Run " + std::to_string(run_id) +
                      " process no longer exists");
            run_info.active = false;
            run_info.pid = 0;
          }
        }
      }

      // Check if all runs are done
      bool any_active = false;
      bool all_max_iterations = true;
      for (const auto &[run_id, run_info] : runs) {
        if (run_info.active) {
          any_active = true;
        }
        if (run_info.iteration_count < max_iterations) {
          all_max_iterations = false;
        }
      }

      if (!any_active && all_max_iterations) {
        logGlobal("All runs completed their iterations, shutting down");
        shutdown_requested = true;
        break;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
  }

  void periodicStatusReport() {
    while (!shutdown_requested) {
      std::this_thread::sleep_for(std::chrono::seconds(10));

      if (shutdown_requested)
        break;

      // Print brief status
      std::ostringstream status;
      status << "STATUS: Run 0 op=" << runs[0].current_operation;

      int active_count = 0;
      for (const auto &[run_id, run_info] : runs) {
        if (run_info.active) {
          active_count++;
        }
      }

      status << ", Active runs=" << active_count;
      logGlobal(status.str());
    }
  }

  void generateSummaryReport() {
    logGlobal("=== EXECUTION SUMMARY ===");

    auto total_duration = std::chrono::steady_clock::now() - global_start_time;
    auto total_seconds = std::chrono::duration<double>(total_duration).count();

    logGlobal("Total execution time: " + std::to_string(total_seconds) +
              " seconds");
    logGlobal("Total launches: " + std::to_string(launch_events.size()));
    logGlobal("Total completions: " + std::to_string(completion_events.size()));

    // Per-run summary
    for (const auto &[run_id, run_info] : runs) {
      logGlobal("Run " + std::to_string(run_id) + ": " +
                std::to_string(run_info.iteration_count) + "/" +
                std::to_string(max_iterations) + " iterations completed (" +
                std::to_string(run_info.assigned_threads) + " threads)");
    }

    // Calculate average execution times
    if (!completion_events.empty()) {
      double total_exec_time = 0;
      int successful_runs = 0;

      for (const auto &event : completion_events) {
        if (event.exit_code == 0) {
          total_exec_time += event.duration_seconds;
          successful_runs++;
        }
      }

      if (successful_runs > 0) {
        double avg_time = total_exec_time / successful_runs;
        logGlobal("Average successful run time: " + std::to_string(avg_time) +
                  " seconds");
      }
    }

    // Memory analysis suggestion
    logGlobal("=== ANALYSIS SUGGESTIONS ===");
    logGlobal("Results organized in: " + base_results_dir);
    logGlobal("Resource monitoring files created by each run in iteration "
              "subdirectories");
    logGlobal(
        "Process logs: run_X_iterY_stdout.log and run_X_iterY_stderr.log");
    logGlobal(
        "Use memory_analyzer.py with launcher_events.csv for global analysis");

    std::string summary_file = base_results_dir + "/execution_summary.txt";
    std::ofstream summary(summary_file);
    if (summary.is_open()) {
      summary << "Schedule Launcher Execution Summary\n";
      summary << "===================================\n\n";
      summary << "Total execution time: " << total_seconds << " seconds\n";
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

  void terminateAllRuns() {
    logGlobal("Terminating all active runs...");
    for (auto &[run_id, run_info] : runs) {
      if (run_info.active && run_info.pid > 0) {
        logGlobal("Terminating Run " + std::to_string(run_id) + " (PID " +
                  std::to_string(run_info.pid) + ")");
        kill(run_info.pid, SIGTERM);

        // Wait a bit for graceful shutdown
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

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
};

void printUsage(const char *program_name) {
  std::cout
      << "Usage: " << program_name
      << " [OPTIONS] --schedule SCHEDULE_FILE --executable EXE_PATH\n"
      << "\nRequired:\n"
      << "  --schedule FILE       JSON schedule file from scheduler.py\n"
      << "  --executable PATH     Path to inference executable\n"
      << "\nOptional:\n"
      << "  --iterations N        Max iterations per run (default: 3)\n"
      << "  --max-threads N       Max system threads (default: auto-detect)\n"
      << "  --exe-args \"ARGS\"     Arguments to pass to executable\n"
      << "  --help                Show this help\n"
      << "\nExample:\n"
      << "  " << program_name << " --schedule best_schedule.json \\\n"
      << "    --executable ./bin/resnet_opt_server_exe \\\n"
      << "    --iterations 5 --max-threads 72 \\\n"
      << "    --exe-args \"--key-mode imperative --depth 8\"\n"
      << "\nOutput:\n"
      << "  Creates launcher_results_TIMESTAMP/ directory with:\n"
      << "    - launcher.log (human-readable log)\n"
      << "    - launcher_events.csv (structured events)\n"
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
  std::string schedule_file;
  std::string executable_path;
  int iterations = 3;
  int max_threads = 0; // 0 means auto-detect
  std::string exe_args_str;

  // Parse command line arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      return 0;
    } else if (arg == "--schedule") {
      if (i + 1 < argc) {
        schedule_file = argv[++i];
      } else {
        std::cerr << "Error: --schedule requires a filename" << std::endl;
        return 1;
      }
    } else if (arg == "--executable") {
      if (i + 1 < argc) {
        executable_path = argv[++i];
      } else {
        std::cerr << "Error: --executable requires a path" << std::endl;
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
  if (schedule_file.empty()) {
    std::cerr << "Error: --schedule is required" << std::endl;
    printUsage(argv[0]);
    return 1;
  }

  if (executable_path.empty()) {
    std::cerr << "Error: --executable is required" << std::endl;
    printUsage(argv[0]);
    return 1;
  }

  // Check if files exist
  if (access(schedule_file.c_str(), R_OK) != 0) {
    std::cerr << "Error: Cannot read schedule file: " << schedule_file
              << std::endl;
    return 1;
  }

  if (access(executable_path.c_str(), X_OK) != 0) {
    std::cerr << "Error: Cannot execute: " << executable_path << std::endl;
    return 1;
  }

  try {
    ScheduleLauncher launcher(schedule_file, executable_path);
    launcher.setMaxIterations(iterations);

    if (max_threads > 0) {
      launcher.setMaxThreads(max_threads);
    }

    if (!exe_args_str.empty()) {
      auto parsed_args = parseArgs(exe_args_str);
      launcher.setExecutableArgs(parsed_args);
    }

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Schedule file: " << schedule_file << std::endl;
    std::cout << "  Executable: " << executable_path << std::endl;
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

  std::cout << "Launcher completed successfully" << std::endl;
  return 0;
}
