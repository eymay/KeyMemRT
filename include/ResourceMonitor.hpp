#pragma once
#include <sys/resource.h>

#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

class ResourceMonitor {
public:
  struct BasicMetrics {
    double timestamp;      // Time since start (seconds)
    double cpu_usage;      // Average CPU usage (%)
    double ram_active_gb;  // Actively used RAM in GB (default memory metric)
    double ram_used_gb;    // Total used RAM in GB
    double ram_total_gb;   // Total RAM in GB
    double process_rss_mb; // Process-specific RSS memory in MB (VmRSS - physical memory actually in RAM)
    double process_vms_mb; // Process virtual memory size (VmSize - total address space including mmap)
    double process_data_mb; // Process heap memory (VmData - malloc/new allocations)
    double process_hwm_mb; // Process peak RSS (VmHWM - high water mark)
    double process_pss_mb; // Process proportional set size (shared memory divided by sharers)
  };

  struct EventMarker {
    std::string name;  // Event name
    double start_time; // Start time (seconds since monitoring began)
    double end_time;   // End time (seconds since monitoring began)
    int thread_id;     // Thread ID
  };

  ResourceMonitor(bool collect_advanced_stats = false)
      : should_run_(true), collect_advanced_stats_(collect_advanced_stats),
        sample_interval_ms_(10) {}

  void start(const std::string &base_filename) {

    should_run_ = true;
    start_time_ = std::chrono::steady_clock::now();

    monitor_thread_ = std::thread([this]() {
      while (should_run_) {
        captureMetrics();
        std::this_thread::sleep_for(
            std::chrono::milliseconds(sample_interval_ms_));
      }
    });

    periodic_save_thread_ = std::thread([this, base_filename]() {
      while (should_run_) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(100 /*sleep interval*/));

        if (should_run_) {
          save_to_file(base_filename);
        }
      }
    });
  }

  void stop() {
    should_run_ = false;
    if (monitor_thread_.joinable()) {
      monitor_thread_.join();
    }

    if (periodic_save_thread_.joinable()) {
      periodic_save_thread_.join();
    }
  }
  ~ResourceMonitor() { stop(); }

  // Mark the start of a named event
  void mark_event_start(const std::string &name) {
    double current_time = getElapsedTime();
    int thread_id = getThreadId();

    std::lock_guard<std::mutex> lock(events_mutex_);
    EventMarker event;
    event.name = name;
    event.start_time = current_time;
    event.end_time = -1; // Not finished yet
    event.thread_id = thread_id;
    events_.push_back(event);
  }

  // Mark the end of a named event
  void mark_event_end(const std::string &name) {
    double current_time = getElapsedTime();
    int thread_id = getThreadId();

    std::lock_guard<std::mutex> lock(events_mutex_);
    // Find the matching start event
    for (auto it = events_.rbegin(); it != events_.rend(); ++it) {
      if (it->name == name && it->thread_id == thread_id &&
          it->end_time == -1) {
        it->end_time = current_time;
        break;
      }
    }
  }

  void save_to_file(const std::string &base_filename) const {
    // Save basic resource usage data
    std::string resource_filename = base_filename + "_basic.csv";
    std::ofstream file(resource_filename);

    if (!file.is_open()) {
      std::cerr << "ERROR: ResourceMonitor failed to open file for writing: "
                << resource_filename << std::endl;
      std::cerr << "Check if directory exists and is writable" << std::endl;
      return;
    }
    file << std::fixed << std::setprecision(6);

    // Basic metrics header
    file << "Time,CPU_Usage,RAM_Active_GB,RAM_Used_GB,RAM_Total_GB,Process_RSS_"
            "MB,Process_VMS_MB,Process_Data_MB,Process_HWM_MB,Process_PSS_MB";
    if (collect_advanced_stats_) {
      file << ",RAM_Free_GB,RAM_Cached_GB,RAM_Buffers_GB,"
           << "Swap_Used_GB,Swap_Free_GB,Swap_Cached_GB,Swap_Total_GB,"
           << "Pages_In,Pages_Out,Swap_Pages_In,Swap_Pages_Out,"
           << "Minor_Faults,Major_Faults";
    }
    file << "\n";

    {
      std::lock_guard<std::mutex> lock(metrics_mutex_);
      for (const auto &sample : basic_metrics_) {
        file << sample.timestamp << "," << sample.cpu_usage << ","
             << sample.ram_active_gb << "," << sample.ram_used_gb << ","
             << sample.ram_total_gb << "," << sample.process_rss_mb << ","
             << sample.process_vms_mb << "," << sample.process_data_mb << ","
             << sample.process_hwm_mb << "," << sample.process_pss_mb;

        if (collect_advanced_stats_ && !advanced_metrics_.empty()) {
          // Write advanced metrics if available
          const auto &advanced =
              advanced_metrics_[&sample - &basic_metrics_[0]];
          file << "," << advanced.ram_free_gb << "," << advanced.ram_cached_gb
               << "," << advanced.ram_buffers_gb << "," << advanced.swap_used_gb
               << "," << advanced.swap_free_gb << "," << advanced.swap_cached_gb
               << "," << advanced.swap_total_gb << "," << advanced.pages_in
               << "," << advanced.pages_out << "," << advanced.swap_pages_in
               << "," << advanced.swap_pages_out << "," << advanced.minor_faults
               << "," << advanced.major_faults;
        }
        file << "\n";
      }
    }

    // Save event timing data
    std::string events_filename = base_filename + "_events.csv";
    std::ofstream events_file(events_filename);
    events_file << std::fixed << std::setprecision(6);
    events_file << "Event,ThreadID,StartTime,EndTime,Duration\n";

    std::lock_guard<std::mutex> lock(events_mutex_);
    for (const auto &event : events_) {
      double duration = event.end_time - event.start_time;
      // Only output completed events
      if (event.end_time > 0) {
        events_file << event.name << "," << event.thread_id << ","
                    << event.start_time << "," << event.end_time << ","
                    << duration << "\n";
      }
    }
  }

  // Configure sampling interval (in milliseconds)
  void set_sample_interval(unsigned int ms) { sample_interval_ms_ = ms; }

private:
  struct AdvancedMetrics {
    double ram_free_gb;
    double ram_cached_gb;
    double ram_buffers_gb;
    double swap_used_gb;
    double swap_free_gb;
    double swap_cached_gb;
    double swap_total_gb;
    unsigned long pages_in;
    unsigned long pages_out;
    unsigned long swap_pages_in;
    unsigned long swap_pages_out;
    long minor_faults;
    long major_faults;
  };

  std::atomic<bool> should_run_;
  bool collect_advanced_stats_;
  unsigned int sample_interval_ms_;
  std::thread monitor_thread_;
  std::thread periodic_save_thread_;
  std::chrono::steady_clock::time_point start_time_;

  mutable std::mutex metrics_mutex_;
  std::vector<BasicMetrics> basic_metrics_;
  std::vector<AdvancedMetrics> advanced_metrics_;

  mutable std::mutex events_mutex_;
  std::vector<EventMarker> events_;

  void captureMetrics() {
    BasicMetrics basic;
    basic.timestamp = getElapsedTime();
    basic.cpu_usage = getCpuUsage();
    getMemoryInfo(basic);

    {
      std::lock_guard<std::mutex> lock(metrics_mutex_);
      basic_metrics_.push_back(basic);

      // Collect advanced metrics if enabled
      if (collect_advanced_stats_) {
        AdvancedMetrics advanced = getAdvancedMetrics();
        advanced_metrics_.push_back(advanced);
      }
    }
  }

public:
  double getElapsedTime() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         start_time_)
        .count();
  }

private:
  int getThreadId() const {
// Simple thread ID implementation
#ifdef _WIN32
    return GetCurrentThreadId();
#else
    return static_cast<int>(pthread_self());
#endif
  }

  double getCpuUsage() {
    // Basic CPU usage implementation
    static double last_idle = 0, last_total = 0;
    std::ifstream stat_file("/proc/stat");
    std::string line;
    std::getline(stat_file, line);

    std::istringstream ss(line);
    std::string cpu;
    unsigned long user, nice, system, idle, iowait, irq, softirq, steal;
    ss >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >>
        steal;

    double total = user + nice + system + idle + iowait + irq + softirq + steal;
    double idle_time = idle;

    double idle_delta = idle_time - last_idle;
    double total_delta = total - last_total;

    last_idle = idle_time;
    last_total = total;

    if (total_delta == 0)
      return 0.0;
    return 100.0 * (1.0 - idle_delta / total_delta);
  }

  void getProcessMemoryMetrics(double &rss_mb, double &vms_mb, double &data_mb,
                                double &hwm_mb, double &pss_mb) {
    // Parse /proc/self/status for VmRSS, VmSize, VmData, VmHWM
    std::ifstream status("/proc/self/status");
    std::string line;
    rss_mb = vms_mb = data_mb = hwm_mb = 0.0;

    while (std::getline(status, line)) {
      std::istringstream iss(line);
      std::string label;
      long value;
      std::string unit;
      iss >> label >> value >> unit;

      if (label == "VmRSS:")
        rss_mb = value / 1024.0; // KB to MB
      else if (label == "VmSize:")
        vms_mb = value / 1024.0;
      else if (label == "VmData:")
        data_mb = value / 1024.0;
      else if (label == "VmHWM:")
        hwm_mb = value / 1024.0;
    }

    // Parse /proc/self/smaps_rollup for PSS (more efficient than smaps)
    std::ifstream smaps("/proc/self/smaps_rollup");
    pss_mb = 0.0;
    while (std::getline(smaps, line)) {
      if (line.substr(0, 4) == "Pss:") {
        std::istringstream iss(line);
        std::string label;
        long value;
        std::string unit;
        iss >> label >> value >> unit;
        pss_mb = value / 1024.0; // KB to MB
        break;
      }
    }

    // Fallback: if smaps_rollup doesn't exist, PSS = RSS (conservative estimate)
    if (pss_mb == 0.0) {
      pss_mb = rss_mb;
    }
  }

  void getMemoryInfo(BasicMetrics &metrics) {
    // Get basic memory info from /proc/meminfo
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    double mem_total = 0, mem_free = 0, mem_available = 0;
    double cached = 0, buffers = 0;

    while (std::getline(meminfo, line)) {
      std::istringstream ss(line);
      std::string key;
      unsigned long value;
      ss >> key >> value;

      if (key == "MemTotal:")
        mem_total = value / 1024.0 / 1024.0;
      else if (key == "MemFree:")
        mem_free = value / 1024.0 / 1024.0;
      else if (key == "MemAvailable:")
        mem_available = value / 1024.0 / 1024.0;
      else if (key == "Cached:")
        cached = value / 1024.0 / 1024.0;
      else if (key == "Buffers:")
        buffers = value / 1024.0 / 1024.0;
    }

    metrics.ram_total_gb = mem_total;
    metrics.ram_used_gb = mem_total - mem_free - cached - buffers;

    // Calculate active memory (total - available)
    // This is a better representation of actual memory pressure
    if (mem_available > 0) {
      metrics.ram_active_gb = mem_total - mem_available;
    } else {
      // Fallback if MemAvailable is not present in older kernels
      metrics.ram_active_gb = metrics.ram_used_gb;
    }

    // Get all process-specific memory metrics
    getProcessMemoryMetrics(metrics.process_rss_mb, metrics.process_vms_mb,
                           metrics.process_data_mb, metrics.process_hwm_mb,
                           metrics.process_pss_mb);
  }

  AdvancedMetrics getAdvancedMetrics() {
    AdvancedMetrics metrics{};

    // Read memory info
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    double mem_total = 0, mem_free = 0, cached = 0, buffers = 0;
    double swap_total = 0, swap_free = 0, swap_cached = 0;

    while (std::getline(meminfo, line)) {
      std::istringstream ss(line);
      std::string key;
      unsigned long value;
      ss >> key >> value;

      if (key == "MemTotal:")
        mem_total = value / 1024.0 / 1024.0;
      else if (key == "MemFree:")
        mem_free = value / 1024.0 / 1024.0;
      else if (key == "Cached:")
        cached = value / 1024.0 / 1024.0;
      else if (key == "Buffers:")
        buffers = value / 1024.0 / 1024.0;
      else if (key == "SwapTotal:")
        swap_total = value / 1024.0 / 1024.0;
      else if (key == "SwapFree:")
        swap_free = value / 1024.0 / 1024.0;
      else if (key == "SwapCached:")
        swap_cached = value / 1024.0 / 1024.0;
    }

    // Read vmstat for page info
    std::ifstream vmstat("/proc/vmstat");
    while (std::getline(vmstat, line)) {
      std::istringstream ss(line);
      std::string key;
      unsigned long value;
      ss >> key >> value;

      if (key == "pgpgin")
        metrics.pages_in = value;
      else if (key == "pgpgout")
        metrics.pages_out = value;
      else if (key == "pswpin")
        metrics.swap_pages_in = value;
      else if (key == "pswpout")
        metrics.swap_pages_out = value;
    }

    // Get fault stats
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
      metrics.minor_faults = usage.ru_minflt;
      metrics.major_faults = usage.ru_majflt;
    }

    // Set values
    metrics.ram_free_gb = mem_free;
    metrics.ram_cached_gb = cached;
    metrics.ram_buffers_gb = buffers;
    metrics.swap_total_gb = swap_total;
    metrics.swap_free_gb = swap_free;
    metrics.swap_cached_gb = swap_cached;
    metrics.swap_used_gb = swap_total - swap_free;

    return metrics;
  }
};
