#ifndef KEYMEMRT_LOGGER_H_
#define KEYMEMRT_LOGGER_H_

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

enum class LogLevel {
  DEBUG,
  INFO,
  WARNING,
  ERROR,
  OFF // No logging
};

class Logger {
public:
  static Logger &getInstance() {
    static Logger instance;
    return instance;
  }

  void setLogLevel(LogLevel level) { logLevel = level; }

  bool isDebugEnabled() const { return logLevel <= LogLevel::DEBUG; }

  bool isInfoEnabled() const { return logLevel <= LogLevel::INFO; }

  bool isWarningEnabled() const { return logLevel <= LogLevel::WARNING; }

  bool isErrorEnabled() const { return logLevel <= LogLevel::ERROR; }

  std::string levelToString(LogLevel level) {
    switch (level) {
    case LogLevel::DEBUG:
      return "DEBUG";
    case LogLevel::INFO:
      return "INFO";
    case LogLevel::WARNING:
      return "WARNING";
    case LogLevel::ERROR:
      return "ERROR";
    case LogLevel::OFF:
      return "OFF";
    default:
      return "UNKNOWN";
    }
  }

  void setLogToFile(bool enable, const std::string &filename = "keymemrt.log") {
    std::lock_guard<std::mutex> lock(logMutex);
    if (enable && !logFile.is_open()) {
      logFile.open(filename, std::ios::out | std::ios::app);
      if (logFile.is_open() && logLevel < LogLevel::OFF) {
        logFile << "\n[" << getTimestamp() << "] [INFO] Logger started"
                << std::endl;
      }
    } else if (!enable && logFile.is_open()) {
      if (logLevel < LogLevel::OFF) {
        logFile << "[" << getTimestamp() << "] [INFO] Logger stopped"
                << std::endl;
      }
      logFile.close();
    }
    logToFile = enable;
  }

  void setLogToConsole(bool enable) { logToConsole = enable; }

  template <typename... Args>
  void debug(const std::string &format, Args... args) {
    if (logLevel <= LogLevel::DEBUG) {
      log(LogLevel::DEBUG, format, args...);
    }
  }

  template <typename... Args>
  void info(const std::string &format, Args... args) {
    if (logLevel <= LogLevel::INFO) {
      log(LogLevel::INFO, format, args...);
    }
  }

  template <typename... Args>
  void warning(const std::string &format, Args... args) {
    if (logLevel <= LogLevel::WARNING) {
      log(LogLevel::WARNING, format, args...);
    }
  }

  template <typename... Args>
  void error(const std::string &format, Args... args) {
    if (logLevel <= LogLevel::ERROR) {
      log(LogLevel::ERROR, format, args...);
    }
  }

private:
  Logger() : logLevel(LogLevel::INFO), logToConsole(true), logToFile(false) {}
  ~Logger() {
    if (logFile.is_open()) {
      logFile.close();
    }
  }

  Logger(const Logger &) = delete;
  Logger &operator=(const Logger &) = delete;

  std::string getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch()) %
                  1000;

    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << now_ms.count();
    return ss.str();
  }

  template <typename... Args>
  void log(LogLevel level, const std::string &format, Args... args) {
    std::lock_guard<std::mutex> lock(logMutex);

    std::string timestamp = getTimestamp();
    std::string levelStr = levelToString(level);

    // Format the message with the provided arguments
    std::string message = formatString(format, args...);

    // Create the full log entry
    std::stringstream logEntry;
    logEntry << "[" << timestamp << "] [" << levelStr << "] " << message;

    // Output to console if enabled
    if (logToConsole) {
      std::cout << logEntry.str() << std::endl;
    }

    // Output to file if enabled
    if (logToFile && logFile.is_open()) {
      logFile << logEntry.str() << std::endl;
    }
  }

  // Simple string formatter that replaces {} placeholders with arguments
  template <typename T, typename... Args>
  std::string formatString(const std::string &format, T value, Args... args) {
    size_t openBracePos = format.find("{}");
    if (openBracePos == std::string::npos) {
      return format;
    }

    std::stringstream ss;
    ss << value;

    return format.substr(0, openBracePos) + ss.str() +
           formatString(format.substr(openBracePos + 2), args...);
  }

  // Base case for the recursion
  std::string formatString(const std::string &format) { return format; }

  LogLevel logLevel;
  bool logToConsole;
  bool logToFile;
  std::ofstream logFile;
  std::mutex logMutex;
};

#define LOG_DEBUG(format, ...)                                                 \
  Logger::getInstance().debug(format, ##__VA_ARGS__)
#define LOG_INFO(format, ...) Logger::getInstance().info(format, ##__VA_ARGS__)
#define LOG_WARNING(format, ...)                                               \
  Logger::getInstance().warning(format, ##__VA_ARGS__)
#define LOG_ERROR(format, ...)                                                 \
  Logger::getInstance().error(format, ##__VA_ARGS__)

#endif // KEYMEMRT_LOGGER_H_"
