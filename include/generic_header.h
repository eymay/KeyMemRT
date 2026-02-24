#ifndef GENERICHEADER_H
#define GENERICHEADER_H

#define LOG_ROT(ctxt, mlir_name)                                               \
  do {                                                                         \
    std::cout << "ROTATION_PROFILE:"                                           \
              << ":INPUT:" << mlir_name << ":LEVEL:" << (ctxt)->GetLevel()     \
              << ":NOISE:" << (ctxt)->GetNoiseScaleDeg()                       \
              << ":TOWERS:" << (ctxt)->GetElements()[0].GetNumOfElements()     \
              << ":LINE:" << __LINE__ << std::endl;                            \
  } while (0)

#include <ResourceMonitor.hpp>
extern std::unique_ptr<ResourceMonitor> monitor;

inline double getCurrentTime() {
  if (monitor) {
    return monitor->getElapsedTime();
  }
  return 0.0;
}

#define LOG_CT(ctxt, mlir_name)                                                \
  do {                                                                         \
    std::cout << "PROFILE:"                                                    \
              << ":OUTPUT:" << mlir_name << ":LEVEL:" << (ctxt)->GetLevel()    \
              << ":NOISE:" << (ctxt)->GetNoiseScaleDeg()                       \
              << ":TOWERS:" << (ctxt)->GetElements()[0].GetNumOfElements()     \
              << ":LINE:" << __LINE__ << ":TIME:" << getCurrentTime() << "\n"; \
  } while (0)

// #include <iostream>
// #include <sstream>
//
// inline double getCurrentMemoryMB() {
//   std::ifstream status("/proc/self/status");
//   std::string line;
//   while (std::getline(stat s, line)) {
//     if (line.substr(0, 6) == "VmRSS:") {
//       std::istringstream iss(line);
//       std::string label, size, unit;
//       iss >> label >> size >> unit;
//       return std::stoll(size) / 1024.0; // Convert KB to MB
//     }
//   }
//   return 0.0;
// }
//

// #define LOG_CT(ctxt, mlir_name)                                                \
//   do {                                                                         \
//     std::cout << "LOG_CT:" << #ctxt << ":TIME:" << getCurrentTime()            \
//               << ":LINE:" << __LINE__ << "\n";                                 \
//   } while (0)
//
#endif
