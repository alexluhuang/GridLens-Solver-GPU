/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   host_platform.cpp
 * @date   2026-10-05
 *
 * @brief Host side of the platform layer (block B9-H). See
 * host_platform.hpp.
 */

#include "host_platform.hpp"

#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <utility>

namespace gridpack {
namespace batchpf {

namespace {

std::string readFile(const std::string &path)
{
  std::ifstream f(path.c_str());
  std::string s;
  if (f) std::getline(f, s);
  return s;
}

/// "2048K" -> bytes
long cacheBytes(const std::string &s)
{
  if (s.empty()) return 0;
  long v = std::atol(s.c_str());
  const char unit = s.back();
  if (unit == 'K') v *= 1024;
  if (unit == 'M') v *= 1024 * 1024;
  return v;
}

/// Size of the cache of a given level for a core
long cacheOfLevel(int cpu, int level)
{
  for (int idx = 0; idx < 8; idx++) {
    const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                             "/cache/index" + std::to_string(idx) + "/";
    const std::string lv = readFile(base + "level");
    if (lv.empty()) break;
    const std::string type = readFile(base + "type");
    if (std::atoi(lv.c_str()) == level && type != "Instruction") {
      return cacheBytes(readFile(base + "size"));
    }
  }
  return 0;
}

/// Online CPUs from "0-9,12,14-19"
std::vector<int> onlineCpus()
{
  std::vector<int> out;
  std::stringstream ss(readFile("/sys/devices/system/cpu/online"));
  std::string range;
  while (std::getline(ss, range, ',')) {
    const std::size_t dash = range.find('-');
    const int a = std::atoi(range.substr(0, dash).c_str());
    const int b = (dash == std::string::npos) ? a : std::atoi(range.substr(dash + 1).c_str());
    for (int c = a; c <= b; c++) out.push_back(c);
  }
  if (out.empty()) {
    const long n = sysconf(_SC_NPROCESSORS_ONLN);
    for (int c = 0; c < n; c++) out.push_back(c);
  }
  return out;
}

/// "CPU part" per processor from /proc/cpuinfo (Arm)
std::map<int, std::string> cpuParts()
{
  std::map<int, std::string> parts;
  std::ifstream f("/proc/cpuinfo");
  std::string line;
  int cpu = -1;
  while (std::getline(f, line)) {
    if (line.compare(0, 9, "processor") == 0) {
      cpu = std::atoi(line.substr(line.find(':') + 1).c_str());
    } else if (line.compare(0, 8, "CPU part") == 0 && cpu >= 0) {
      std::string p = line.substr(line.find(':') + 1);
      p.erase(0, p.find_first_not_of(" \t"));
      parts[cpu] = p;
    }
  }
  return parts;
}

}  // namespace

CpuTopology discoverTopology()
{
  CpuTopology t;
  const std::map<int, std::string> parts = cpuParts();
  for (int c : onlineCpus()) {
    CpuCore core;
    core.id = c;
    core.l2_bytes = cacheOfLevel(c, 2);
    core.l3_bytes = cacheOfLevel(c, 3);
    core.capacity = std::atoi(readFile("/sys/devices/system/cpu/cpu" +
                                       std::to_string(c) + "/cpu_capacity").c_str());
    const auto it = parts.find(c);
    if (it != parts.end()) core.part = it->second;
    t.cores.push_back(core);
  }
  // Tell performance cores apart by L2 size, else by CPU part, else by
  // capacity; if nothing differs, all cores count as performance cores.
  std::set<long> l2;
  std::set<std::string> kinds;
  int cap_max = 0, cap_min = 1 << 30;
  for (const CpuCore &c : t.cores) {
    if (c.l2_bytes > 0) l2.insert(c.l2_bytes);
    if (!c.part.empty()) kinds.insert(c.part);
    if (c.capacity > 0) {
      cap_max = std::max(cap_max, c.capacity);
      cap_min = std::min(cap_min, c.capacity);
    }
  }
  if (l2.size() > 1) {
    const long big = *l2.rbegin();
    for (CpuCore &c : t.cores) c.performance = (c.l2_bytes == big);
    t.basis = "L2 cache size";
  } else if (kinds.size() > 1 && cap_max > 0) {
    // the part whose cores have the higher average capacity
    std::map<std::string, std::pair<long, int>> sum;
    for (const CpuCore &c : t.cores) {
      sum[c.part].first += c.capacity;
      sum[c.part].second++;
    }
    std::string best;
    double best_avg = -1.0;
    for (const auto &s : sum) {
      const double avg = static_cast<double>(s.second.first) / s.second.second;
      if (avg > best_avg) {
        best_avg = avg;
        best = s.first;
      }
    }
    for (CpuCore &c : t.cores) c.performance = (c.part == best);
    t.basis = "CPU part";
  } else if (cap_max > 0 && cap_max > cap_min + cap_min / 5) {
    for (CpuCore &c : t.cores) c.performance = (c.capacity * 10 >= cap_max * 9);
    t.basis = "cpu_capacity";
  } else {
    t.basis = "all cores alike";
  }
  for (const CpuCore &c : t.cores) t.performance_count += c.performance ? 1 : 0;
  return t;
}

std::string describeTopology(const CpuTopology &t)
{
  std::ostringstream os;
  os << t.cores.size() << " online cores, " << t.performance_count
     << " performance and " << (t.cores.size() - t.performance_count)
     << " efficiency (classified by " << t.basis << ")";
  std::set<std::string> parts;
  for (const CpuCore &c : t.cores) {
    if (!c.part.empty()) parts.insert(c.part);
  }
  if (!parts.empty()) {
    os << "; CPU parts:";
    for (const std::string &p : parts) os << " " << p;
  }
  return os.str();
}

std::string applyBinding(const CpuTopology &t, CpuBinding policy, int local_rank,
                         int local_size, const std::vector<int> &accelerator_local)
{
  if (policy == CpuBinding::None) return "CPU binding: none (Execution/cpuBinding)";
  cpu_set_t current;
  CPU_ZERO(&current);
  if (sched_getaffinity(0, sizeof(current), &current) != 0) {
    return "CPU binding: affinity unavailable; left unchanged";
  }
  const int allowed = CPU_COUNT(&current);
  if (policy == CpuBinding::Auto &&
      allowed < static_cast<int>(t.cores.size())) {
    return "CPU binding: kept the launcher's binding (" + std::to_string(allowed) +
           " cores)";
  }
  // Core order: performance cores (most capable first), then efficiency
  std::vector<CpuCore> perf, eff;
  for (const CpuCore &c : t.cores) (c.performance ? perf : eff).push_back(c);
  auto byCap = [](const CpuCore &a, const CpuCore &b) {
    return a.capacity != b.capacity ? a.capacity > b.capacity : a.id < b.id;
  };
  std::sort(perf.begin(), perf.end(), byCap);
  std::sort(eff.begin(), eff.end(), byCap);
  std::vector<int> order;
  for (const CpuCore &c : perf) order.push_back(c.id);
  for (const CpuCore &c : eff) order.push_back(c.id);
  // Leave the last (least capable) core for the operating system
  std::vector<int> usable = order;
  if (usable.size() > 2) usable.pop_back();
  // Accelerator ranks take two cores each, first; then the others one each
  std::vector<int> ranks;
  for (int r : accelerator_local) {
    if (r < local_size) ranks.push_back(r);
  }
  for (int r = 0; r < local_size; r++) {
    if (std::find(ranks.begin(), ranks.end(), r) == ranks.end()) ranks.push_back(r);
  }
  std::size_t next = 0;
  std::vector<int> mine;
  for (int r : ranks) {
    const bool acc = std::find(accelerator_local.begin(), accelerator_local.end(), r) !=
                     accelerator_local.end();
    const int want = acc ? 2 : 1;
    for (int k = 0; k < want; k++) {
      const int core = usable[next % usable.size()];
      next++;
      if (r == local_rank) mine.push_back(core);
    }
  }
  if (next > usable.size()) {
    return "CPU binding: more ranks than cores; left unchanged";
  }
  cpu_set_t set;
  CPU_ZERO(&set);
  for (int c : mine) CPU_SET(c, &set);
  if (sched_setaffinity(0, sizeof(set), &set) != 0) {
    return "CPU binding: sched_setaffinity failed; left unchanged";
  }
  std::ostringstream os;
  os << "CPU binding: local rank " << local_rank << " on core";
  for (int c : mine) os << " " << c;
  return os.str();
}

double hostAvailableBytes()
{
  std::ifstream f("/proc/meminfo");
  std::string key;
  double kb = 0.0;
  std::string unit;
  while (f >> key >> kb >> unit) {
    if (key == "MemAvailable:") return kb * 1024.0;
  }
  return 0.0;
}

double processResidentBytes()
{
  std::ifstream f("/proc/self/statm");
  double size = 0.0, resident = 0.0;
  f >> size >> resident;
  return resident * static_cast<double>(sysconf(_SC_PAGESIZE));
}

}  // namespace batchpf
}  // namespace gridpack
