/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   settings.cpp
 * @date   2026-10-05
 *
 * @brief Settings resolver for the GPU batch path (block B14). See
 * settings.hpp; the keys, values and defaults are those of appendix G of
 * the architecture guide.
 */

#include "settings.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <set>
#include <sstream>
#include <utility>

#include "gridpack/configuration/configuration.hpp"

namespace gridpack {
namespace batchpf {

namespace {

const char *const kGpuPath = "Configuration.Contingency_analysis.GPUBatch";
const char *const kExecPath = "Configuration.Contingency_analysis.Execution";
using Cursor = gridpack::utility::Configuration::CursorPtr;

std::string trim(const std::string &in)
{
  const std::size_t a = in.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return std::string();
  const std::size_t b = in.find_last_not_of(" \t\r\n");
  return in.substr(a, b - a + 1);
}

std::string lower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

/// Raw text of a key, or false if absent
bool text(const Cursor &c, const std::string &key, std::string *out)
{
  if (!c) return false;
  std::string v;
  if (!c->get(key, &v)) return false;
  *out = trim(v);
  return true;
}

[[noreturn]] void bad(const std::string &block, const std::string &key,
                      const std::string &value, const std::string &expected)
{
  throw SettingsError("Contingency_analysis/" + block + "/" + key + " must be " +
                      expected + ", not '" + value + "'");
}

/// Enumerated value: one of the listed lower-case names
template <class T>
void readEnum(const Cursor &c, const std::string &block, const std::string &key,
              const std::vector<std::pair<std::string, T>> &choices,
              Setting<T> *out)
{
  std::string v;
  if (!text(c, key, &v)) return;
  const std::string lv = lower(v);
  std::string list;
  for (const auto &ch : choices) {
    if (lv == ch.first) {
      out->value = ch.second;
      out->source = Source::Xml;
      return;
    }
    list += (list.empty() ? "" : ", ") + ch.first;
  }
  bad(block, key, v, "one of: " + list);
}

void readBool(const Cursor &c, const std::string &block, const std::string &key,
              Setting<bool> *out)
{
  readEnum<bool>(c, block, key, {{"true", true}, {"false", false}}, out);
}

/// Integer, optionally allowing "auto" (stored as auto_value)
void readInt(const Cursor &c, const std::string &block, const std::string &key,
             long lo, long hi, bool allow_auto, int auto_value, Setting<int> *out)
{
  std::string v;
  if (!text(c, key, &v)) return;
  if (allow_auto && lower(v) == "auto") {
    out->value = auto_value;
    out->source = Source::Xml;
    return;
  }
  char *end = nullptr;
  errno = 0;
  const long x = std::strtol(v.c_str(), &end, 10);
  if (v.empty() || *end != '\0' || errno != 0 || x < lo || x > hi) {
    std::ostringstream os;
    os << (allow_auto ? "'auto' or " : "") << "an integer from " << lo << " to " << hi;
    bad(block, key, v, os.str());
  }
  out->value = static_cast<int>(x);
  out->source = Source::Xml;
}

void readDouble(const Cursor &c, const std::string &block, const std::string &key,
                double lo, double hi, Setting<double> *out)
{
  std::string v;
  if (!text(c, key, &v)) return;
  char *end = nullptr;
  errno = 0;
  const double x = std::strtod(v.c_str(), &end);
  if (v.empty() || *end != '\0' || errno != 0 || !(x >= lo && x <= hi)) {
    std::ostringstream os;
    os << "a number from " << lo << " to " << hi;
    bad(block, key, v, os.str());
  }
  out->value = x;
  out->source = Source::Xml;
}

/// Reject keys that are not in the list (catches typos, RT-5)
void checkKeys(const Cursor &c, const std::string &block,
               const std::set<std::string> &known)
{
  if (!c) return;
  gridpack::utility::Configuration::ChildElements kids;
  c->children(kids);
  for (const auto &k : kids) {
    if (k.name == "<xmlcomment>" || k.name == "<xmlattr>") continue;
    if (known.count(k.name) == 0) {
      std::string list;
      for (const std::string &s : known) list += (list.empty() ? "" : ", ") + s;
      throw SettingsError("unknown setting Contingency_analysis/" + block + "/" +
                          k.name + " (known: " + list + ")");
    }
  }
}

std::string envValue(const char *name)
{
  const char *v = std::getenv(name);
  return v ? std::string(v) : std::string();
}

}  // namespace

const char *sourceName(Source s)
{
  switch (s) {
    case Source::Xml: return "configuration.xml";
    case Source::Environment: return "environment";
    case Source::Detected: return "detected";
    default: return "default";
  }
}

ResolvedSettings resolveSettings(gridpack::utility::Configuration *config)
{
  ResolvedSettings r;
  GpuBatchSettings &g = r.gpu;
  ExecutionSettings &e = r.exec;
  const Cursor gc = config->getCursor(kGpuPath);
  const Cursor ec = config->getCursor(kExecPath);
  const std::string G = "GPUBatch";
  const std::string E = "Execution";

  // Block absent: accelerator off, stock GridPACK (RT-4)
  g.block_present = static_cast<bool>(gc);
  if (g.block_present) g.enabled = {Enabled::Auto, Source::Default};
  if (gc) {
    checkKeys(gc, G, {"enabled", "onUnavailable", "pluginPath", "backend",
                      "device", "batchSize", "maxValidatedBatch", "backfill",
                      "threadsPerBlock", "memoryProfile", "memoryHeadroomGB",
                      "maxMemoryGB", "formulation", "plannerOrdering",
                      "pivotTolerance", "solvePlacement", "warmStart",
                      "refinementSteps", "health", "shadowFraction",
                      "telemetry", "profilerRanges"});
    readEnum<Enabled>(gc, G, "enabled",
                      {{"auto", Enabled::Auto}, {"on", Enabled::On}, {"off", Enabled::Off}},
                      &g.enabled);
    readEnum<OnUnavailable>(gc, G, "onUnavailable",
                            {{"fallback", OnUnavailable::Fallback},
                             {"error", OnUnavailable::Error}},
                            &g.on_unavailable);
    std::string path;
    if (text(gc, "pluginPath", &path)) g.plugin_path = {path, Source::Xml};
    readEnum<int>(gc, G, "backend",
                  {{"auto", BATCHPF_BACKEND_AUTO}, {"cudss", BATCHPF_BACKEND_CUDSS},
                   {"alg2", BATCHPF_BACKEND_ALG2},
                   {"cpu_reference", BATCHPF_BACKEND_CPU_REFERENCE}},
                  &g.backend);
    readInt(gc, G, "device", 0, 1023, false, 0, &g.device);
    readInt(gc, G, "batchSize", 1, 65536, true, 0, &g.batch_size);
    readInt(gc, G, "maxValidatedBatch", 1, 65536, false, 0, &g.max_validated_batch);
    readBool(gc, G, "backfill", &g.backfill);
    readInt(gc, G, "threadsPerBlock", 32, 1024, true, 0, &g.threads_per_block);
    if (g.threads_per_block.value % 32 != 0) {
      bad(G, "threadsPerBlock", std::to_string(g.threads_per_block.value),
          "'auto' or a multiple of 32");
    }
    readEnum<int>(gc, G, "memoryProfile",
                  {{"auto", BATCHPF_MEMORY_AUTO}, {"unified", BATCHPF_MEMORY_UNIFIED},
                   {"coherent", BATCHPF_MEMORY_COHERENT},
                   {"discrete", BATCHPF_MEMORY_DISCRETE}},
                  &g.memory_profile);
    readDouble(gc, G, "memoryHeadroomGB", 0.0, 1.0e6, &g.memory_headroom_gb);
    readDouble(gc, G, "maxMemoryGB", 0.0, 1.0e6, &g.max_memory_gb);
    std::string form;
    if (text(gc, "formulation", &form)) {
      const std::string lf = lower(form);
      if (lf == "grouped") {
        throw SettingsError("Contingency_analysis/GPUBatch/formulation 'grouped' is "
                            "reserved and not available in this version; use "
                            "'superset'");
      }
      if (lf != "superset") bad(G, "formulation", form, "'superset'");
      g.formulation = {lf, Source::Xml};
    }
    readEnum<int>(gc, G, "plannerOrdering",
                  {{"amd", BATCHPF_ORDERING_AMD}, {"colamd", BATCHPF_ORDERING_COLAMD}},
                  &g.planner_ordering);
    readDouble(gc, G, "pivotTolerance", 0.0, 1.0, &g.pivot_tolerance);
    readEnum<int>(gc, G, "solvePlacement",
                  {{"gpu", BATCHPF_SOLVE_GPU}, {"host", BATCHPF_SOLVE_HOST}},
                  &g.solve_placement);
    readEnum<int>(gc, G, "warmStart",
                  {{"base_case", BATCHPF_WARM_START_BASE_CASE},
                   {"raw", BATCHPF_WARM_START_RAW}},
                  &g.warm_start);
    readInt(gc, G, "refinementSteps", 0, 100, false, 0, &g.refinement_steps);
    const Cursor hc = gc->getCursor("health");
    if (hc) {
      const std::string H = "GPUBatch/health";
      checkKeys(hc, H, {"residualLimit", "pivotLimit", "checkNonFinite"});
      readDouble(hc, H, "residualLimit", 0.0, 1.0, &g.residual_limit);
      readDouble(hc, H, "pivotLimit", 0.0, 1.0, &g.pivot_limit);
      readBool(hc, H, "checkNonFinite", &g.check_nonfinite);
    }
    readDouble(gc, G, "shadowFraction", 0.0, 1.0, &g.shadow_fraction);
    readEnum<int>(gc, G, "telemetry",
                  {{"off", BATCHPF_TELEMETRY_OFF}, {"summary", BATCHPF_TELEMETRY_SUMMARY},
                   {"detailed", BATCHPF_TELEMETRY_DETAILED}},
                  &g.telemetry);
    readBool(gc, G, "profilerRanges", &g.profiler_ranges);
  }
  // Plugin location: XML, then the environment, then next to ca.x
  if (g.plugin_path.source == Source::Default) {
    const std::string env = envValue("GRIDPACK_BATCHPF_PLUGIN_PATH");
    if (!env.empty()) g.plugin_path = {env, Source::Environment};
  }

  if (ec) {
    checkKeys(ec, E, {"acceleratorRanks", "cpuBinding", "logLevel"});
    std::string ranks;
    if (text(ec, "acceleratorRanks", &ranks) && lower(ranks) != "auto") {
      std::vector<int> list;
      std::string tok;
      std::istringstream is(ranks);
      while (std::getline(is, tok, ',')) {
        std::istringstream ws(tok);
        std::string piece;
        while (ws >> piece) {
          char *end = nullptr;
          const long x = std::strtol(piece.c_str(), &end, 10);
          if (*end != '\0' || x < 0 || x > 1000000) {
            bad(E, "acceleratorRanks", ranks,
                "'auto' or a list of rank numbers separated by commas or spaces");
          }
          list.push_back(static_cast<int>(x));
        }
      }
      if (list.empty()) bad(E, "acceleratorRanks", ranks, "'auto' or a list of ranks");
      std::sort(list.begin(), list.end());
      list.erase(std::unique(list.begin(), list.end()), list.end());
      e.accelerator_ranks = {list, Source::Xml};
    } else if (!ranks.empty()) {
      e.accelerator_ranks.source = Source::Xml;
    }
    readEnum<CpuBinding>(ec, E, "cpuBinding",
                         {{"auto", CpuBinding::Auto}, {"none", CpuBinding::None},
                          {"performance_first", CpuBinding::PerformanceFirst}},
                         &e.cpu_binding);
    readEnum<int>(ec, E, "logLevel",
                  {{"error", BATCHPF_LOG_ERROR}, {"warn", BATCHPF_LOG_WARN},
                   {"info", BATCHPF_LOG_INFO}, {"debug", BATCHPF_LOG_DEBUG}},
                  &e.log_level);
  }
  r.cuda_visible_devices = envValue("CUDA_VISIBLE_DEVICES");
  r.jit_cache_path = envValue("CUDA_CACHE_PATH");
  return r;
}

std::vector<std::string> describeSettings(const ResolvedSettings &s)
{
  const GpuBatchSettings &g = s.gpu;
  const ExecutionSettings &e = s.exec;
  std::vector<std::string> out;
  auto line = [&](const std::string &key, const std::string &value, Source src) {
    out.push_back(key + " = " + value + "  [" + sourceName(src) + "]");
  };
  auto num = [](double x) {
    std::ostringstream os;
    os << x;
    return os.str();
  };
  auto autoInt = [](int v) { return v == 0 ? std::string("auto") : std::to_string(v); };
  static const std::array<const char *, 4> backends = {"auto", "cudss", "alg2", "cpu_reference"};
  static const std::array<const char *, 4> profiles = {"auto", "unified", "coherent", "discrete"};
  static const std::array<const char *, 3> tele = {"off", "summary", "detailed"};
  static const std::array<const char *, 4> levels = {"error", "warn", "info", "debug"};
  line("GPUBatch/enabled",
       g.enabled.value == Enabled::On ? "on" : g.enabled.value == Enabled::Auto ? "auto" : "off",
       g.enabled.source);
  if (g.enabled.value == Enabled::Off) return out;
  line("GPUBatch/onUnavailable",
       g.on_unavailable.value == OnUnavailable::Error ? "error" : "fallback",
       g.on_unavailable.source);
  line("GPUBatch/pluginPath",
       g.plugin_path.value.empty() ? "(next to ca.x)" : g.plugin_path.value,
       g.plugin_path.source);
  line("GPUBatch/backend", backends.at(g.backend.value), g.backend.source);
  line("GPUBatch/device", std::to_string(g.device.value), g.device.source);
  line("GPUBatch/batchSize", autoInt(g.batch_size.value), g.batch_size.source);
  line("GPUBatch/maxValidatedBatch",
       g.max_validated_batch.value == 0 ? "backend default"
                                        : std::to_string(g.max_validated_batch.value),
       g.max_validated_batch.source);
  line("GPUBatch/backfill", g.backfill.value ? "true" : "false", g.backfill.source);
  line("GPUBatch/threadsPerBlock", autoInt(g.threads_per_block.value),
       g.threads_per_block.source);
  line("GPUBatch/memoryProfile", profiles.at(g.memory_profile.value), g.memory_profile.source);
  line("GPUBatch/memoryHeadroomGB",
       g.memory_headroom_gb.value < 0.0 ? "by memory profile" : num(g.memory_headroom_gb.value),
       g.memory_headroom_gb.source);
  line("GPUBatch/maxMemoryGB",
       g.max_memory_gb.value > 0.0 ? num(g.max_memory_gb.value) : "none",
       g.max_memory_gb.source);
  line("GPUBatch/formulation", g.formulation.value, g.formulation.source);
  line("GPUBatch/plannerOrdering",
       g.planner_ordering.value == BATCHPF_ORDERING_COLAMD ? "colamd" : "amd",
       g.planner_ordering.source);
  line("GPUBatch/pivotTolerance", num(g.pivot_tolerance.value), g.pivot_tolerance.source);
  line("GPUBatch/solvePlacement",
       g.solve_placement.value == BATCHPF_SOLVE_HOST ? "host" : "gpu",
       g.solve_placement.source);
  line("GPUBatch/warmStart",
       g.warm_start.value == BATCHPF_WARM_START_RAW ? "raw" : "base_case",
       g.warm_start.source);
  line("GPUBatch/refinementSteps", std::to_string(g.refinement_steps.value),
       g.refinement_steps.source);
  line("GPUBatch/health/residualLimit", num(g.residual_limit.value), g.residual_limit.source);
  line("GPUBatch/health/pivotLimit", num(g.pivot_limit.value), g.pivot_limit.source);
  line("GPUBatch/health/checkNonFinite", g.check_nonfinite.value ? "true" : "false",
       g.check_nonfinite.source);
  line("GPUBatch/shadowFraction", num(g.shadow_fraction.value), g.shadow_fraction.source);
  line("GPUBatch/telemetry", tele.at(g.telemetry.value), g.telemetry.source);
  line("GPUBatch/profilerRanges", g.profiler_ranges.value ? "true" : "false",
       g.profiler_ranges.source);
  std::string ranks;
  for (int x : e.accelerator_ranks.value) ranks += (ranks.empty() ? "" : ",") + std::to_string(x);
  line("Execution/acceleratorRanks", ranks.empty() ? "auto" : ranks,
       e.accelerator_ranks.source);
  line("Execution/cpuBinding",
       e.cpu_binding.value == CpuBinding::None ? "none"
           : e.cpu_binding.value == CpuBinding::PerformanceFirst ? "performance_first"
                                                                 : "auto",
       e.cpu_binding.source);
  line("Execution/logLevel", levels.at(e.log_level.value), e.log_level.source);
  line("CUDA_VISIBLE_DEVICES",
       s.cuda_visible_devices.empty() ? "(not set)" : s.cuda_visible_devices,
       s.cuda_visible_devices.empty() ? Source::Default : Source::Environment);
  if (!s.jit_cache_path.empty()) {
    line("CUDA_CACHE_PATH", s.jit_cache_path, Source::Environment);
  }
  return out;
}

}  // namespace batchpf
}  // namespace gridpack
