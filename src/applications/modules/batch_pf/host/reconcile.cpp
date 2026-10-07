/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   reconcile.cpp
 * @date   2026-10-05
 *
 * @brief Reconciler (block B11.4): result files in event order.
 */

#include "reconcile.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <memory>

namespace gridpack {
namespace batchpf {

OutcomeCoverage checkOutcomeCoverage(int expected, const std::vector<int> &indices)
{
  OutcomeCoverage result;
  std::vector<int> seen(expected + 1, 0);
  for (int event : indices) {
    if (event == 0) continue;
    if (event < 1 || event > expected) {
      result.unexpected.push_back(event);
    } else {
      seen[event]++;
    }
  }
  for (int event = 1; event <= expected; event++) {
    if (seen[event] == 0) result.missing.push_back(event);
    if (seen[event] > 1) result.duplicates.push_back(event);
  }
  return result;
}

std::vector<PartRun> scanPartRuns(const std::string &path, int part, std::size_t *rows)
{
  std::vector<PartRun> runs;
  std::ifstream in(path.c_str(), std::ios::in | std::ios::binary);
  if (!in) return runs;
  std::string line;
  std::streamoff pos = 0;
  bool unterminated = false;
  while (std::getline(in, line)) {
    // getline drops the newline; a last row without one ends the file
    unterminated = in.eof();
    const std::streamoff size =
        static_cast<std::streamoff>(line.size()) + (unterminated ? 0 : 1);
    if (!line.empty()) {
      char *end = nullptr;
      long event = std::strtol(line.c_str(), &end, 10);
      // a row without an event index stays with the rows before it
      if (end == line.c_str() && !runs.empty()) event = runs.back().event;
      if (!runs.empty() && runs.back().event == event) {
        runs.back().length += size;
      } else {
        PartRun r;
        r.event = event;
        r.part = part;
        r.offset = pos;
        r.length = size;
        runs.push_back(r);
      }
      (*rows)++;
    } else if (!runs.empty()) {
      runs.back().length += size;   // keep blank lines where they were
    }
    pos += size;
  }
  // The last run reaches the end of the file
  if (unterminated && !runs.empty()) runs.back().add_newline = true;
  return runs;
}

std::size_t appendPartsByEvent(const std::vector<std::string> &parts, std::ostream &out)
{
  std::vector<PartRun> runs;
  std::size_t rows = 0;
  const int nparts = static_cast<int>(parts.size());
  for (int p = 0; p < nparts; p++) {
    const std::vector<PartRun> mine = scanPartRuns(parts[p], p, &rows);
    runs.insert(runs.end(), mine.begin(), mine.end());
  }
  std::stable_sort(runs.begin(), runs.end(),
                   [](const PartRun &a, const PartRun &b) { return a.event < b.event; });

  std::vector<std::unique_ptr<std::ifstream>> files(nparts);
  std::vector<char> buf(1 << 20);
  for (const PartRun &r : runs) {
    std::unique_ptr<std::ifstream> &in = files[r.part];
    if (!in) {
      in = std::make_unique<std::ifstream>(parts[r.part].c_str(),
                                           std::ios::in | std::ios::binary);
    }
    in->clear();
    in->seekg(r.offset);
    std::streamoff left = r.length;
    char last = '\n';
    while (left > 0 && *in) {
      const std::streamsize want =
          static_cast<std::streamsize>(std::min(left, static_cast<std::streamoff>(buf.size())));
      in->read(buf.data(), want);
      const std::streamsize got = in->gcount();
      if (got <= 0) break;
      out.write(buf.data(), got);
      last = buf[got - 1];
      left -= got;
    }
    if (last != '\n') out.put('\n');   // the file's last row had no newline
  }
  return rows;
}

}  // namespace batchpf
}  // namespace gridpack
