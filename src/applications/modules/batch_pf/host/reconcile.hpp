/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   reconcile.hpp
 * @date   2026-10-05
 *
 * @brief Reconciler (block B11.4): result files in event order.
 *
 * GridPACK's driver concatenates the per-rank part files of csv_flat,
 * csv_delta and the violations table rank by rank, so rows appear in the
 * order cases finished. With the batch path that order depends on batch
 * timing as well, so the guide asks for final files sorted by event index
 * (scenario R5, step 3). The convergence table is already sorted by the
 * driver.
 */

#ifndef GRIDPACK_BATCHPF_HOST_RECONCILE_HPP
#define GRIDPACK_BATCHPF_HOST_RECONCILE_HPP

#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

namespace gridpack {
namespace batchpf {

/**
 * Append the rows of the given part files to out, ordered by event index
 * (the integer in each row's first column). Rows of the same event keep
 * their order. Missing part files are skipped; the part files are left in
 * place for the caller to remove.
 *
 * Each rank writes all rows of a case together, so a part file is a series
 * of runs of rows with one event index. The runs are found in one pass and
 * then copied in event order, so memory grows with the number of cases, not
 * with the number of rows (a csv_delta study can have hundreds of millions).
 *
 * @return number of rows written
 */
std::size_t appendPartsByEvent(const std::vector<std::string> &parts, std::ostream &out);

/// Consecutive rows of one part file that belong to one event
struct PartRun {
  long event = 0;
  int part = 0;
  std::streamoff offset = 0;
  std::streamoff length = 0;
  bool add_newline = false;   // the file's last row has no newline; one is added
};

/**
 * The runs appendPartsByEvent() finds in one part file, in file order, and
 * the number of rows in them. A missing file has none. Copying the runs of
 * all parts in event order (a stable sort of the parts' runs, taken in part
 * order) gives exactly appendPartsByEvent()'s output; the driver's parallel
 * writer does that from every rank at once.
 */
std::vector<PartRun> scanPartRuns(const std::string &path, int part, std::size_t *rows);

struct OutcomeCoverage {
  std::vector<int> missing;
  std::vector<int> duplicates;
  std::vector<int> unexpected;
  bool complete() const noexcept
  {
    return missing.empty() && duplicates.empty() && unexpected.empty();
  }
};

// Event zero is the base case. Contingencies have indices 1..expected.
OutcomeCoverage checkOutcomeCoverage(int expected, const std::vector<int> &indices);

}  // namespace batchpf
}  // namespace gridpack

#endif
