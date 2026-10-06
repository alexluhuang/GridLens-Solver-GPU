/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   ca_parquet.hpp
 * @date   2026-10-06
 *
 * @brief outputFormat=parquet: the csv_delta branch table as two columnar
 * Apache Parquet files.
 *
 * csv_delta repeats about twenty base-case columns on every row and writes
 * every number as text. The parquet format keeps the same rows and values
 * in two tables:
 *
 *   <outputFile>_branches.parquet  one row per monitored branch: identity,
 *                                  ratings and base-case values
 *   <outputFile>_flows/part-NNNNN.parquet
 *                                  one row per (case, branch): the
 *                                  contingency values. Each file holds a
 *                                  consecutive range of events, so reading
 *                                  the files in name order gives event order
 *                                  (pyarrow, Polars, DuckDB and Spark read
 *                                  the directory as one table).
 *
 * Joining the two on branch_id and the contingency table on event_idx gives
 * csv_delta's rows; the delta columns are differences of stored columns.
 * Values are the doubles csv_delta prints, so rounding them as csv_delta
 * does reproduces its text. The other outputs (convergence, violations,
 * contingencies, buses, summary) are those of csv_delta.
 *
 * Each rank writes its cases to an uncompressed part file, one row group per
 * case. At the end every rank writes one consecutive range of events from
 * all parts, compressing in parallel (guide scenario R5: event order). A
 * single-file merge on one rank took 20-40 s on 7k-10k-bus grids, most of
 * it compression; the parallel write removes that. Parquet
 * support is optional at build time (Apache Arrow's C++ Parquet library);
 * without it, available() is false and the driver rejects the format at
 * start-up. This header has no Arrow types, so the driver needs no Arrow
 * headers.
 */

#ifndef CA_PARQUET_HPP_
#define CA_PARQUET_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gridpack {
namespace contingency_analysis {

/// Identity, ratings and base-case values of one monitored branch
struct ParquetBranchRow {
  int32_t branch_id = 0;
  int32_t from_bus = 0;
  int32_t to_bus = 0;
  std::string ckt;
  double base_kv_from = 0.0;
  double base_kv_to = 0.0;
  int32_t area_from = 0;
  int32_t area_to = 0;
  double base_rate_mva = 0.0;
  double cont_rate_mva = 0.0;
  double base_p_mw = 0.0;
  double base_q_mvar = 0.0;
  double base_mva = 0.0;
  double base_loading_pct = 0.0;
  double v_from_base = 0.0;
  double v_to_base = 0.0;
  double ang_from_base = 0.0;
  double ang_to_base = 0.0;
};

/// Writer of one rank's flows part file, and of one range of the final table
class ParquetFlows {
 public:
  /// True if this ca.x was built with Parquet support
  static bool available();

  /// Write <path> with one row per branch (call on one rank)
  static void writeBranches(const std::string &path,
                            const std::vector<ParquetBranchRow> &rows);

  /// One case's rows in a part file
  struct RowGroup {
    int32_t event_idx = 0;
    int32_t part = 0;    // index into the parts list (the writing rank)
    int32_t index = 0;   // row group number in that part
    int64_t rows = 0;
  };

  /**
   * Write the listed row groups, in the given order, to <path> (one rank's
   * share of the final table). Row groups are combined into larger ones.
   * @return rows written
   */
  static int64_t writeRange(const std::vector<std::string> &parts,
                            const std::vector<RowGroup> &groups,
                            const std::string &path);

  /// Row groups written to this rank's part so far (part index set to rank)
  const std::vector<RowGroup> &rowGroups() const;

  /// Open the part file of this rank (created on the first case)
  ParquetFlows(const std::string &part_path, int rank);
  ~ParquetFlows();
  ParquetFlows(const ParquetFlows &) = delete;
  ParquetFlows &operator=(const ParquetFlows &) = delete;

  /// Append one row of the current case
  void add(int32_t event_idx, int32_t branch_id, double p_mw, double q_mvar,
           double mva, double loading_pct, double v_from, double v_to,
           double ang_from, double ang_to);

  /// Write the current case's rows as one row group
  void endCase();

  /// Finish the part file
  void close();

 private:
  struct Impl;
  std::unique_ptr<Impl> p_impl;
};

}  // namespace contingency_analysis
}  // namespace gridpack

#endif
