/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   ca_parallel_write.hpp
 * @date   2026-10-07
 *
 * @brief Merging the per-rank part files of a result table from all ranks
 * at once.
 *
 * During the case loop every rank streams its rows (csv_flat, csv_delta,
 * violations) to its own part file. The final table used to be written by
 * rank 0 alone, which read and wrote every byte: 8.7 s of a 42 s ACTIVSg10k
 * study for a 17 GB csv_flat table. Here each rank finds the runs of rows of
 * one case in its own part, the ranks exchange only the run sizes (a few
 * numbers per case), and every rank computes the same layout of the final
 * file. Each rank then copies its own runs to their offsets in the final
 * file, so the copying is spread over all ranks.
 *
 * The bytes are those of the single-rank merge: rank order then file order
 * (the CPU path), or event order with ties kept in rank and file order (the
 * batch path, batchpf::appendPartsByEvent()).
 */

#ifndef CA_PARALLEL_WRITE_HPP_
#define CA_PARALLEL_WRITE_HPP_

#include <mpi.h>

#include <string>

namespace gridpack {
namespace contingency_analysis {

/**
 * Write out_file = header + the rows of every rank's part file, and remove
 * the part files. Collective over comm.
 * @param part this rank's part file (it may not exist: no rows)
 * @param by_event order rows by event index (the first column) instead of
 *        by rank
 * @return the number of rows, counted as the single-rank merge counted them
 *         (non-blank rows by event, newline characters by rank), on every rank
 */
long long writePartsInParallel(MPI_Comm comm, const std::string &out_file,
                               const std::string &header, const std::string &part,
                               bool by_event);

}  // namespace contingency_analysis
}  // namespace gridpack

#endif
