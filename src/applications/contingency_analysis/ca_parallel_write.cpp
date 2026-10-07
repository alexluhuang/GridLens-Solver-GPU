/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   ca_parallel_write.cpp
 * @date   2026-10-07
 *
 * @brief Merging per-rank part files from all ranks at once (see
 * ca_parallel_write.hpp).
 */

#include "ca_parallel_write.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <vector>

#include "gridpack/applications/modules/batch_pf/host/reconcile.hpp"

namespace gridpack {
namespace contingency_analysis {

namespace {

[[noreturn]] void fail(MPI_Comm comm, const std::string &what)
{
  std::fprintf(stderr, "ERROR: writing results: %s: %s\n", what.c_str(), std::strerror(errno));
  MPI_Abort(comm, 1);
  std::abort();   // MPI_Abort does not return
}

/// Write all of buf at offset
void writeAt(MPI_Comm comm, int fd, const char *buf, std::size_t len, off_t offset,
             const std::string &name)
{
  while (len > 0) {
    const ssize_t n = ::pwrite(fd, buf, len, offset);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) fail(comm, "write " + name);
    buf += n;
    len -= static_cast<std::size_t>(n);
    offset += n;
  }
}

/// One run of rows in the final file, in the order of the final file
struct Entry {
  long long event = 0;
  int rank = 0;
  int index = 0;          // run number within its rank's part
  long long length = 0;   // bytes written, including an added newline
};

}  // namespace

long long writePartsInParallel(MPI_Comm comm, const std::string &out_file,
                               const std::string &header, const std::string &part,
                               bool by_event)
{
  int rank = 0, size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // This rank's runs: per case when ordering by event, else the whole part
  std::vector<batchpf::PartRun> runs;
  std::size_t scanned_rows = 0;
  if (by_event) {
    runs = batchpf::scanPartRuns(part, rank, &scanned_rows);
  } else {
    struct stat st {};
    if (::stat(part.c_str(), &st) == 0 && st.st_size > 0) {
      batchpf::PartRun r;
      r.part = rank;
      r.length = st.st_size;
      runs.push_back(r);
    }
  }

  // Exchange (event, length) of every run; every rank lays out the file
  std::vector<long long> mine;
  for (const batchpf::PartRun &r : runs) {
    mine.push_back(r.event);
    mine.push_back(r.length + (r.add_newline ? 1 : 0));
  }
  int nmine = static_cast<int>(mine.size());
  std::vector<int> counts(size), displs(size, 0);
  MPI_Allgather(&nmine, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
  for (int p = 1; p < size; p++) displs[p] = displs[p - 1] + counts[p - 1];
  std::vector<long long> all(static_cast<std::size_t>(displs.back()) + counts.back());
  MPI_Allgatherv(mine.data(), nmine, MPI_LONG_LONG, all.data(), counts.data(),
                 displs.data(), MPI_LONG_LONG, comm);
  std::vector<Entry> layout;
  for (int p = 0; p < size; p++) {
    for (int k = 0; k < counts[p] / 2; k++) {
      Entry e;
      e.event = all[displs[p] + 2 * k];
      e.length = all[displs[p] + 2 * k + 1];
      e.rank = p;
      e.index = k;
      layout.push_back(e);
    }
  }
  // Rank order, then file order; by event, a stable sort keeps that order
  // among the runs of one event, as appendPartsByEvent() does
  if (by_event) {
    std::stable_sort(layout.begin(), layout.end(),
                     [](const Entry &a, const Entry &b) { return a.event < b.event; });
  }
  std::vector<off_t> where(runs.size(), 0);
  off_t total = static_cast<off_t>(header.size());
  for (const Entry &e : layout) {
    if (e.rank == rank) where[e.index] = total;
    total += e.length;
  }

  // Rank 0 creates the file at its final size, then every rank fills in its runs
  if (rank == 0) {
    const int fd = ::open(out_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (fd < 0) fail(comm, "create " + out_file);
    writeAt(comm, fd, header.data(), header.size(), 0, out_file);
    if (::ftruncate(fd, total) != 0) fail(comm, "size " + out_file);
    if (::close(fd) != 0) fail(comm, "close " + out_file);
  }
  MPI_Barrier(comm);
  long long newlines = 0;
  if (!runs.empty()) {
    const int in = ::open(part.c_str(), O_RDONLY);
    if (in < 0) fail(comm, "open " + part);
    const int out = ::open(out_file.c_str(), O_WRONLY);
    if (out < 0) fail(comm, "open " + out_file);
    std::vector<char> buf(8 << 20);
    for (std::size_t i = 0; i < runs.size(); i++) {
      off_t src = runs[i].offset, dst = where[i];
      long long left = runs[i].length;
      while (left > 0) {
        const std::size_t want =
            static_cast<std::size_t>(std::min<long long>(left, static_cast<long long>(buf.size())));
        const ssize_t got = ::pread(in, buf.data(), want, src);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) fail(comm, "read " + part);
        if (!by_event) newlines += std::count(buf.data(), buf.data() + got, '\n');
        writeAt(comm, out, buf.data(), static_cast<std::size_t>(got), dst, out_file);
        src += got;
        dst += got;
        left -= got;
      }
      if (runs[i].add_newline) writeAt(comm, out, "\n", 1, dst, out_file);
    }
    if (::close(in) != 0) fail(comm, "close " + part);
    if (::close(out) != 0) fail(comm, "close " + out_file);
  }
  std::remove(part.c_str());
  long long rows = by_event ? static_cast<long long>(scanned_rows) : newlines;
  // Every rank adds its count after its copies are closed, so the file is
  // complete when this returns
  MPI_Allreduce(MPI_IN_PLACE, &rows, 1, MPI_LONG_LONG, MPI_SUM, comm);
  return rows;
}

}  // namespace contingency_analysis
}  // namespace gridpack
