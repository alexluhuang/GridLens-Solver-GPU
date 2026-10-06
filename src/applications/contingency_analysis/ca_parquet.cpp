/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   ca_parquet.cpp
 * @date   2026-10-06
 *
 * @brief outputFormat=parquet writer (see ca_parquet.hpp).
 */

#include "ca_parquet.hpp"

#include <stdexcept>

#ifdef GRIDPACK_CA_HAVE_PARQUET

#include <algorithm>
#include <cstdio>
#include <utility>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/util/compression.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/properties.h>

namespace gridpack {
namespace contingency_analysis {

namespace {

/// Merged files are written in row groups of about this many rows
const int64_t kMergedRowGroup = 1 << 20;

void check(const arrow::Status &s, const std::string &what)
{
  if (!s.ok()) throw std::runtime_error("parquet output: " + what + ": " + s.ToString());
}

template <class T>
T value(arrow::Result<T> r, const std::string &what)
{
  check(r.status(), what);
  return std::move(r).ValueUnsafe();
}

/// Final files: zstd where this Arrow build has it, else snappy, else none.
/// Part files are temporary and read back once, so they stay uncompressed.
/// Dictionary encoding only pays for the integer key columns; for unique
/// doubles it is built and then abandoned, which made writing 50% slower.
std::shared_ptr<parquet::WriterProperties> writerProperties(bool compress)
{
  parquet::WriterProperties::Builder b;
  b.disable_dictionary();
  if (!compress) {
    b.compression(parquet::Compression::UNCOMPRESSED);
  } else if (arrow::util::Codec::IsAvailable(arrow::Compression::ZSTD)) {
    b.compression(parquet::Compression::ZSTD);
  } else if (arrow::util::Codec::IsAvailable(arrow::Compression::SNAPPY)) {
    b.compression(parquet::Compression::SNAPPY);
  }
  return b.build();
}

std::shared_ptr<arrow::Schema> flowsSchema()
{
  return arrow::schema({
      arrow::field("event_idx", arrow::int32(), false),
      arrow::field("branch_id", arrow::int32(), false),
      arrow::field("cont_p_mw", arrow::float64(), false),
      arrow::field("cont_q_mvar", arrow::float64(), false),
      arrow::field("cont_mva", arrow::float64(), false),
      arrow::field("cont_loading_pct", arrow::float64(), false),
      arrow::field("v_from_cont", arrow::float64(), false),
      arrow::field("v_to_cont", arrow::float64(), false),
      arrow::field("ang_from_cont", arrow::float64(), false),
      arrow::field("ang_to_cont", arrow::float64(), false)});
}

std::shared_ptr<arrow::Array> int32Array(const std::vector<int32_t> &v)
{
  arrow::Int32Builder b;
  check(b.AppendValues(v.data(), static_cast<int64_t>(v.size())), "int32 column");
  return value(b.Finish(), "int32 column");
}

std::shared_ptr<arrow::Array> doubleArray(const std::vector<double> &v)
{
  arrow::DoubleBuilder b;
  check(b.AppendValues(v.data(), static_cast<int64_t>(v.size())), "double column");
  return value(b.Finish(), "double column");
}

std::unique_ptr<parquet::arrow::FileWriter> openWriter(const std::string &path,
                                                       const arrow::Schema &schema,
                                                       bool compress)
{
  std::shared_ptr<arrow::io::FileOutputStream> out =
      value(arrow::io::FileOutputStream::Open(path), "open " + path);
  std::shared_ptr<parquet::ArrowWriterProperties> arrow_props =
      parquet::ArrowWriterProperties::Builder().build();
  return value(parquet::arrow::FileWriter::Open(schema, arrow::default_memory_pool(), out,
                                                writerProperties(compress), arrow_props),
               "start " + path);
}

}  // namespace

struct ParquetFlows::Impl {
  std::string path;
  int rank = 0;
  std::vector<RowGroup> groups;
  std::unique_ptr<parquet::arrow::FileWriter> writer;
  std::shared_ptr<arrow::Schema> schema = flowsSchema();
  std::vector<int32_t> event, branch;
  std::vector<double> col[8];
};

bool ParquetFlows::available() { return true; }

void ParquetFlows::writeBranches(const std::string &path,
                                 const std::vector<ParquetBranchRow> &rows)
{
  std::vector<int32_t> id, from, to, area_from, area_to;
  std::vector<double> d[12];
  arrow::StringBuilder ckt;
  for (const ParquetBranchRow &r : rows) {
    id.push_back(r.branch_id);
    from.push_back(r.from_bus);
    to.push_back(r.to_bus);
    check(ckt.Append(r.ckt), "ckt column");
    area_from.push_back(r.area_from);
    area_to.push_back(r.area_to);
    const double values[12] = {r.base_kv_from, r.base_kv_to, r.base_rate_mva, r.cont_rate_mva,
                               r.base_p_mw, r.base_q_mvar, r.base_mva, r.base_loading_pct,
                               r.v_from_base, r.v_to_base, r.ang_from_base, r.ang_to_base};
    for (int k = 0; k < 12; k++) d[k].push_back(values[k]);
  }
  std::shared_ptr<arrow::Schema> schema = arrow::schema({
      arrow::field("branch_id", arrow::int32(), false),
      arrow::field("from_bus", arrow::int32(), false),
      arrow::field("to_bus", arrow::int32(), false),
      arrow::field("ckt", arrow::utf8(), false),
      arrow::field("base_kv_from", arrow::float64(), false),
      arrow::field("base_kv_to", arrow::float64(), false),
      arrow::field("area_from", arrow::int32(), false),
      arrow::field("area_to", arrow::int32(), false),
      arrow::field("base_rate_mva", arrow::float64(), false),
      arrow::field("cont_rate_mva", arrow::float64(), false),
      arrow::field("base_p_mw", arrow::float64(), false),
      arrow::field("base_q_mvar", arrow::float64(), false),
      arrow::field("base_mva", arrow::float64(), false),
      arrow::field("base_loading_pct", arrow::float64(), false),
      arrow::field("v_from_base", arrow::float64(), false),
      arrow::field("v_to_base", arrow::float64(), false),
      arrow::field("ang_from_base", arrow::float64(), false),
      arrow::field("ang_to_base", arrow::float64(), false)});
  std::vector<std::shared_ptr<arrow::Array>> arrays = {
      int32Array(id), int32Array(from), int32Array(to), value(ckt.Finish(), "ckt column"),
      doubleArray(d[0]), doubleArray(d[1]), int32Array(area_from), int32Array(area_to)};
  for (int k = 2; k < 12; k++) arrays.push_back(doubleArray(d[k]));
  std::shared_ptr<arrow::Table> table = arrow::Table::Make(schema, arrays);
  std::unique_ptr<parquet::arrow::FileWriter> w = openWriter(path, *schema, true);
  check(w->WriteTable(*table, std::max<int64_t>(1, table->num_rows())), "write " + path);
  check(w->Close(), "close " + path);
}

ParquetFlows::ParquetFlows(const std::string &part_path, int rank)
    : p_impl(std::make_unique<Impl>())
{
  p_impl->path = part_path;
  p_impl->rank = rank;
}

const std::vector<ParquetFlows::RowGroup> &ParquetFlows::rowGroups() const
{
  return p_impl->groups;
}

ParquetFlows::~ParquetFlows()
{
  try {
    close();
  } catch (...) {
    // close() reports errors when called explicitly; a destructor must not throw
  }
}

void ParquetFlows::add(int32_t event_idx, int32_t branch_id, double p_mw, double q_mvar,
                       double mva, double loading_pct, double v_from, double v_to,
                       double ang_from, double ang_to)
{
  Impl &d = *p_impl;
  d.event.push_back(event_idx);
  d.branch.push_back(branch_id);
  const double values[8] = {p_mw, q_mvar, mva, loading_pct, v_from, v_to, ang_from, ang_to};
  for (int k = 0; k < 8; k++) d.col[k].push_back(values[k]);
}

void ParquetFlows::endCase()
{
  Impl &d = *p_impl;
  if (d.event.empty()) return;
  if (!d.writer) d.writer = openWriter(d.path, *d.schema, false);
  std::vector<std::shared_ptr<arrow::Array>> arrays = {int32Array(d.event), int32Array(d.branch)};
  for (int k = 0; k < 8; k++) arrays.push_back(doubleArray(d.col[k]));
  std::shared_ptr<arrow::Table> table = arrow::Table::Make(d.schema, arrays);
  // One row group per case, so cases can be ordered without reading rows
  check(d.writer->WriteTable(*table, table->num_rows()), "write " + d.path);
  RowGroup g;
  g.event_idx = d.event.front();
  g.part = d.rank;
  g.index = static_cast<int32_t>(d.groups.size());
  g.rows = table->num_rows();
  d.groups.push_back(g);
  d.event.clear();
  d.branch.clear();
  for (int k = 0; k < 8; k++) d.col[k].clear();
}

void ParquetFlows::close()
{
  Impl &d = *p_impl;
  endCase();
  if (d.writer) {
    check(d.writer->Close(), "close " + d.path);
    d.writer.reset();
  }
}

int64_t ParquetFlows::writeRange(const std::vector<std::string> &parts,
                                 const std::vector<RowGroup> &groups,
                                 const std::string &path)
{
  if (groups.empty()) return 0;
  std::vector<std::unique_ptr<parquet::arrow::FileReader>> readers(parts.size());
  std::shared_ptr<arrow::Schema> schema = flowsSchema();
  std::unique_ptr<parquet::arrow::FileWriter> w = openWriter(path, *schema, true);
  std::vector<std::shared_ptr<arrow::Table>> pending;
  int64_t pending_rows = 0, rows = 0;
  auto flush = [&]() {
    if (pending.empty()) return;
    std::shared_ptr<arrow::Table> t = value(arrow::ConcatenateTables(pending), "combine " + path);
    check(w->WriteTable(*t, std::max<int64_t>(1, t->num_rows())), "write " + path);
    pending.clear();
    pending_rows = 0;
  };
  for (const RowGroup &g : groups) {
    std::unique_ptr<parquet::arrow::FileReader> &r = readers.at(g.part);
    if (!r) {
      std::shared_ptr<arrow::io::ReadableFile> in =
          value(arrow::io::ReadableFile::Open(parts[g.part]), "open " + parts[g.part]);
      r = value(parquet::arrow::OpenFile(in, arrow::default_memory_pool()),
                "read " + parts[g.part]);
    }
    std::shared_ptr<arrow::Table> t = value(r->ReadRowGroup(g.index), "read " + parts[g.part]);
    rows += t->num_rows();
    pending_rows += t->num_rows();
    pending.push_back(std::move(t));
    if (pending_rows >= kMergedRowGroup) flush();
  }
  flush();
  check(w->Close(), "close " + path);
  return rows;
}

}  // namespace contingency_analysis
}  // namespace gridpack

#else  // built without Parquet support

namespace gridpack {
namespace contingency_analysis {

struct ParquetFlows::Impl {};

namespace {
[[noreturn]] void unavailable()
{
  throw std::runtime_error("this ca.x was built without Parquet support");
}
}  // namespace

bool ParquetFlows::available() { return false; }
void ParquetFlows::writeBranches(const std::string &, const std::vector<ParquetBranchRow> &)
{
  unavailable();
}
int64_t ParquetFlows::writeRange(const std::vector<std::string> &,
                                 const std::vector<RowGroup> &, const std::string &)
{
  unavailable();
}
const std::vector<ParquetFlows::RowGroup> &ParquetFlows::rowGroups() const { unavailable(); }
ParquetFlows::ParquetFlows(const std::string &, int) { unavailable(); }
ParquetFlows::~ParquetFlows() = default;
void ParquetFlows::add(int32_t, int32_t, double, double, double, double, double, double,
                       double, double)
{
  unavailable();
}
void ParquetFlows::endCase() { unavailable(); }
void ParquetFlows::close() { unavailable(); }

}  // namespace contingency_analysis
}  // namespace gridpack

#endif
