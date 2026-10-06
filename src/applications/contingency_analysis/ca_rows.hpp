/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   ca_rows.hpp
 * @date   2026-10-06
 *
 * @brief Helpers for writing the per-branch result tables (csv_flat,
 * csv_delta) from numbers.
 *
 * The tables used to be built from text: GridPACK formatted every bus and
 * branch with "%20.12e", gathered the strings, and the driver parsed them
 * back with sscanf before formatting each row again with an ostream. Reading
 * the solved state directly is several times faster. To keep every output
 * byte-identical, round13() reproduces the value that print-and-parse
 * produced, and RowText formats numbers exactly as the ostream did (both use
 * correctly rounded conversions; checked over two million values, including
 * ties, signed zeros and non-finite values).
 */

#ifndef CA_ROWS_HPP_
#define CA_ROWS_HPP_

#include <charconv>
#include <cstddef>
#include <string>

namespace gridpack {
namespace contingency_analysis {

/// The value printing with "%20.12e" and reading back with "%lf" gives
inline double round13(double x)
{
  char buf[64];
  const std::to_chars_result r =
      std::to_chars(buf, buf + sizeof(buf), x, std::chars_format::scientific, 12);
  double y = 0.0;
  std::from_chars(buf, r.ptr, y);
  return y;
}

/**
 * The token sscanf("%15s") reads from a circuit tag printed with "%s".
 * Returns false when reading the rest of the printed line would not line up
 * with the fields that follow (an empty tag, a tag longer than 15 characters
 * or one with inner spaces); such branches keep the text route.
 */
inline bool scanToken(const std::string &tag, std::string *token)
{
  const std::size_t begin = tag.find_first_not_of(" \t\n\v\f\r");
  if (begin == std::string::npos) return false;
  std::size_t end = tag.find_first_of(" \t\n\v\f\r", begin);
  if (end == std::string::npos) end = tag.size();
  if (end - begin > 15) return false;
  if (tag.find_first_not_of(" \t\n\v\f\r", end) != std::string::npos) return false;
  token->assign(tag, begin, end - begin);
  return true;
}

/// Row text built in one buffer; numbers formatted like an ostream in fixed mode
class RowText {
 public:
  void clear() { p_text.clear(); }
  const std::string &str() const { return p_text; }
  bool empty() const { return p_text.empty(); }

  RowText &add(const std::string &s)
  {
    p_text += s;
    return *this;
  }
  RowText &add(const char *s)
  {
    p_text += s;
    return *this;
  }
  RowText &add(char c)
  {
    p_text += c;
    return *this;
  }
  RowText &add(int v)
  {
    char buf[16];
    const std::to_chars_result r = std::to_chars(buf, buf + sizeof(buf), v);
    p_text.append(buf, r.ptr);
    return *this;
  }
  /// Same text as `ostream << std::fixed << std::setprecision(digits) << v`
  RowText &fixed(double v, int digits)
  {
    char buf[400];   // enough for the largest double in fixed notation
    const std::to_chars_result r =
        std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::fixed, digits);
    p_text.append(buf, r.ptr);
    return *this;
  }

 private:
  std::string p_text;
};

}  // namespace contingency_analysis
}  // namespace gridpack

#endif
