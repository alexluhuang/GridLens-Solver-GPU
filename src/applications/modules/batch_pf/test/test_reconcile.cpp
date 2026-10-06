/*
 * Copyright (c) 2013 Battelle Memorial Institute
 * Licensed under modified BSD License. See LICENSE in the top level directory.
 */
#include "../host/reconcile.hpp"

#include <chrono>
#include <cstddef>
#include <exception>
#include <string>
#include <system_error>
#include <vector>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {

struct Scratch {
  std::filesystem::path path = std::filesystem::temp_directory_path() /
      ("batchpf-reconcile-" + std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()));

  Scratch() { std::filesystem::create_directory(path); }
  ~Scratch()
  {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
  Scratch(const Scratch &) = delete;
  Scratch &operator=(const Scratch &) = delete;
  Scratch(Scratch &&) = delete;
  Scratch &operator=(Scratch &&) = delete;
};

void check(const Scratch &scratch, const std::vector<std::string> &contents,
           const std::string &expected, std::size_t expected_rows)
{
  std::vector<std::string> parts;
  for (const auto &content : contents) {
    const auto path = scratch.path / std::to_string(parts.size());
    std::ofstream file(path, std::ios::binary);
    file << content;
    parts.push_back(path.string());
  }
  // A rank with no results may have no part file.
  parts.push_back((scratch.path / "absent").string());
  std::ostringstream out;
  const auto rows = gridpack::batchpf::appendPartsByEvent(parts, out);
  if (rows != expected_rows || out.str() != expected) {
    throw std::runtime_error("rows were lost or reordered within an event");
  }
}

}  // namespace

int main()
{
  try {
    using gridpack::batchpf::checkOutcomeCoverage;
    if (!checkOutcomeCoverage(0, {0}).complete() ||
        !checkOutcomeCoverage(3, {2, 0, 1, 3}).complete()) {
      throw std::runtime_error("complete study rejected");
    }
    const auto faults = checkOutcomeCoverage(4, {0, 1, 1, 3, -1, 5});
    if (faults.complete() || faults.missing != std::vector<int>{2, 4} ||
        faults.duplicates != std::vector<int>{1} ||
        faults.unexpected != std::vector<int>{-1, 5}) {
      throw std::runtime_error("missing or repeated outcomes went undetected");
    }
    const Scratch scratch;
    check(scratch, {}, "", 0);
    check(scratch, {"", ""}, "", 0);
    check(scratch, {"12,c\n12,d\n2,b\n", "0,base\n1,a\n"},
          "0,base\n1,a\n2,b\n12,c\n12,d\n", 5);
    check(scratch, {"2,a\n\n1,b\n2,c", "1,d\r\n"},
          "1,b\n1,d\r\n2,a\n\n2,c\n", 4);
    check(scratch, {"1,a", "2,b"}, "1,a\n2,b\n", 2);
    // Exercise copying across the internal buffer boundary.
    const std::string large = "3," + std::string(2 * 1024 * 1024, 'x') + "\n";
    check(scratch, {large + "1,a\n", "2,b\n"}, "1,a\n2,b\n" + large, 3);
    std::cout << "No errors detected\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "failure detected: " << e.what() << '\n';
    return 1;
  }
}
