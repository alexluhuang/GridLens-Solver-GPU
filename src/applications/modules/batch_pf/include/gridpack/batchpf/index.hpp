/*
 * Copyright (c) 2013 Battelle Memorial Institute
 * Licensed under modified BSD License. See LICENSE in the top level directory.
 */
#ifndef GRIDPACK_BATCHPF_INDEX_HPP
#define GRIDPACK_BATCHPF_INDEX_HPP

namespace gridpack {
namespace batchpf {

// Distinct indices cannot be substituted for one another. Conversion to
// plain integers is explicit at GridPACK and plugin boundaries (ADR-18).
template <class Tag>
struct Index {
  int value = -1;
  constexpr Index() noexcept = default;
  constexpr explicit Index(int index) noexcept : value(index) {}
};

struct BusTag;
struct CaseTag;
struct BranchTag;
struct MemberTag;
using BusIndex = Index<BusTag>;
using CaseIndex = Index<CaseTag>;
using BranchIndex = Index<BranchTag>;
using MemberIndex = Index<MemberTag>;

}  // namespace batchpf
}  // namespace gridpack

#endif
