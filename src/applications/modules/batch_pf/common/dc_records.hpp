/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   dc_records.hpp
 * @date   2026-10-07
 *
 * @brief Conversion between the plugin interface's two-terminal dc line
 * records (batchpf_dc_*) and GridPACK's dc line model (pf_hvdc.hpp). Used
 * on both sides of the interface: ca.x builds the records from its network
 * and reads final operating points back; the plugin turns them into the
 * model data its kernels evaluate. No CUDA.
 */

#ifndef GRIDPACK_BATCHPF_COMMON_DC_RECORDS_HPP
#define GRIDPACK_BATCHPF_COMMON_DC_RECORDS_HPP

#include "gridpack/batchpf/batchpf_plugin.h"
#include "pf_hvdc.hpp"

namespace gridpack {
namespace batchpf {

using gridpack::powerflow::HVDCConverter;
using gridpack::powerflow::HVDCConverterState;
using gridpack::powerflow::HVDCLineData;
using gridpack::powerflow::HVDCMode;
using gridpack::powerflow::HVDCSolution;

/// dc line data of an interface record; the converter buses are local indices
HVDCLineData dcLineData(const batchpf_dc_line &r);

/**
 * Interface record of a dc line
 * @param line dc line data
 * @param rect_bus local index of the rectifier bus
 * @param inv_bus local index of the inverter bus
 * @param reference operating point every solve starts from
 */
batchpf_dc_line dcLineRecord(const HVDCLineData &line, int rect_bus, int inv_bus,
                             const HVDCSolution &reference);

/// Operating point of an interface record
HVDCSolution dcSolution(const batchpf_dc_state &r);

/// Interface record of an operating point
batchpf_dc_state dcState(const HVDCSolution &s);

}  // namespace batchpf
}  // namespace gridpack

#endif
