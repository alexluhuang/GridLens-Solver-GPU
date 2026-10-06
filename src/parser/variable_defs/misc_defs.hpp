/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/*
 * This file contains definitions for miscellaneous parameters that can be read
 * in from PTI format files. Each parameter has a corresponding macro that can
 * be used as a unique string to identify the parameter. The use of macros
 * instead of using strings directly will provide extra safety by forcing
 * compiler errors in the case of typos or spelling mistakes.
 */

/**
 *  Variables that can be associated more than once for a bus or a branch can be
 *  indexed by an integer to distinguish different instances. For example,
 *  multiple generators can be associated with a bus and multiple transmission
 *  elements can be associated with a branch. The variables that have an associated
 *  index are denoted with the keyword "indexed".
 */

#ifndef _MISC_VAR_HPP_
#define _MISC_VAR_HPP_

// CASE DATA
/**
 * 0: base case
 * 1: add information to existing case
 * type: integer
 */
#define CASE_ID "CASE_ID"

/**
 * System base MVS. 
 * Default value is 100.0 MVA
 * type: real float
 */
#define CASE_SBASE "CASE_SBASE"

// AREA DATA
/**
 * Total number of area fields
 * type: integer
 */
#define AREA_TOTAL "AREA_TOTAL"

/**
 * Area number
 * type: integer
 * indexed
 */
#define AREAINTG_NUMBER "AREAINTG_NUMBER"

/**
 * Bus number of the area slack bus
 * type: integer
 * indexed
 */
#define AREAINTG_ISW "AREAINTG_ISW"

/**
 * Desired net interchange leaving the area (export); entered in MW
 * type: real float
 * indexed
 */
#define AREAINTG_PDES "AREAINTG_PDES"

/**
 * Interchange tolerance bandwidth; entered in MW
 * type: real float
 * indexed
 */
#define AREAINTG_PTOL "AREAINTG_PTOL"

/**
 * Area name
 * type: string
 * indexed
 */
#define AREAINTG_NAME "AREAINTG_NAME"

// TRANSFORMER IMPEDANCE CORRECTION
/**
 * Impedance correction table number
 * type: integer
 */
#define XFMR_CORR_TABLE_NUMBER "XFMR_CORR_TABLE_NUMBER"

/**
 * Either off-nominal turns ratio in pu or phase shift angle in degrees
 * type: real float
 */
#define XFMR_CORR_TABLE_T1  "XFMR_CORR_TABLE_T1"
#define XFMR_CORR_TABLE_T2  "XFMR_CORR_TABLE_T2"
#define XFMR_CORR_TABLE_T3  "XFMR_CORR_TABLE_T3"
#define XFMR_CORR_TABLE_T4  "XFMR_CORR_TABLE_T4"
#define XFMR_CORR_TABLE_T5  "XFMR_CORR_TABLE_T5"
#define XFMR_CORR_TABLE_T6  "XFMR_CORR_TABLE_T6"
#define XFMR_CORR_TABLE_T7  "XFMR_CORR_TABLE_T7"
#define XFMR_CORR_TABLE_T8  "XFMR_CORR_TABLE_T8"
#define XFMR_CORR_TABLE_T9  "XFMR_CORR_TABLE_T9"
#define XFMR_CORR_TABLE_T10 "XFMR_CORR_TABLE_T10"
#define XFMR_CORR_TABLE_T11 "XFMR_CORR_TABLE_T11"

/**
 * Scaling factor by which transformer nominal impedance is to be multiplied to obtain
 * the actual transformer impedance for the corresponding "Ti"
 * type: real float
 */
#define XFMR_CORR_TABLE_F1  "XFMR_CORR_TABLE_F1"
#define XFMR_CORR_TABLE_F2  "XFMR_CORR_TABLE_F2"
#define XFMR_CORR_TABLE_F3  "XFMR_CORR_TABLE_F3"
#define XFMR_CORR_TABLE_F4  "XFMR_CORR_TABLE_F4"
#define XFMR_CORR_TABLE_F5  "XFMR_CORR_TABLE_F5"
#define XFMR_CORR_TABLE_F6  "XFMR_CORR_TABLE_F6"
#define XFMR_CORR_TABLE_F7  "XFMR_CORR_TABLE_F7"
#define XFMR_CORR_TABLE_F8  "XFMR_CORR_TABLE_F8"
#define XFMR_CORR_TABLE_F9  "XFMR_CORR_TABLE_F9"
#define XFMR_CORR_TABLE_F10 "XFMR_CORR_TABLE_F10"
#define XFMR_CORR_TABLE_T11 "XFMR_CORR_TABLE_T11"


// MULTISECTION LINE GROUPING 
/**
 * Two-character upper case alphanumeric multisection line grouping identifier.
 * type: string
 */
#define MULTI_SEC_LINE_ID "MULTI_SEC_LINE_ID"

/**
 * Metered end flag
 * <= 1 to designate from bus as the metered end
 * >= 2 to designate to bus as metered end
 * type: integer
 */
#define MULTI_SEC_LINE_MET "MULTI_SEC_LINE_MET"

/**
 * Bus numbers of the "dummy buses" connected by the branches that comprise this multisection
 * line grouping. No defaults allowed.
 * type: integer
 */
#define MULTI_SEC_LINE_DUM1 "MULTI_SEC_LINE_DUM1"
#define MULTI_SEC_LINE_DUM2 "MULTI_SEC_LINE_DUM2"
#define MULTI_SEC_LINE_DUM3 "MULTI_SEC_LINE_DUM3"
#define MULTI_SEC_LINE_DUM4 "MULTI_SEC_LINE_DUM4"
#define MULTI_SEC_LINE_DUM5 "MULTI_SEC_LINE_DUM5"
#define MULTI_SEC_LINE_DUM6 "MULTI_SEC_LINE_DUM6"
#define MULTI_SEC_LINE_DUM7 "MULTI_SEC_LINE_DUM7"
#define MULTI_SEC_LINE_DUM8 "MULTI_SEC_LINE_DUM8"
#define MULTI_SEC_LINE_DUM9 "MULTI_SEC_LINE_DUM9"


// ZONE DATA
/**
 * Total number of zone fields
 * type: integer
 */
#define ZONE_TOTAL "ZONE_TOTAL"

/**
 * Zone Number
 * type: integer
 * indexed
 */
#define ZONE_NUMBER "ZONE_NUMBER"

/**
 * Zone Name
 * type: string
 * indexed
 */
#define ZONE_NAME "ZONE_NAME"


// INTERAREA TRANSFER
/**
 * From area number of interarea transfer
 * type: integer
 */
#define INTERAREA_TRANSFER_FROM "INTERAREA_TRANSFER_FROM"

/**
 * To area number of interarea transfer
 * type: integer
 */
#define INTERAREA_TRANSFER_TO "INTERAREA_TRANSFER_TO"

/**
 * Single-character (0 through 9 or A through Z) upper case interarea transfer identifier
 * used to distinguish among multiple transfers
 * type: character
 */
#define INTERAREA_TRANSFER_TRID "INTERAREA_TRANSFER_TRID"

/**
 * MW comprising this transfer
 * type: real float
 */
#define INTERAREA_TRANSFER_PTRAN "INTERAREA_TRANSFER_PTRAN"


// OWNER
/**
 * Total number of owner fields
 * type: integer
 */
#define OWNER_TOTAL "OWNER_TOTAL"

/**
 * Owner number
 * type: integer
 * indexed
 */
#define OWNER_NUMBER "OWNER_NUMBER"

/**
 * Owner name
 * type: string
 * indexed
 */
#define OWNER_NAME "OWNER_NAME"

// TWO-TERMINAL DC LINE DATA (network level, indexed by dc line)
/**
 * Total number of two-terminal dc lines
 * type: integer
 */
#define HVDC_LINE_TOTAL "HVDC_LINE_TOTAL"

/**
 * Dc line name
 * type: string
 * indexed
 */
#define HVDC_LINE_NAME "HVDC_LINE_NAME"

/**
 * Control mode: 0 blocked, 1 power, 2 current
 * type: integer
 * indexed
 */
#define HVDC_LINE_MDC "HVDC_LINE_MDC"

/**
 * Dc line resistance; entered in ohms
 * type: real float
 * indexed
 */
#define HVDC_LINE_RDC "HVDC_LINE_RDC"

/**
 * Power (MW) or current (amps) demand. For power control a positive value
 * is desired rectifier power and a negative value desired inverter power
 * type: real float
 * indexed
 */
#define HVDC_LINE_SETVL "HVDC_LINE_SETVL"

/**
 * Scheduled compounded dc voltage; entered in kV
 * type: real float
 * indexed
 */
#define HVDC_LINE_VSCHD "HVDC_LINE_VSCHD"

/**
 * Mode switch dc voltage; entered in kV
 * type: real float
 * indexed
 */
#define HVDC_LINE_VCMOD "HVDC_LINE_VCMOD"

/**
 * Compounding resistance; entered in ohms
 * type: real float
 * indexed
 */
#define HVDC_LINE_RCOMP "HVDC_LINE_RCOMP"

/**
 * Current margin, fraction of the order the inverter takes over when the
 * rectifier is at its minimum firing angle
 * type: real float
 * indexed
 */
#define HVDC_LINE_DELTI "HVDC_LINE_DELTI"

/**
 * Converter data at the rectifier (RECT) and inverter (INV) ends: ac bus
 * number, number of bridges, maximum and minimum firing (rectifier) or
 * extinction (inverter) angle in degrees, commutating resistance and
 * reactance per bridge in ohms, primary base ac voltage in kV, transformer
 * ratio, tap setting and tap limits and step
 * type: integer (BUS), real float (others)
 * indexed
 */
#define HVDC_RECT_BUS "HVDC_RECT_BUS"
#define HVDC_RECT_NB "HVDC_RECT_NB"
#define HVDC_RECT_ANMX "HVDC_RECT_ANMX"
#define HVDC_RECT_ANMN "HVDC_RECT_ANMN"
#define HVDC_RECT_RC "HVDC_RECT_RC"
#define HVDC_RECT_XC "HVDC_RECT_XC"
#define HVDC_RECT_EBAS "HVDC_RECT_EBAS"
#define HVDC_RECT_TR "HVDC_RECT_TR"
#define HVDC_RECT_TAP "HVDC_RECT_TAP"
#define HVDC_RECT_TMX "HVDC_RECT_TMX"
#define HVDC_RECT_TMN "HVDC_RECT_TMN"
#define HVDC_RECT_STP "HVDC_RECT_STP"
#define HVDC_INV_BUS "HVDC_INV_BUS"
#define HVDC_INV_NB "HVDC_INV_NB"
#define HVDC_INV_ANMX "HVDC_INV_ANMX"
#define HVDC_INV_ANMN "HVDC_INV_ANMN"
#define HVDC_INV_RC "HVDC_INV_RC"
#define HVDC_INV_XC "HVDC_INV_XC"
#define HVDC_INV_EBAS "HVDC_INV_EBAS"
#define HVDC_INV_TR "HVDC_INV_TR"
#define HVDC_INV_TAP "HVDC_INV_TAP"
#define HVDC_INV_TMX "HVDC_INV_TMX"
#define HVDC_INV_TMN "HVDC_INV_TMN"
#define HVDC_INV_STP "HVDC_INV_STP"

#endif /* _MISC_VAR_HPP_ */
