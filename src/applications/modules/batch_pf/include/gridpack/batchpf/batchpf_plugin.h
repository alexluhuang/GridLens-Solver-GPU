/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   batchpf_plugin.h
 * @date   2026-10-05
 *
 * @brief Interface between ca.x and the GPU batch power flow plugin (I-10).
 *
 * ca.x is built without CUDA. The GPU code lives in a shared object that ca.x
 * loads at run time, so the same ca.x runs with or without a GPU (guide
 * sections 5.3.6 and ADR-16). The two sides may be built by different
 * compilers at different times, so this interface follows the rules for a
 * stable boundary:
 *
 *  - Plain C: functions taking opaque handles and plain records. No C++
 *    types or exceptions cross it.
 *  - Every record starts with its size and a version, so either side can
 *    tell an older or newer peer apart and only read fields it knows.
 *  - The plugin exports one function, batchpf_get_api(), which fills a table
 *    of function pointers. A different major version is rejected; a newer
 *    minor version only appends to the table and to records.
 *  - Every call returns a status code; batchpf_api::last_error() gives the
 *    message for the last failure on a session.
 *  - Arrays are always passed with their length. Each side frees what it
 *    allocated: the caller owns every buffer it passes in and must keep it
 *    alive until the call that uses it has finished.
 *
 * Index conventions: buses are numbered 0..n_bus-1 in GridPACK's local order.
 * Off-diagonal admittances are given per directed bus pair ("edge") in CSR
 * form by row bus. Angles are in radians, voltages and powers per unit
 * unless stated otherwise.
 */

#ifndef GRIDPACK_BATCHPF_PLUGIN_H
#define GRIDPACK_BATCHPF_PLUGIN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Interface version implemented by this header */
enum {
  BATCHPF_API_MAJOR = 1,
  BATCHPF_API_MINOR = 0
};

/* Status codes. Expected outcomes such as a case that does not converge are
 * reported per case in batchpf_outcome, not as errors. */
typedef int32_t batchpf_status;
enum {
  BATCHPF_OK = 0,
  BATCHPF_ERR_INVALID_ARGUMENT = 1,
  BATCHPF_ERR_VERSION = 2,          /* incompatible interface version */
  BATCHPF_ERR_NO_DEVICE = 3,        /* no usable GPU or driver */
  BATCHPF_ERR_UNAVAILABLE = 4,      /* requested backend or feature missing */
  BATCHPF_ERR_OUT_OF_MEMORY = 5,
  BATCHPF_ERR_BACKEND = 6,          /* error inside a solver backend */
  BATCHPF_ERR_CUDA = 7,             /* CUDA runtime or kernel error */
  BATCHPF_ERR_STATE = 8,            /* call made in the wrong order */
  BATCHPF_ERR_TIMEOUT = 9,          /* wait() timed out; not a failure */
  BATCHPF_ERR_INTERNAL = 10
};

/* Batch linear solver backends (GPUBatch/backend) */
enum {
  BATCHPF_BACKEND_AUTO = 0,
  BATCHPF_BACKEND_CUDSS = 1,
  BATCHPF_BACKEND_ALG2 = 2,         /* custom batched LU (Zhou et al.) */
  BATCHPF_BACKEND_CPU_REFERENCE = 3 /* KLU per member on the CPU */
};

/* Memory paradigm (GPUBatch/memoryProfile, guide section 8.8) */
enum {
  BATCHPF_MEMORY_AUTO = 0,
  BATCHPF_MEMORY_UNIFIED = 1,       /* integrated GPU, one physical memory */
  BATCHPF_MEMORY_COHERENT = 2,      /* separate memory, hardware coherence */
  BATCHPF_MEMORY_DISCRETE = 3       /* separate memory, explicit copies */
};

/* Fill-reducing ordering for planning (GPUBatch/plannerOrdering) */
enum {
  BATCHPF_ORDERING_AMD = 0,
  BATCHPF_ORDERING_COLAMD = 1
};

/* Where triangular solves run (GPUBatch/solvePlacement) */
enum {
  BATCHPF_SOLVE_GPU = 0,
  BATCHPF_SOLVE_HOST = 1
};

/* Starting point for contingency cases (GPUBatch/warmStart) */
enum {
  BATCHPF_WARM_START_BASE_CASE = 0, /* base-case solution (ADR-05) */
  BATCHPF_WARM_START_RAW = 1        /* the values GridPACK resets to */
};

/* Telemetry detail (GPUBatch/telemetry) */
enum {
  BATCHPF_TELEMETRY_OFF = 0,
  BATCHPF_TELEMETRY_SUMMARY = 1,
  BATCHPF_TELEMETRY_DETAILED = 2
};

/* Log levels, most severe first */
enum {
  BATCHPF_LOG_ERROR = 0,
  BATCHPF_LOG_WARN = 1,
  BATCHPF_LOG_INFO = 2,
  BATCHPF_LOG_DEBUG = 3
};

/* Bus roles; the same numbers as PSS/E bus types */
enum {
  BATCHPF_BUS_PQ = 1,
  BATCHPF_BUS_PV = 2,
  BATCHPF_BUS_REF = 3,
  BATCHPF_BUS_ISOLATED = 4
};

/* Per-case result status */
enum {
  BATCHPF_CASE_CONVERGED = 0,       /* solve() would have returned true */
  BATCHPF_CASE_DIVERGED = 1,        /* iteration limit or mismatch blow-up */
  BATCHPF_CASE_FLAGGED = 2,         /* failed a numerical health check */
  BATCHPF_CASE_NOT_RUN = 3
};

/* Health events, bit flags in batchpf_outcome::health_events */
enum {
  BATCHPF_HEALTH_SMALL_PIVOT = 1,
  BATCHPF_HEALTH_NONFINITE = 2,
  BATCHPF_HEALTH_RESIDUAL = 4,
  BATCHPF_HEALTH_ITERATION_LIMIT = 8,
  BATCHPF_HEALTH_MISMATCH_GROWTH = 16,
  BATCHPF_HEALTH_STAGNATION = 32
};

/* Messages from the plugin go to ca.x's log through this callback */
typedef void (*batchpf_log_fn)(void *user, int32_t level, const char *message);

/* Effective GPUBatch settings, resolved by ca.x (guide section 8.16) */
typedef struct batchpf_settings {
  uint32_t struct_size;
  uint32_t struct_version;
  int32_t backend;              /* BATCHPF_BACKEND_* */
  int32_t device;               /* index among visible devices */
  int32_t batch_size;           /* 0 = automatic */
  int32_t max_validated_batch;  /* 0 = backend default cap */
  int32_t backfill;             /* refill finished slots during a batch */
  int32_t threads_per_block;    /* 0 = automatic, else a multiple of 32 */
  int32_t memory_profile;       /* BATCHPF_MEMORY_* */
  int32_t planner_ordering;     /* BATCHPF_ORDERING_* */
  int32_t solve_placement;      /* BATCHPF_SOLVE_* */
  int32_t refinement_steps;
  int32_t health_check_nonfinite;
  int32_t telemetry;            /* BATCHPF_TELEMETRY_* */
  int32_t profiler_ranges;
  int32_t log_level;            /* BATCHPF_LOG_* */
  double memory_headroom_bytes; /* kept free on unified memory */
  double max_memory_bytes;      /* 0 = no cap */
  double host_available_bytes;  /* OS-reported available memory */
  double host_reserved_bytes;   /* used by GridPACK ranks on this node */
  double pivot_tolerance;       /* reference factorization pivoting */
  double health_residual_limit; /* relative linear residual limit */
  double health_pivot_limit;    /* smallest pivot relative to largest */
  const char *plugin_dir;       /* directory holding backend plugins */
  batchpf_log_fn log_fn;
  void *log_user;
} batchpf_settings;

/* Result of probing the GPU (guide section 8.8.1) */
typedef struct batchpf_device_info {
  uint32_t struct_size;
  uint32_t struct_version;
  int32_t device_count;
  int32_t device;
  int32_t cc_major;
  int32_t cc_minor;
  int32_t integrated;
  int32_t concurrent_managed_access;
  int32_t pageable_memory_access;
  int32_t pageable_uses_host_page_tables;
  int32_t host_native_atomics;
  int32_t gpudirect_rdma;
  int32_t dmabuf;
  int32_t multiprocessors;
  int32_t driver_version;
  int32_t runtime_version;
  int32_t memory_profile;       /* selected BATCHPF_MEMORY_* */
  int32_t backend;              /* effective BATCHPF_BACKEND_* (after plan) */
  int32_t batch_size;           /* effective batch size (after plan) */
  int32_t threads_per_block;    /* effective (after plan) */
  double total_memory_bytes;
  double free_memory_bytes;     /* as reported by the CUDA runtime */
  double budget_bytes;          /* memory the plugin allows itself */
  char name[256];
  char backend_version[64];
} batchpf_device_info;

/* The superset model (I-4). All arrays are read during set_model() only. */
typedef struct batchpf_model {
  uint32_t struct_size;
  uint32_t struct_version;
  int32_t n_bus;
  int32_t n_edge;
  double sbase;                 /* MVA */
  const int32_t *bus_type;      /* n_bus, BATCHPF_BUS_* at case start */
  const double *g_diag;         /* n_bus, admittance diagonal */
  const double *b_diag;
  const double *p0;             /* n_bus, scheduled net injection */
  const double *q0;
  const double *v_init;         /* n_bus, values GridPACK resets to */
  const double *theta_init;
  const double *v_base;         /* n_bus, base-case solution */
  const double *theta_base;
  const double *ql;             /* n_bus, in-service constant-power Q load */
  const double *ip;             /* n_bus, in-service constant current load */
  const double *iq;
  const double *yp;             /* n_bus, in-service constant admittance load */
  const double *yq;
  const double *qmax;           /* n_bus, in-service generator Q limits */
  const double *qmin;
  const int32_t *row_start;     /* n_bus + 1 */
  const int32_t *edge_col;      /* n_edge, column bus of each edge */
  const int32_t *edge_mate;     /* n_edge, index of the reverse edge */
  const double *edge_g;         /* n_edge, off-diagonal admittance */
  const double *edge_b;
} batchpf_model;

/* Newton and controller rules, copied from GridPACK's settings */
typedef struct batchpf_solver_params {
  uint32_t struct_size;
  uint32_t struct_version;
  double tolerance;             /* Powerflow/tolerance */
  double damping_factor;        /* Powerflow/dampingFactor */
  double qlim_deadband;         /* Powerflow/qlimDeadband (MVAr) */
  int32_t max_iteration;        /* Powerflow/maxIteration */
  int32_t pf_qlim;              /* Powerflow/qlim: limits inside solve() */
  int32_t max_controller_iterations;
  int32_t ca_qlim;              /* Contingency_analysis/qlim: extra check */
  int32_t warm_start;           /* BATCHPF_WARM_START_* */
  int32_t reserved;
} batchpf_solver_params;

/* A bus value changed by a contingency (absolute values) */
typedef struct batchpf_bus_update {
  int32_t bus;
  int32_t type;                 /* BATCHPF_BUS_* */
  double g_diag;
  double b_diag;
  double p0;
  double q0;
  double qmax;
  double qmin;
} batchpf_bus_update;

/* An edge admittance changed by a contingency (absolute values) */
typedef struct batchpf_edge_update {
  int32_t edge;
  int32_t reserved;
  double g;
  double b;
} batchpf_edge_update;

/* One contingency case (a batch member, I-6) */
typedef struct batchpf_case {
  int64_t case_id;              /* caller's id, echoed in the outcome */
  int32_t n_bus_updates;
  int32_t n_edge_updates;
  const batchpf_bus_update *bus_updates;
  const batchpf_edge_update *edge_updates;
  int32_t slack_bus;            /* reference bus of this case */
  int32_t reserved;
} batchpf_case;

/* A batch of cases. cases[] and the arrays it points to must stay valid
 * until wait() reports the batch finished. */
typedef struct batchpf_batch {
  uint32_t struct_size;
  uint32_t struct_version;
  int32_t n_cases;
  int32_t reserved;
  const batchpf_case *cases;
} batchpf_batch;

/* Outcome of one case (I-7) */
typedef struct batchpf_outcome {
  int64_t case_id;
  int32_t status;               /* BATCHPF_CASE_* */
  int32_t health_events;        /* BATCHPF_HEALTH_* bits */
  int32_t iterations;           /* Newton iterations of the last solve */
  int32_t total_iterations;     /* all Newton iterations */
  int32_t controller_iterations;
  int32_t solves;               /* 2 if the extra Q-limit check re-solved */
  int32_t pv_to_pq;             /* reactive-limit conversions */
  int32_t max_p_bus;            /* local bus of largest P mismatch, or -1 */
  int32_t max_q_bus;
  int32_t reserved;
  double final_tolerance;       /* infinity norm of the last mismatch */
  double max_p_mismatch;        /* MW */
  double max_q_mismatch;        /* MVAr */
} batchpf_outcome;

/* Caller-provided output buffers for a batch (I-7 and I-9). Per-bus arrays
 * are case-major: entry (case c, bus k) is at c * n_bus + k. */
typedef struct batchpf_results {
  uint32_t struct_size;
  uint32_t struct_version;
  int32_t n_cases;
  int32_t n_bus;
  batchpf_outcome *outcomes;    /* n_cases */
  double *v;                    /* n_cases * n_bus */
  double *theta;                /* n_cases * n_bus */
  int32_t *qlim_conversion;     /* n_cases * n_bus: 0, +1 (at Qmax), -1 */
  double *q_required;           /* n_cases * n_bus: MVAr at conversion */
} batchpf_results;

/* Totals collected by the plugin (guide section 8.12) */
typedef struct batchpf_diagnostics {
  uint32_t struct_size;
  uint32_t struct_version;
  int32_t backend;
  int32_t batch_size;
  int32_t jacobian_rows;
  int32_t levels_factor;        /* dependency levels of the factorization */
  int32_t levels_lower;
  int32_t levels_upper;
  int64_t jacobian_nnz;         /* superset pattern */
  int64_t minimal_nnz;          /* standard layout of the base case */
  int64_t factor_nnz;           /* entries of L and U */
  int64_t factor_flops;
  int64_t batches;
  int64_t cases;
  int64_t converged;
  int64_t diverged;
  int64_t flagged;
  int64_t newton_steps;         /* member iterations, summed */
  int64_t slot_steps;           /* slot iterations incl. idle slots */
  double plan_seconds;
  double setup_seconds;
  double seconds_materialize;
  double seconds_mismatch;
  double seconds_jacobian;
  double seconds_factor;
  double seconds_solve;
  double seconds_update;
  double seconds_qlim;
  double seconds_exchange;
  double seconds_total;
  double bytes_factor;          /* estimated bytes moved, for bandwidth */
  double bytes_solve;
  double bytes_jacobian;
  double bytes_mismatch;
} batchpf_diagnostics;

typedef struct batchpf_session batchpf_session;   /* opaque */

/* Function table filled by batchpf_get_api() */
typedef struct batchpf_api {
  uint32_t struct_size;
  uint32_t api_major;
  uint32_t api_minor;
  uint32_t reserved;
  const char *plugin_version;   /* static strings owned by the plugin */
  const char *build_info;

  /* Probe the GPU without creating a session. Calls cudaGetDeviceCount()
   * first, so it fails cleanly when no GPU or driver is present. */
  batchpf_status (*probe)(int32_t device, batchpf_device_info *info,
                          char *error, size_t error_size);

  /* Create a session for the given settings. A backend of CPU_REFERENCE
   * does not need a GPU. */
  batchpf_status (*session_create)(const batchpf_settings *settings,
                                   batchpf_session **session,
                                   char *error, size_t error_size);
  void (*session_destroy)(batchpf_session *session);
  const char *(*last_error)(const batchpf_session *session);
  batchpf_status (*get_device_info)(batchpf_session *session,
                                    batchpf_device_info *info);

  /* Give the plugin the network (copied). */
  batchpf_status (*set_model)(batchpf_session *session,
                              const batchpf_model *model);

  /* One-time planning and backend setup. Returns the number of cases the
   * caller should put in each batch. */
  batchpf_status (*plan)(batchpf_session *session,
                         const batchpf_solver_params *params,
                         int64_t expected_cases, int32_t *batch_capacity);

  /* Start solving a batch; returns at once with a ticket. */
  batchpf_status (*submit)(batchpf_session *session,
                           const batchpf_batch *batch,
                           batchpf_results *results, int64_t *ticket);

  /* Wait for a batch: timeout_ms < 0 waits forever, 0 only polls.
   * Returns BATCHPF_ERR_TIMEOUT if it is still running. */
  batchpf_status (*wait)(batchpf_session *session, int64_t ticket,
                         int32_t timeout_ms);

  batchpf_status (*get_diagnostics)(batchpf_session *session,
                                    batchpf_diagnostics *diagnostics);
} batchpf_api;

/* The single symbol a core plugin exports. api->struct_size must be set by
 * the caller; the plugin fills at most that many bytes. */
typedef batchpf_status (*batchpf_get_api_fn)(uint32_t requested_major,
                                             batchpf_api *api);
batchpf_status batchpf_get_api(uint32_t requested_major, batchpf_api *api);

#ifdef __cplusplus
}
#endif

#endif /* GRIDPACK_BATCHPF_PLUGIN_H */
