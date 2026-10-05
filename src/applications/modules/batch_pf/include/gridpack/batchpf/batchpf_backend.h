/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
/**
 * @file   batchpf_backend.h
 * @date   2026-10-05
 *
 * @brief Interface between the core GPU plugin and batch linear solver
 * backend plugins (I-11, carrying the I-8 contract of guide section 5.3.5).
 *
 * A backend factors and solves many sparse systems that share one sparsity
 * pattern. The core plugin builds the Jacobians; the backend only does the
 * linear algebra. Backends that need extra libraries (cuDSS) are separate
 * shared objects, so a missing library disables that backend only.
 *
 * The same rules as batchpf_plugin.h apply: plain C, size-and-version
 * prefixed records, one exported function filling a table, status codes,
 * caller-owned buffers.
 *
 * Value layout ("case-interleaved", guide section 8.4.1): for a matrix with
 * nnz stored entries and a batch capacity B, entry p of member b is at
 * values[p * B + b]. Vectors of length n use x[i * B + b]. Neighboring
 * members of the same entry are neighbors in memory, which lets GPU threads
 * that work on different members read memory together.
 *
 * All pointers passed to refactorize() and solve() live in the memory kind
 * given at setup (device memory for GPU backends), and the calls are
 * ordered on the stream given at setup.
 */

#ifndef GRIDPACK_BATCHPF_BACKEND_H
#define GRIDPACK_BATCHPF_BACKEND_H

#include <stddef.h>
#include <stdint.h>
#include "gridpack/batchpf/batchpf_plugin.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  BATCHPF_BACKEND_API_MAJOR = 1,
  BATCHPF_BACKEND_API_MINOR = 0
};

/* Memory kinds a backend accepts (bit flags) */
enum {
  BATCHPF_MEMKIND_DEVICE = 1,
  BATCHPF_MEMKIND_HOST = 2,
  BATCHPF_MEMKIND_MANAGED = 4
};

/* Per-member status written by refactorize() and solve() */
enum {
  BATCHPF_MEMBER_OK = 0,
  BATCHPF_MEMBER_SMALL_PIVOT = 1,
  BATCHPF_MEMBER_NONFINITE = 2,
  BATCHPF_MEMBER_SKIPPED = 3
};

typedef struct batchpf_backend_caps {
  uint32_t struct_size;
  uint32_t struct_version;
  int32_t max_batch;            /* 0 = no fixed limit */
  int32_t validated_batch;      /* largest batch size validated so far */
  int32_t member_masking;       /* inactive members cost no work */
  int32_t iterative_refinement;
  int32_t failed_member_reporting;
  int32_t memory_kinds;         /* BATCHPF_MEMKIND_* bits */
  char name[32];
  char version[64];
} batchpf_backend_caps;

/* One-time plan for a backend (I-5) */
typedef struct batchpf_backend_plan {
  uint32_t struct_size;
  uint32_t struct_version;
  int32_t n;                    /* rows (and columns) */
  int32_t batch_capacity;       /* B, also the stride of every buffer */
  int64_t nnz;
  const int32_t *row_ptr;       /* host, n + 1: shared CSR pattern */
  const int32_t *col_idx;       /* host, nnz */
  const int32_t *row_perm;      /* host, n, fill-reducing order, or NULL */
  const int32_t *col_perm;      /* host, n, or NULL */
  const double *reference_values; /* host, nnz: a typical member */
  int32_t memory_kind;          /* BATCHPF_MEMKIND_* used for the buffers */
  int32_t device;
  void *stream;                 /* cudaStream_t (NULL for host backends) */
  int32_t refinement_steps;
  int32_t threads_per_block;    /* 0 = backend decides */
  double pivot_limit;           /* relative pivot threshold for flagging */
  batchpf_log_fn log_fn;
  void *log_user;
  int32_t log_level;
  int32_t reserved;
} batchpf_backend_plan;

typedef struct batchpf_backend batchpf_backend;   /* opaque */

typedef struct batchpf_backend_api {
  uint32_t struct_size;
  uint32_t api_major;
  uint32_t api_minor;
  uint32_t reserved;
  const char *name;
  batchpf_status (*capabilities)(batchpf_backend_caps *caps);
  batchpf_status (*setup)(const batchpf_backend_plan *plan,
                          batchpf_backend **backend,
                          char *error, size_t error_size);
  /* Numeric factorization of active members (mask[b] != 0). values holds
   * nnz * B entries in the interleaved layout. member_status has B
   * entries. */
  batchpf_status (*refactorize)(batchpf_backend *backend,
                                const double *values, const int32_t *mask,
                                int32_t *member_status);
  /* Solve A x = rhs for active members, using the last factorization. rhs
   * and x hold n * B entries; they may not alias. */
  batchpf_status (*solve)(batchpf_backend *backend, const double *rhs,
                          double *x, const int32_t *mask,
                          int32_t *member_status);
  void (*teardown)(batchpf_backend *backend);
  const char *(*last_error)(const batchpf_backend *backend);
} batchpf_backend_api;

typedef batchpf_status (*batchpf_get_backend_api_fn)(
    uint32_t requested_major, batchpf_backend_api *api);
batchpf_status batchpf_get_backend_api(uint32_t requested_major,
                                       batchpf_backend_api *api);

#ifdef __cplusplus
}
#endif

#endif /* GRIDPACK_BATCHPF_BACKEND_H */
