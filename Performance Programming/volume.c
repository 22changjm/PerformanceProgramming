#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

// Include OpenMP
#include <omp.h>

#include "volume.h"

inline double volume_get(volume_t* v, int x, int y, int d) {
  return v->weights[((v->width * y) + x) * v->depth + d];
}

inline void volume_set(volume_t* v, int x, int y, int d, double value) {
  v->weights[((v->width * y) + x) * v->depth + d] = value;
}

volume_t* make_volume(int width, int height, int depth, double value) {
  volume_t* new_vol = malloc(sizeof(struct volume));
  new_vol->weights = malloc(sizeof(double) * width * height * depth);

  new_vol->width  = width;
  new_vol->height = height;
  new_vol->depth  = depth;

  return new_vol;
}

void copy_volume(volume_t* dest, volume_t* src) {
  assert(dest->width == src->width);
  assert(dest->height == src->height);
  assert(dest->depth == src->depth);

  double* dest_weights = dest->weights;
  double* src_weights = src->weights;
  int size = dest->width * dest->height * dest->depth;
  for (int i = 0; i < size; i += 128) {
    _mm256_storeu_pd(dest_weights + i, _mm256_loadu_pd(src_weights + i));
    _mm256_storeu_pd(dest_weights + i + 4, _mm256_loadu_pd(src_weights + i + 4));
    _mm256_storeu_pd(dest_weights + i + 8, _mm256_loadu_pd(src_weights + i + 8));
    _mm256_storeu_pd(dest_weights + i + 12, _mm256_loadu_pd(src_weights + i + 12));
    _mm256_storeu_pd(dest_weights + i + 16, _mm256_loadu_pd(src_weights + i + 16));
    _mm256_storeu_pd(dest_weights + i + 20, _mm256_loadu_pd(src_weights + i + 20));
    _mm256_storeu_pd(dest_weights + i + 24, _mm256_loadu_pd(src_weights + i + 24));
    _mm256_storeu_pd(dest_weights + i + 28, _mm256_loadu_pd(src_weights + i + 28));
    _mm256_storeu_pd(dest_weights + i + 32, _mm256_loadu_pd(src_weights + i + 32));
    _mm256_storeu_pd(dest_weights + i + 36, _mm256_loadu_pd(src_weights + i + 36));
    _mm256_storeu_pd(dest_weights + i + 40, _mm256_loadu_pd(src_weights + i + 40));
    _mm256_storeu_pd(dest_weights + i + 44, _mm256_loadu_pd(src_weights + i + 44));
    _mm256_storeu_pd(dest_weights + i + 48, _mm256_loadu_pd(src_weights + i + 48));
    _mm256_storeu_pd(dest_weights + i + 52, _mm256_loadu_pd(src_weights + i + 52));
    _mm256_storeu_pd(dest_weights + i + 56, _mm256_loadu_pd(src_weights + i + 56));
    _mm256_storeu_pd(dest_weights + i + 60, _mm256_loadu_pd(src_weights + i + 60));
    _mm256_storeu_pd(dest_weights + i + 64, _mm256_loadu_pd(src_weights + i + 64));
    _mm256_storeu_pd(dest_weights + i + 68, _mm256_loadu_pd(src_weights + i + 68));
    _mm256_storeu_pd(dest_weights + i + 72, _mm256_loadu_pd(src_weights + i + 72));
    _mm256_storeu_pd(dest_weights + i + 76, _mm256_loadu_pd(src_weights + i + 76));
    _mm256_storeu_pd(dest_weights + i + 80, _mm256_loadu_pd(src_weights + i + 80));
    _mm256_storeu_pd(dest_weights + i + 84, _mm256_loadu_pd(src_weights + i + 84));
    _mm256_storeu_pd(dest_weights + i + 88, _mm256_loadu_pd(src_weights + i + 88));
    _mm256_storeu_pd(dest_weights + i + 92, _mm256_loadu_pd(src_weights + i + 92));
    _mm256_storeu_pd(dest_weights + i + 96, _mm256_loadu_pd(src_weights + i + 96));
    _mm256_storeu_pd(dest_weights + i + 100, _mm256_loadu_pd(src_weights + i + 100));
    _mm256_storeu_pd(dest_weights + i + 104, _mm256_loadu_pd(src_weights + i + 104));
    _mm256_storeu_pd(dest_weights + i + 108, _mm256_loadu_pd(src_weights + i + 108));
    _mm256_storeu_pd(dest_weights + i + 112, _mm256_loadu_pd(src_weights + i + 112));
    _mm256_storeu_pd(dest_weights + i + 116, _mm256_loadu_pd(src_weights + i + 116));
    _mm256_storeu_pd(dest_weights + i + 120, _mm256_loadu_pd(src_weights + i + 120));
    _mm256_storeu_pd(dest_weights + i + 124, _mm256_loadu_pd(src_weights + i + 124));
  }
}

void free_volume(volume_t* v) {
  free(v->weights);
  free(v);
}
