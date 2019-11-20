#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

// Include OpenMP
#include <omp.h>

#include "layers.h"
#include "volume.h"


conv_layer_t* make_conv_layer(int input_width, int input_height, int input_depth, int filter_width, int num_filters,
                              int stride, int pad) {
  conv_layer_t* l = (conv_layer_t*)malloc(sizeof(conv_layer_t));

  l->output_depth = num_filters;
  l->filter_width = filter_width;
  l->input_depth  = input_depth;
  l->input_width  = input_width;
  l->input_height = input_height;

  l->filter_height = l->filter_width;
  l->stride        = stride;
  l->pad = pad;

  l->output_width = (input_width + pad * 2 - filter_width) /
                    stride + 1;
  l->output_height = (input_height + pad * 2 - filter_width) /
                     stride + 1;

  l->filters = malloc(sizeof(volume_t*) * num_filters);
  volume_t** filter = l->filters;
  for (int i = 0; i < num_filters; i++) {
    *(filter + i) = make_volume(filter_width, filter_width, input_depth, 0.0);
  }

  l->bias   = 0.0;
  l->biases = make_volume(1, 1, num_filters, 0.0);

  return l;
}

// Performs the forward pass for a convolutional layer by convolving each one
// of the filters with a particular input, and placing the result in the output
// array.
//
// One way to think about convolution in this case is that we have one of the
// layer's filters (a 3D array) that is superimposed on one of the layer's
// inputs (a second 3D array) that has been implicitly padded with zeros. Since
// convolution is a sum of products (described below), we don't actually have
// to add any zeros to the input volume since those terms will not contribute
// to the convolution. Instead, for each position in the filter, we just make
// sure that we are in bounds for the input volume.
//
// Essentially, the filter is "sliding" across the input, in both the x and y
// directions, where we increment our position in each direction by using the
// stride parameter.
//
// At each position, we compute the sum of the elementwise product of the filter
// and the part of the array it's covering. For instance, let's consider a 2D
// case, where the filter (on the left) is superimposed on some part of the
// input (on the right).
//
//   Filter             Input
//  -1  0  1           1  2  3
//  -1  0  1           4  5  6
//  -1  0  1           7  8  9
//
// Here, the sum of the elementwise product is:
//    Filter[0][0] * Input[0][0] + Filter[0][1] * Input[0][1] + ...
//    = -1 * 1 + 0 * 2 + ... + 0 * 8 + 1 * 9
//    = 6
//
// The 3D case is essentially the same, we just have to sum over the other
// dimension as well. Also, since volumes are internally represented as 1D
// arrays, we must use the volume_get and volume_set commands to access elements
// at a coordinate (x, y, d). Finally, we add the corresponding bias for the
// filter to the sum before putting it into the output volume.
void conv_forward(conv_layer_t* l, volume_t** inputs, volume_t** outputs, int start, int end) {
  int stride = l->stride;
  volume_t* in  = inputs[0];
  volume_t* out = outputs[0];
  int in_height = in->height;
  int in_width = in->width;
  int in_depth = in->depth;
  double* in_weights = in->weights;
  int l_output_height = l->output_height;
  int l_output_depth = l->output_depth;
  volume_t** l_filters = l->filters;
    for (int f = 0; f < l_output_depth; f++) {
      volume_t* filter = l_filters[f];
      int filter_height = filter->height;
      int filter_width = filter->width;
      int filter_depth = filter->depth;
      double* filter_weights = filter->weights;
      int y = -2;
      for (int out_y = 0; out_y < l_output_height; y += stride, out_y++) {
        int x = -2;
        for (int out_x = 0; out_x < l_output_height; x += stride, out_x++) {

          // Take sum of element-wise product
          double sum = 0.0;
          for (int fy = 0; fy < filter_height; fy++) {
            int in_y = y + fy;
            for (int fx = 0; fx < filter_width; fx++) {
              int in_x = x + fx;
              if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                double* filter_n = filter_weights + ((filter_width * fy) + fx) * filter_depth;
                double* in_n = in_weights + ((in_width * in_y) + in_x) * in_depth;
                __m256d sum_vector = _mm256_setzero_pd();
                if (filter_depth == 16) {
                    __m256d filter_vector = _mm256_loadu_pd((double const *)filter_n);
                    __m256d in_vector = _mm256_loadu_pd((double const *)in_n);
                    __m256d mul =  _mm256_mul_pd(filter_vector, in_vector);
                    sum_vector = _mm256_add_pd(sum_vector,mul);
                    filter_vector = _mm256_loadu_pd((double const *)filter_n + 4);
                    in_vector = _mm256_loadu_pd((double const *)in_n + 4);
                    mul =  _mm256_mul_pd(filter_vector, in_vector);
                    sum_vector = _mm256_add_pd(sum_vector,mul);
                    filter_vector = _mm256_loadu_pd((double const *)filter_n + 8);
                    in_vector = _mm256_loadu_pd((double const *)in_n + 8);
                    mul =  _mm256_mul_pd(filter_vector, in_vector);
                    sum_vector = _mm256_add_pd(sum_vector,mul);
                    filter_vector = _mm256_loadu_pd((double const *)filter_n + 12);
                    in_vector = _mm256_loadu_pd((double const *)in_n + 12);
                    mul =  _mm256_mul_pd(filter_vector, in_vector);
                    sum_vector = _mm256_add_pd(sum_vector,mul);
                    sum += sum_vector[0] + sum_vector[1] + sum_vector[2] + sum_vector[3];
                  }
                else if (filter_depth == 20) {
                    __m256d filter_vector = _mm256_loadu_pd((double const *)filter_n);
                    __m256d in_vector = _mm256_loadu_pd((double const *)in_n);
                    __m256d mul =  _mm256_mul_pd(filter_vector, in_vector);
                    sum_vector = _mm256_add_pd(sum_vector,mul);
                    filter_vector = _mm256_loadu_pd((double const *)filter_n + 4);
                    in_vector = _mm256_loadu_pd((double const *)in_n + 4);
                    mul =  _mm256_mul_pd(filter_vector, in_vector);
                    sum_vector = _mm256_add_pd(sum_vector,mul);
                    filter_vector = _mm256_loadu_pd((double const *)filter_n + 8);
                    in_vector = _mm256_loadu_pd((double const *)in_n + 8);
                    mul =  _mm256_mul_pd(filter_vector, in_vector);
                    sum_vector = _mm256_add_pd(sum_vector,mul);
                    filter_vector = _mm256_loadu_pd((double const *)filter_n + 12);
                    in_vector = _mm256_loadu_pd((double const *)in_n + 12);
                    mul =  _mm256_mul_pd(filter_vector, in_vector);
                    sum_vector = _mm256_add_pd(sum_vector,mul);
                    filter_vector = _mm256_loadu_pd((double const *)filter_n + 16);
                    in_vector = _mm256_loadu_pd((double const *)in_n + 16);
                    mul =  _mm256_mul_pd(filter_vector, in_vector);
                    sum_vector = _mm256_add_pd(sum_vector,mul);
                    sum += sum_vector[0] + sum_vector[1] + sum_vector[2] + sum_vector[3];

              }
              else if (filter_depth == 3) {
                __m256d filter_vector = _mm256_loadu_pd((double const *)filter_n);
                __m256d in_vector = _mm256_loadu_pd((double const *)in_n);
                __m256d mul =  _mm256_mul_pd(filter_vector, in_vector);
                sum_vector = _mm256_add_pd(sum_vector,mul);
                sum += sum_vector[0] + sum_vector[1] + sum_vector[2];
              }
              }
            }
          }

          sum += l->biases->weights[f];
          out->weights[((out->width * out_y) + out_x) * out->depth + f] = sum;
        }
      }
    }
}


void conv_load(conv_layer_t* l, const char* file_name) {
  int filter_width;
  int filter_height;
  int depth;
  int filters;

  FILE* fin = fopen(file_name, "r");

  fscanf(fin, "%d %d %d %d", &filter_width, &filter_height, &depth, &filters);
  assert(filter_width == l->filter_width);
  assert(filter_height == l->filter_height);
  assert(depth == l->input_depth);
  assert(filters == l->output_depth);

  for (int f = 0; f < filters; f++) {
    volume_t* fil = l->filters[f];
    double* fil_weights = fil->weights;
    int fil_width = fil->width;
    int fil_depth = fil->depth;

    for (int x = 0; x < filter_width; x++) {
      for (int y = 0; y < filter_height; y++) {
        for (int d = 0; d < depth; d++) {
          double val;
          fscanf(fin, "%lf", &val);
          fil_weights[((fil_width * y) + x) * fil_depth + d] = val;
        }
      }
    }
  }

  double* biase_weights = l->biases->weights;
  for (int d = 0; d < l->output_depth; d++) {
    double val;
    fscanf(fin, "%lf", &val);
    biase_weights[d] = val;
  }

  fclose(fin);
}

relu_layer_t* make_relu_layer(int input_width, int input_height, int input_depth) {
  relu_layer_t* l = (relu_layer_t*)malloc(sizeof(relu_layer_t));

  l->input_depth  = input_depth;
  l->input_width  = input_width;
  l->input_height = input_height;

  l->output_width  = input_width;
  l->output_height = input_height;
  l->output_depth  = input_depth;

  return l;
}

// Applies the Rectifier Linear Unit (ReLU) function to the input, which sets
// output(x, y, d) to max(0.0, input(x, y, d)).
void relu_forward(relu_layer_t* l, volume_t** inputs, volume_t** outputs, int start, int end) {
  __m256d zero = _mm256_setzero_pd();
  int size = l->input_height * l->input_width * l->input_depth;
  for (int i = start; i <= end; i++) {
    volume_t* input = inputs[i];
    volume_t* output = outputs[i];

    double* input_weights = input->weights;
    double* output_weights = output->weights;
    for (int i = 0; i < size; i+= 64) {
      __m256d input_vec = _mm256_loadu_pd(input_weights + i);
      _mm256_storeu_pd(output_weights + i, _mm256_max_pd(input_vec, zero));
      input_vec = _mm256_loadu_pd(input_weights + i + 4);
      _mm256_storeu_pd(output_weights + i + 4, _mm256_max_pd(input_vec, zero));
      input_vec = _mm256_loadu_pd(input_weights + i + 8);
      _mm256_storeu_pd(output_weights + i + 8, _mm256_max_pd(input_vec, zero));
      input_vec = _mm256_loadu_pd(input_weights + i + 12);
      _mm256_storeu_pd(output_weights + i + 12, _mm256_max_pd(input_vec, zero));
      input_vec = _mm256_loadu_pd(input_weights + i + 16);
      _mm256_storeu_pd(output_weights + i + 16, _mm256_max_pd(input_vec, zero));
      input_vec = _mm256_loadu_pd(input_weights + i + 20);
      _mm256_storeu_pd(output_weights + i + 20, _mm256_max_pd(input_vec, zero));
      input_vec = _mm256_loadu_pd(input_weights + i + 24);
      _mm256_storeu_pd(output_weights + i + 24, _mm256_max_pd(input_vec, zero));
      input_vec = _mm256_loadu_pd(input_weights + i + 28);
      _mm256_storeu_pd(output_weights + i + 28, _mm256_max_pd(input_vec, zero));
      input_vec = _mm256_loadu_pd(input_weights + i + 32);
      _mm256_storeu_pd(output_weights + i + 32, _mm256_max_pd(input_vec, zero));
      input_vec = _mm256_loadu_pd(input_weights + i + 36);
      _mm256_storeu_pd(output_weights + i + 36, _mm256_max_pd(input_vec, zero));
      input_vec = _mm256_loadu_pd(input_weights + i + 40);
      _mm256_storeu_pd(output_weights + i + 40, _mm256_max_pd(input_vec, zero));
      input_vec = _mm256_loadu_pd(input_weights + i + 44);
      _mm256_storeu_pd(output_weights + i + 44, _mm256_max_pd(input_vec, zero));
      input_vec = _mm256_loadu_pd(input_weights + i + 48);
      _mm256_storeu_pd(output_weights + i + 48, _mm256_max_pd(input_vec, zero));
      input_vec = _mm256_loadu_pd(input_weights + i + 52);
      _mm256_storeu_pd(output_weights + i + 52, _mm256_max_pd(input_vec, zero));
      input_vec = _mm256_loadu_pd(input_weights + i + 56);
      _mm256_storeu_pd(output_weights + i + 56, _mm256_max_pd(input_vec, zero));
      input_vec = _mm256_loadu_pd(input_weights + i + 60);
      _mm256_storeu_pd(output_weights + i + 60, _mm256_max_pd(input_vec, zero));
    }
  }
}

pool_layer_t* make_pool_layer(int input_width, int input_height, int input_depth, int pool_width, int stride) {
  pool_layer_t* l = (pool_layer_t*)malloc(sizeof(pool_layer_t));

  l->pool_width   = pool_width;
  l->input_depth  = input_depth;
  l->input_width  = input_width;
  l->input_height = input_height;

  l->pool_height = l->pool_width;
  l->stride      = stride;
  l->pad         = 0;

  l->output_depth  = input_depth;

  l->output_height = l->output_width  = (input_width  - pool_width) / stride + 1;

  return l;
}

// This is like the convolutional layer in that we are sliding across the input
// volume, but instead of having a filter that we use to find the sum of an
// elementwise product, we instead just output the max value of some part of
// the image. For instance, if we consider a 2D case where the following is the
// part of the input that we are considering:
//
//     1 3 5
//     4 2 1
//     2 2 2
//
// then the value of the corresponding element in the output is 5 (since that
// is the maximum element). This effectively compresses the input.
void pool_forward(pool_layer_t* l, volume_t** inputs, volume_t** outputs, int start, int end) {
    volume_t* in  = inputs[0];
    volume_t* out = outputs[0];
    double* out_weights = out->weights;
    double* in_weights = in->weights;
    int in_width = in->width;
    int in_depth = in->depth;
    int out_width = out->width;
    int out_depth = out->depth;
    int output_depth = l->output_depth;
    int output_width = l->output_width;
    for (int d = 0; d < output_depth; d++) {
      int x = 0;
      for (int out_x = 0; out_x < output_width; x += 2, out_x++) {
        int y = 0;
        for (int out_y = 0; out_y < output_width; y += 2, out_y++) {

          double max = -INFINITY;
          for (int fy = y; fy < 2 + y; fy++)  {
            for (int fx = x; fx < 2 + x; fx++) {
              double v = in_weights[((in_width * fy) + fx) * in_depth + d];
              if (v > max) {
                max = v;
              }
            }
          }

          out_weights[((out_width * out_y) + out_x) * out_depth + d] = max;
        }
      }
    }
}

fc_layer_t* make_fc_layer(int input_width, int input_height, int input_depth, int num_neurons) {
  fc_layer_t* l = (fc_layer_t*)malloc(sizeof(fc_layer_t));

  l->output_depth = num_neurons;
  l->input_depth  = input_depth;
  l->input_width  = input_width;
  l->input_height = input_height;

  int num_inputs = l->num_inputs = l->input_width * l->input_height * l->input_depth;
  l->output_width  = 1;
  l->output_height = 1;

  l->filters = (volume_t**)malloc(sizeof(volume_t*) * num_neurons);
  volume_t** l_filters = l->filters;
  for (int i = 0; i < num_neurons; i++) {
    l_filters[i] = make_volume(1, 1, num_inputs, 0.0);
  }

  l->bias   = 0.0;
  l->biases = make_volume(1, 1, num_neurons, 0.0);

  return l;
}

// Computes the dot product (i.e. the sum of the elementwise product) of the
// input's weights with each of the filters. Note that these filters are not
// the same as the filters for the convolutional layer.
void fc_forward(fc_layer_t* l, volume_t** inputs, volume_t** outputs, int start, int end) {
    volume_t* in  = inputs[0];
    volume_t* out = outputs[0];
    double* in_weights = in->weights;
    volume_t** l_filters = l->filters;
    double* l_biases_weight = l->biases->weights;
    double* out_weights = out->weights;
    for (int i = 0; i < 10; i++) {
      double* l_weights = l_filters[i]->weights;
      double dot = 0.0;
      __m256d dot_vector = _mm256_setzero_pd();
      for (int d = 0; d < 320; d += 32) {
        __m256d in_vector = _mm256_loadu_pd(in_weights + d);
        __m256d l_vector = _mm256_loadu_pd(l_weights + d);
        __m256d mul =  _mm256_mul_pd(in_vector, l_vector);
        dot_vector = _mm256_add_pd(dot_vector, mul);
        in_vector = _mm256_loadu_pd(in_weights + d + 4);
        l_vector = _mm256_loadu_pd(l_weights + d + 4);
        mul =  _mm256_mul_pd(in_vector, l_vector);
        dot_vector = _mm256_add_pd(dot_vector, mul);
        in_vector = _mm256_loadu_pd(in_weights + d + 8);
        l_vector = _mm256_loadu_pd(l_weights + d + 8);
        mul =  _mm256_mul_pd(in_vector, l_vector);
        dot_vector = _mm256_add_pd(dot_vector, mul);
        in_vector = _mm256_loadu_pd(in_weights + d + 12);
        l_vector = _mm256_loadu_pd(l_weights + d + 12);
        mul =  _mm256_mul_pd(in_vector, l_vector);
        dot_vector = _mm256_add_pd(dot_vector, mul);
        in_vector = _mm256_loadu_pd(in_weights + d + 16);
        l_vector = _mm256_loadu_pd(l_weights + d + 16);
        mul =  _mm256_mul_pd(in_vector, l_vector);
        dot_vector = _mm256_add_pd(dot_vector, mul);
        in_vector = _mm256_loadu_pd(in_weights + d + 20);
        l_vector = _mm256_loadu_pd(l_weights + d + 20);
        mul =  _mm256_mul_pd(in_vector, l_vector);
        dot_vector = _mm256_add_pd(dot_vector, mul);
        in_vector = _mm256_loadu_pd(in_weights + d + 24);
        l_vector = _mm256_loadu_pd(l_weights + d + 24);
        mul =  _mm256_mul_pd(in_vector, l_vector);
        dot_vector = _mm256_add_pd(dot_vector, mul);
        in_vector = _mm256_loadu_pd(in_weights + d + 28);
        l_vector = _mm256_loadu_pd(l_weights + d + 28);
        mul =  _mm256_mul_pd(in_vector, l_vector);
        dot_vector = _mm256_add_pd(dot_vector, mul);
      }
      dot += dot_vector[0] + dot_vector[1] + dot_vector[2] + dot_vector[3] + l_biases_weight[i];
      out_weights[i] = dot;
    }
}

void fc_load(fc_layer_t* l, const char* filename) {
  FILE* fin = fopen(filename, "r");

  int num_inputs;
  int output_depth;
  fscanf(fin, "%d %d", &num_inputs, &output_depth);
  assert(output_depth == l->output_depth);
  assert(num_inputs == l->num_inputs);

  for (int i = 0; i < output_depth; i++) {
    for (int j = 0; j < num_inputs; j++) {
      fscanf(fin, "%lf", &(l->filters[i]->weights[j]));
    }
  }

  for (int i = 0; i < l->output_depth; i++) {
    fscanf(fin, "%lf", &(l->biases->weights[i]));
  }

  fclose(fin);
}

softmax_layer_t* make_softmax_layer(int input_width, int input_height, int input_depth) {
  softmax_layer_t* l = (softmax_layer_t*)malloc(sizeof(softmax_layer_t));

  l->input_depth  = input_depth;
  l->input_width  = input_width;
  l->input_height = input_height;

  l->output_width  = 1;
  l->output_height = 1;
  int l_output_depth = l->output_depth  = input_width * input_height * input_depth;

  l->likelihoods = (double*)malloc(sizeof(double) * l_output_depth);

  return l;
}

// This function converts an input's weights array into a probability
// distribution by using the following formula:
//
// likelihood[i] = exp(in->weights[i]) / sum(exp(in->weights))
//
// To increase the numerical stability of taking the exponential of a value, we
// subtract the maximum input weights from each weight before taking the
// exponential. This yields exactly the same results as the expression above,
// but is more resilient to floating point errors.
void softmax_forward(softmax_layer_t* l, volume_t** inputs, volume_t** outputs, int start, int end) {
  double likelihoods[l->output_depth];
  int out_depth = l->output_depth;
  double* in_weights = &(inputs[0]->weights[0]);
  double* out_weights = &(outputs[0]->weights[0]);

    // Compute max activation (used to compute exponentials)
    double amax = *(in_weights);
    for (double* i = in_weights; i < in_weights + out_depth; i++) {
      if (*(i) > amax) {
        amax = *(i);
      }
    }

    // Compute exponentials in a numerically stable way
    double total = 0.0;
    int s = 0;
    for (double* i = in_weights; i < in_weights + out_depth; i++, s++) {
      double e = exp(*(i) - amax);
      total += e;
      likelihoods[s] = e;
    }

    // Normalize and output to sum to one
    s = 0;
    for (double* i = out_weights; i < out_weights + out_depth; i++, s++) {
      *(i) = likelihoods[s] / total;
    }
}
