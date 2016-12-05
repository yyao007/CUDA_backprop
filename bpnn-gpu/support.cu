#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "support.h"

/*** Allocate 1d array of floats ***/

float *alloc_1d_dbl(int n)
{
  float *new1;

  new1 = (float *) malloc ((unsigned) (n * sizeof (float)));
  if (new1 == NULL) {
    printf("ALLOC_1D_DBL: Couldn't allocate array of floats\n");
    return (NULL);
  }
  return (new1);
}


/*** Allocate 2d array of floats ***/

float **alloc_2d_dbl(int m, int n)
{
  int i;
  float **new1;

  new1 = (float **) malloc ((unsigned) (m * sizeof (float *)));
  if (new1 == NULL) {
    printf("ALLOC_2D_DBL: Couldn't allocate array of dbl ptrs\n");
    return (NULL);
  }

  for (i = 0; i < m; i++) {
    new1[i] = alloc_1d_dbl(n);
  }

  return (new1);
}


void bpnn_randomize_weights(float **w, int m, int n)
{
  int i, j;

  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
     w[i][j] = (float) rand()/RAND_MAX;
    //  w[i][j] = dpn1();
    }
  }
}

void bpnn_randomize_row(float *w, int m)
{
  int i;
  for (i = 0; i <= m; i++) {
     //w[i] = (float) rand()/RAND_MAX;
   w[i] = 0.1;
    }
}


void bpnn_zero_weights(float **w, int m, int n)
{
  int i, j;

  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
      w[i][j] = 0.0;
    }
  }
}


void bpnn_initialize(int seed)
{
  printf("Random number generator seed: %d\n", seed);
  srand(seed);
}


BPNN *bpnn_internal_create(int n_in, int n_hidden, int n_out)
{
  BPNN *newnet;

  newnet = (BPNN *) malloc (sizeof (BPNN));
  if (newnet == NULL) {
    printf("BPNN_CREATE: Couldn't allocate neural network\n");
    return (NULL);
  }

  newnet->input_n = n_in;
  newnet->hidden_n = n_hidden;
  newnet->output_n = n_out;
  newnet->input_units = alloc_1d_dbl(n_in + 1);
  newnet->hidden_units = alloc_1d_dbl(n_hidden + 1);
  newnet->output_units = alloc_1d_dbl(n_out + 1);

  newnet->hidden_delta = alloc_1d_dbl(n_hidden + 1);
  newnet->output_delta = alloc_1d_dbl(n_out + 1);
  newnet->target = alloc_1d_dbl(n_out + 1);

  newnet->input_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
  newnet->hidden_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);

  newnet->input_prev_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
  newnet->hidden_prev_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);

  return (newnet);
}


void bpnn_free(BPNN *net)
{
  int n1, n2, i;

  n1 = net->input_n;
  n2 = net->hidden_n;

  free((char *) net->input_units);
  free((char *) net->hidden_units);
  free((char *) net->output_units);

  free((char *) net->hidden_delta);
  free((char *) net->output_delta);
  free((char *) net->target);

  for (i = 0; i <= n1; i++) {
    free((char *) net->input_weights[i]);
    free((char *) net->input_prev_weights[i]);
  }
  free((char *) net->input_weights);
  free((char *) net->input_prev_weights);

  for (i = 0; i <= n2; i++) {
    free((char *) net->hidden_weights[i]);
    free((char *) net->hidden_prev_weights[i]);
  }
  free((char *) net->hidden_weights);
  free((char *) net->hidden_prev_weights);

  free((char *) net);
}


/*** Creates a new fully-connected network from scratch,
     with the given numbers of input, hidden, and output units.
     Threshold units are automatically included.  All weights are
     randomly initialized.

     Space is also allocated for temporary storage (momentum weights,
     error computations, etc).
***/

BPNN *bpnn_create(int n_in, int n_hidden, int n_out)
{

  BPNN *newnet;

  newnet = bpnn_internal_create(n_in, n_hidden, n_out);

#ifdef INITZERO
  bpnn_zero_weights(newnet->input_weights, n_in, n_hidden);
#else
  bpnn_randomize_weights(newnet->input_weights, n_in, n_hidden);
#endif
  bpnn_randomize_weights(newnet->hidden_weights, n_hidden, n_out);
  bpnn_zero_weights(newnet->input_prev_weights, n_in, n_hidden);
  bpnn_zero_weights(newnet->hidden_prev_weights, n_hidden, n_out);
  bpnn_randomize_row(newnet->target, n_out);
  return (newnet);
}

// BPNN *bpnn_allocate_device(int in, int hid, int out) {
//   cudaError_t cuda_ret;
//   BPNN *net_d;

//   cuda_ret = cudaMalloc((void**)&net_d, sizeof(BPNN));
//   if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
//   // cuda_ret = cudaMalloc((void*)&net_d->input_n, sizeof(int));
//   // if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
//   // cuda_ret = cudaMalloc((void*)&net_d->hidden_n, sizeof(int));
//   // if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
//   // cuda_ret = cudaMalloc((void*)&net_d->output_n, sizeof(int));
//   // if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
//   cuda_ret = cudaMalloc((void**)&(net_d->input_units), (in+1)*sizeof(float));
//   if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
//   cuda_ret = cudaMalloc((void**)&net_d->hidden_units, (hid+1)*sizeof(float));
//   if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
//   cuda_ret = cudaMalloc((void**)&net_d->output_units, (out+1)*sizeof(float));
//   if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
//   cuda_ret = cudaMalloc((void**)&net_d->hidden_delta, (hid+1)*sizeof(float));
//   if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
//   cuda_ret = cudaMalloc((void**)&net_d->output_delta, (out+1)*sizeof(float));
//   if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
//   cuda_ret = cudaMalloc((void**)&net_d->target, (out+1)*sizeof(float));
//   if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

//   cuda_ret = cudaMalloc((void***)&net_d->input_weights, (in+1)*sizeof(float));
//   if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
//   for (int i = 0; i <= in; ++i) {
//     cuda_ret = cudaMalloc((void**)&net_d->input_weights[i], (hid+1)*sizeof(float));
//     if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
//   }

//   cuda_ret = cudaMalloc((void***)&net_d->hidden_weights, (hid+1)*sizeof(float));
//   if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
//   for (int i = 0; i <= hid; ++i) {
//     cuda_ret = cudaMalloc((void**)&net_d->hidden_weights[i], (out+1)*sizeof(float));
//     if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
//   }

//   cuda_ret = cudaMalloc((void***)&net_d->input_prev_weights, (in+1)*sizeof(float));
//   if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
//   for (int i = 0; i <= in; ++i) {
//     cuda_ret = cudaMalloc((void**)&net_d->input_prev_weights[i], (hid+1)*sizeof(float));
//     if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
//   }

//   cuda_ret = cudaMalloc((void***)&net_d->hidden_prev_weights, (hid+1)*sizeof(float));
//   if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
//   for (int i = 0; i <= hid; ++i) {
//     cuda_ret = cudaMalloc((void**)&net_d->hidden_prev_weights[i], (out+1)*sizeof(float));
//     if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
//   }

//   return net_d;
// }

void bpnn_copy_device(BPNN *net_d, BPNN *net_h, int in, int hid, int out) {
  cudaError_t cuda_ret;
  
  cuda_ret = cudaMemcpy(&net_d->input_n, &net_h->input_n, sizeof(int), cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
  cuda_ret = cudaMemcpy(&net_d->hidden_n, &net_h->hidden_n, sizeof(int), cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
  cuda_ret = cudaMemcpy(&net_d->output_n, &net_h->output_n, sizeof(int), cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
  cuda_ret = cudaMemcpy(net_d->input_units, net_h->input_units, (in+1)*sizeof(float), cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
  cuda_ret = cudaMemcpy(net_d->hidden_units, net_h->hidden_units, (hid+1)*sizeof(float), cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
  cuda_ret = cudaMemcpy(net_d->output_units, net_h->output_units, (out+1)*sizeof(float), cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
  cuda_ret = cudaMemcpy(net_d->hidden_delta, net_h->hidden_delta, (hid+1)*sizeof(float), cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
  cuda_ret = cudaMemcpy(net_d->output_delta, net_h->output_delta, (out+1)*sizeof(float), cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
  cuda_ret = cudaMemcpy(net_d->target, net_h->target, (out+1)*sizeof(float), cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");

  for (int i = 0; i <= in; ++i) {
    cuda_ret = cudaMemcpy(net_d->input_weights[i], net_h->input_weights[i], (hid+1)*sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
  }
  for (int i = 0; i <= hid; ++i) {
    cuda_ret = cudaMemcpy(net_d->hidden_weights[i], net_h->hidden_weights[i], (out+1)*sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
  }
  for (int i = 0; i <= in; ++i) {
    cuda_ret = cudaMemcpy(net_d->input_prev_weights[i], net_h->input_prev_weights[i], (hid+1)*sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
  }
  for (int i = 0; i <= hid; ++i) {
    cuda_ret = cudaMemcpy(net_d->hidden_prev_weights[i], net_h->hidden_prev_weights[i], (out+1)*sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
  }
}

void load(BPNN *net, int layer_size)
{
  float *units;
  int nr, nc, imgsize, i, j, k;

  nr = layer_size;
  
  imgsize = nr * nc;
  units = net->input_units;

  k = 1;
  for (i = 0; i < nr; i++) {
    units[k] = (float) rand()/RAND_MAX ;
    k++;
    }
}

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

