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

float *alloc_2d_dbl(int m, int n)
{
  float *new1;

  new1 = (float *) malloc ((unsigned) (m * n * sizeof (float)));
  if (new1 == NULL) {
    printf("ALLOC_2D_DBL: Couldn't allocate array of dbl ptrs\n");
    return (NULL);
  }
  return (new1);
}


void bpnn_randomize_weights(float *w, int m, int n)
{
  int i, j;

  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
     w[i*(n+1)+j] = (float) rand()/RAND_MAX;
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


void bpnn_zero_weights(float *w, int m, int n)
{
  int i, j;

  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
      w[i*(n+1)+j] = 0.0;
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
  free(net->input_units);
  free(net->hidden_units);
  free(net->output_units);

  free(net->hidden_delta);
  free(net->output_delta);
  free(net->target);

  free(net->input_weights);
  free(net->input_prev_weights);
  free(net->hidden_weights);
  free(net->hidden_prev_weights);

  free(net);
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


void load(BPNN *net, int layer_size)
{
  float *units;
  int nr, i, k;

  nr = layer_size;
  units = net->input_units;

  k = 1;
  for (i = 0; i < nr; i++) {
    units[k] = (float) rand()/RAND_MAX ;
    k++;
    }
}

void bpnn_save_dbg(BPNN *net, const char *filename) {
  int n1, n2, n3, i, j;
  float *w;

  FILE *pFile;
  pFile = fopen( filename, "w+" );

  n1 = net->input_n;  n2 = net->hidden_n;  n3 = net->output_n;
  fprintf(pFile, "Saving %dx%dx%d network\n", n1, n2, n3);

  w = net->hidden_weights;
  for (i = 0; i <= n2; i++) {
    for (j = 0; j <= n3; j++) {
      fprintf(pFile, "%d,%d,%f\n", i,j,w[i*(n3+1)+j]);
    }
  }

  fclose(pFile);
  return;
}

// void compare_result(const char *gpu, const char *cpu) {
//   fd_gpu = fopen(gpu, 'r');
//   fd_cpu = fopen(cpu, 'r');

  
// }

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

