#ifndef __FILEH__
#define __FILEH__

#include <sys/time.h>

#define BIGRND 0x7fffffff

#define ETA 0.3       //eta value
#define MOMENTUM 0.3  //momentum value


typedef struct {
  int input_n;                  /* number of input units */
  int hidden_n;                 /* number of hidden units */
  int output_n;                 /* number of output units */

  float *input_units;          /* the input units */
  float *hidden_units;         /* the hidden units */
  float *output_units;         /* the output units */

  float *hidden_delta;         /* storage for hidden unit error */
  float *output_delta;         /* storage for output unit error */

  float *target;               /* storage for target vector */

  float *input_weights;       /* weights from input to hidden layer */
  float *hidden_weights;      /* weights from hidden to output layer */

                                /*** The next two are for momentum ***/
  float *input_prev_weights;  /* previous change on input to hidden wgt */
  float *hidden_prev_weights; /* previous change on hidden to output wgt */
} BPNN;


/*** User-level functions ***/

void bpnn_initialize(int);

BPNN *bpnn_create(int, int, int);
BPNN *bpnn_allocate_device(int, int, int);
void bpnn_copy_device(BPNN *, BPNN *, int, int, int);
void bpnn_train_kernel_device1(BPNN *);
void bpnn_train_kernel_device(BPNN *);
void bpnn_free(BPNN *);
void bpnn_save_dbg(BPNN *, const char *);
void compare_result(const char *, const char *);

void load(BPNN *, int);

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

#ifdef __cplusplus
extern "C" {
#endif

void startTime(Timer* timer);
void stopTime(Timer* timer);
float elapsedTime(Timer timer);

#ifdef __cplusplus
}
#endif

#define FATAL(msg, ...) \
    do {\
        fprintf(stderr, "[%s:%d] "msg"\n", __FILE__, __LINE__, ##__VA_ARGS__);\
        exit(-1);\
    } while(0)

#if __BYTE_ORDER != __LITTLE_ENDIAN
# error "File I/O is not implemented for this system: wrong endianness."
#endif

#endif
