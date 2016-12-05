#include "support.h"

#define BLOCK_SIZE 512
#define GRID_SIZE 512


__device__ float ABS(float x) {
	return ((x > 0.0)? x : -x);
}        

/*** The squashing function.  Currently, it's a sigmoid. ***/
__device__ float squash(float x) {
  return (1.0 / (1.0 + exp(-x)));
}

__global__ void bpnn_layerforward_kernel(float *l1, float *l2, float *w, float *middle, int n1, int n2) {
	__shared__ float private_w[BLOCK_SIZE*2], private_l1[BLOCK_SIZE*2], private_l2[16+1];

	int stride = 2 * blockDim.x * gridDim.x;
	int start = 2 * blockDim.x * blockIdx.x;
	int t = threadIdx.x;
	
	/*** Set up thresholding unit ***/
	if (start + t == 0){
		l1[0] = 1.0;
	}
	if (start + t <= n2) {
		private_l2[t] = 0.0;
	}
	__syncthreads();

	for (int i = 0; i <= n1; i += stride) {
		private_l1[t] = (i+start+t <= n1)? l1[i+start+t] : 0.0;
		private_l1[t+blockDim.x] = (i+start+t+blockDim.x <= n1)? l1[i+start+t+blockDim.x] : 0.0;
		__syncthreads();

		for (int j = 1; j <= n2; ++j) {
			private_w[t] = ((i+start+t) <= n1)? w[(i+start+t)*(n2+1) + j] : 0.0;
			private_w[t+blockDim.x] = ((i+start+t+blockDim.x) <= n1)? w[(i+start+t+blockDim.x)*(n2+1) + j] : 0.0;
			__syncthreads();

			// multiply private_w and private_l1
			private_w[t] *= private_l1[t];
			private_w[t+blockDim.x] *= private_l1[t+blockDim.x];
			__syncthreads();

			// reduction part
			for (int rstride = blockDim.x; rstride > 0; rstride >>= 1) {
		    	if (t < rstride) {
		    		private_w[t] += private_w[t+rstride];
		    	}
		    	__syncthreads();
    		}

			if (n1 < 2*BLOCK_SIZE) {
				if (start + t == 0) {
		    		private_l2[j] += private_w[0];	
		    	}
		    	continue;
	    	}	

		    if (t == 0) {
		    	middle[blockIdx.x] = private_w[0];
		    }
	    	__syncthreads();

		    // take another reduction on middle to compute final result.
		    private_w[t] = (start+t < GRID_SIZE)? middle[start+t] : 0.0;
		    private_w[t+blockDim.x] = (start+t+blockDim.x < GRID_SIZE)? middle[start+t+blockDim.x] : 0.0;
		    __syncthreads();

		    for (int rstride = blockDim.x; rstride > 0; rstride >>= 1) {
		    	if (t < rstride) {
		    		private_w[t] += private_w[t+rstride];
		    	}
		    	__syncthreads();
		    }
		    
		    // store into l2
		    if (start + t == 0) {
		    	private_l2[j] += private_w[0];	
		    }
		    __syncthreads();
		}
	}

	if (start + t <= n2 && start + t > 0) {
		l2[start+t] = squash(private_l2[start+t]);
		// printf("l2[%d] = %f\n", start+t, l2[start+t]);
	}

}

__global__ void bpnn_output_error_kernel(float *delta, float *target, float *output, 
									int nj, float *err) {
	int j;
	float o, t, errsum;
	if (threadIdx.x == 0) {
		errsum = 0.0;
		for (j = 1; j <= nj; j++) {
			o = output[j];
			t = target[j];
			delta[j] = o * (1.0 - o) * (t - o);
			errsum += ABS(delta[j]);
		}
		// *err = errsum;
	}
}

__global__ void bpnn_hidden_error_kernel(float *delta_h, int nh, float *delta_o, int no, 
						float *who, float *hidden, float *err) {
	int k;
	float sum;
	int t = threadIdx.x;

	__shared__ float h[BLOCK_SIZE], d_o[BLOCK_SIZE], errsum;

	if (t == 0) {
		errsum = 0.0;
	}

  	if (t > 0 && t <= no) {
  		d_o[t] = delta_o[t];
	}

	__syncthreads();

  	if (t <= nh) {
	  	h[t] = hidden[t];
	  	sum = 0.0;
	  	for (k = 1; k <= no; ++k) {
	  		sum += d_o[k] * who[t*(no+1)+k];
	  	}
  		delta_h[t] = h[t] * (1.0 - h[t]) * sum;
  		// printf("hidden_delta[%d] = %f\n", t, delta_h[t]);
  		atomicAdd(&errsum, ABS(delta_h[t]));
  	}
  	__syncthreads();

  	if (t == 0) {
  		// *err = errsum;
  	}
}	

__global__ void bpnn_adjust_weights_kernel(float *delta, int ndelta, float *ly, 
									int nly, float *w, float *oldw) {
	
	__shared__ float private_ly[BLOCK_SIZE*2], private_delta[16+1];
	float new_dw;

	int stride = 2 * blockDim.x * gridDim.x;
	int start = 2 * blockDim.x * blockIdx.x;
	int t = threadIdx.x;

	// initialize shared memory
	if (t == 0) {
		ly[0] = 1.0;
	}
	if (start + t <= ndelta) {
		private_delta[start+t] = delta[start+t];
	}
	__syncthreads();

	for (int i = 0; i <= nly; i += stride) {
		int k = i + start + t;
		private_ly[t] = (k <= nly)? ly[k] : 0.0;
		private_ly[t+blockDim.x] = (k+blockDim.x <= nly)? ly[k+blockDim.x] : 0.0;
		__syncthreads();

		for (int j = 1; j <= ndelta; ++j) {
			int count = 0;
			while (count < 2 && k <= nly) {
				new_dw = ETA * private_delta[j] * private_ly[t] + MOMENTUM * oldw[k*(ndelta+1)+j];
				w[k*(ndelta+1)+j] += new_dw;
				oldw[k*(ndelta+1)+j] = new_dw;	
				// printf("w[%d][%d] = %f\n", start+t, 0, w[k*(ndelta+1)+0]);	
				// printf("w[%d][%d] = %f\n", start+t, j, w[k*(ndelta+1)+j]);	
				k += blockDim.x;
				count += 1;
			}
			__syncthreads();
		}
	}
}


void bpnn_train_kernel_device(int in_n, int hid_n, int out_n, float *input_units, float *hidden_units, 
					float *output_units, float *hidden_delta, float *output_delta, float *target, 
					float *input_weights, float *hidden_weights, float *input_prev_weights, 
					float *hidden_prev_weights) {
	
	float out_err, hid_err; 
	float *middle;
	cudaError_t cuda_ret;

	printf("Performing GPU computation...");

	cuda_ret = cudaMalloc((void**)&(middle), (GRID_SIZE)*sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMemset(hidden_units, 0, hid_n*sizeof(int));
	if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory");

	// dim3 grid_2d()
	dim3 grid(GRID_SIZE, 1, 1);
	dim3 block(BLOCK_SIZE, 1, 1);

	bpnn_layerforward_kernel<<<grid, block>>>(input_units, hidden_units, input_weights, middle, in_n, hid_n);
	bpnn_layerforward_kernel<<<grid, block>>>(hidden_units, output_units, hidden_weights, middle, hid_n, out_n);
	bpnn_output_error_kernel<<<1, BLOCK_SIZE>>>(output_delta, target, output_units, out_n, &out_err);
	bpnn_hidden_error_kernel<<<1, BLOCK_SIZE>>>(hidden_delta, hid_n, output_delta, out_n, hidden_weights, hidden_units, &hid_err);
	bpnn_adjust_weights_kernel<<<grid, block>>>(output_delta, out_n, hidden_units, hid_n, hidden_weights, hidden_prev_weights);
	bpnn_adjust_weights_kernel<<<grid, block>>>(hidden_delta, hid_n, input_units, in_n, input_weights, input_prev_weights);
}


