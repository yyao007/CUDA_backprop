#define BLOCK_SIZE 512
#define GRID_SIZE 512


/*** The squashing function.  Currently, it's a sigmoid. ***/
__device__ float squash(float x)
{
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
			private_w[t] *= l1[t];
			private_w[t+blockDim.x] *= l1[t+blockDim.x];
			__syncthreads();

			int size = n1;
			int block = 1;
    		while (true) {
    			// reduction part
				for (int rstride = blockDim.x; rstride > 0; rstride >>= 1) {
			    	if (t < rstride) {
			    		private_w[t] += private_w[t+rstride];
			    	}
			    	__syncthreads();
	    		}

				if (size < 2*BLOCK_SIZE) {
					__syncthreads();
		    		break;
		    	}			    
			    if (t == 0) {
			    	size = GRID_SIZE/block;
			    	block *= BLOCK_SIZE;
			    	middle[blockIdx.x] = private_w[0];
			    }
		    	__syncthreads();

			    // take another reduction on middle to compute final result.
			    private_w[t] = (start+t < GRID_SIZE)? middle[start+t] : 0.0;
			    private_w[t+blockDim.x] = (start+t+blockDim.x < GRID_SIZE)? middle[start+t+blockDim.x] : 0.0;
			    __syncthreads();
		    }
		    // store into l2
		    if (start + t == 0) {
		    	l2[j] += private_w[0];	
		    }
		    __syncthreads();
		}
	}

	if (start + t <= n2 && start + t > 0) {
		l2[start+t] = squash(l2[start+t]);
		printf("l2[%d] = %f\n", start+t, l2[start+t]);
	}

}




void bpnn_train_kernel_device(int in_n, int hid_n, int out_n, float *input_units, float *hidden_units, 
					float *output_units, float *hidden_delta, float *output_delta, float *target, 
					float *input_weights, float *hidden_weights, float *input_prev_weights, 
					float *hidden_prev_weights) {
	
	float out_err, hid_err; 
	float *middle;
	cudaError_t cuda_ret;

	printf("Performing GPU computation\n");

	// cuda_ret = cudaMemset(&in, in_n, sizeof(int));
	// if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory");
	// cuda_ret = cudaMemset(&hid, hid_n, sizeof(int));
	// if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory");
	// cuda_ret = cudaMemset(&out, out_n, sizeof(int));
	// if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory");

	cuda_ret = cudaMalloc((void**)&(middle), (GRID_SIZE)*sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMemset(hidden_units, 0, hid_n*sizeof(int));
	if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory");

	// dim3 grid_2d()
	dim3 grid_in(GRID_SIZE, 1, 1);
	dim3 block_in(BLOCK_SIZE, 1, 1);

	bpnn_layerforward_kernel<<<grid_in, block_in>>>(input_units, hidden_units, input_weights, middle, in_n, hid_n);
	

	// bpnn_layerforward(net->input_units, net->hidden_units,net->input_weights, in, hid);
	// bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
	// bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
	// bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  
	// bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);
	// bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in, net->input_weights, net->input_prev_weights);
}


