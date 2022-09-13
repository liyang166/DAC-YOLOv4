#include "avgpool_layer.h"
#include "dark_cuda.h"
#include "utils.h"
#include <stdio.h>
#include "convolutional_layer.h"
#include "gemm.h"

avgpool_layer make_avgpool_layer(int batch, int w, int h, int c, int channelpool)
{

	avgpool_layer l = { (LAYER_TYPE)0 };
	l.type = AVGPOOL;
	l.batch = batch;
	l.h = h;
	l.w = w;
	l.c = c;
	l.channelpool = channelpool;
	l.inputs = h*w*c;
	//根据配置设置输出尺寸
	if (!channelpool) {
		l.out_w = 1;
		l.out_h = 1;
		l.out_c = l.c;
		fprintf(stderr, "avg                          %4d x%4d x%4d ->   %4d \n", w, h, c, c);
	}
	else {
		l.out_w = l.w;//通道方向平均池化输出宽度为输入的宽度
		l.out_h = l.h;//通道方向平均池化输出高度为输入的高度
		l.out_c = 1;//通道方向平均池化输出通道数为1
		l.bflops = (l.c * l.out_h * l.out_w) / 1000000000.;
		//设置网络输出图
		fprintf(stderr, "avg                          %4d x%4d x%4d ->   %4d x%4d x%4d\n", w, h, c, l.out_w, l.out_h, l.out_c, l.bflops);//
	}
	l.outputs = l.out_h * l.out_w * l.out_c;
	int output_size = l.out_h * l.out_w * l.out_c * batch;
	//int output_size = l.outputs * batch;
	l.output = (float*)xcalloc(output_size, sizeof(float));
	l.delta = (float*)xcalloc(output_size, sizeof(float));
	l.forward = forward_avgpool_layer;
	l.backward = backward_avgpool_layer;

#ifdef GPU
	l.forward_gpu = forward_avgpool_layer_gpu;
	l.backward_gpu = backward_avgpool_layer_gpu;
	l.output_gpu = cuda_make_array(l.output, output_size);
	l.delta_gpu = cuda_make_array(l.delta, output_size);
#endif  // GPU

    return l;
}

void resize_avgpool_layer(avgpool_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    l->inputs = h*w*l->c;
       
    if(l->channelpool){
		l->out_w = w;//根据载入resize图片的尺寸修改输出尺寸
		l->out_h = h;
	}
    //l->out_c = l->c;这一句不能添加，会导致通道数错误，也是段错误
    l->outputs = l->out_w * l->out_h * l->out_c;
    int output_size = l->outputs * l->batch;

    if (l->train)l->delta = (float*)xrealloc(l->delta, output_size * sizeof(float));
    l->output = (float*)xrealloc(l->output, output_size * sizeof(float));/**/

//增加：申请GPU处理所需的显存空间
#ifdef GPU
    CHECK_CUDA(cudaFree(l->output_gpu));
    l->output_gpu  = cuda_make_array(l->output, output_size);

    if (l->train) {
        CHECK_CUDA(cudaFree(l->delta_gpu));
        l->delta_gpu = cuda_make_array(l->delta, output_size);
    }
#endif

}

void forward_avgpool_layer(const avgpool_layer l, network_state state)
{
    int b,i,k;
    if(!l.channelpool)//如果不使用通道方向池化，处理方法不变
	{
		for(b = 0; b < l.batch; ++b)
		{
			for(k = 0; k < l.c; ++k)
			{
				int out_index = k + b*l.c;
				l.output[out_index] = 0;
				for(i = 0; i < l.h*l.w; ++i){
					int in_index = i + l.h*l.w*(k + b*l.c);
					l.output[out_index] += state.input[in_index];
				}
				l.output[out_index] /= l.h*l.w;
			}
		}
	}
    else{//如果使用通道方向池化，增加如下内容
        for(b = 0; b < l.batch; ++b){             
            for(i = 0; i < l.h*l.w; ++i){
                int out_index = i + b*l.h*l.w;
	        l.output[out_index] = 0;
                for(k = 0; k < l.c; ++k){
		    int in_index = k + l.c*(i + b*l.h*l.w);
		    l.output[out_index] += state.input[in_index];           
                }
		    l.output[out_index] /= l.c;
            }
        }
    }   
}

void backward_avgpool_layer(const avgpool_layer l, network_state state)
{
    int b,i,k;
    if(!l.channelpool){//如果不使用通道方向池化，处理方法不变
        for(b = 0; b < l.batch; ++b){
            for(k = 0; k < l.c; ++k){
                int out_index = k + b*l.c;
                for(i = 0; i < l.h*l.w; ++i){
                    int in_index = i + l.h*l.w*(k + b*l.c);
                    state.delta[in_index] += l.delta[out_index] / (l.h*l.w);
                }
            }
        }
    }
    else{//如果使用通道方向池化，增加如下内容
        for(b = 0; b < l.batch; ++b){
            for(i = 0; i < l.h*l.w; ++i){
                int out_index = i + b*l.h*l.w;
                for(k = 0; k < l.c; ++k){
					int in_index = k + l.c*(i + b*l.h*l.w);
                    state.delta[in_index] += l.delta[out_index] / (l.c);
                }
            }
        }
    }
}
