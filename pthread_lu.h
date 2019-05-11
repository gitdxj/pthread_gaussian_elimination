#ifndef PTHREAD_LU_H_INCLUDED
#define PTHREAD_LU_H_INCLUDED

#define MAX_THREAD_NUM 64

#include <iostream>
#include <pthread.h>
#include <semaphore.h>  // 信号量
#include <windows.h>  // 高精度计时
#include <x86intrin.h>  // 会把本机支持的SSE、AVX库全部导入
#include "matrix.h"

//thread_data 是用作线程参数的结构体
typedef struct Thread_Data {
	float** Matrix;
	int N;
	int k;
	int numThreads;
	int thread_No;
} thread_data;


void *eliminate_lu(void *threadarg);
void *eliminate_std(void *threadarg);
void *eliminate_lu_sse(void *threadarg);
void *eliminate_lu_avx(void *threadarg);
void *persis_thread_lu(void *threadarg);
void test_std(int N, int numThreads, long long& time_interval);
void test_lu_sinthread(int N, long long& time_interval);
void test_lu(int N, int numThreads, long long& time_interval);
void test_lu_sse(int N, int numThreads, long long& time_interval);
void test_lu_avx(int N, int numThreads, long long& time_interval);
void test_persis_lu(int N, int numThreads, long long& time_interval);

#endif // PTHREAD_LU_H_INCLUDED
