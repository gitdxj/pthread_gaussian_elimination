#ifndef PTHREAD_LU_H_INCLUDED
#define PTHREAD_LU_H_INCLUDED

#include <iostream>
#include <pthread.h>
#include <semaphore.h>  // �ź���
#include <windows.h>  // �߾��ȼ�ʱ
#include <x86intrin.h>  // ��ѱ���֧�ֵ�SSE��AVX��ȫ������
#include "matrix.h"

//thread_data �������̲߳����Ľṹ��
typedef struct Thread_Data {
	float** Matrix;
	int N;
	int k;
	int numThreads;
	int thread_No;
} thread_data;

void *eliminate_lu(void *threadarg);
void *eliminate_std(void *threadarg);
void test_std(int N, int numThreads, long long& time_interval);
void test_lu(int N, int numThreads, long long& time_interval);

#endif // PTHREAD_LU_H_INCLUDED