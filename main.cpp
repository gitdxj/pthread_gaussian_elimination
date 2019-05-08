#include <iostream>
#include <pthread.h>
#include <semaphore.h>
#include <windows.h>
#include <cstdlib>  // �������������
#include <iomanip>  // ������ʽ�����
#include <algorithm>
#include <x86intrin.h>  // ��ѱ���֧�ֵ�SSE��AVX��ȫ������
#include <fstream>
#include <time.h>
using namespace std;

void matrix_initialize(float **Matrix, int N);
void copy_matrix(float** dst, float** src, int N);
void LU(float**Matrix, int N);
void show_matrix(float **Matrix, int N);
void *hello(void *arg);
bool swap_rows(float **Matrix, int N, int k);
void *eliminate(void *threadarg);
void *eliminate_lu(void *threadarg);

struct Thread_Data {
	float** Matrix;
	int N;
	int k;
	int numThreads;
} thread_data;

sem_t sem_parent;
sem_t sem_children;

int main()
{
    int N = 6;  // �������״ΪN*N
    int numThreads = 8;  // �߳�����
    pthread_t *thread_handle = new pthread_t[numThreads];
    int errcode;

    float **Matrix = new float*[N];
    for(int i=0; i<N; i++)
        Matrix[i] = new float[N];
    matrix_initialize(Matrix, N);
    show_matrix(Matrix, N);

    float **m = new float*[N];
    for(int i=0; i<N; i++)
        m[i] = new float[N];
    copy_matrix(m, Matrix, N);

    thread_data.Matrix = Matrix;
	thread_data.N = N;
	thread_data.numThreads = numThreads;

    int *index = new int[numThreads];
	for(int i = 0; i < numThreads; i++)
    {
    	index[i] = i;
    }

	//Gaussian Elimination
//	for (int k=0; k < N-1; k++){
//		// ���д��ϵ��½��м���
//		swap_rows(Matrix, N, k);
//
//		thread_data.k = k;  // k������е��˵�k��
//
//		for (int thread_index = 0; thread_index < numThreads; thread_index++){
//			pthread_create(&thread_handle[thread_index], NULL, eliminate, (void*)&index[thread_index]);
//		}
//
//		for (int thread_index = 0; thread_index < numThreads; thread_index++){
//			pthread_join(thread_handle[thread_index], NULL);
//		}
//	}

    LU(m, N);
    show_matrix(m, N);


    for (int k=0; k < N; k++){

        if(0 == Matrix[k][k])  // ���A(k,k)��λ��Ϊ0�Ļ����ʹӺ�����һ�в�Ϊ0�Ļ���
        {
            bool swapped = swap_rows(Matrix, N, k);

            if(!swapped)   // ��������κ�һ�еĵ�k�ж�û�в���0��ͷ�ľ�ֱ������һ��k
                continue;
        }

		thread_data.k = k;  // k������е��˵�k��
		for(int j=k+1; j<N; j++)
            Matrix[k][j] = Matrix[k][j]/Matrix[k][k];
        Matrix[k][k] = 1.0;

		for(int i=0; i<numThreads; i++){
			pthread_create(&thread_handle[i], NULL, eliminate_lu, (void*)&index[i]);
		}

		for(int i=0; i<numThreads; i++){
			pthread_join(thread_handle[i], NULL);
		}

	}

	show_matrix(Matrix, N);


    return 0;
}

/** Ϊ�����и�λֵ�����ֵ**/
void matrix_initialize(float **Matrix, int N)
{
    srand((unsigned)time(NULL));  // ʱ��������
    for(int i = 0; i<N; i++)
        for(int j = 0; j<N; j++)
                Matrix[i][j] = rand()%10;  // ���ֵȡ10����
}

/**�����̨��ӡ����**/
void show_matrix(float **Matrix, int N)
{
    for(int i = 0; i<N; i++)
        {
            for(int j = 0; j<N; j++){
                   cout << fixed/*��С����ʽ��������ÿ�ѧ��������*/
                        << setprecision(1)/*����С�����һλ*/
                        << setw(6)/*ָ��������Ϊ6�������ÿո���*/
                        << right/*���Ҷ���*/ << Matrix[i][j];
            }
            cout << endl;
        }
    cout << endl;
}

/**��ͨ��LU�㷨**/
void LU(float **Matrix, int N)
{
    for(int k = 0; k<N; k++)
    {
        if(0 == Matrix[k][k])  // ���A(k,k)��λ��Ϊ0�Ļ����ʹӺ�����һ�в�Ϊ0�Ļ���
        {
            bool swapped = swap_rows(Matrix, N, k);

            if(!swapped)   // ��������κ�һ�еĵ�k�ж�û�в���0��ͷ�ľ�ֱ������һ��k
                continue;
        }
        for(int j = k+1; j<N; j++)
            Matrix[k][j] = Matrix[k][j] / Matrix[k][k];
        Matrix[k][k] = 1.0;
        for(int i = k+1; i<N; i++){
            for(int j = k+1; j<N; j++)
                Matrix[i][j] = Matrix[i][j] - Matrix[i][k] * Matrix[k][j];
            Matrix[i][k] = 0;
        }
    }
}

void copy_matrix(float** dst, float** src, int N)  // �Ѿ���src��ֵ��������dst
{
    for(int i = 0; i<N; i++)
        for(int j = 0; j<N; j++)
            dst[i][j] = src[i][j];
}

/**���Զ��̵߳�hello����**/
void *hello(void *arg)
{
    int *thread_No = (int*) arg;
//    cout<<"this is thread"<<*thread_No<<endl;
    printf("Hello from thread %d \n", *thread_No);
}

/**
��Matrix(k,k)Ϊ0��ʱ�򣬴�k�������ҳ�һ��iʹ��Matrix(i,k)��Ϊ0����i�к�k�л���
��Ϊ��Matrix(k,k)Ϊ0ʱ���ᵼ�³���Ϊ0����
param N:��ζ����Ϊ��СΪN*N
param k:����Matrix(k,k)==0ʱ��k
�����������м�Ļ����򷵻�true
δ���л�������ζ�Ŵ�k�����µ�ÿһ����k�е�λ�ö�Ϊ0���򷵻�false
**/
bool swap_rows(float **Matrix, int N, int k)
{
//    cout<<"Matrix("<<k<<","<<k<<") = 0"<<endl;
    for(int i = k+1; i<N; i++){
        if(0 != Matrix[i][k]){  // ���ֵ�Matrix(i,k)��Ϊ0�����õ�i�к͵�k��λ�û���
            for(int j = k; j<N; j++)
                swap(Matrix[k][j], Matrix[i][j]);
            cout << "row " << k << " and row " << i << " swapped��"<<endl;
//            show_matrix(Matrix, N); cout<<endl;
            return true;
        }
        else if(N-1 == i)  // k�����ÿһ���ڵ�k�ж���0
                return false;
    }
}

void *eliminate(void *threadarg){
	int *thread_index = (int*)threadarg;
	float** Matrix = thread_data.Matrix;
	int N = thread_data.N;
	int k = thread_data.k;
	int numThreads = thread_data.numThreads;
	int i, j;
	float m;

	for (i=k+1+ *thread_index; i<N; i += numThreads){
		m = Matrix[i][k]/Matrix[k][k];
		for (j=k; j<N; j++){
			Matrix[i][j] = Matrix[i][j] - (Matrix[k][j] * m);
//			Matrix[i][j] = Matrix[i][ij] - (Matrix[i][k] * Matrix[k][j]);
		}
	}
}

void *eliminate_lu(void *threadarg)
{
    int *thread_index = (int*)threadarg;
	float** Matrix = thread_data.Matrix;
	int N = thread_data.N;
	int k = thread_data.k;
	int numThreads = thread_data.numThreads;
	int i, j;
	float m;

	for (i=k+1+ *thread_index; i<N; i += numThreads){
		for (j=k+1; j<N; j++){
			Matrix[i][j] = Matrix[i][j] - (Matrix[k][j] * Matrix[i][k]);
//			Matrix[i][j] = Matrix[i][ij] - (Matrix[i][k] * Matrix[k][j]);
		}
		Matrix[i][k] = 0;
	}
//	sem_wait(&sem_children);
}


void test(int N, int numThreads)

