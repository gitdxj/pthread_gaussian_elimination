#include <iostream>
#include <pthread.h>
#include <semaphore.h>
#include <windows.h>
#include <cstdlib>  // 用来产生随机数
#include <iomanip>  // 用来格式化输出
#include <algorithm>
#include <x86intrin.h>  // 会把本机支持的SSE、AVX库全部导入
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
    int N = 6;  // 矩阵的形状为N*N
    int numThreads = 8;  // 线程数量
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
//		// 按行从上到下进行计算
//		swap_rows(Matrix, N, k);
//
//		thread_data.k = k;  // k代表进行到了第k行
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

        if(0 == Matrix[k][k])  // 如果A(k,k)的位置为0的话，就从后面找一行不为0的互换
        {
            bool swapped = swap_rows(Matrix, N, k);

            if(!swapped)   // 如果下面任何一行的第k列都没有不是0打头的就直接跳下一个k
                continue;
        }

		thread_data.k = k;  // k代表进行到了第k行
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

/** 为矩阵中各位值赋随机值**/
void matrix_initialize(float **Matrix, int N)
{
    srand((unsigned)time(NULL));  // 时间作种子
    for(int i = 0; i<N; i++)
        for(int j = 0; j<N; j++)
                Matrix[i][j] = rand()%10;  // 随机值取10以内
}

/**向控制台打印矩阵**/
void show_matrix(float **Matrix, int N)
{
    for(int i = 0; i<N; i++)
        {
            for(int j = 0; j<N; j++){
                   cout << fixed/*以小数形式输出（不用科学计数法）*/
                        << setprecision(1)/*保留小数点后一位*/
                        << setw(6)/*指定输出宽度为6，不足用空格补齐*/
                        << right/*向右对齐*/ << Matrix[i][j];
            }
            cout << endl;
        }
    cout << endl;
}

/**普通的LU算法**/
void LU(float **Matrix, int N)
{
    for(int k = 0; k<N; k++)
    {
        if(0 == Matrix[k][k])  // 如果A(k,k)的位置为0的话，就从后面找一行不为0的互换
        {
            bool swapped = swap_rows(Matrix, N, k);

            if(!swapped)   // 如果下面任何一行的第k列都没有不是0打头的就直接跳下一个k
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

void copy_matrix(float** dst, float** src, int N)  // 把矩阵src的值赋给矩阵dst
{
    for(int i = 0; i<N; i++)
        for(int j = 0; j<N; j++)
            dst[i][j] = src[i][j];
}

/**测试多线程的hello函数**/
void *hello(void *arg)
{
    int *thread_No = (int*) arg;
//    cout<<"this is thread"<<*thread_No<<endl;
    printf("Hello from thread %d \n", *thread_No);
}

/**
当Matrix(k,k)为0的时候，从k行向下找出一行i使得Matrix(i,k)不为0，将i行和k行互换
因为当Matrix(k,k)为0时，会导致除数为0问题
param N:意味矩阵为大小为N*N
param k:出现Matrix(k,k)==0时的k
若进行了两行间的互换则返回true
未进行互换（意味着从k行向下的每一行在k列的位置都为0）则返回false
**/
bool swap_rows(float **Matrix, int N, int k)
{
//    cout<<"Matrix("<<k<<","<<k<<") = 0"<<endl;
    for(int i = k+1; i<N; i++){
        if(0 != Matrix[i][k]){  // 发现第Matrix(i,k)不为0，就让第i行和第k行位置互换
            for(int j = k; j<N; j++)
                swap(Matrix[k][j], Matrix[i][j]);
            cout << "row " << k << " and row " << i << " swapped："<<endl;
//            show_matrix(Matrix, N); cout<<endl;
            return true;
        }
        else if(N-1 == i)  // k下面的每一行在第k列都是0
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

