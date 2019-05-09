#include "pthread_lu.h"

using namespace std;

void *eliminate_std(void *threadarg){
	thread_data *data = (thread_data*)threadarg;
	float** Matrix = data->Matrix;
	int N = data->N;
	int k = data->k;
	int numThreads = data->numThreads;
	int thread_No = data->thread_No;
	int i, j;
	float m;

	for (i=k+1+thread_No; i<N; i += numThreads){
		m = Matrix[i][k]/Matrix[k][k];
		for (j=k; j<N; j++){
			Matrix[i][j] = Matrix[i][j] - (Matrix[k][j] * m);
		}
	}
}

void *eliminate_lu(void *threadarg)
{
    thread_data *data = (thread_data*)threadarg;
	float** Matrix = data->Matrix;
	int N = data->N;
	int k = data->k;
	int numThreads = data->numThreads;
	int thread_No = data->thread_No;
	int i, j;

	for (i=k+1+thread_No; i<N; i += numThreads)
    {
        for (j=k+1; j<N; j++)
            Matrix[i][j] = Matrix[i][j] - (Matrix[k][j] * Matrix[i][k]);
		Matrix[i][k] = 0;
	}
}

/**使用了sse**/
void *eliminate_lu_sse(void *threadarg)
{
    thread_data *data = (thread_data*)threadarg;
	float** Matrix = data->Matrix;
	int N = data->N;
	int k = data->k;
	int numThreads = data->numThreads;
	int thread_No = data->thread_No;
	int i, j;

	for (i=k+1+thread_No; i<N; i += numThreads)
    {
        __m128 A_i_k = _mm_set_ps1(Matrix[i][k]);
//             __m128 A_i_k = _mm_load1_ps(Matrix[i]+k);
        for(int j = N-4; j>k; j-=4)
            {
                __m128 A_k_j = _mm_loadu_ps(Matrix[k]+j);
                __m128 t = _mm_mul_ps(A_k_j, A_i_k);
                __m128 A_i_j = _mm_loadu_ps(Matrix[i]+j);
                A_i_j = _mm_sub_ps(A_i_j, t);
                _mm_storeu_ps(Matrix[i]+j, A_i_j);
            }
        for(int j = k+1; j<k+1+(N-k-1)%4; j++)  // 不能被4整除的部分
            Matrix[i][j] = Matrix[i][j] - Matrix[i][k] * Matrix[k][j];
        Matrix[i][k] = 0.0;
    }
}


/**使用了avx**/
void *eliminate_lu_avx(void *threadarg)
{
    thread_data *data = (thread_data*)threadarg;
	float** Matrix = data->Matrix;
	int N = data->N;
	int k = data->k;
	int numThreads = data->numThreads;
	int thread_No = data->thread_No;
	int i, j;

	for (i=k+1+thread_No; i<N; i += numThreads)
    {
        __m256 A_i_k = _mm256_set1_ps(Matrix[i][k]);

        for(int j = N-8; j>k; j-=8)
        {
            __m256 A_k_j = _mm256_loadu_ps(Matrix[k]+j);
            __m256 t = _mm256_mul_ps(A_k_j, A_i_k);
            __m256 A_i_j = _mm256_loadu_ps(Matrix[i]+j);
            A_i_j = _mm256_sub_ps(A_i_j, t);
            _mm256_storeu_ps(Matrix[i]+j, A_i_j);
        }
        for(int j = k+1; j<k+1+(N-k-1)%8; j++)  // 不能被4整除的部分
            Matrix[i][j] = Matrix[i][j] - Matrix[i][k] * Matrix[k][j];
        Matrix[i][k] = 0.0;
    }
}


/**标准的单线程算法，用来作参考**/
void test_lu_sinthread(int N, long long& time_interval)
{
    long long head, tail, freq;  // 用于高精度计时
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);

    float **Matrix = new float*[N];
    for(int i=0; i<N; i++)
        Matrix[i] = new float[N];
    matrix_initialize(Matrix, N);

    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    gaussian_elimination_lu(Matrix, N);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    time_interval = (tail - head) * 1000.0 / freq ;
}

void test_std(int N, int numThreads, long long& time_interval)
{

    long long head, tail, freq;  // 用于高精度计时
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);

    // 创建矩阵并初始化随机值
    float **Matrix = new float*[N];
    for(int i=0; i<N; i++)
        Matrix[i] = new float[N];
    matrix_initialize(Matrix, N);

    // thread handle
    pthread_t *thread_handle = new pthread_t[numThreads];

    // 线程编号
    int *index = new int[numThreads];
	for(int i = 0; i < numThreads; i++)
    	index[i] = i;

    thread_data *thread_arg = new thread_data[numThreads];
    for(int i = 0; i<numThreads; i++)
    {
        thread_arg[i].Matrix = Matrix;
        thread_arg[i].N = N;
        thread_arg[i].numThreads = numThreads;
        thread_arg[i].thread_No = index[i];
    }

    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    // 计算
    for (int k=0; k < N-1; k++){
        if(0 == Matrix[k][k])  // 如果A(k,k)的位置为0的话，就从后面找一行不为0的互换
        {
            bool swapped = swap_rows(Matrix, N, k);

            if(!swapped)   // 如果下面任何一行的第k列都没有不是0打头的就直接跳下一个k
                continue;
        }

		for(int i = 0; i<numThreads; i++)  // k代表进行到了第k行
            thread_arg[i].k = k;
		for (int thread_index = 0; thread_index < numThreads; thread_index++){
			pthread_create(&thread_handle[thread_index], NULL, eliminate_std, (void*)&thread_arg[thread_index]);
		}

		for (int thread_index = 0; thread_index < numThreads; thread_index++){
			pthread_join(thread_handle[thread_index], NULL);
		}
	}
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    time_interval = (tail - head) * 1000.0 / freq ;

    if(N<10)
        show_matrix(Matrix, N);

	// 回收内存
	for(int i=0; i<N; i++)
        delete []Matrix[i];
    delete []Matrix;
}



void test_lu(int N, int numThreads, long long& time_interval)
{
    long long head, tail, freq;  // 用于高精度计时
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);

    // 创建矩阵并初始化随机值
    float **Matrix = new float*[N];
    for(int i=0; i<N; i++)
        Matrix[i] = new float[N];
    matrix_initialize(Matrix, N);

    // 初始化Thread_Data
    struct Thread_Data data;
    data.Matrix = Matrix;
    data.N = N;
    data.numThreads = numThreads;

    // thread handle
    pthread_t *thread_handle = new pthread_t[numThreads];

    // 线程编号
    int *index = new int[numThreads];
	for(int i = 0; i < numThreads; i++)
    	index[i] = i;

    thread_data *thread_arg = new thread_data[numThreads];
    for(int i = 0; i<numThreads; i++)
    {
        thread_arg[i].Matrix = Matrix;
        thread_arg[i].N = N;
        thread_arg[i].numThreads = numThreads;
        thread_arg[i].thread_No = index[i];
    }

    QueryPerformanceCounter((LARGE_INTEGER *)&head);

    // 计算
    for (int k=0; k < N; k++){

		if(0 == Matrix[k][k])  // 如果A(k,k)的位置为0的话，就从后面找一行不为0的互换
        {
            bool swapped = swap_rows(Matrix, N, k);

            if(!swapped)   // 如果下面任何一行的第k列都没有不是0打头的就直接跳下一个k
                continue;
        }

		for(int j=k+1; j<N; j++)
            Matrix[k][j] = Matrix[k][j]/Matrix[k][k];
        Matrix[k][k] = 1.0;

		for(int i = 0; i<numThreads; i++)  // k代表进行到了第k行
            thread_arg[i].k = k;

		for (int thread_index = 0; thread_index < numThreads; thread_index++){
			pthread_create(&thread_handle[thread_index], NULL, eliminate_lu, (void*)&thread_arg[thread_index]);
		}

		for (int thread_index = 0; thread_index < numThreads; thread_index++){
			pthread_join(thread_handle[thread_index], NULL);
		}
	}

    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    time_interval = (tail - head) * 1000.0 / freq ;

	if(N<10)
        show_matrix(Matrix, N);

	// 回收内存
	for(int i=0; i<N; i++)
        delete []Matrix[i];
    delete []Matrix;
}

void test_lu_sse(int N, int numThreads, long long& time_interval)
{
    long long head, tail, freq;  // 用于高精度计时
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);

    // 创建矩阵并初始化随机值
    float **Matrix = new float*[N];
    for(int i=0; i<N; i++)
        Matrix[i] = new float[N];
    matrix_initialize(Matrix, N);

    // 初始化Thread_Data
    struct Thread_Data data;
    data.Matrix = Matrix;
    data.N = N;
    data.numThreads = numThreads;

    // thread handle
    pthread_t *thread_handle = new pthread_t[numThreads];

    // 线程编号
    int *index = new int[numThreads];
	for(int i = 0; i < numThreads; i++)
    	index[i] = i;

    thread_data *thread_arg = new thread_data[numThreads];
    for(int i = 0; i<numThreads; i++)
    {
        thread_arg[i].Matrix = Matrix;
        thread_arg[i].N = N;
        thread_arg[i].numThreads = numThreads;
        thread_arg[i].thread_No = index[i];
    }

    QueryPerformanceCounter((LARGE_INTEGER *)&head);

    // 计算
    for (int k=0; k < N; k++){

		if(0 == Matrix[k][k])  // 如果A(k,k)的位置为0的话，就从后面找一行不为0的互换
        {
            bool swapped = swap_rows(Matrix, N, k);

            if(!swapped)   // 如果下面任何一行的第k列都没有不是0打头的就直接跳下一个k
                continue;
        }

		__m128 A_k_k = _mm_set_ps1(Matrix[k][k]);
		for(int j = N-4; j>k; j-=4)
            {
                __m128 A_k_j = _mm_loadu_ps(Matrix[k]+j);
                A_k_j = _mm_div_ps(A_k_j, A_k_k);
                _mm_storeu_ps(Matrix[k]+j, A_k_j);
            }
        for(int j = k+1; j<k+1+(N-k-1)%4; j++)  // 不能被4整除的部分
            Matrix[k][j] = Matrix[k][j] / Matrix[k][k];
        Matrix[k][k] = 1.0;

		for(int i = 0; i<numThreads; i++)  // k代表进行到了第k行
            thread_arg[i].k = k;

		for (int thread_index = 0; thread_index < numThreads; thread_index++){
			pthread_create(&thread_handle[thread_index], NULL, eliminate_lu_sse, (void*)&thread_arg[thread_index]);
		}

		for (int thread_index = 0; thread_index < numThreads; thread_index++){
			pthread_join(thread_handle[thread_index], NULL);
		}
    }

    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    time_interval = (tail - head) * 1000.0 / freq ;

	if(N<=10){
        show_matrix(Matrix, N);
	}

	// 回收内存
	for(int i=0; i<N; i++)
        delete []Matrix[i];
    delete []Matrix;
}


void test_lu_avx(int N, int numThreads, long long& time_interval)
{
    long long head, tail, freq;  // 用于高精度计时
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);

    // 创建矩阵并初始化随机值
    float **Matrix = new float*[N];
    for(int i=0; i<N; i++)
        Matrix[i] = new float[N];
    matrix_initialize(Matrix, N);

    // 初始化Thread_Data
    struct Thread_Data data;
    data.Matrix = Matrix;
    data.N = N;
    data.numThreads = numThreads;

    // thread handle
    pthread_t *thread_handle = new pthread_t[numThreads];

    // 线程编号
    int *index = new int[numThreads];
	for(int i = 0; i < numThreads; i++)
    	index[i] = i;

    thread_data *thread_arg = new thread_data[numThreads];
    for(int i = 0; i<numThreads; i++)
    {
        thread_arg[i].Matrix = Matrix;
        thread_arg[i].N = N;
        thread_arg[i].numThreads = numThreads;
        thread_arg[i].thread_No = index[i];
    }

    QueryPerformanceCounter((LARGE_INTEGER *)&head);

    // 计算
    for (int k=0; k < N; k++){

		if(0 == Matrix[k][k])  // 如果A(k,k)的位置为0的话，就从后面找一行不为0的互换
        {
            bool swapped = swap_rows(Matrix, N, k);

            if(!swapped)   // 如果下面任何一行的第k列都没有不是0打头的就直接跳下一个k
                continue;
        }

        __m256 A_k_k = _mm256_set1_ps(Matrix[k][k]);
		for(int j = N-8; j>k; j-=8)
        {
            __m256 A_k_j = _mm256_loadu_ps(Matrix[k]+j);
            A_k_j = _mm256_div_ps(A_k_j, A_k_k);
            _mm256_storeu_ps(Matrix[k]+j, A_k_j);
        }
        for(int j = k+1; j<k+1+(N-k-1)%8; j++)  // 不能被8整除的部分
            Matrix[k][j] = Matrix[k][j] / Matrix[k][k];
        Matrix[k][k] = 1.0;

		for(int i = 0; i<numThreads; i++)  // k代表进行到了第k行
            thread_arg[i].k = k;

		for (int thread_index = 0; thread_index < numThreads; thread_index++){
			pthread_create(&thread_handle[thread_index], NULL, eliminate_lu_avx, (void*)&thread_arg[thread_index]);
		}

		for (int thread_index = 0; thread_index < numThreads; thread_index++){
			pthread_join(thread_handle[thread_index], NULL);
		}
    }

    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    time_interval = (tail - head) * 1000.0 / freq ;

	if(N<=10){
        show_matrix(Matrix, N);
	}

	// 回收内存
	for(int i=0; i<N; i++)
        delete []Matrix[i];
    delete []Matrix;

}
