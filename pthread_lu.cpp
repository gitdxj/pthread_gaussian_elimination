#include "pthread_lu.h"

using namespace std;

thread_data persis_thread_data;
sem_t sem_parent;
sem_t sem_children[MAX_THREAD_NUM];

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

/**ʹ����sse**/
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
        for(int j = k+1; j<k+1+(N-k-1)%4; j++)  // ���ܱ�4�����Ĳ���
            Matrix[i][j] = Matrix[i][j] - Matrix[i][k] * Matrix[k][j];
        Matrix[i][k] = 0.0;
    }
}


/**ʹ����avx**/
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
        for(int j = k+1; j<k+1+(N-k-1)%8; j++)  // ���ܱ�4�����Ĳ���
            Matrix[i][j] = Matrix[i][j] - Matrix[i][k] * Matrix[k][j];
        Matrix[i][k] = 0.0;
    }
}


/**��׼�ĵ��߳��㷨���������ο�**/
void test_lu_sinthread(int N, long long& time_interval)
{
    long long head, tail, freq;  // ���ڸ߾��ȼ�ʱ
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

    long long head, tail, freq;  // ���ڸ߾��ȼ�ʱ
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);

    // �������󲢳�ʼ�����ֵ
    float **Matrix = new float*[N];
    for(int i=0; i<N; i++)
        Matrix[i] = new float[N];
    matrix_initialize(Matrix, N);

    // thread handle
    pthread_t *thread_handle = new pthread_t[numThreads];

    // �̱߳��
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
    // ����
    for (int k=0; k < N-1; k++){
        if(0 == Matrix[k][k])  // ���A(k,k)��λ��Ϊ0�Ļ����ʹӺ�����һ�в�Ϊ0�Ļ���
        {
            bool swapped = swap_rows(Matrix, N, k);

            if(!swapped)   // ��������κ�һ�еĵ�k�ж�û�в���0��ͷ�ľ�ֱ������һ��k
                continue;
        }

		for(int i = 0; i<numThreads; i++)  // k������е��˵�k��
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

	// �����ڴ�
	for(int i=0; i<N; i++)
        delete []Matrix[i];
    delete []Matrix;
}



void test_lu(int N, int numThreads, long long& time_interval)
{
    long long head, tail, freq;  // ���ڸ߾��ȼ�ʱ
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);

    // �������󲢳�ʼ�����ֵ
    float **Matrix = new float*[N];
    for(int i=0; i<N; i++)
        Matrix[i] = new float[N];
    matrix_initialize(Matrix, N);

    // thread handle
    pthread_t *thread_handle = new pthread_t[numThreads];

    // �̱߳��
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

    // ����
    for (int k=0; k < N; k++){

		if(0 == Matrix[k][k])  // ���A(k,k)��λ��Ϊ0�Ļ����ʹӺ�����һ�в�Ϊ0�Ļ���
        {
            bool swapped = swap_rows(Matrix, N, k);

            if(!swapped)   // ��������κ�һ�еĵ�k�ж�û�в���0��ͷ�ľ�ֱ������һ��k
                continue;
        }

		for(int j=k+1; j<N; j++)
            Matrix[k][j] = Matrix[k][j]/Matrix[k][k];
        Matrix[k][k] = 1.0;

		for(int i = 0; i<numThreads; i++)  // k������е��˵�k��
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

	// �����ڴ�
	for(int i=0; i<N; i++)
        delete []Matrix[i];
    delete []Matrix;
}

void test_lu_sse(int N, int numThreads, long long& time_interval)
{
    long long head, tail, freq;  // ���ڸ߾��ȼ�ʱ
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);

    // �������󲢳�ʼ�����ֵ
    float **Matrix = new float*[N];
    for(int i=0; i<N; i++)
        Matrix[i] = new float[N];
    matrix_initialize(Matrix, N);


    // thread handle
    pthread_t *thread_handle = new pthread_t[numThreads];

    // �̱߳��
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

    // ����
    for (int k=0; k < N; k++){

		if(0 == Matrix[k][k])  // ���A(k,k)��λ��Ϊ0�Ļ����ʹӺ�����һ�в�Ϊ0�Ļ���
        {
            bool swapped = swap_rows(Matrix, N, k);

            if(!swapped)   // ��������κ�һ�еĵ�k�ж�û�в���0��ͷ�ľ�ֱ������һ��k
                continue;
        }

		__m128 A_k_k = _mm_set_ps1(Matrix[k][k]);
		for(int j = N-4; j>k; j-=4)
            {
                __m128 A_k_j = _mm_loadu_ps(Matrix[k]+j);
                A_k_j = _mm_div_ps(A_k_j, A_k_k);
                _mm_storeu_ps(Matrix[k]+j, A_k_j);
            }
        for(int j = k+1; j<k+1+(N-k-1)%4; j++)  // ���ܱ�4�����Ĳ���
            Matrix[k][j] = Matrix[k][j] / Matrix[k][k];
        Matrix[k][k] = 1.0;

		for(int i = 0; i<numThreads; i++)  // k������е��˵�k��
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

	// �����ڴ�
	for(int i=0; i<N; i++)
        delete []Matrix[i];
    delete []Matrix;
}


void test_lu_avx(int N, int numThreads, long long& time_interval)
{
    long long head, tail, freq;  // ���ڸ߾��ȼ�ʱ
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);

    // �������󲢳�ʼ�����ֵ
    float **Matrix = new float*[N];
    for(int i=0; i<N; i++)
        Matrix[i] = new float[N];
    matrix_initialize(Matrix, N);

    // thread handle
    pthread_t *thread_handle = new pthread_t[numThreads];

    // �̱߳��
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

    // ����
    for (int k=0; k < N; k++){

		if(0 == Matrix[k][k])  // ���A(k,k)��λ��Ϊ0�Ļ����ʹӺ�����һ�в�Ϊ0�Ļ���
        {
            bool swapped = swap_rows(Matrix, N, k);

            if(!swapped)   // ��������κ�һ�еĵ�k�ж�û�в���0��ͷ�ľ�ֱ������һ��k
                continue;
        }

        __m256 A_k_k = _mm256_set1_ps(Matrix[k][k]);
		for(int j = N-8; j>k; j-=8)
        {
            __m256 A_k_j = _mm256_loadu_ps(Matrix[k]+j);
            A_k_j = _mm256_div_ps(A_k_j, A_k_k);
            _mm256_storeu_ps(Matrix[k]+j, A_k_j);
        }
        for(int j = k+1; j<k+1+(N-k-1)%8; j++)  // ���ܱ�8�����Ĳ���
            Matrix[k][j] = Matrix[k][j] / Matrix[k][k];
        Matrix[k][k] = 1.0;

		for(int i = 0; i<numThreads; i++)  // k������е��˵�k��
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

	// �����ڴ�
	for(int i=0; i<N; i++)
        delete []Matrix[i];
    delete []Matrix;

}

void* persis_thread_lu(void *threadarg)
{
    int *thread_No = (int*)threadarg;
//    cout << *thread_No << "���̴߳����ɹ�"<<endl;
	float** Matrix = persis_thread_data.Matrix;
	int N = persis_thread_data.N;
	int numThreads = persis_thread_data.numThreads;
    int k;
    int i, j;
    while(k < N){
//        cout << *thread_No << ',' << "k = " << k<<endl;
        k = persis_thread_data.k;
        for (i=k+1+*thread_No; i<N; i += numThreads)
        {
            for (j=k+1; j<N; j++)
                Matrix[i][j] = Matrix[i][j] - (Matrix[k][j] * Matrix[i][k]);
            Matrix[i][k] = 0;
        }
        sem_post(&sem_parent);  //�������߳�
        if(N-1 == k)  // �������ȫ������ˣ����˳�����߳�
            pthread_exit(NULL);
        else  // ���ȫ������û��ɣ��������Լ�
            sem_wait(&sem_children[*thread_No]);  // �����Լ�
    }
}


/**����ķ����������ѭ��ִ��һ�ξʹ���numThreads���̣߳�ѭ������ʱ����
��һ����ֻ����һ���̣߳�ȫ��������ɺ�����**/
void test_persis_lu(int N, int numThreads, long long& time_interval)
{
    long long head, tail, freq;  // ���ڸ߾��ȼ�ʱ
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);

    // �������󲢳�ʼ�����ֵ
    float **Matrix = new float*[N];
    for(int i=0; i<N; i++)
        Matrix[i] = new float[N];
    matrix_initialize(Matrix, N);


    // ������һ����������֤���׼ȷ�Ե�
//    float **Matrix1 = new float*[N];
//    for(int i=0; i<N; i++)
//        Matrix1[i] = new float[N];
//    copy_matrix(Matrix1, Matrix, N);
//    gaussian_elimination_lu(Matrix1, N);
//    show_matrix(Matrix1, N);

    // thread handle
    pthread_t *thread_handle = new pthread_t[numThreads];

    // �̱߳��
    int *index = new int[numThreads];
	for(int i = 0; i < numThreads; i++)
    	index[i] = i;


    /**
    persis_thread_data���������cpp�ļ��Ŀ�ͷ����һ��ȫ�ֱ���
    ���߳̿���ֱ�Ӵ�persis_thread_data�ж�ȡ���ݣ�����е���k�С��߳���������ָ��
    �������߳�ʱ���ݵĲ���ֻ���̵߳ı��
    **/
    persis_thread_data.k = 0;
    persis_thread_data.Matrix = Matrix;
    persis_thread_data.N = N;
    persis_thread_data.numThreads = numThreads;


    /**���̵߳��ź���Ϊsem_parent
    ���̵߳��ź�����ŵ�һ������sem_children�У�����������֧��64�����̣߳��������pthread_lu.h���޸���һֵ
    ��Щ�ź����������ڴ�cpp�ļ��Ŀ�ʼ��**/
    sem_init(&sem_parent, 0, 0);
    for(int i=0; i<numThreads; i++)
        sem_init(&sem_children[i], 0, 0);

    QueryPerformanceCounter((LARGE_INTEGER *)&head);

    // ����
    for (int k=0; k < N; k++){
//        cout << "main_thread: k = " << k <<endl;
        persis_thread_data.k = k;

		if(0 == Matrix[k][k])  // ���A(k,k)��λ��Ϊ0�Ļ����ʹӺ�����һ�в�Ϊ0�Ļ���
        {
            bool swapped = swap_rows(Matrix, N, k);
            if(!swapped)   // ��������κ�һ�еĵ�k�ж�û�в���0��ͷ�ľ�ֱ������һ��k
                continue;
        }

		for(int j=k+1; j<N; j++)
            Matrix[k][j] = Matrix[k][j]/Matrix[k][k];
        Matrix[k][k] = 1.0;



        if(0 == k){ // ��һ��ѭ�������̣߳�֮���ѭ����Ȼʹ����Щ�߳�
            for (int thread_index = 0; thread_index < numThreads; thread_index++){
                pthread_create(&thread_handle[thread_index], NULL, persis_thread_lu, (void*)(index+thread_index));
            }
            // �������̣߳��ô��߳�ִ��
            for(int _=0; _<numThreads; _++)
                sem_wait(&sem_parent);
        }
        else{
            /**���̻߳������߳�Ȼ�����߳�������
            ���߳������һ���ֵ�����󣬻ỽ�����̲߳����Լ����߳�������
            ���߳�ִ�С����߳�����--->���߳�ִ�С����߳����� ѭ������ֱ���������**/
            for(int i=0; i<numThreads; i++)
                sem_post(&sem_children[i]);
            for(int _=0; _<numThreads; _++)
                sem_wait(&sem_parent);
        }

	}

    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    time_interval = (tail - head) * 1000.0 / freq ;

    for(int i=0; i<numThreads; i++)
        sem_destroy(&sem_children[i]);
    sem_destroy(&sem_parent);

	if(N<10)
        show_matrix(Matrix, N);

	// �����ڴ�
	for(int i=0; i<N; i++)
        delete []Matrix[i];
    delete []Matrix;
}
