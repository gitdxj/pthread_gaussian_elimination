#include "matrix.h"

using namespace std;

/**Ϊ�����и�λֵ�����ֵ**/
void matrix_initialize(float **Matrix, int N, int scale)
{
    srand((unsigned)time(NULL));  // ʱ��������
    for(int i = 0; i<N; i++)
        for(int j = 0; j<N; j++)
                Matrix[i][j] = rand()%scale;  // ���ֵȡ10����
}

/**�Ѿ���src��ֵ��������dst**/
void copy_matrix(float** dst, float** src, int N)
{
    for(int i = 0; i<N; i++)
        for(int j = 0; j<N; j++)
            dst[i][j] = src[i][j];
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
//            cout << "row " << k << " and row " << i << " swapped��"<<endl;
//            show_matrix(Matrix, N); cout<<endl;
            return true;
        }
        else if(N-1 == i)  // k�����ÿһ���ڵ�k�ж���0
                return false;
    }
}

/**��ͨ��LU�㷨**/
void gaussian_elimination_lu(float **Matrix, int N)
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


/**SSEδ�����**/
void gaussian_elimination_sse_unaligned(float **Matrix, int N, bool p45)
{
    for(int k = 0; k<N; k++)
    {
        // ��ʼ�ǽ��Matrix(k,k)Ϊ0������
        if(0 == Matrix[k][k])  // ���A(k,k)��λ��Ϊ0�Ļ����ʹӺ�����һ�в�Ϊ0�Ļ���
        {
            bool swapped = swap_rows(Matrix, N, k);

            if(!swapped)   // ��������κ�һ�еĵ�k�ж�û�в���0��ͷ�ľ�ֱ������һ��k
                continue;
        }

        __m128 A_k_k = _mm_set_ps1(Matrix[k][k]);
//        __m128 A_k_k = _mm_load1_ps(Matrix[k]+k);
        int part_2 = (N-k-1)%4;
        if(p45){  // p45Ϊtrue�����45�е�ѭ��������
            for(int j = N-4; j>k; j-=4)
            {
                __m128 A_k_j = _mm_loadu_ps(Matrix[k]+j);
                A_k_j = _mm_div_ps(A_k_j, A_k_k);
                _mm_storeu_ps(Matrix[k]+j, A_k_j);
            }
            for(int j = k+1; j<k+1+part_2; j++)  // ���ܱ�4�����Ĳ���
                Matrix[k][j] = Matrix[k][j] / Matrix[k][k];
            }
        else
            for(int j = k+1; j<N; j++)
                Matrix[k][j] = Matrix[k][j] / Matrix[k][k];



        Matrix[k][k] = 1.0;

        for(int i = k+1; i<N; i++)
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
            for(int j = k+1; j<k+1+part_2; j++)  // ���ܱ�4�����Ĳ���
                Matrix[i][j] = Matrix[i][j] - Matrix[i][k] * Matrix[k][j];
            Matrix[i][k] = 0.0;
        }
//        if(k == 2)
//            show_matrix(Matrix, N);
    }
}

/**AVXδ�����**/
void gaussian_elimination_avx_unaligned(float **Matrix, int N, bool p45)
{
    for(int k = 0; k<N; k++)
    {
        // ��ʼ�ǽ��Matrix(k,k)Ϊ0������
        if(0 == Matrix[k][k])  // ���A(k,k)��λ��Ϊ0�Ļ����ʹӺ�����һ�в�Ϊ0�Ļ���
        {
            bool swapped = swap_rows(Matrix, N, k);

            if(!swapped)   // ��������κ�һ�еĵ�k�ж�û�в���0��ͷ�ľ�ֱ������һ��k
                continue;
        }

        __m256 A_k_k = _mm256_set1_ps(Matrix[k][k]);

        int part_2 = (N-k-1)%8;

        if(p45){  // ��4��5�еĴ���������
            for(int j = N-8; j>k; j-=8)
            {
                __m256 A_k_j = _mm256_loadu_ps(Matrix[k]+j);
                A_k_j = _mm256_div_ps(A_k_j, A_k_k);
                _mm256_storeu_ps(Matrix[k]+j, A_k_j);
            }
            for(int j = k+1; j<k+1+part_2; j++)  // ���ܱ�8�����Ĳ���
                Matrix[k][j] = Matrix[k][j] / Matrix[k][k];
        }
        else  // ��4��5�еĴ��벻������
            for(int j = k+1; j<N; j++)
                Matrix[k][j] = Matrix[k][j] / Matrix[k][k];

        Matrix[k][k] = 1.0;

        for(int i = k+1; i<N; i++)
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
            for(int j = k+1; j<k+1+part_2; j++)  // ���ܱ�4�����Ĳ���
                Matrix[i][j] = Matrix[i][j] - Matrix[i][k] * Matrix[k][j];
            Matrix[i][k] = 0.0;
        }

    }
}



