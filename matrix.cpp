#include "matrix.h"

using namespace std;

/**为矩阵中各位值赋随机值**/
void matrix_initialize(float **Matrix, int N, int scale)
{
    srand((unsigned)time(NULL));  // 时间作种子
    for(int i = 0; i<N; i++)
        for(int j = 0; j<N; j++)
                Matrix[i][j] = rand()%scale;  // 随机值取10以内
}

/**把矩阵src的值赋给矩阵dst**/
void copy_matrix(float** dst, float** src, int N)
{
    for(int i = 0; i<N; i++)
        for(int j = 0; j<N; j++)
            dst[i][j] = src[i][j];
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
//            cout << "row " << k << " and row " << i << " swapped："<<endl;
//            show_matrix(Matrix, N); cout<<endl;
            return true;
        }
        else if(N-1 == i)  // k下面的每一行在第k列都是0
                return false;
    }
}

/**普通的LU算法**/
void gaussian_elimination_lu(float **Matrix, int N)
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


/**SSE未对齐的**/
void gaussian_elimination_sse_unaligned(float **Matrix, int N, bool p45)
{
    for(int k = 0; k<N; k++)
    {
        // 开始是解决Matrix(k,k)为0的问题
        if(0 == Matrix[k][k])  // 如果A(k,k)的位置为0的话，就从后面找一行不为0的互换
        {
            bool swapped = swap_rows(Matrix, N, k);

            if(!swapped)   // 如果下面任何一行的第k列都没有不是0打头的就直接跳下一个k
                continue;
        }

        __m128 A_k_k = _mm_set_ps1(Matrix[k][k]);
//        __m128 A_k_k = _mm_load1_ps(Matrix[k]+k);
        int part_2 = (N-k-1)%4;
        if(p45){  // p45为true，则把45行的循环向量化
            for(int j = N-4; j>k; j-=4)
            {
                __m128 A_k_j = _mm_loadu_ps(Matrix[k]+j);
                A_k_j = _mm_div_ps(A_k_j, A_k_k);
                _mm_storeu_ps(Matrix[k]+j, A_k_j);
            }
            for(int j = k+1; j<k+1+part_2; j++)  // 不能被4整除的部分
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
            for(int j = k+1; j<k+1+part_2; j++)  // 不能被4整除的部分
                Matrix[i][j] = Matrix[i][j] - Matrix[i][k] * Matrix[k][j];
            Matrix[i][k] = 0.0;
        }
//        if(k == 2)
//            show_matrix(Matrix, N);
    }
}

/**AVX未对齐的**/
void gaussian_elimination_avx_unaligned(float **Matrix, int N, bool p45)
{
    for(int k = 0; k<N; k++)
    {
        // 开始是解决Matrix(k,k)为0的问题
        if(0 == Matrix[k][k])  // 如果A(k,k)的位置为0的话，就从后面找一行不为0的互换
        {
            bool swapped = swap_rows(Matrix, N, k);

            if(!swapped)   // 如果下面任何一行的第k列都没有不是0打头的就直接跳下一个k
                continue;
        }

        __m256 A_k_k = _mm256_set1_ps(Matrix[k][k]);

        int part_2 = (N-k-1)%8;

        if(p45){  // 若4到5行的代码向量化
            for(int j = N-8; j>k; j-=8)
            {
                __m256 A_k_j = _mm256_loadu_ps(Matrix[k]+j);
                A_k_j = _mm256_div_ps(A_k_j, A_k_k);
                _mm256_storeu_ps(Matrix[k]+j, A_k_j);
            }
            for(int j = k+1; j<k+1+part_2; j++)  // 不能被8整除的部分
                Matrix[k][j] = Matrix[k][j] / Matrix[k][k];
        }
        else  // 若4到5行的代码不向量化
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
            for(int j = k+1; j<k+1+part_2; j++)  // 不能被4整除的部分
                Matrix[i][j] = Matrix[i][j] - Matrix[i][k] * Matrix[k][j];
            Matrix[i][k] = 0.0;
        }

    }
}



