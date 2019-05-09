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
            cout << "row " << k << " and row " << i << " swapped："<<endl;
//            show_matrix(Matrix, N); cout<<endl;
            return true;
        }
        else if(N-1 == i)  // k下面的每一行在第k列都是0
                return false;
    }
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





