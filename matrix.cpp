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
            cout << "row " << k << " and row " << i << " swapped��"<<endl;
//            show_matrix(Matrix, N); cout<<endl;
            return true;
        }
        else if(N-1 == i)  // k�����ÿһ���ڵ�k�ж���0
                return false;
    }
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





