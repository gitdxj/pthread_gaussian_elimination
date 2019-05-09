#include <iostream>
#include <fstream>
#include "matrix.h"
#include "pthread_lu.h"
using namespace std;

void *hello(void *arg);



int main()
{
    ofstream outFile;
    outFile.open("data.csv", ios::out);
    int start = 100;
    int num_iteration = 20;
    int *N = new int[num_iteration];
    for(int i = 0; i<num_iteration; i++){
        N[i] = start;
        start = start+100;
    }
    int repetition = 5;
    int numThreads = 8;
    long long time1, time2, time3, time4;
    long long mean1=0;
    long long mean2=0;
    long long mean3=0;
    long long mean4=0;
    for(int i = 0; i<num_iteration; i++)
    {
        for(int _ = 0; _<repetition; _++)
            {
                test_lu_sinthread(N[i], time1);
                test_lu(N[i],numThreads, time2);
                test_lu_sse(N[i], numThreads, time3);
                test_lu_avx(N[i], numThreads, time4);
                mean1 += time1/repetition;
                mean2 += time2/repetition;
                mean3 += time3/repetition;
                mean4 += time4/repetition;
            }
        cout << "N = " << N[i] <<endl;
        cout << "单线程："<<mean1<<endl
            << "多线程：" <<mean2<<endl
            << "多线程+SSE："<<mean3<<endl
            << "多线程+AVX："<<mean4<<endl;
        outFile<<N[i]<<','<<mean1<<','<<mean2<<','<<mean3<<','<<mean4<<endl;
    }
    outFile.close();
}


/**测试多线程的hello函数**/
void *hello(void *arg)
{
    int *thread_No = (int*) arg;
//    cout<<"this is thread"<<*thread_No<<endl;
    printf("Hello from thread %d \n", *thread_No);
}






