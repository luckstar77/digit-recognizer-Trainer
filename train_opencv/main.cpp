#include <opencv2/opencv.hpp>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc,char** argv)
{
    vector<string> img_path;//輸入文件名變量
    vector<int> img_catg;
    int nLine = 0;
    string buf;
    ifstream svm_data( argv[1] );//訓練樣本圖片的路徑都寫在這個txt文件中，使用bat批處理文件可以得到這個txt文件
    unsigned long n;
    while( svm_data )//將訓練樣本文件依次讀取進來
    {
        if( getline( svm_data, buf ) )
        {
            nLine ++;
            if( nLine % 2 == 0 )//注：奇數行是圖片全路徑，偶數行是標籤
            {
                img_catg.push_back( atoi( buf.c_str() ) );//atoi將字符串轉換成整型，標誌(0,1，2，...，9)，注意這裡至少要有兩個類別，否則會出錯
            }
            else
            {
                img_path.push_back( buf );//圖像路徑
            }
        }
    }
    svm_data.close();//關閉文件
    CvMat *data_mat, *res_mat;
    int nImgNum = nLine / 2; //nImgNum是樣本數量，只有文本行數的一半，另一半是標籤
    data_mat = cvCreateMat( nImgNum, 1296, CV_32FC1 );  //第二個參數，即矩陣的列是由下面的descriptors的大小決定的，可以由descriptors.size()得到，且對於不同大小的輸入訓練圖片，這個值是不同的
    cvSetZero( data_mat );
    //類型矩陣,存儲每個樣本的類型標誌
    res_mat = cvCreateMat( nImgNum, 1, CV_32FC1 );
    cvSetZero( res_mat );
    IplImage* src;
    IplImage* trainImg=cvCreateImage(cvSize(48,48),8,3);//需要分析的圖片，這裡默認設定圖片是28*28大小，所以上面定義了324，如果要更改圖片大小，可以先用debug查看一下descriptors是多少，然後設定好再運行
    
    //處理HOG特徵
    for( string::size_type i = 0; i != img_path.size(); i++ )
    {
        src=cvLoadImage(img_path[i].c_str(),1);
        if( src == NULL )
        {
            cout<<" can not load the image: "<<img_path[i].c_str()<<endl;
            continue;
        }
        
        cout<<" 處理： "<<img_path[i].c_str()<<endl;
        
        cvResize(src,trainImg);
        HOGDescriptor *hog=new HOGDescriptor(cvSize(48,48),cvSize(24,24),cvSize(12,12),cvSize(6,6),9);
        vector<float>descriptors;//存放結果
        hog->compute(trainImg, descriptors,Size(1,1), Size(0,0)); //Hog特徵計算
        cout<<"HOG dims: "<<descriptors.size()<<endl;
        n=0;
        for(vector<float>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)
        {
            cvmSet(data_mat,i,n,*iter);//存儲HOG特徵
            n++;
        }
        cvmSet( res_mat, i, 0, img_catg[i] );
        cout<<" 處理完畢: "<<img_path[i].c_str()<<" "<<img_catg[i]<<endl;
    }
    
    
    CvSVM svm;//新建一個SVM
    CvSVMParams param;//這裡是SVM訓練相關參數
    CvTermCriteria criteria;
    criteria = cvTermCriteria( CV_TERMCRIT_EPS, 1000, FLT_EPSILON );
    param = CvSVMParams( CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.09, 1.0, 10.0, 0.5, 1.0, NULL, criteria );
    
    svm.train( data_mat, res_mat, NULL, NULL, param );//訓練數據
    //保存訓練好的分類器
    svm.save( argv[2] );
    
    //檢測樣本
    IplImage *test;
    char result[512];
    vector<string> img_tst_path;
    ifstream img_tst( argv[3] );  //加載需要預測的圖片集合，這個文本里存放的是圖片全路徑，不要標籤
    while( img_tst )
    {
        if( getline( img_tst, buf ) )
        {
            img_tst_path.push_back( buf );
        }
    }
    img_tst.close();
    
    ofstream predict_txt( argv[4] );//把預測結果存儲在這個文本中
    for( string::size_type j = 0; j != img_tst_path.size(); j++ )//依次遍歷所有的待檢測圖片
    {
        test = cvLoadImage( img_tst_path[j].c_str(), 1);
        if( test == NULL )
        {
            cout<<" can not load the image: "<<img_tst_path[j].c_str()<<endl;
            continue;
        }
        IplImage* trainTempImg=cvCreateImage(cvSize(48,48),8,3);
        cvZero(trainTempImg);
        cvResize(test,trainTempImg);
        HOGDescriptor *hog=new HOGDescriptor(cvSize(48,48),cvSize(24,24),cvSize(12,12),cvSize(6,6),9);
        vector<float>descriptors;//結果數組
        hog->compute(trainTempImg, descriptors,Size(1,1), Size(0,0));
        cout<<"HOG dims: "<<descriptors.size()<<endl;
        CvMat* SVMtrainMat=cvCreateMat(1,descriptors.size(),CV_32FC1);
        int n=0;
        for(vector<float>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)
        {
            cvmSet(SVMtrainMat,0,n,*iter);
            n++;
        }
        
        int ret = svm.predict(SVMtrainMat);//檢測結果
        sprintf( result, "%s  %d\r\n",img_tst_path[j].c_str(),ret );
        predict_txt<<result;  //輸出檢測結果到文本
    }
    predict_txt.close();
    cvReleaseMat( &data_mat );
    cvReleaseMat( &res_mat );
    cvReleaseImage(&test);
    cvReleaseImage(&trainImg);
    return 0;
}
