/*************************************************************************
    > File Name: detectFeatures.cpp
    > Author: xiang gao
    > Mail: gaoxiang12@mails.tsinghua.edu.cn
    > 特征提取与匹配
    > Created Time: 2015年07月18日 星期六 16时00分21秒
 ************************************************************************/
#include <algorithm>
#include<iostream>
#include "slamBase.h"
using namespace std;

// OpenCV 特征检测模块
#include <opencv2/features2d/features2d.hpp>
// #include <opencv2/nonfree/nonfree.hpp> // use this if you want to use SIFT or SURF
#include <opencv2/calib3d/calib3d.hpp>

int main( int argc, char** argv )
{
    string inFileName;


    inFileName = "/home/jin/Downloads/data/RV_Data/Translation/Y1/frm_0001.dat";
    SR4kFRAME f1 = readSRFrame(inFileName) ; 

    inFileName = "/home/jin/Downloads/data/RV_Data/Translation/Y2/frm_0001.dat";
    SR4kFRAME f2 = readSRFrame(inFileName) ; 

    // 声明并从data文件夹里读取两个rgb与深度图
    // cv::Mat rgb1 = cv::imread( "/home/jin/Downloads/partVII/data/test/color_d1_1.png");
    // cv::Mat rgb2 = cv::imread( "/home/jin/Downloads/partVII/data/test/color_d1_2.png");

    cv::Mat rgb1 = f1.rgb;
    cv::Mat rgb2 = f2.rgb;

    cv::Mat depth1 = f1.depthXYZ;
    cv::Mat depth2 = f2.depthXYZ;

    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();

    vector< cv::KeyPoint > kp1, kp2; //关键点
    detector->detect( rgb1, kp1 );  //提取关键点
    detector->detect( rgb2, kp2 );

    cout<<"Key points of two images: "<<kp1.size()<<", "<<kp2.size()<<endl;
       
    // 计算描述子
    cv::Mat desp1, desp2;
    descriptor->compute( rgb1, kp1, desp1 );
    descriptor->compute( rgb2, kp2, desp2 );
    // 匹配描述子
    vector< cv::DMatch > matches; 
    cv::BFMatcher matcher;
    matcher.match( desp1, desp2, matches );
    // cout<<"Find total "<<matches.size()<<" matches."<<endl;

    // // 可视化：显示匹配的特征
    cv::Mat imgMatches;
    // cv::drawMatches( rgb1, kp1, rgb2, kp2, matches, imgMatches );
    // cv::imshow( "matches", imgMatches );
    // cv::imwrite( "./data/matches.png", imgMatches );
    // cv::waitKey( 0 );

    // 筛选匹配，把距离太大的去掉
    // 这里使用的准则是去掉大于四倍最小距离的匹配
    vector< cv::DMatch > goodMatches;
    int minDis = 9999;
    for ( size_t i=0; i<matches.size(); i++ )
    {
        if ( matches[i].distance < minDis )
            minDis = matches[i].distance;
    }

    cout<<"min dis = "<<minDis<<endl;
    minDis = max(20,minDis);
    for ( size_t i=0; i<matches.size(); i++ )
    {
        if (matches[i].distance < 5*minDis)
            goodMatches.push_back( matches[i] );
    }

    // // 显示 good matches

    // cv::drawMatches( rgb1, kp1, rgb2, kp2, goodMatches, imgMatches );
    // cv::imshow( "good matches", imgMatches );
    // cv::imwrite( "./data/good_matches.png", imgMatches );
    // cv::waitKey(0);

    // 计算图像间的运动关系
    // 关键函数：cv::solvePnPRansac()
    // 为调用此函数准备必要的参数
    
    // 第一个帧的三维点
    vector<cv::Point3f> pts_obj;

    vector<cv::Point3f> pts_src;
    vector<cv::Point3f> pts_dst;
    // 第二个帧的图像点
    vector< cv::Point2f > pts_img;

    // 相机内参
    CAMERA_INTRINSIC_PARAMETERS C;
    C.cx = 325.5;
    C.cy = 253.5;
    C.fx = 518.0;
    C.fy = 519.0;
    C.scale = 1000.0;

    int matchNum = min(goodMatches.size(), min( kp1.size(),kp2.size() ) );
    cout<<"good matches="<<matchNum<<endl;

    for (size_t i=0; i<matchNum; i++)
    {
        // query 是第一个, train 是第二个
        if (goodMatches[i].queryIdx > matchNum)
            continue;
        cv::Point2f p1 = kp1[goodMatches[i].queryIdx].pt;
        cv::Point2f p2 = kp2[goodMatches[i].queryIdx].pt;
        // 获取d是要小心！x是向右的，y是向下的，所以y才是行，x是列！
        if (depth1.at<double>(int(p1.x),int(p1.y),2) ==0 || depth2.at<double>(int(p2.x),int(p2.y),2)==0 )
            continue;
        
        // 将(u,v,d)转成(x,y,z)
        cv::Point3f pd1;
        pd1.x = depth1.at<double>(int(p1.x),int(p1.y),0);
        pd1.y = depth1.at<double>(int(p1.x),int(p1.y),1);
        pd1.z = depth1.at<double>(int(p1.x),int(p1.y),2);

        pts_src.push_back( pd1 );
        pts_img.push_back( cv::Point2f( kp2[goodMatches[i].trainIdx].pt ) );

        cv::Point3f pd2;

        pd2.x = depth2.at<double>(int(p2.x),int(p2.y),0);
        pd2.y = depth2.at<double>(int(p2.x),int(p2.y),1);
        pd2.z = depth2.at<double>(int(p2.x),int(p2.y),2);
        pts_dst.push_back( pd2 ); 

        // cout<<"src.size "<<pts_src.size()<<endl;

    }



    double camera_matrix_data[3][3] = {
        {C.fx, 0, C.cx},
        {0, C.fy, C.cy},
        {0, 0, 1}
    };

    // 构建相机矩阵
    cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );
    cv::Mat rvec, tvec, inliers;




    // // 求解pnp
    // cv::solvePnPRansac( pts_src, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 1.0, .99, inliers );
    // cout<<"inliers: "<<inliers.rows<<endl;
    // cout<<"R="<<rvec<<endl;
    // cout<<"t="<<tvec<<endl;

    // // 画出inliers匹配 
    // vector< cv::DMatch > matchesShow;
    // for (size_t i=0; i<inliers.rows; i++)
    // {
    //     matchesShow.push_back( goodMatches[inliers.ptr<int>(i)[0]] );    
    // }
    // cv::drawMatches( rgb1, kp1, rgb2, kp2, matchesShow, imgMatches );
    // cv::imshow( "inlier matches", imgMatches );
    // cv::imwrite( "./data/inliers.png", imgMatches );
    // cv::waitKey( 0 );

    cv::Mat outM3by4 = cv::Mat::zeros(3,4,CV_64F);



    cout<<"src.size "<<pts_src.size()<<endl;
    cv::estimateAffine3D(pts_src,pts_dst,outM3by4,inliers,3,0.9999);


    cv::Mat rmat = outM3by4(cv::Rect(0,0,3,3));
    cv::Mat rvecN;

    // cv::Rodrigues(rvec,rmat);
    cv::Rodrigues(rmat,rvecN);

    cv::Mat tvecN = outM3by4(cv::Rect(3,0,1,3));

    
    cout<<"affine M = "<<outM3by4<<endl;
    cout<<"R="<<rvecN<<endl;
    cout<<"t="<<tvecN<<endl;
    cout<<"inliers: "<<inliers.rows<<endl;



    return 0;
}
