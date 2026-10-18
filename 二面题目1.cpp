#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    vector<Mat> imgs;
    vector<string> names;
    string listFile = "Resources/calibratecamera.txt";
    ifstream fin(listFile);

    if (!fin.is_open()) {
        cout << "请确保文件存在且路径正确。" << endl;
        return -1;
    }

    string imageName;
    while (getline(fin, imageName)) {

        if (imageName.empty()) continue;

        Mat img1 = imread(imageName);
        if (img1.empty()) {
            cout << "读取失败: " << imageName << endl;
            continue;
        }
        imgs.push_back(img1);
        names.push_back(imageName);
    }
    fin.close();
    if (imgs.empty()) {
        cout << "没有读到有效图片" << endl;
        return -1;
    }
    cout << "成功读取 " << imgs.size() << " 张图片" << endl;
    Size boardSize(11, 8);
    float squareSize = 50.0f; 
    vector<vector<Point2f>> imagePoints; 
    vector<vector<Point3f>> objectPoints; 
    vector<Point3f> objp;
    for (int i = 0; i < boardSize.height; ++i) {
        for (int j = 0; j < boardSize.width; ++j) {
            objp.push_back(Point3f(j * squareSize, i * squareSize, 0));
        }
    }
    cout << "开始检测角点..." << endl;
    int successCount = 0;

    for (size_t i = 0; i < imgs.size(); ++i) {
        Mat gray;

        if (imgs[i].channels() > 1) {
            cvtColor(imgs[i], gray, COLOR_BGR2GRAY);
        }
        else {
            gray = imgs[i].clone();
        }

        vector<Point2f> corners;
  
        bool found = findChessboardCorners(gray, boardSize, corners,
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);
        if (found) {           ;
            cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),  TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.001));

            imagePoints.push_back(corners);
            objectPoints.push_back(objp);
            successCount++;
        }
        else {
            cout << "警告: 图片 " << names[i] << " 未检测到棋盘格角点。" << endl;
        }
    }

    cout << "成功检测到 " << successCount << " 张图片的角点。" << endl;

    if (successCount < 3) {
        cout << "错误: 成功标定的图片数量太少 (<3)，无法进行准确标定。" << endl;
        return -1;
    }

    cout << "正在进行相机标定..." << endl;

    Mat cameraMatrix = Mat::eye(3, 3, CV_64F); 
    Mat distCoeffs = Mat::zeros(8, 1, CV_64F); 
    vector<Mat> rvecs, tvecs;

    double rms = calibrateCamera(objectPoints, imagePoints, imgs[0].size(),
        cameraMatrix, distCoeffs, rvecs, tvecs,
        CALIB_FIX_K3 | CALIB_ZERO_TANGENT_DIST);
    cout << "标定完成！" << endl;
    cout << "重投影误差 (RMS): " << rms << endl;
    cout << "\n--- 相机内参矩阵 ---" << endl;
    cout << cameraMatrix << endl;

    cout << "\n--- 畸变系数 ---" << endl;
    cout << distCoeffs << endl;

    // 保存结果到文件
    FileStorage fs("calibration_result.yml", FileStorage::WRITE);
    if (fs.isOpened()) {
        fs << "camera_matrix" << cameraMatrix;
        fs << "distortion_coefficients" << distCoeffs;
        fs << "reprojection_error" << rms;
        fs.release();
        cout << "\n标定结果已保存到 calibration_result.yml" << endl;
    }
    else {
        cout << "\n无法保存结果文件。" << endl;
    }

    if (!imagePoints.empty()) {
        Mat resultImg = imgs.back().clone(); 
        int lastSuccessIdx = -1;
        for (int k = 0; k < imgs.size(); ++k) {
        }
        Mat displayImg = imgs[0].clone();
        for (size_t i = 0; i < imgs.size(); ++i) {
            Mat gray;
            if (imgs[i].channels() > 1) cvtColor(imgs[i], gray, COLOR_BGR2GRAY);
            else gray = imgs[i];

            vector<Point2f> corners;
            bool found = findChessboardCorners(gray, boardSize, corners);
            if (found) {
                drawChessboardCorners(displayImg, boardSize, corners, true);
                break;
            }
        }
        imshow("Calibration Check - Corners Detected", displayImg);
        cout << "按任意键退出..." << endl;
        waitKey(0);
        destroyAllWindows();
    }

    return 0;
}
