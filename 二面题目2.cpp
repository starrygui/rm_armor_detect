#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <vector>
using namespace std;
using namespace cv;
struct Params
{
    bool autoTune = true;
    float whitePercentile = 0.999f;
    int separationThreshold_RED = 120;
    int separationThreshold_BLUE = 140;
    int grayTHreshold = 40;
    int whiteThreshold = 180;
    int minLightArea = 60;
    int maxLightArea = 6000;
    float minLightRatio = 1.8f;
    float maxLightRatio = 20.0f;
    float minLightWidth = 1.5f;
    float maxAngleDiff = 15.0f;
    float maxLengthDiff = 0.6f;
    float minRatio = 1.2f;
    float maxRatio = 2.8f;
    float pairMinRatioLo = 1.0f;
    float pairMaxRatioHi = 5.0f;


    float armorWidthMm = 135.0f;
    float armorHeightMm = 55.0f;
};
struct LightBar {
    RotatedRect rect;
    Point2f center;
    float length = 0;
    float width = 0;
    float ratio = 0;
    float angle = 0;
};
struct Pose {
    bool valid = false;
    Vec3d rvec = { 0, 0, 0 };
    Vec3d tvec = { 0, 0, 0 };
    Mat R;
    double yaw = 0;
    double pitch = 0;
    double roll = 0;
    double distance = 0;
    double yawOffset = 0;
    double pitchOffset = 0;
};
struct Armor {
    RotatedRect rect;
    vector<Point2f> points;
    Point2f center;
    float width = 0;
    float height = 0;
    float ratio = 0;
    int li = -1;
    int ri = -1;
    float score = 0;
    Pose pose;
};
struct Timing {
    double total = 0;
    double pnp = 0;
};

using Clock = chrono::high_resolution_clock;
static inline double msSince(const Clock::time_point& t0) {
    return chrono::duration<double, milli>(Clock::now() - t0).count();
}
class Armor_detect
    {
    public:
        enum Enemy_Color { ENEMY_RED = 0, ENEMY_BLUE = 1 };
        int _enemyColorAsint = ENEMY_RED;
        Params _para;
        Timing _timing;
        Mat _K;
        Mat _D;
        bool __hasCameraMatrix = false;
        vector<Point3f> _armorObjPts;
        /* Armor_detect() {
             updateArmorObjPts();
         }*/
        vector<Armor> detect(const Mat& src)
        {
            _lastLights.clear();
            _lastArmors.clear();
            if (src.empty()) return {};
            const auto tAll = Clock::now();
            preprocess(src);
            estimateThresholds();
            colorSeparation();
            vector<LightBar> lights = findLightBars();
            vector<Armor> armors = matchArmors(lights);

            /* const auto tPnp = Clock::now();
             if (_hasCameraMatrix) {
                 for (auto& a : armors)
                     solveArmorPose(a);
             }*/
             /*_timing.pnp = msSince(tPnp);*/
            _timing.total = msSince(tAll);

            _lastLights = lights;
            _lastArmors = armors;
            return armors;
        }
        void setEnemyColor(Enemy_Color c)
        { 
            _enemyColorAsint = static_cast<int>(c);
        }
        Enemy_Color enemyColor() const 
        { 
            return (_enemyColorAsint == ENEMY_BLUE) ? ENEMY_BLUE : ENEMY_RED;
        }
        Mat drawOn(Mat img, const vector<LightBar>& lights, const vector<Armor>& armors, const Timing& tm)const
        {
            for (const auto& l : lights)
            {
                Point2f pts[4];
                l.rect.points(pts);
                for (int k = 0; k < 4; k++)
                {
                    line(img, pts[k], pts[(k + 1) % 4], Scalar(0, 255, 0), 2, LINE_AA);
                    for (const auto& a : armors)
                    {
                        rectangle(img, boundingRect(a.points), Scalar(0, 0, 255), 2, LINE_AA);
                        for (size_t k = 0; k < a.points.size(); k++)
                            line(img, a.points[k], a.points[(k + 1) % 4], Scalar(0, 250, 255), 1, LINE_AA);
                        circle(img, a.center, 5, Scalar(0, 0, 255), -1, LINE_AA);
                        drawMarker(img, a.center, Scalar(0, 0, 255), MARKER_CROSS, 16, 2, LINE_AA);
                        const Rect rb = boundingRect(a.points);
                    }
                }
            }
        }
    private:
        void updateArmorObjPts() {
            const float w = _para.armorWidthMm * 0.5f;
            const float h = _para.armorHeightMm * 0.5f;
            _armorObjPts = { Point3f(-w, -h, 0), Point3f(w, -h, 0), Point3f(w, h, 0), Point3f(-w, h, 0) };
        }
        void preprocess(const Mat& src)
        {
            _src = src.clone();
            if (_src.channels() == 1)
                cvtColor(_src, _src, COLOR_GRAY2BGR);
            if (_src.channels() == 4)
                cvtColor(_src, _src, COLOR_BGRA2GRAY);
            split(_src, _splitSrc);
            _blueCh = _splitSrc[0];
            _redCh = _splitSrc[2];
        }
        void estimateThresholds() {
            if (!_para.autoTune) return;
            const int step = max(1, (int)sqrt(_graySrc.total() / 60000.0));
            int hist[256] = { 0 };
            long long count = 0;
            for (int y = 0; y < _graySrc.rows; y += step) {
                const uchar* p = _graySrc.ptr<uchar>(y);
                for (int x = 0; x < _graySrc.cols; x += step)
                {
                    hist[p[x]]++;
                    count++;
                }
            }
            if (count <= 0) return;
            const double total = static_cast<double>(count);
            auto quantile = [&](double q) {
                double acc = 0;
                for (int i = 0; i < 256; i++) {
                    acc += hist[i];
                    if (acc >= total * q)
                        return i;
                }
                return 255;
                };
            const int pWhite = quantile(_para.whitePercentile);
            int white = min(250, max(80, pWhite - 8));
            const double maxCoreRatio = 0.005;
            int coreCount = 0;
            for (int i = white + 1; i < 256; i++) coreCount += hist[i];
            while (white < 250 && coreCount > total * maxCoreRatio) { coreCount -= hist[white + 1]; white++; }
            _para.whiteThreshold = white;
            _para.grayTHreshold = min(100, max(10, static_cast<int>(_para.whiteThreshold * 0.30f)));
            Mat diff;
            if (enemyColor() == ENEMY_RED)
                subtract(_redCh, _blueCh, diff);
            else
                subtract(_blueCh, _redCh, diff);
            double maxDiff = 0;
            cv::minMaxLoc(diff, nullptr, &maxDiff, nullptr, nullptr);
            const int sepTh = max(20, static_cast<int>(maxDiff * 0.5));
            if (enemyColor() == ENEMY_RED) _para.separationThreshold_RED = sepTh;
            else                           _para.separationThreshold_BLUE = sepTh;
        }
        void colorSeparation() 
        {
            const bool red = (enemyColor() == ENEMY_RED);
            const Mat& enemyCh = red ? _redCh : _blueCh;
            const Mat& otherCh = red ? _blueCh : _redCh;
            subtract(enemyCh, otherCh, _sepA);
            threshold(_sepA, _sepA, red ? _para.separationThreshold_RED : _para.separationThreshold_BLUE, 255, THRESH_BINARY);
            inRange(_graySrc, Scalar(_para.grayTHreshold), Scalar(_para.whiteThreshold), _sepAGray);
            bitwise_and(_sepA, _sepAGray, _binary);
            const Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
            morphologyEx(_binary, _binary, MORPH_OPEN, kernel);
            morphologyEx(_binary, _binary, MORPH_CLOSE, kernel);
        }
        vector<Armor> matchArmors(const vector<LightBar>& lights) 
        {
            vector<Armor> armors;
            const float loRatio = min(_para.minRatio, _para.pairMinRatioLo);
            const float hiRatio = max(_para.maxRatio, _para.pairMaxRatioHi);
            for (size_t i = 0; i < lights.size(); i++) {
                for (size_t j = i + 1; j < lights.size(); j++) {
                    LightBar a = lights[i], b = lights[j];
                    if (fabs(a.center.x - b.center.x) < 8.0f) continue;
                    if (fabs(a.center.y - b.center.y) > max(a.length, b.length) * 0.5f) continue;
                    if (fabs(a.length - b.length) / max(a.length, b.length) > _para.maxLengthDiff) continue;
                    if (fabs(a.angle - b.angle) > _para.maxAngleDiff) continue;
                    const float width = static_cast<float>(norm(a.center - b.center));
                    const float height = (a.length + b.length) * 0.5f;
                    if (height < 1e-3f) continue;
                    const float ratio = width / height;
                    if (ratio < loRatio || ratio > hiRatio) continue;
                    if (a.center.x > b.center.x) swap(a, b);
                    Point2f aPts[4], bPts[4];
                    a.rect.points(aPts); b.rect.points(bPts);
                    vector<Point2f> allPts(aPts, aPts + 4);
                    allPts.insert(allPts.end(), bPts, bPts + 4);
                    Armor armor;
                    armor.li = (int)i; armor.ri = (int)j;
                    armor.rect = minAreaRect(allPts);
                    armor.center = (a.center + b.center) * 0.5f;
                    armor.width = width; armor.height = height; armor.ratio = ratio;
                    armor.rect.points(armor.points);
                    armor.score = fabs(a.length - b.length) / max(a.length, b.length)
                        + fabs(a.angle - b.angle) / max(1.0f, _para.maxAngleDiff)
                        + fabs(ratio - 1.9f) / 1.9f;
                    armors.push_back(armor);
                }
            }
            sort(armors.begin(), armors.end(), [](const Armor& x, const Armor& y) { return x.score < y.score; });
            vector<char> used(lights.size(), 0);
            vector<Armor> picked;
            for (const auto& a : armors) {
                if (a.li < 0 || a.ri < 0) continue;
                if (used[a.li] || used[a.ri]) continue;
                used[a.li] = used[a.ri] = 1;
                picked.push_back(a);
            }
            sort(picked.begin(), picked.end(), [](const Armor& x, const Armor& y) { return x.center.x < y.center.x; });
            return picked;
        }
       
        const vector<LightBar>& lastLights() const { return _lastLights; }
        const vector<Armor>& lastArmors() const { return _lastArmors; }

        vector<LightBar> findLightBars() {
            vector<LightBar> lights;
            Mat binCopy = _binary.clone();
            vector<vector<Point>> contours;
            findContours(binCopy, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            for (const auto& c : contours) {
                const float area = static_cast<float>(contourArea(c));
                if (area < _para.minLightArea || area > _para.maxLightArea) continue;
                const RotatedRect rect = minAreaRect(c);
                const float w = rect.size.width, h = rect.size.height;
                if (w <= 0 || h <= 0) continue;
                LightBar lb;
                lb.rect = rect; lb.center = rect.center;
                lb.length = max(w, h); lb.width = min(w, h);
                lb.ratio = lb.length / max(1e-6f, lb.width);
                if (lb.ratio < _para.minLightRatio || lb.ratio > _para.maxLightRatio) continue;
                if (lb.width < _para.minLightWidth) continue;
                const float axis = (w >= h) ? rect.angle : (rect.angle + 90.0f);
                lb.angle = axis - 90.0f * std::floor((axis + 45.0f) / 90.0f);
                lights.push_back(lb);
            }
            sort(lights.begin(), lights.end(), [](const LightBar& a, const LightBar& b) { return a.center.x < b.center.x; });
            return lights;
        }
    Mat _src, _graySrc, _sepA, _sepAGray, _binary, _blueCh, _redCh;
    vector<Mat>_splitSrc;
    vector<LightBar> _lastLights;
    vector<Armor> _lastArmors;
};

 
    


    vector<Point2f> orderArmorPoints(const vector<Point2f>& pts) const {
        vector<Point2f> p = pts;
        int idxLT = 0, idxRB = 0;
        double minSum = 1e18, maxSum = -1e18;
        for (int i = 0; i < 4; i++) {
            double s = p[i].x + p[i].y;
            if (s < minSum) { minSum = s; idxLT = i; }
            if (s > maxSum) { maxSum = s; idxRB = i; }
        }
        vector<int> rest;
        for (int i = 0; i < 4; i++) if (i != idxLT && i != idxRB) rest.push_back(i);
        int idxRT = rest[0], idxLB = rest[1];
        if (p[rest[0]].x - p[rest[0]].y < p[rest[1]].x - p[rest[1]].y) { idxLB = rest[0]; idxRT = rest[1]; }
        else { idxLB = rest[1]; idxRT = rest[0]; }
        return { p[idxLT], p[idxRT], p[idxRB], p[idxLB] };
    }

   

int main(int argc, char** argv)
{
    string videoPath = "Resources/clip.avi";
   /* string calibPath = "Resources/camera_params.yml";*/

    /*if (argc > 1) videoPath = argv[1];
    if (argc > 2) calibPath = argv[2];*/

    Armor_detect detector;
    detector.setEnemyColor(Armor_detect::ENEMY_RED);

    VideoCapture cap;
    if (videoPath == "0") cap.open(0);
    else                  cap.open(videoPath);

    if (!cap.isOpened()) {
        printf("视频/摄像头打开失败: %s\n", videoPath.c_str());
        return -1;
    }

    // 关掉自动曝光和自动白平衡（避免光线变化导致二值化阈值乱跳）
    cap.set(CAP_PROP_AUTO_EXPOSURE, 0.25);
    cap.set(CAP_PROP_AUTO_WB, 0);

    const int imgW = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    const int imgH = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    printf("视频分辨率: %dx%d\n", imgW, imgH);

    Mat frame;
    while (cap.read(frame)) {
        if (frame.empty()) break;

        vector<Armor> armors = detector.detect(frame);

        printf("装甲板=%d | 耗时=%.2fms (PnP %.2fms)\n",
            (int)armors.size(), detector._timing.total, detector._timing.pnp);
        for (size_t i = 0; i < armors.size(); i++) {
            const auto& a = armors[i];
            if (a.pose.valid) {
                printf("  [#%zu] d=%.2fm off=(%.2f, %.2f)deg\n",
                    i, a.pose.distance, a.pose.yawOffset, a.pose.pitchOffset);
            }
        }

        Mat show = detector.drawOn(frame.clone(), detector._lastLight, armors, detector._timing);
        imshow("Result", show);

        const int k = waitKey(1);
        if (k == 27 || k == 'q') break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}