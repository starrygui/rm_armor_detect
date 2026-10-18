#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace cv;
using namespace std;
struct Params
{
    bool autoTune = true;
    float whitePercentile = 0.999f;

    int separationThreshold_RED = 120;
    int separationThreshold_BLUE = 140;
    int grayTHreshold = 40;
    int whiteThreshold = 180;

    int minLightArea = 60;
    int maxLightArea = 120;
    float minLightRatio = 2.5f;
    float maxLightRatio = 6.0f;
    float minLightWidth = 2.0f;

    float maxAngleDiff = 15.0f;
    float maxLengthDiff = 0.6f;
    float minRatio = 1.2f;
    float maxRatio = 2.8f;
    float pairMinRatioLo = 1.0f;
    float pairMaxRatioHi = 5.0f;

    float armorWidthMm = 56e-3;
    float armorHeightMm = 135e-3;
};
struct LightBar {
    RotatedRect rect;
    Point2f center;
    float length = 0;
    float width = 0;
    float ratio = 0;
    float angle = 0;
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
};
struct Timing {
    double total = 0;
};
using Clock = chrono::high_resolution_clock;
static inline double msSince(const Clock::time_point& t0) {
    return chrono::duration<double, milli>(Clock::now() - t0).count();
}
class ArmorDetector
{
public:
    enum EnemyColor { ENEMY_RED = 0, ENEMY_BLUE = 1 };
    Params _para;
    int _enemyColorAsint = ENEMY_BLUE;
    Timing _timing;
    vector<Point3f> _armorObjPts;
    ArmorDetector()
    {
        updateArmorObjPts();
    }
    vector<Armor> detect(const Mat& src) {
        _lastLights.clear();
        _lastArmors.clear();
        if (src.empty()) return {};
        const auto tAll = Clock::now();
        preprocess(src);
        estimateThresholds();
        colorSeparation();
        vector<LightBar> lights = findLightBars();
        vector<Armor> armors = matchArmors(lights);
        _timing.total = msSince(tAll);
        _lastLights = lights;
        _lastArmors = armors;
        return armors;
    }
    void setEnemyColor(EnemyColor c)
    {
        _enemyColorAsint = static_cast<int>(c);
    }

    EnemyColor enemyColor() const
    {
        return (_enemyColorAsint == ENEMY_BLUE) ? ENEMY_BLUE : ENEMY_RED;
    }
    const vector<LightBar>& lastLights() const
    {
        return _lastLights;
    }
    const vector<Armor>& lastArmors() const
    {
        return _lastArmors;
    }
    Mat drawOn(Mat img, const vector<LightBar>& lights, const vector<Armor>& armors, const Timing& tm) const {
        for (const auto& l : lights)
        {
            Point2f pts[4]; l.rect.points(pts);
            for (int k = 0; k < 4; k++) line(img, pts[k], pts[(k + 1) % 4], Scalar(0, 255, 0), 2, LINE_AA);
        }
        for (const auto& a : armors) {
            rectangle(img, boundingRect(a.points), Scalar(0, 0, 255), 2, LINE_AA);
            for (size_t k = 0; k < a.points.size(); k++)
                line(img, a.points[k], a.points[(k + 1) % a.points.size()], Scalar(0, 220, 255), 1, LINE_AA);
            drawMarker(img, a.center, Scalar(0, 0, 255), MARKER_CROSS, 10, 2, LINE_AA);
        }
        return img;
    }
private:
    void updateArmorObjPts() {
        const float w = _para.armorWidthMm * 0.5f;
        const float h = _para.armorHeightMm * 0.5f;
        _armorObjPts = { Point3f(-w, -h, 0), Point3f(w, -h, 0), Point3f(w, h, 0), Point3f(-w, h, 0) };
    }
    void preprocess(const Mat& src) {
        _src = src.clone();
        if (_src.channels() == 1)      cvtColor(_src, _src, COLOR_GRAY2BGR);
        else if (_src.channels() == 4) cvtColor(_src, _src, COLOR_BGRA2BGR);
        cvtColor(_src, _graySrc, COLOR_BGR2GRAY);
        if (_blueCh.size() != _src.size() || _blueCh.type() != CV_8UC1)
            _blueCh.create(_src.size(), CV_8UC1);
        if (_redCh.size() != _src.size() || _redCh.type() != CV_8UC1)
            _redCh.create(_src.size(), CV_8UC1);
        split(_src, _splitSrc);
        _blueCh = _splitSrc[0];
        _redCh = _splitSrc[2];
    }

    void estimateThresholds() {
        if (!_para.autoTune) return;
        const int step = max(1, (int)std::sqrt(_graySrc.total() / 60000.0));
        int hist[256] = { 0 };
        long long count = 0;
        for (int y = 0; y < _graySrc.rows; y += step) {
            const uchar* p = _graySrc.ptr<uchar>(y);
            for (int x = 0; x < _graySrc.cols; x += step) { hist[p[x]]++; count++; }
        }
        if (count <= 0) return;
        const double total = static_cast<double>(count);
        auto quantile = [&](double q) {
            double acc = 0;
            for (int i = 0; i < 256; i++)
            {
                acc += hist[i]; if (acc >= total * q)
                    return i;
            }
            return 255;
            };
        const int pWhite = quantile(_para.whitePercentile);
        int white = min(250, max(80, pWhite - 8));
        const double maxCoreRatio = 0.005;
        int coreCount = 0;
        for (int i = white + 1; i < 256; i++) coreCount += hist[i];
        while (white < 250 && coreCount > total * maxCoreRatio)
        {
            coreCount -= hist[white + 1];
            white++;
        }
        _para.whiteThreshold = white;
        _para.grayTHreshold = min(100, max(10, static_cast<int>(_para.whiteThreshold * 0.30f)));
        Mat diff;
        if (enemyColor() == ENEMY_RED)
            subtract(_redCh, _blueCh, diff);
        else
            subtract(_blueCh, _redCh, diff);
        double maxDiff = 0;
        minMaxLoc(diff, nullptr, &maxDiff, nullptr, nullptr);
        const int sepTh = max(20, static_cast<int>(maxDiff * 0.5));
        if (enemyColor() == ENEMY_RED) _para.separationThreshold_RED = sepTh;
        else                           _para.separationThreshold_BLUE = sepTh;
    }
    void colorSeparation() {
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

    vector<LightBar> findLightBars() {
        vector<LightBar> lights;
        Mat binCopy = _binary.clone();
        vector<vector<Point>> contours;
        findContours(binCopy, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        for (const auto& c : contours) {
            const float area = static_cast<float>(contourArea(c));
            if (area < _para.minLightArea || area > _para.maxLightArea)
                continue;
            const RotatedRect rect = minAreaRect(c);
            const float w = rect.size.width, h = rect.size.height;
            if (w <= 0 || h <= 0) continue;
            LightBar lb;
            lb.rect = rect;
            lb.center = rect.center;
            lb.length = max(w, h); lb.width = min(w, h);
            lb.ratio = lb.length / lb.width;
            if (lb.ratio < _para.minLightRatio || lb.ratio > _para.maxLightRatio) continue;
            if (lb.width < _para.minLightWidth) continue;
            const float axis = (w >= h) ? rect.angle : (rect.angle + 90.0f);
            lb.angle = axis - 90.0f * std::floor((axis + 45.0f) / 90.0f);
            lights.push_back(lb);
        }
        sort(lights.begin(), lights.end(), [](const LightBar& a, const LightBar& b) { return a.center.x < b.center.x; });
        return lights;
    }
    vector<Armor> matchArmors(const vector<LightBar>& lights) {
        vector<Armor> armors;
        const float loRatio = min(_para.minRatio, _para.pairMinRatioLo);
        const float hiRatio = max(_para.maxRatio, _para.pairMaxRatioHi);
        for (size_t i = 0; i < lights.size(); i++) {
            for (size_t j = i + 1; j < lights.size(); j++) {
                LightBar a = lights[i], b = lights[j];
                if (fabs(a.length - b.length) / max(a.length, b.length) > _para.maxLengthDiff) continue;//长度差
                if (fabs(a.angle - b.angle) > _para.maxAngleDiff) continue;//角度差
                const float width = static_cast<float>(norm(a.center - b.center));//像素的宽
                const float height = (a.length + b.length) * 0.5f;
                const float ratio = width / height;
                if (ratio < loRatio || ratio > hiRatio)
                    continue;
                if (a.center.x > b.center.x) swap(a, b);
                Point2f aPts[4], bPts[4];
                a.rect.points(aPts); b.rect.points(bPts);
                vector<Point2f> allPts(aPts, aPts + 4);
                allPts.insert(allPts.end(), bPts, bPts + 4);
                Armor armor;
                armor.li = (int)i; armor.ri = (int)j;
                armor.rect = minAreaRect(allPts);
                armor.center = (a.center + b.center) * 0.5f;
                armor.width = width;
                armor.height = height;
                armor.ratio = ratio;
                armor.rect.points(armor.points);
                armors.push_back(armor);
            }
        }
        vector<char> used(lights.size(), 0);
        vector<Armor> picked;
        for (const auto& a : armors) {
            if (a.li < 0 || a.ri < 0) continue;
            if (used[a.li] || used[a.ri]) continue;// 如果有灯条被占用，跳过
            used[a.li] = used[a.ri] = 1;// 标记灯条已用
            picked.push_back(a);
        }
        sort(picked.begin(), picked.end(), [](const Armor& x, const Armor& y) { return x.center.x < y.center.x; });
        return picked;
    }
    Mat _src, _graySrc, _sepA, _sepAGray, _binary, _blueCh, _redCh;
    vector<Mat> _splitSrc;
    vector<LightBar> _lastLights;
    vector<Armor> _lastArmors;
};


int main()
{
    string videoPath = "Resources/clip.avi";

    ArmorDetector detector;
    detector.setEnemyColor(ArmorDetector::ENEMY_BLUE);


    VideoCapture cap;
    if (videoPath == "0") cap.open(0);
    else                  cap.open(videoPath);

    if (!cap.isOpened()) {
        printf("视频/摄像头打开失败: %s\n", videoPath.c_str());
        return -1;
    }
    Mat frame;
    while (cap.read(frame)) {
        if (frame.empty()) break;

        vector<Armor> armors = detector.detect(frame);

        printf("装甲板=%d | 耗时=%.2fms (PnP %.2fms)\n",
            (int)armors.size(), detector._timing.total, detector._timing.pnp);
        Mat show = detector.drawOn(frame.clone(), detector.lastLights(), armors, detector._timing);
        imshow("Result", show);
        const int k = waitKey(1);
        if (k == 27 || k == 'q') break;
    }
    cap.release();
    destroyAllWindows();
    return 0;
}
