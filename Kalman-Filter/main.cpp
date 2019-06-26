/******************************************
 * OpenCV: Ball Tracking using a
 * Kalman Filter     
 * 
 * https://www.myzhar.com/blog/tutorials/tutorial-opencv-ball-tracker-using-kalman-filter/
 ******************************************/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp> // trackbar
#include <opencv2/imgproc/imgproc.hpp> // colour conversion and blur
#include <opencv2/video/video.hpp>

#include <iostream>
#include <vector>

using namespace std;

#define MIN_H_RED 0
#define MAX_H_RED 9 // were found to be good values using trackbar
//Blue: 200 to 300
//Red: 0 to 10 and 160 to 180

//Trackbar variables
const int max_value_H = 360 / 2;
const int max_value = 255;
int low_H = 0;
int high_H = max_value_H;
static void on_low_H_thresh_trackbar(int, void *)
{
    low_H = min(high_H - 1, low_H);
    cv::setTrackbarPos("Low H", "Threshold", low_H);
}
static void on_high_H_thresh_trackbar(int, void *)
{
    high_H = max(high_H, low_H + 1);
    cv::setTrackbarPos("High H", "Threshold", high_H);
}

int main()
{
    // Camera frame
    cv::Mat frame;

    // >>>> Kalman Filter
    int stateSize = 6;
    int measSize = 4;
    int contrSize = 0;

    unsigned int type = CV_32F;
    cv::KalmanFilter kf(stateSize, measSize, contrSize, type);

    cv::Mat state(stateSize, 1, type); // [x,y,v_x,v_y,w,h]
    cv::Mat meas(measSize, 1, type);   // [z_x,z_y,z_w,z_h]
    //cv::Mat procNoise(stateSize, 1, type)
    // [E_x,E_y,E_v_x,E_v_y,E_w,E_h]

    // Transition State Matrix A
    // Note: set dT at each processing step!
    // [ 1 0 dT 0  0 0 ]
    // [ 0 1 0  dT 0 0 ]
    // [ 0 0 1  0  0 0 ]
    // [ 0 0 0  1  0 0 ]
    // [ 0 0 0  0  1 0 ]
    // [ 0 0 0  0  0 1 ]
    cv::setIdentity(kf.transitionMatrix);

    // Measure Matrix H
    // [ 1 0 0 0 0 0 ]
    // [ 0 1 0 0 0 0 ]
    // [ 0 0 0 0 1 0 ]
    // [ 0 0 0 0 0 1 ]
    kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
    kf.measurementMatrix.at<float>(0) = 1.0f;
    kf.measurementMatrix.at<float>(7) = 1.0f;
    kf.measurementMatrix.at<float>(16) = 1.0f;
    kf.measurementMatrix.at<float>(23) = 1.0f;

    // Process Noise Covariance Matrix Q
    // [ Ex   0   0     0     0    0  ]
    // [ 0    Ey  0     0     0    0  ]
    // [ 0    0   Ev_x  0     0    0  ]
    // [ 0    0   0     Ev_y  0    0  ]
    // [ 0    0   0     0     Ew   0  ]
    // [ 0    0   0     0     0    Eh ]
    //cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2));
    kf.processNoiseCov.at<float>(0) = 1e-2;
    kf.processNoiseCov.at<float>(7) = 1e-2;
    kf.processNoiseCov.at<float>(14) = 5.0f;
    kf.processNoiseCov.at<float>(21) = 5.0f;
    kf.processNoiseCov.at<float>(28) = 1e-2;
    kf.processNoiseCov.at<float>(35) = 1e-2;

    // Measures Noise Covariance Matrix R
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));
    // <<<< Kalman Filter

    // Camera Index
    int idx = 1;

    // Camera Capture
    cv::VideoCapture cap;

    // >>>>> Camera Settings
    if (!cap.open(idx))
    {
        cout << "Webcam not connected.\n"
             << "Please verify\n";
        return EXIT_FAILURE;
    }

    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1024);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 768);
    // <<<<< Camera Settings

    cout << "\nHit 'q' to exit...\n";

    char ch = 0;

    double ticks = 0;
    bool found = false;

    int notFoundCount = 0;

    // ------- TRACKBARS -----
    cv::namedWindow("Tracking");
    cv::namedWindow("Threshold");
    // Trackbars to set thresholds for HSV values
    cv::createTrackbar("Low H", "Threshold", &low_H, max_value_H, on_low_H_thresh_trackbar);
    cv::createTrackbar("High H", "Threshold", &high_H, max_value_H, on_high_H_thresh_trackbar);

    // >>>>> Main loop
    while (ch != 'q' && ch != 'Q')
    {
        double precTick = ticks;
        ticks = (double)cv::getTickCount();

        double dT = (ticks - precTick) / cv::getTickFrequency(); //seconds

        // Frame acquisition
        cap >> frame;

        cv::Mat res;
        frame.copyTo(res);

        if (found)
        {
            // >>>> Matrix A
            kf.transitionMatrix.at<float>(2) = dT;
            kf.transitionMatrix.at<float>(9) = dT;
            // <<<< Matrix A

            //cout << "dT:" << endl
            //   << dT << endl;

            state = kf.predict();
            //cout << "State post:" << endl
            //   << state << endl;

            cv::Rect predRect;
            predRect.width = state.at<float>(4);
            predRect.height = state.at<float>(5);
            predRect.x = state.at<float>(0) - predRect.width / 2;
            predRect.y = state.at<float>(1) - predRect.height / 2;

            cv::Point center;
            center.x = state.at<float>(0);
            center.y = state.at<float>(1);
            cv::circle(res, center, 2, CV_RGB(255, 0, 0), -1);

            cv::rectangle(res, predRect, CV_RGB(255, 0, 0), 2);
        }

        // ---- SMOOTHEN NOISE ----
        // Types of Filters: https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html
        cv::Mat frame_blur1;
        cv::GaussianBlur(frame, frame_blur1, cv::Size(5, 5), 3.0, 3.0);
        cv::Mat frame_blur;
        cv::medianBlur(frame_blur1, frame_blur, 5);

        //------- HSV CONVERSION ----
        //convert BGR colourspace to HSV for later thresholding
        //https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
        cv::Mat frame_HSV;
        cv::cvtColor(frame_blur, frame_HSV, CV_BGR2HSV); // (src, dest, type)

        // ------- THRESHOLDING COLOUR -----
        //Thresholding in HSV is better than in RGB
        //https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html

        cv::Mat rangeRes = cv::Mat::zeros(frame.size(), CV_8UC1);

        //inRange(FRAME_SOURCE, SCALAR_LOW, SCALAR_HIGH, FRAME_DEST)
        cv::inRange(frame_HSV, cv::Scalar(MIN_H_RED, 100, 100),
                    cv::Scalar(MAX_H_RED, 255, 255), rangeRes);

        // >>>>> Improving the result
        cv::erode(rangeRes, rangeRes, cv::Mat(), cv::Point(-1, -1), 2);
        cv::dilate(rangeRes, rangeRes, cv::Mat(), cv::Point(-1, -1), 2);
        // <<<<< Improving the result

        // Threshold viewing
        cv::imshow("Threshold", rangeRes);

        // >>>>> Contours detection
        vector<vector<cv::Point>> contours;
        cv::findContours(rangeRes, contours, CV_RETR_EXTERNAL,
                         CV_CHAIN_APPROX_NONE);
        // <<<<< Contours detection

        // >>>>> Filtering
        vector<vector<cv::Point>> balls;
        vector<cv::Rect> ballsBox;
        for (size_t i = 0; i < contours.size(); i++)
        {
            cv::Rect bBox;
            bBox = cv::boundingRect(contours[i]);

            float ratio = (float)bBox.width / (float)bBox.height;
            if (ratio > 1.0f)
                ratio = 1.0f / ratio;

            // Searching for a bBox almost square
            if (ratio > 0.75 && bBox.area() >= 400)
            {
                balls.push_back(contours[i]);
                ballsBox.push_back(bBox);
            }
        }
        // <<<<< Filtering

        //cout << "Balls found:" << ballsBox.size() << endl;

        // >>>>> Detection result
        for (size_t i = 0; i < balls.size(); i++)
        {
            cv::drawContours(res, balls, i, CV_RGB(20, 150, 20), 1);
            cv::rectangle(res, ballsBox[i], CV_RGB(0, 255, 0), 2);

            cv::Point center;
            center.x = ballsBox[i].x + ballsBox[i].width / 2;
            center.y = ballsBox[i].y + ballsBox[i].height / 2;
            cv::circle(res, center, 2, CV_RGB(20, 150, 20), -1);

            stringstream sstr;
            sstr << "(" << center.x << "," << center.y << ")";
            cv::putText(res, sstr.str(),
                        cv::Point(center.x + 3, center.y - 3),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(20, 150, 20), 2);
        }
        // <<<<< Detection result

        // >>>>> Kalman Update
        if (balls.size() == 0)
        {
            notFoundCount++;
            //cout << "notFoundCount:" << notFoundCount << endl;
            if (notFoundCount >= 100)
            {
                found = false;
            }
            /*else
                kf.statePost = state;*/
        }
        else
        {
            notFoundCount = 0;

            meas.at<float>(0) = ballsBox[0].x + ballsBox[0].width / 2;
            meas.at<float>(1) = ballsBox[0].y + ballsBox[0].height / 2;
            meas.at<float>(2) = (float)ballsBox[0].width;
            meas.at<float>(3) = (float)ballsBox[0].height;

            if (!found) // First detection!
            {
                // >>>> Initialization
                kf.errorCovPre.at<float>(0) = 1; // px
                kf.errorCovPre.at<float>(7) = 1; // px
                kf.errorCovPre.at<float>(14) = 1;
                kf.errorCovPre.at<float>(21) = 1;
                kf.errorCovPre.at<float>(28) = 1; // px
                kf.errorCovPre.at<float>(35) = 1; // px

                state.at<float>(0) = meas.at<float>(0);
                state.at<float>(1) = meas.at<float>(1);
                state.at<float>(2) = 0;
                state.at<float>(3) = 0;
                state.at<float>(4) = meas.at<float>(2);
                state.at<float>(5) = meas.at<float>(3);
                // <<<< Initialization

                kf.statePost = state;

                found = true;
            }
            else
                kf.correct(meas); // Kalman Correction

            //cout << "Measure matrix:" << endl
            //   << meas << endl;
        }
        // <<<<< Kalman Update

        // Final result
        cv::imshow("Tracking", res);

        // User key
        ch = cv::waitKey(1);
    }

    return EXIT_SUCCESS;
}