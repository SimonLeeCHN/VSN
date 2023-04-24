#ifndef UTILS_H
#define UTILS_H

#include "QImage"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"

void Mat2QImage(cv::Mat mat,QImage &img );
void QImage2Mat(QImage img,cv::Mat& mat);
bool GetMatFromFile(QString file,cv::Mat &mat);

#endif // UTILS_H
