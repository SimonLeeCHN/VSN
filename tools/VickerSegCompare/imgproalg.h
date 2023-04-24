#ifndef IMGPROALG_H
#define IMGPROALG_H

#include <QObject>
#include <QString>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"

class ImgProAlg : public QObject
{
    Q_OBJECT

public:
    static cv::Rect AlgCalBoundingBox(cv::Mat inMat, int rotateAngle, bool showMat);
    static int AlgCalRotateAngle(cv::Mat inMat);
    void AlgBatchTest(QString gtDir,QString cpDir,QString outFile);

public slots:
    void slotCalBoundingBox(cv::Mat inMat, int rotateAngle, bool showMat);
    void slotCalRotateAngle(cv::Mat inMat);
    void slotBatchTest(QString gtDir,QString cpDir,QString outFile);

signals:
    void sigCalBoundingBoxFin(cv::Rect outRect);
    void sigCalRotateAngleFin(int angle);

};

#endif // IMGPROALG_H
