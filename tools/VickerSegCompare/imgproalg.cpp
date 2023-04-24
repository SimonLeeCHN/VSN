#include "imgproalg.h"

#include "QDir"
#include "QFile"
#include "QFileInfo"
#include "QRect"

/*
 *  计算包围框
 *  考虑旋转角度,返回一个cv::Rect类型包围框
 */
cv::Rect ImgProAlg::AlgCalBoundingBox(cv::Mat inMat, int rotateAngle, bool showMat)
{
    cv::Point2f centerPoint(inMat.cols/2,inMat.rows/2);
    auto rotationMat = cv::getRotationMatrix2D(centerPoint,-rotateAngle,1);    //opencv与qt里旋转方向相反
    cv::warpAffine(inMat,inMat,rotationMat,cv::Size(inMat.cols,inMat.rows));

    if(showMat)
        cv::imshow("cloneMat", inMat);

    //计算包围框
    auto _ret = cv::boundingRect(inMat);
    return _ret;
}

/*
 *  通过枚举方式计算传入维氏压痕图像最大包围框的角度
 *  枚举旋转，d1+d2最大即可
 */
int ImgProAlg::AlgCalRotateAngle(cv::Mat inMat)
{
    /*
     * 自适应裁剪，避免后续处理耗时过长
     */
//    auto _bb = cv::boundingRect(inMat);
//    QRect _qbb(QPoint(_bb.x,_bb.y),QSize(_bb.width,_bb.height));

//    //扩展并裁切，注意不要越界
//    _qbb.adjust(-50,-50,50,50);
//    _qbb = _qbb.intersected(QRect(0,0,inMat.cols,inMat.rows));
//    _bb = cv::Rect(_qbb.x(),_qbb.y(),_qbb.width(),_qbb.height());
//    inMat = inMat(_bb);                     //裁切

//    //缩减图像
//    auto _bb = cv::boundingRect(inMat);
//    if(_bb.width > 500)
//    {
//        double _fac = double(500.0/inMat.rows);
//        cv::resize(inMat,inMat,cv::Size(inMat.cols * _fac,inMat.rows * _fac));
//    }

    cv::Mat _cloneMat = inMat.clone();
    cv::Point2f _centerPoint(_cloneMat.cols/2,_cloneMat.rows/2);

    /*
     * 最大值只会在压痕图像旋转左右45度的范围内出现，先旋转到头，再枚举到另一端
     */
    int _maxD = 0;
    int _maxAngle = 0;
    for(int angle = -45; angle <= 45; angle++)
    {
        //旋转图像
        auto _rotation = cv::getRotationMatrix2D(_centerPoint, angle, 1);
        cv::warpAffine(inMat,_cloneMat,_rotation,cv::Size(_cloneMat.cols,_cloneMat.rows));

        //求包围框
        auto _bb = cv::boundingRect(_cloneMat);
        int _D = _bb.width + _bb.height;

        if(_D >= _maxD)
        {
            _maxD = _D;
            _maxAngle = angle;
        }
    }

    return -_maxAngle;                  //opencv与qt里旋转方向相反
}

void ImgProAlg::AlgBatchTest(QString gtDir, QString cpDir, QString outFile)
{

}


/******SLOT******/
void ImgProAlg::slotCalBoundingBox(cv::Mat inMat, int rotateAngle, bool showMat)
{
    auto _ret = this->AlgCalBoundingBox(inMat, rotateAngle, showMat);
    emit sigCalBoundingBoxFin(_ret);
}

void ImgProAlg::slotCalRotateAngle(cv::Mat inMat)
{
    auto _maxAngle = this->AlgCalRotateAngle(inMat);

    emit sigCalRotateAngleFin(_maxAngle);
}

void ImgProAlg::slotBatchTest(QString gtDir, QString cpDir, QString outFile)
{

}
