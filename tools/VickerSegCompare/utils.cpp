#include "utils.h"

/*  Mat <-> QImage convert functions    */

/*
 *  Mat转QImage
 *  img参数为引用方式
 */
void Mat2QImage(cv::Mat mat,QImage &img )
{
    if(mat.channels() == 4)
    {
        img = QImage(mat.data, mat.cols, mat.rows, mat.step,QImage::Format_ARGB32);
    }else if(mat.channels() == 3)
    {
        img = QImage(mat.data, mat.cols, mat.rows, mat.step,QImage::Format_RGB888).rgbSwapped();
    }else if(mat.channels() == 1)
    {
        img = QImage(mat.data, mat.cols, mat.rows, mat.step,QImage::Format_Indexed8);
    }
}

/*
 *  QImage转mat
 *  mat参数为引用方式
 */
void QImage2Mat(QImage img,cv::Mat& mat)
{
    cv::Mat _mat;
    switch (img.format())
    {
        case QImage::Format_ARGB32:
        case QImage::Format_RGB32:
        case QImage::Format_ARGB32_Premultiplied:
        {
            _mat = cv::Mat(img.height(), img.width(), CV_8UC4, (void*)img.constBits(), img.bytesPerLine());
            break;
        }
        case QImage::Format_RGB888:
        {
            _mat = cv::Mat(img.height(), img.width(), CV_8UC3, (void*)img.constBits(), img.bytesPerLine());
            break;
        }
        case QImage::Format_Indexed8:
        {
            _mat = cv::Mat(img.height(), img.width(), CV_8UC1, (void*)img.constBits(), img.bytesPerLine());
            break;
        }
        default:
        {
            break;
        }
    }

    //这里需返回clone否则会因为上面临时变量作用域解除而导致析构
    mat =  _mat.clone();
}

/*
 *  读取文件到mat
 *  mat参数为引用方式
 */
bool GetMatFromFile(QString file,cv::Mat &mat)
{
    QImage _Img;
    cv::Mat _mat;
    if(_Img.load(file))
    {
        QImage2Mat(_Img,_mat);
        mat = _mat.clone();
        return true;
    }

    return false;
}
