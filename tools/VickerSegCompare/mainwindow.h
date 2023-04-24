#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QDragEnterEvent>
#include <QDropEvent>
#include "QGraphicsScene"
#include "QThread"

#include "BoundingBox.h"
#include "LeeGraphicsView.h"
#include "imgproalg.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;

    ImgProAlg* m_pImgProAlg = nullptr;              //图像计算线程
    QThread m_ImgProAlgThread;

    BoundingBox* m_pBoundingBox = nullptr;
    QGraphicsScene m_Scene;
    QGraphicsPixmapItem* m_pScenePixmap = nullptr;

    struct{
        cv::Mat picMat;
        QImage picImg;
        unsigned int indentationSpace = 0;
    }gGroundTrue;
    struct{
        cv::Mat picMat;
        QImage picImg;
        unsigned int indentationSpace = 0;
        double dice = 0;
    }gCompareImg;

    void UpdateDetailsShow();
    void DoImgCalculation();
    void OpenGroundTrueImg(QString filename);
    void OpenCompareImg(QString filename);

protected:
    void dropEvent(QDropEvent *event);
    void dragEnterEvent(QDragEnterEvent *e);

signals:
    void sigReqCalBoundingBox(cv::Mat inMat, int rotateAngle, bool showMat);
    void sigReqCalRotateAngle(cv::Mat inMat);

public slots:
    void slotBoxGeometryChange();
    void slotCalBoundingBoxFin(cv::Rect outRect);
    void slotCalRotateAngleFin(int angle);

private slots:
    void on_BTN_OpenGT_clicked();

    void on_BTN_OpenCI_clicked();

    void on_HS_Rotation_sliderMoved(int position);

    void on_HS_Rotation_sliderReleased();
    void on_AC_BatchTest_triggered();
};
#endif // MAINWINDOW_H
