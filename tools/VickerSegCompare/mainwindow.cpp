#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "batchtestdlg.h"

#include "QFileDialog"
#include "QPixmap"
#include "QDebug"
#include "QImage"
#include "QPainter"
#include "QDropEvent"
#include "QMimeData"
#include "QList"
#include "QUrl"

#include "utils.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"

using namespace std;
using namespace cv;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    /*
     * 注册信号槽用自定义变量类型
     */
    qRegisterMetaType<cv::Mat>("cv::Mat");
    qRegisterMetaType<cv::Rect>("cv::Rect");

    /*
     * 图像计算线程
     */
    m_pImgProAlg = new ImgProAlg();
    m_pImgProAlg->moveToThread(&m_ImgProAlgThread);
    connect(&m_ImgProAlgThread,SIGNAL(finished()),m_pImgProAlg,SLOT(deleteLater()));

    //计算线程 - 计算包围框
    connect(this,SIGNAL(sigReqCalBoundingBox(cv::Mat,int,bool)),m_pImgProAlg,SLOT(slotCalBoundingBox(cv::Mat, int, bool)));
    connect(m_pImgProAlg,SIGNAL(sigCalBoundingBoxFin(cv::Rect)),this,SLOT(slotCalBoundingBoxFin(cv::Rect)));

    //计算线程 - 计算最大D时的旋转角度
    connect(this,SIGNAL(sigReqCalRotateAngle(cv::Mat)),m_pImgProAlg,SLOT(slotCalRotateAngle(cv::Mat)));
    connect(m_pImgProAlg,SIGNAL(sigCalRotateAngleFin(int)),this,SLOT(slotCalRotateAngleFin(int)));

    //计算线程 - 启动
    m_ImgProAlgThread.start();

    /*
     * Graph设置
     */
    //设置scene
    m_Scene.setBackgroundBrush(QBrush(Qt::darkGray));

    //添加测量框
    m_pBoundingBox = new BoundingBox();
    m_Scene.addItem(m_pBoundingBox);
    m_pBoundingBox->setVisible(true);
    m_pBoundingBox->setZValue(99);
    connect(m_pBoundingBox,SIGNAL(sigGeometryChange()),this,SLOT(slotBoxGeometryChange()));

    //图片层
    m_pScenePixmap = m_Scene.addPixmap(QPixmap());

    //graphicsview 设置
    ui->graphicsView->setScene(&m_Scene);
    ui->graphicsView->EnableDrag(true);

}

MainWindow::~MainWindow()
{
    cv::destroyAllWindows();

    m_ImgProAlgThread.quit();
    m_ImgProAlgThread.wait();

    delete m_pBoundingBox;
    delete m_pScenePixmap;
    delete ui;
}

/*
 *  更新显示信息
 */
void MainWindow::UpdateDetailsShow()
{
    ui->PTE_Detail->clear();

    //GT
    ui->PTE_Detail->insertPlainText(QString("Ground True:\n Space: %1\n").arg(QString::number(gGroundTrue.indentationSpace)));

    ui->PTE_Detail->insertPlainText(QString("\n"));

    //比较图
    ui->PTE_Detail->insertPlainText(QString("Compare Image:\n Space: %1\n").arg(QString::number(gCompareImg.indentationSpace)));
    ui->PTE_Detail->insertPlainText(QString(" Dice(GT,CI): %1\n DiceLoss: %2\n").arg(QString::number(gCompareImg.dice)).arg(QString::number(1 - gCompareImg.dice)));
    QRectF _measureRect = m_pBoundingBox->GetRealBoundRect();
    ui->PTE_Detail->insertPlainText(QString(" d1: %1\n d2: %2\n").arg(QString::number(_measureRect.width())).arg(QString::number(_measureRect.height())));

    //旋转角度
    ui->PTE_Detail->insertPlainText(QString("\nAngle: %1\n").arg(QString::number(ui->HS_Rotation->sliderPosition())));
}

void MainWindow::DoImgCalculation()
{
    if(gGroundTrue.picMat.data != nullptr)
    {
        //计算groundtrue面积
        gGroundTrue.indentationSpace = cv::countNonZero(gGroundTrue.picMat);

        if(gCompareImg.picMat.data != nullptr)
        {
            //计算面积
            gCompareImg.indentationSpace = cv::countNonZero(gCompareImg.picMat);

            //计算dice
            if(gGroundTrue.picMat.data != nullptr)
            {
                cv::Mat _intersectMat;
                cv::bitwise_and(gGroundTrue.picMat, gCompareImg.picMat, _intersectMat);
                int _intersectCount = cv::countNonZero(_intersectMat);
                gCompareImg.dice = double(2.0*_intersectCount / double(gGroundTrue.indentationSpace + gCompareImg.indentationSpace + 0.000001));        //避免除数为0
            }

            //求解摆正压痕所需旋转的角度
            if(ui->AC_EnableAutoRotate->isChecked())
            {
                this->setEnabled(false);
                emit sigReqCalRotateAngle(gCompareImg.picMat);
            }
            else
            {
                //触发计算包围框
                this->on_HS_Rotation_sliderReleased();
            }
        }
    }
}

void MainWindow::OpenGroundTrueImg(QString filename)
{
    //载入图片
    gGroundTrue.picMat = cv::imread(filename.toLocal8Bit().toStdString(),cv::IMREAD_GRAYSCALE);
    Mat2QImage(gGroundTrue.picMat,gGroundTrue.picImg);
    ui->LBL_GroundTruePic->setPixmap(QPixmap::fromImage(gGroundTrue.picImg));

    //显示文件名
    auto splitList = filename.split('/');
    ui->LBL_GroundTrueName->setText(splitList.last());

    //执行图像计算
    this->DoImgCalculation();

    //显示计算数据
    this->UpdateDetailsShow();
}

void MainWindow::OpenCompareImg(QString filename)
{
    //载入图片时初始化
    ui->HS_Rotation->setSliderPosition(0);
    m_pScenePixmap->setRotation(0);             //重置旋转
    m_pScenePixmap->setPixmap(QPixmap());       //清空图片
    ui->PTE_Detail->clear();                    //清除数据

    //载入图片
    gCompareImg.picMat = cv::imread(filename.toLocal8Bit().toStdString(),cv::IMREAD_GRAYSCALE);
    Mat2QImage(gCompareImg.picMat,gCompareImg.picImg);
    m_pScenePixmap->setPixmap(QPixmap::fromImage(gCompareImg.picImg));
    m_pScenePixmap->setTransformOriginPoint(QPointF(gCompareImg.picImg.width()/2,gCompareImg.picImg.height()/2));
    m_Scene.setSceneRect(gCompareImg.picImg.rect());
    ui->graphicsView->ZoomView(LeeGraphicsView::eZoomFill);

    //显示文件名
    auto splitList = filename.split('/');
    ui->LBL_CompareImgName->setText(splitList.last());

    //执行图像计算
    this->DoImgCalculation();

    //显示计算数据
    this->UpdateDetailsShow();
}

/*
 *  测量框发射几何变化信号
 *  更新D1,D2测量量
 *  D1：测量框宽度
 *  D2: 测量框高度
 */
void MainWindow::slotBoxGeometryChange()
{
    this->UpdateDetailsShow();
}

void MainWindow::slotCalBoundingBoxFin(cv::Rect outRect)
{
    QRectF _bb(QPointF(outRect.x,outRect.y),QSizeF(outRect.width,outRect.height));
    _bb = _bb.adjusted(0.25,0.25,0.25,0.25);              //微调边缘以消除画笔宽度影响

    m_pBoundingBox->SetBoundRectSize(_bb);
    m_pBoundingBox->setPos(_bb.topLeft());
}

void MainWindow::slotCalRotateAngleFin(int angle)
{
    this->setEnabled(true);

    ui->HS_Rotation->setSliderPosition(angle);
    m_pScenePixmap->setRotation(angle);

    //触发计算包围框
    this->on_HS_Rotation_sliderReleased();
}

void MainWindow::dragEnterEvent(QDragEnterEvent *e)
{
    e->acceptProposedAction();
}

/*
 *  允许通过拖拽的方式打开图片文件
 *  靠左边打开为GT，靠右边打开为比较图
 */
void MainWindow::dropEvent(QDropEvent *event)
{
    auto urls = event->mimeData()->urls();
    if(urls.isEmpty())
        return;

    QString filepath = urls.first().toLocalFile();
    QPoint dropPos = event->position().toPoint();

    if(dropPos.x() <= ui->LBL_GroundTruePic->geometry().width())
    {
        //打开GT图
        this->OpenGroundTrueImg(filepath);
    }else
    {
        //打开比较图
        this->OpenCompareImg(filepath);
    }

}

void MainWindow::on_BTN_OpenGT_clicked()
{
    //获取文件名
    QString fileName = QFileDialog::getOpenFileName(this,tr("Open Image"), ".", tr("Image Files (*.png *.jpg *.bmp)"));
    if(fileName.isEmpty())
        return;

    //载入GroundTrue图片
    this->OpenGroundTrueImg(fileName);
}

void MainWindow::on_BTN_OpenCI_clicked()
{
    //载入与显示图片
    QString fileName = QFileDialog::getOpenFileName(this,tr("Open Image"), ".", tr("Image Files (*.png *.jpg *.bmp)"));
    if(fileName.isEmpty())
        return;

    //载入比较图片
    this->OpenCompareImg(fileName);
}

void MainWindow::on_HS_Rotation_sliderMoved(int position)
{
    Q_UNUSED(position)

    //避免传入空图像时调用旋转
    if(m_pScenePixmap->pixmap().isNull())
        return;

    m_pScenePixmap->setRotation(position);
    this->UpdateDetailsShow();
}

void MainWindow::on_HS_Rotation_sliderReleased()
{
    //请求线程计算包围框
    emit sigReqCalBoundingBox(gCompareImg.picMat.clone(),ui->HS_Rotation->sliderPosition(),ui->AC_EnableImshow->isChecked());
}

void MainWindow::on_AC_BatchTest_triggered()
{
    BatchTestDlg dlg;
    dlg.exec();
}

