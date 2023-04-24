#include "batchtestdlg.h"
#include "ui_batchtestdlg.h"

#include "QFile"
#include "QTextStream"
#include "QFileDialog"
#include "QFileInfo"
#include "QMessageBox"
#include "QStringList"

#include "imgproalg.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"

BatchTestDlg::BatchTestDlg(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::BatchTestDlg)
{
    ui->setupUi(this);
}

BatchTestDlg::~BatchTestDlg()
{
    delete ui;
}

void BatchTestDlg::DoBatchTest()
{
    this->setEnabled(false);

    QDir gtPath(ui->LE_gtPath->text());
    QDir cpPath(ui->LE_comparePath->text());
    QString outputFileName = ui->LE_outputPath->text();

    //创建结果文件
    QFile _outFile(outputFileName);
    if(!_outFile.open(QFile::ReadWrite|QFile::Append))
    {
        QMessageBox::critical(this,"Output File error","Output File error");
    }
    _outFile.write(gtPath.path().toLocal8Bit() + '\n');
    _outFile.write(cpPath.path().toLocal8Bit() + '\n');
    _outFile.write(QString("Name,gtArea,cpArea,Dice,d1,d2\n").toLocal8Bit());

    //读取文件
    QStringList _nameFilters;
    _nameFilters<<"*.png"<<"*.jpg"<<"*.bmp";
    auto gtFileList = gtPath.entryList(_nameFilters,QDir::Files);
    auto cpFileList = cpPath.entryList(_nameFilters,QDir::Files);
    assert(gtFileList.count() == cpFileList.count());

    ui->progressBar->setValue(0);
    ui->progressBar->setMaximum(gtFileList.count());

    //遍历对比每个图片
    for(int _index = 0 ; _index < gtFileList.count() ; _index++)
    {
        //读入图像
        QString _gtFileName = gtPath.filePath(gtFileList.value(_index));
        QString _cpFileName = cpPath.filePath(cpFileList.value(_index));
        cv::Mat gtMat = cv::imread(_gtFileName.toLocal8Bit().toStdString(),cv::IMREAD_GRAYSCALE);
        cv::Mat cpMat = cv::imread(_cpFileName.toLocal8Bit().toStdString(),cv::IMREAD_GRAYSCALE);
        if(gtMat.size != cpMat.size)
            cv::resize(cpMat,cpMat,gtMat.size());

        //计算面积
        int gtArea = cv::countNonZero(gtMat);
        int cpArea = cv::countNonZero(cpMat);

        //计算Dice
        cv::Mat _intersectMat;
        cv::bitwise_and(gtMat, cpMat, _intersectMat);
        double dice = double(2.0 * cv::countNonZero(_intersectMat) / double(gtArea + cpArea + 0.000001));        //避免除数为0

        //求解d1与d2
        int _angle = ImgProAlg::AlgCalRotateAngle(cpMat);
        cv::Rect _bb = ImgProAlg::AlgCalBoundingBox(cpMat,_angle,ui->CB_EnableMatShow->isChecked());
        int d1 = _bb.width;
        int d2 = _bb.height;

        //写入文件
        QString _str = gtFileList.value(_index) + ',';      //name
        _str.append(QString::number(gtArea) + ',');         //gtArea
        _str.append(QString::number(cpArea) + ',');         //cpArea
        _str.append(QString::number(dice) + ',');           //dice
        _str.append(QString::number(d1) + ',');             //d1
        _str.append(QString::number(d2) + '\n');            //d2
        _outFile.write(_str.toLocal8Bit());

        //进度条
        ui->progressBar->setValue(ui->progressBar->value() + 1);
        qApp->processEvents();
    }

    _outFile.close();

    this->setEnabled(true);
    QMessageBox::information(this,"BatchTest end","Batch test end successfully");

}

void BatchTestDlg::on_BTN_gtPath_clicked()
{
    ui->LE_gtPath->setText(QFileDialog::getExistingDirectory(this
                                                             ,"Open ground true dir"
                                                             ,"./"
                                                             ,QFileDialog::ShowDirsOnly|QFileDialog::ReadOnly));
}


void BatchTestDlg::on_BTN_comparePath_clicked()
{
    ui->LE_comparePath->setText(QFileDialog::getExistingDirectory(this
                                                             ,"Open compare dir"
                                                             ,"./"
                                                             ,QFileDialog::ShowDirsOnly|QFileDialog::ReadOnly));
}


void BatchTestDlg::on_BTN_outputPath_clicked()
{
    ui->LE_outputPath->setText(QFileDialog::getSaveFileName(this,"save file at","./","*.csv"));
}


void BatchTestDlg::on_BTN_Start_clicked()
{
    /*
     *  检查路径是否合法
     */
    QFileInfo probe;

    probe.setFile(ui->LE_gtPath->text());
    if((true == ui->LE_gtPath->text().isEmpty())
            || (false == probe.isDir()))
    {
        QMessageBox::critical(this,"Empty or not dir","ground true is empty or not valid dir");
        return;
    }

    probe.setFile(ui->LE_comparePath->text());
    if((true == ui->LE_comparePath->text().isEmpty())
            || (false == probe.isDir()))
    {
        QMessageBox::critical(this,"Empty or not dir","compare is empty or not valid dir");
        return;
    }

    probe.setFile(ui->LE_outputPath->text());
    if((true == ui->LE_comparePath->text().isEmpty())
            || (true == probe.isFile()))
    {
        QMessageBox::critical(this,"Empty or file exist","outfile path is empty or have a exist file");
        return;
    }

    this->DoBatchTest();
}

