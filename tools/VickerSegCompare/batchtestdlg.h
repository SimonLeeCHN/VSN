#ifndef BATCHTESTDLG_H
#define BATCHTESTDLG_H

#include <QDialog>

namespace Ui {
class BatchTestDlg;
}

class BatchTestDlg : public QDialog
{
    Q_OBJECT

public:
    explicit BatchTestDlg(QWidget *parent = nullptr);
    ~BatchTestDlg();

    void DoBatchTest();

private slots:
    void on_BTN_gtPath_clicked();

    void on_BTN_comparePath_clicked();

    void on_BTN_outputPath_clicked();

    void on_BTN_Start_clicked();

private:
    Ui::BatchTestDlg *ui;
};

#endif // BATCHTESTDLG_H
