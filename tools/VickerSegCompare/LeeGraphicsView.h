/*
 *  重载实现GraphicsView
 */
#ifndef LEEGRAPHICSVIEW_H
#define LEEGRAPHICSVIEW_H

#include <QObject>
#include <QPoint>
#include "QGraphicsView"
#include "QWheelEvent"

class LeeGraphicsView : public QGraphicsView
{
    Q_OBJECT
public:
    LeeGraphicsView(QWidget* parent = nullptr);
    ~LeeGraphicsView() override;

    enum {eZoomFill,eZoomOrigin,eZoomUp,eZoomDown};
    void ZoomView(int type);
    void EnableDrag(bool state);

private:
    qreal m_qrZoomScale;

    QPoint m_MoveStartPos;

protected:
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;

signals:
    void sigMousePress(QMouseEvent* event);
    void sigMouseMove(QMouseEvent* event);
    void sigMouseRelease(QMouseEvent* event);
};

#endif // LEEGRAPHICSVIEW_H
