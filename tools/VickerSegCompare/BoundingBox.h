/*
 *  实现一个可以单独调节四条边的矩形框
 */
#ifndef GRAPHICSRECT_H
#define GRAPHICSRECT_H

#include <QObject>
#include "QGraphicsItem"
#include "QPoint"
#include "QRect"
#include "QPainter"
#include "QPen"

class BoundingBox : public QObject,public QGraphicsItem
{
    Q_OBJECT
    Q_INTERFACES(QGraphicsItem)
public:
    BoundingBox();
    ~BoundingBox() override;
    QRectF boundingRect() const override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = nullptr) override;
    QPainterPath shape() const override;

    QRectF GetRealBoundRect();
    void SetBoundRectSize(QRectF newRect);
    int GetHoverSide();
    void ClearSelectSide();
    void MoveSelectSide(qreal step);

    enum {eSideNone=1,eSideTop,eSideBottom,eSideLeft,eSideRight};

private:
    QRectF m_BoundRect;

    int m_iHoverSide;
    int m_iSelectSide;
    bool m_bIsAdjusting;

protected:
    void hoverMoveEvent(QGraphicsSceneHoverEvent *event) override;
    void hoverLeaveEvent(QGraphicsSceneHoverEvent *event) override;
    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;

signals:
    void sigGeometryChange();       //几何发生变换信号
    void sigSelectSide(int side);
};

#endif // GRAPHICSRECT_H
