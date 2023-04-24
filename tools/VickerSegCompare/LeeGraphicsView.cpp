#include "LeeGraphicsView.h"
#include "math.h"
#include "QDebug"

#define ZOOM_STEP   0.2

LeeGraphicsView::LeeGraphicsView(QWidget *parent) :
    QGraphicsView(parent)
{
    m_qrZoomScale = 0;

    //缩放跟随鼠标位置
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    setResizeAnchor(QGraphicsView::AnchorUnderMouse);
}

LeeGraphicsView::~LeeGraphicsView()
{

}

void LeeGraphicsView::ZoomView(int type)
{
    //根据类型进行缩放
    switch (type)
    {
        case LeeGraphicsView::eZoomFill:
        {   //缩放到填充视图

            //重置回原始图形变化矩阵
            this->resetTransform();

            //计算缩放倍率
            m_qrZoomScale = double(this->rect().height()) / double(this->scene()->sceneRect().height());

            this->scale(m_qrZoomScale,m_qrZoomScale);

            break;
        }
        case LeeGraphicsView::eZoomOrigin:
        {
            //重置回原始图形变化矩阵
            this->resetTransform();

            break;
        }
        case LeeGraphicsView::eZoomUp:
        {
            //简单放大
            qreal _scale = 1 + ZOOM_STEP;
            this->scale(_scale,_scale);
            m_qrZoomScale+= _scale;

            break;
        }
        case LeeGraphicsView::eZoomDown:
        {
            //简单缩小
            qreal _scale = 1 - ZOOM_STEP;
            this->scale(_scale,_scale);
            m_qrZoomScale+= _scale;

            break;
        }
        default:
        {
            break;
        }
    }
}

void LeeGraphicsView::EnableDrag(bool state)
{
    if(state)
    {
        //使能拖动
        setDragMode(QGraphicsView::ScrollHandDrag);
    }
    else
    {
        //失能拖动
        setDragMode(QGraphicsView::NoDrag);
    }
}

/******EVENTS******/

/*
 *  滚轮事件
 */
void LeeGraphicsView::wheelEvent(QWheelEvent *event)
{
    if(event->angleDelta().y() > 0)
    {
        this->ZoomView(LeeGraphicsView::eZoomUp);
    }
    else
    {
        this->ZoomView(LeeGraphicsView::eZoomDown);
    }

    event->accept();
}

/*
 *  鼠标左键被按下事件
 */
void LeeGraphicsView::mousePressEvent(QMouseEvent *event)
{
//    qDebug()<<"1"<<event->pos();
//    qDebug()<<"2"<<mapToScene(event->pos());
//    qDebug()<<"3"<<this->scene()->itemAt(mapToScene(event->pos()),this->transform());
//    qDebug()<<"4"<<this->scene()->items(mapToScene(event->pos()));
//
//    /*
//     *  若下方没有其他item，发射信号,接受事件
//     *  若有其他item，继续向下传递
//     */
//    auto _tempPoint = this->scene()->items(mapToScene(event->pos()));
//    if(_tempPoint.count() == 1)
//    {
//        m_bIsPressed = true;
//        emit sigMousePress(event);
//        event->accept();
//    }


    emit sigMousePress(event);

    QGraphicsView::mousePressEvent(event);

}

/*
 *  鼠标移动事件
 */
void LeeGraphicsView::mouseMoveEvent(QMouseEvent *event)
{
    emit sigMouseMove(event);
    QGraphicsView::mouseMoveEvent(event);
}

/*
 *  鼠标释放事件
 */
void LeeGraphicsView::mouseReleaseEvent(QMouseEvent *event)
{
    emit sigMouseRelease(event);
    QGraphicsView::mouseReleaseEvent(event);
}
