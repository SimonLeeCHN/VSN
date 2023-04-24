#include "BoundingBox.h"
#include "QGraphicsSceneHoverEvent"
#include "QGraphicsScene"
#include "QGraphicsItem"
#include "QDebug"
#include "QCursor"
#include "QBrush"
#include "QPen"
#include "QColor"

#define RECT_LINEWIDTH  1                               //绘制边界笔宽度
#define RECT_BRUSHCOLOR QColor(255,0,0,0)               //测量框内部填充颜色

BoundingBox::BoundingBox()
{
    m_BoundRect.setRect(0,0,100,100);

    m_iHoverSide = BoundingBox::eSideNone;
    m_iSelectSide = BoundingBox::eSideNone;
    m_bIsAdjusting = false;

    this->setAcceptHoverEvents(true);                   //接受悬停事件
    this->setFlag(QGraphicsItem::ItemIsMovable,true);   //整体可移动
}
BoundingBox::~BoundingBox()
{

}

/*
 *  重载实现
 *  返回区域包围框
 *  继承QGraphicsItem后必须实现，以供给其他系统函数使用
 *  包围框比实际大一点，留下余量以防重绘不干净
 */
QRectF BoundingBox::boundingRect() const
{
    QRectF _tempRect = m_BoundRect.adjusted(-RECT_LINEWIDTH*6,-RECT_LINEWIDTH*6,RECT_LINEWIDTH*6,RECT_LINEWIDTH*6);

//    QRect _tempRect = m_BoundRect;
//    _tempRect.setLeft(_tempRect.left() - RECT_LINEWIDTH*2);
//    _tempRect.setTop(_tempRect.top() - RECT_LINEWIDTH*2);
//    _tempRect.setWidth(_tempRect.width() + RECT_LINEWIDTH*4);
//    _tempRect.setHeight(_tempRect.height() + RECT_LINEWIDTH*4);

    return _tempRect;
}

/*
 *  重载实现
 *  绘制你的图形
 *  注意：这里的绘制是按照m_BoundRect左上角开始向右下角绘制，而非从中心点开始绘制
 *  即调用继承来的setPos方法后，是矩形左上角点对齐
 */
void BoundingBox::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    Q_UNUSED(option)
    Q_UNUSED(widget)

    painter->save();

    QPen _pen;
    _pen.setWidth(RECT_LINEWIDTH);

    /*
     * 绘制四边
     */
    if(m_iSelectSide == BoundingBox::eSideTop)
    {
        _pen.setColor(Qt::blue);
        painter->setPen(_pen);
    }
    else
    {
        _pen.setColor(Qt::red);
        painter->setPen(_pen);
    }
    painter->drawLine(m_BoundRect.topLeft(),QPointF(m_BoundRect.left() + m_BoundRect.width(),m_BoundRect.top()));    //top

    if(m_iSelectSide == BoundingBox::eSideBottom)
    {
        _pen.setColor(Qt::blue);
        painter->setPen(_pen);
    }
    else
    {
        _pen.setColor(Qt::red);
        painter->setPen(_pen);
    }
    painter->drawLine(QPointF(m_BoundRect.left(),m_BoundRect.top()+m_BoundRect.height())
                      ,QPointF(m_BoundRect.left()+m_BoundRect.width(),m_BoundRect.top()+m_BoundRect.height()));      //bottom

    if(m_iSelectSide == BoundingBox::eSideLeft)
    {
        _pen.setColor(Qt::blue);
        painter->setPen(_pen);
    }
    else
    {
        _pen.setColor(Qt::red);
        painter->setPen(_pen);
    }
    painter->drawLine(m_BoundRect.topLeft(),QPointF(m_BoundRect.left(),m_BoundRect.top()+m_BoundRect.height()));     //left

    if(m_iSelectSide == BoundingBox::eSideRight)
    {
        _pen.setColor(Qt::blue);
        painter->setPen(_pen);
    }
    else
    {
        _pen.setColor(Qt::red);
        painter->setPen(_pen);
    }
    painter->drawLine(QPointF(m_BoundRect.left() + m_BoundRect.width(),m_BoundRect.top())
                      ,QPointF(m_BoundRect.left()+m_BoundRect.width(),m_BoundRect.top()+m_BoundRect.height()));      //right

    /*
     *  绘制内部填充色
     */
    QBrush _brush(RECT_BRUSHCOLOR);
    painter->setBrush(_brush);
    painter->setPen(Qt::NoPen);

    painter->drawRect(this->m_BoundRect);

    painter->restore();
}

/*
 *  重载实现
 */
QPainterPath BoundingBox::shape() const
{
    QPainterPath _path;
    _path.addRect(m_BoundRect);
    return _path;
}

/******FUNCS******/

/*
 *  获取实际矩形框尺寸
 *  GraphicsItem默认的boundingRect是在 QRectF BoundingBox::boundingRect() const实现的，
 *  由于绘制时需考虑重绘问题故大了一圈
 */
QRectF BoundingBox::GetRealBoundRect()
{
    return m_BoundRect;
}

/*
 *  设置新矩形框尺寸
 *  这里只使用新矩形框的长度和宽度，内部维护的矩形永远左上角在(0,0)
 *  切记框尺寸可能发生改变的地方都要调用prepareGeometryChange
 */
void BoundingBox::SetBoundRectSize(QRectF newRect)
{
    prepareGeometryChange();

    m_BoundRect.setWidth(newRect.width());
    m_BoundRect.setHeight(newRect.height());

    //发送变换信号，使得可以更新D1D2
    emit sigGeometryChange();
}

/*
 *  获取当前鼠标悬浮在哪个边
 */
int BoundingBox::GetHoverSide()
{
    return m_iHoverSide;
}

/*
 *  清除选择边
 */
void BoundingBox::ClearSelectSide()
{
    m_iSelectSide = BoundingBox::eSideNone;
}

/*
 *  传入step，移动m_iSelectSide所选的边
 *  step为符号数，例如对于底边，正数向下移动step长度，负数向上移动step长度
 */
void BoundingBox::MoveSelectSide(qreal step)
{
    //面积发生变换，切记调用prepareGeometryChange()
    prepareGeometryChange();

    if(m_iSelectSide != BoundingBox::eSideNone)
    {
        qreal _temp;
        QRectF _tempRect = m_BoundRect;

        switch (m_iSelectSide)
        {
            case BoundingBox::eSideTop:
            {
                _temp = _tempRect.top() + step;
                if(_temp < ((_tempRect.top() + _tempRect.height()) - RECT_LINEWIDTH*2))
                    _tempRect.setTop(_temp);

                break;
            }
            case BoundingBox::eSideBottom:
            {
                _temp = _tempRect.bottom() + step;
                if(_temp > (_tempRect.top() + RECT_LINEWIDTH*2))
                    _tempRect.setBottom(_temp);

                break;
            }
            case BoundingBox::eSideLeft:
            {
                _temp = _tempRect.left() + step;
                if(_temp < ((_tempRect.left() + _tempRect.width()) - RECT_LINEWIDTH*2))
                    _tempRect.setLeft(_temp);

                break;
            }
            case BoundingBox::eSideRight:
            {
                _temp = _tempRect.right() + step;
                if(_temp > (_tempRect.left() + RECT_LINEWIDTH*2))
                    _tempRect.setRight(_temp);

                break;
            }
            default:
            {
                break;
            }

        }

        /*
         *  将_tempRect的左上角作为item的坐标移动
         *  修改m_BoundRect的长宽
         *  确保内部矩形m_BoundRect的左上角在(0,0)
         */
        QPointF _pos = this->pos();
        _pos.setX(_pos.x() + _tempRect.x());
        _pos.setY(_pos.y() + _tempRect.y());
        this->setPos(_pos);
        m_BoundRect.setWidth(_tempRect.width());
        m_BoundRect.setHeight(_tempRect.height());

        //发送变换信号，使得可以更新D1D2
        emit sigGeometryChange();
    }
}

/******EVENTS******/

/*
 *  鼠标在boundingRect内悬浮移动事件
 */
void BoundingBox::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{
    if(abs(event->pos().y() - m_BoundRect.top()) <= RECT_LINEWIDTH * 15)
    {   //上边沿
        m_iHoverSide = BoundingBox::eSideTop;
        setCursor(Qt::SizeVerCursor);
    }
    else if(abs(event->pos().y() - (m_BoundRect.top() + m_BoundRect.height())) <= RECT_LINEWIDTH * 15)
    {   //下边沿
        m_iHoverSide = BoundingBox::eSideBottom;
        setCursor(Qt::SizeVerCursor);
    }
    else if(abs(event->pos().x() - m_BoundRect.left()) <= RECT_LINEWIDTH * 15)
    {   //左边沿
        m_iHoverSide = BoundingBox::eSideLeft;
        setCursor(Qt::SizeHorCursor);
    }
    else if(abs(event->pos().x() - m_BoundRect.right()) <= RECT_LINEWIDTH * 15)
    {   //右边沿
        m_iHoverSide = BoundingBox::eSideRight;
        setCursor(Qt::SizeHorCursor);
    }
    else
    {   //没有在边缘
        m_iHoverSide = BoundingBox::eSideNone;
        setCursor(Qt::SizeAllCursor);
    }
}

/*
 *  鼠标在boundingRect内悬浮离开事件
 */
void BoundingBox::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    Q_UNUSED(event)

    m_iHoverSide = BoundingBox::eSideNone;
    setCursor(Qt::ArrowCursor);
}

/*
 *  鼠标在boundingRect内按压事件
 *  每次按压更新m_iSelectSide
 */
void BoundingBox::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    m_bIsAdjusting = true;
    m_iSelectSide = m_iHoverSide;

    //发送选择边变化信号，供界面可控按钮状态切换
    emit sigSelectSide(m_iSelectSide);

    this->scene()->update();                //刷新以绘制被选中的边界颜色

    QGraphicsItem::mousePressEvent(event);
}

/*
 *  鼠标在boundingRect内移动事件
 *  该事件必须在鼠标被按压的状态下才会触发
 */
void BoundingBox::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    //面积发生变换，切记调用prepareGeometryChange()
    prepareGeometryChange();

    if(m_bIsAdjusting)
    {
        QRectF _tempRect = m_BoundRect;

        switch (m_iSelectSide)
        {
            case BoundingBox::eSideTop:
            {
                if(event->pos().y() < ((_tempRect.top() + _tempRect.height()) - RECT_LINEWIDTH*2))
                {
                    _tempRect.setTop(event->pos().y());
                }

                break;
            }
            case BoundingBox::eSideBottom:
            {
                if(event->pos().y() > (_tempRect.top() + RECT_LINEWIDTH*2))
                {
                    _tempRect.setBottom(event->pos().y());
                }

                break;
            }
            case BoundingBox::eSideLeft:
            {
                if(event->pos().x() < ((_tempRect.left() + _tempRect.width()) - RECT_LINEWIDTH*2))              //移动左边沿时不超过右边沿
                {
                    _tempRect.setLeft(event->pos().x());
                }

                break;
            }
            case BoundingBox::eSideRight:
            {
                if(event->pos().x() > (_tempRect.left() + RECT_LINEWIDTH*2))
                {
                    _tempRect.setRight(event->pos().x());
                }

                break;
            }
            case BoundingBox::eSideNone:
            {
                //没有选择到边缘，传递给原始事件，实现移动
                QGraphicsItem::mouseMoveEvent(event);
                break;
            }
            default:
            {
                break;
            }
        }

        /*
         *  将_tempRect的左上角作为item的坐标移动
         *  修改m_BoundRect的长宽
         *  确保内部矩形m_BoundRect的左上角在(0,0)
         */
        QPointF _pos = this->pos();
        _pos.setX(_pos.x() + _tempRect.x());
        _pos.setY(_pos.y() + _tempRect.y());
        this->setPos(_pos);
        m_BoundRect.setWidth(_tempRect.width());
        m_BoundRect.setHeight(_tempRect.height());

        //发送变换信号，使得可以更新D1D2
        emit sigGeometryChange();

    }

}

/*
 *  鼠标在boundingRect内释放事件
 */
void BoundingBox::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    m_bIsAdjusting = false;

    QGraphicsItem::mouseReleaseEvent(event);
}
