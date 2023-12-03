import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


def conv_img2cvmat(inimg):
    convmat = np.asarray(inimg)
    return convmat


def conv_cvmat2img(inmat):
    convimg = Image.fromarray(inmat)
    return convimg


def conv_tensor2img(t):
    unloader = transforms.ToPILImage()
    image = t.cpu().clone()  # clone the tensor
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    return image
    # plt.imshow(image)
    # plt.show()

def get_largest_area(inimg: np.ndarray):
    """
    搜索输入的opencv图像中的最大轮廓，并返回仅绘制有该最大轮廓的图像
    :param inimg: ndarray类型，opencv图像
    :return: 绘制有最大轮廓的outimg；在inimg上提取出的轮廓
    """
    outimg = np.zeros(inimg.shape[0:2], np.uint8)

    # 搜索图像内区域
    _contours, _ = cv2.findContours(inimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(_contours) == 0:
        return outimg, _contours

    # 寻找最大的面积
    _maxAreaIndex = 0
    _maxArea = 0
    for i in range(len(_contours)):
        _indexArea = cv2.contourArea(_contours[i])
        if _maxArea < _indexArea:
            _maxArea = _indexArea
            _maxAreaIndex = i

    # 在输出图上绘制最大轮廓区域
    cv2.drawContours(outimg, _contours, _maxAreaIndex, 255, thickness=-1)
    # cv2.imshow("outimg", outimg)
    # cv2.imshow("inimg", inimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return outimg, _contours


if __name__ == '__main__':
    img = Image.open('../../Dataset/testpics/out/ct1_HV0.1_1_VSN.png')

    convmat = conv_img2cvmat(img)
    cv2.imshow("convmat", convmat)
    get_largest_area(convmat)

    convimg = conv_cvmat2img(convmat)
    convimg.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # get_largest_area(cv2.imread('../Dataset/testpics/out/ct1_HV0.1_1_VSN.png', cv2.CV_8UC1))
