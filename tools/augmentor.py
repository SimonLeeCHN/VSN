# 导入数据增强工具
import Augmentor

# 确定原始图像存储路径以及掩码文件存储路径
p = Augmentor.Pipeline("manual_labeled/imgs/")
p.ground_truth("manual_labeled/masks/")

# # 图像旋转： 按照概率0.8执行，最大左旋角度10，最大右旋角度10
# p.rotate(probability=0.8, max_left_rotation=25, max_right_rotation=25)
#
# # 图像左右互换： 按照概率0.5执行
# p.flip_left_right(probability=0.5)
#
# # 图像上下翻转: 概率0.5
# p.flip_top_bottom(probability=0.5)
#
# # 图像放大缩小：
# p.zoom_random(probability=0.3, percentage_area=0.95)

p.random_distortion(probability=0.2, grid_width=4, grid_height=4, magnitude=4)
p.rotate_random_90(probability=0.8)
p.flip_random(probability=0.8)

# 最终扩充的数据样本数
p.sample(200)
