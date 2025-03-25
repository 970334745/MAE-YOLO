import warnings, os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    # 代表用cpu训练 不推荐！没意义！ 而且有些模块不能在cpu上跑
# os.environ["CUDA_VISIBLE_DEVICES"]="0"     # 代表用第一张卡进行训练  0：第一张卡 1：第二张卡
# 多卡训练参考<YOLOV8V10配置文件.md>下方常见错误和解决方案
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# BILIBILI UP 魔傀面具
# 训练参数官方详解链接：https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings:~:text=a%20training%20run.-,Train%20Settings,-The%20training%20settings

# 指定显卡和多卡训练问题 统一都在<YOLOV8V10配置文件.md>下方常见错误和解决方案。
# 训练过程中loss出现nan，可以尝试关闭AMP，就是把下方amp=False的注释去掉。
# 训练时候输出的AMP Check使用的YOLOv8n的权重不是代表载入了预训练权重的意思，只是用于测试AMP，正常的不需要理会。
# 整合多个创新点的B站视频链接:https://www.bilibili.com/video/BV15H4y1Y7a2/
# 更多问题解答请看使用说明.md下方<常见疑问>

# YOLOV8源码常见疑问解答小课堂
# 1. [关于配置文件中Optimizer参数为auto的时候，究竟Optimizer会怎么选用呢？](https://www.bilibili.com/video/BV1K34y1w7cZ/)
# 2. [best.pt究竟是根据什么指标来保存的?](https://www.bilibili.com/video/BV1jN411M7MA/)
# 3. [数据增强在yolov8中的应用](https://www.bilibili.com/video/BV1aQ4y1g7ah/)
# 4. [如何添加FPS计算代码和FPS的相关的一些疑问](https://www.bilibili.com/video/BV1Sw411g7DD/)
# 5. [预测框粗细颜色修改与精度小数位修改](https://www.bilibili.com/video/BV12K421a7rH/)
# 6. [导出改进/剪枝的onnx模型和讲解onnx-opset和onnxsim的作用](https://www.bilibili.com/video/BV1CK421e7Y3/)
# 7. [YOLOV8模型详细讲解(包含该如何改进YOLOV8)(刚入门小白，需要改进YOLOV8的同学必看！)](https://www.bilibili.com/video/BV1Ms421u7VH/)
# 8. [学习率变化问题](https://www.bilibili.com/video/BV1frnferEL1/)

# 一些非常推荐小白看的视频链接
# 1. [YOLOV8模型详细讲解(包含该如何改进YOLOV8)(刚入门小白，需要改进YOLOV8的同学必看！)](https://www.bilibili.com/video/BV1Ms421u7VH/)
# 2. [提升多少才能发paper？轻量化需要看什么指标？需要轻量化到什么程度才能发paper？这期给大家一一解答！](https://www.bilibili.com/video/BV1QZ421M7gu/)
# 3. [深度学习实验部分常见疑问解答！(小白刚入门必看！少走弯路！少自我内耗！)](https://www.bilibili.com/video/BV1Bz421B7pC/)
#     ```
#     1. 如何衡量自己的所做的工作量够不够？
#     2. 为什么别人的论文说这个模块对xxx有作用，但是我自己用的时候还掉点了？
#     3. 提升是和什么模型相比呢 比如和yolov8这种基础模型比还是和别人提出的目前最好的模型比
#     4. 对比不同的模型的时候，输入尺寸，学习率，学习次数这些是否需要一致？
#     ```
# 4. [深度学习实验部分常见疑问解答二！(小白刚入门必看！少走弯路！少自我内耗！)](https://www.bilibili.com/video/BV1ZM4m1m785/)
#     ```
#     1. 为什么我用yolov8自带的coco8、coco128训练出来的效果很差？
#     2. 我的数据集很大，机器跑得慢，我是否可以用数据集的百分之10的数据去测试这个改进点是否有效？有效再跑整个数据集？
#     ```
# 5. [深度学习实验部分常见疑问解答三！(怎么判断模型是否收敛？模型过拟合怎么办？)](https://www.bilibili.com/video/BV11S421d76P/)
# 6. [YOLO系列模型训练结果详细解答！(训练过程的一些疑问，该放哪个文件运行出来的结果、参数量计算量在哪里看..等等问题)](https://www.bilibili.com/video/BV11b421J7Vx/)
# 7. [深度学习论文实验中新手非常容易陷入的一个误区：抱着解决xxx问题的心态去做实验](https://www.bilibili.com/video/BV1kkkvYJEHG/)
# 8. [深度学习实验准备-数据集怎么选？有哪些需要注意的点？](https://www.bilibili.com/video/BV11zySYvEhs/)
# 9. [深度学习炼丹必备必看必须知道的小技巧！](https://www.bilibili.com/video/BV1q3SZYsExc/)

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8n.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='/root/code/dataset/dataset_visdrone/data.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=32,
                close_mosaic=0,
                workers=8, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                optimizer='SGD', # using SGD
                # device='0,1', # 指定显卡和多卡训练参考<YOLOV8V10配置文件.md>下方常见错误和解决方案
                # patience=0, # set 0 to close earlystop.
                # resume=True, # 断点续训,YOLO初始化时选择last.pt,例如YOLO('last.pt')
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )