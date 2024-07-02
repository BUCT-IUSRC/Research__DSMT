# DSMT: RGB-D dual-stream multi-task model for multi-granularity semantic understanding
## Overeview
基于RGB-D图像构建的检测、分割、分类多任务模型。

## Doc
### 0. Before Start 环境配置和数据集准备
#### (a) Environment
```bash
$ python -m pip install -r requirements.txt    
```
#### (b) Dataset Prepare 数据集准备
检测、分割、分类数据集分别存放在data/det、 data/seg 和 data/cls 下，RGB图像和深度图像共用标注。如需修改读取方式可在 utils/datasets.py和SegmentationDataset.py中实现。
### 1. Inference
#### (a) 普通图片推理
```bash
$ python detect.py --weights ./mtm.pt或其他模型 --source data/images --conf 0.25 --img-size 1024  
```
结果图片在runs/detect文件夹中，也可以推理同时显示。  
```bash
$ python detect.py --weights ./mtm.pt或其他模型 --source data/images --conf 0.25 --img-size 1024 --view-img  
```
同原版YOLOV5，--weights写你的pt文件，--source写图片文件夹或者视频文件的路径，--conf检测阈值，--img-size为resize到模型的目标长边尺寸  
### 2. Test 训练后测试模型
```bash
$ python test-decoupled.py --data cityscapes_det.yaml --segdata ./data/citys --weights ./pspv5s.pt --img-size 1024 --base-size 1024
```
对比原版多两个参数: --segdata后写Cityscapes数据集的文件夹地址(现在只支持这个，可以参考SegmentationDataset.py自行扩展)  
检测长边和分割长边参数分离，--img-size是检测长边 --base-size是分割长边,我的配置是把Cityscapes放在1024*512尺寸下推理，比较能兼顾速度精度，训练也是以此为目的调参的.  
如果训练后测试你自己的数据集，用test_custom.py（训练中train_custom.py会测）  
```bash
$ python test_custom.py --data 你的.yaml --segdata 你的分割数据路径 --weights ./pspv5s.pt --img-size 1024 --base-size 1024
```
### 3. Train 
```bash
$ python train.py --data cityscapes_det.yaml --cfg mtm_pdlk.yaml --batch-size 18 --epochs 300 --weights ./mtm.pt --workers 8 --label-smoothing 0.1 --img-size 832 --noautoanchor
```
### 4. Code Guide 我修改了什么
1. common.py  
加入了一些新的模块，加入分类头。
2. yolo.py  
yolov5的模型主架构代码，包括Model类和检测要用的Detect类,以下部分请重点关注：    
   (1) Model的初始化函数中，我在save中**手动添加了37层**(分割层号，检测是38)。原代码forward_onece采用了for循环前向推理，将后续会用到的层结果保存在列表中(会用到哪些层由parse函数对yaml配置文件解析得到，在初始化函数中调用了parse，需要保存的中间层号在save列表，forward时候按照save序号将对应层中间结果存入y列表)，目前的方法中由于我手动加入37层，检测层运行结束后，会返回x(检测结果)、y [-2] (分割结果)和y [22] (分类结果)。因此若修改了配置文件增加了新的层（例如给最新的P6模型增加分割层），务必修改Model的初始化函数把37换成新的分割层号（这确实不是个好接口，赶时间，另外别把37改成-2，看yolo原版代码就知道这么改不管用）。另外yolov5原作者在很多代码中默认了检测层是最后一层，务必在配置中把检测层放在最后一层。（3）加入了深度图像读取，在forward_onece里实验。  
   (2) Model的解析函数parse_model从yaml文件解析配置，如果想增加新的模块首先在common.py或yolo.py中实现该类，在parse_model中仿照写出该类的解析方法，再在配置文件中写入配置。如果仿照我的分割头类接口设计新增分割头，仅需实现类，在parse_model的解析分割头的支持列表中加入该类名即可。   
3. train_decoupled.py   
   加入了解耦头，检测精度有上升但另外两项任务精度轻微下降。	
4. models/mtm_pdlk.yaml  
   模型配置文件，包括了检测、分割、分类的类别数、深度骨干的层数和模型大小控制参数。    
   
5. data/cityscapes_det.yaml  
检测数据集配置，同原版，新增了分割、分类数据集地址，train.py读分割、分类数据地址是按这里配置的  
   
6. test.py  
   新增了分类测试函数  
   
7. utils/loss.py  
   新增解耦损失函数。   
   
8. utils/metrics.py  
   新增了fitness3函数用于train时候选模型，包括P，R，AP@.5，AP@.5:.95、mIoU和acc的比例。新增了计算mIoU函数。  
   
9. detect.py  
   新增了画分割和叠加图、同尺寸图保存视频以及用于提交的trainid转id功能（见上面推理部分），修改了开cudnn.benchmark的情况    
     
10. test_decoupled.py  
   加入解耦头之后的测试文件。
  
