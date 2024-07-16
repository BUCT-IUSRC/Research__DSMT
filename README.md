## DSMT: RGB-D dual-stream multi-task model for multi-granularity semantic understanding
Shilong Wu, Sihan Chen, Yadong Wang, Hao Liu, Zirui Tao, Yaohui Zhu, Tianyu Shen*, Kunfeng Wang*
(*Corresponding authors)
## Framework Overview
We propose a dual-stream multi-task (DSMT) model for realizing three different granularities of semantic understanding simultaneously: object detection, semantic segmentation, and scene classification.We utilize two parallel backbone networks to process both RGB and depth images. Subsequently, the features extracted from these two branches are interacted and calibrated using a Coordinate Attention Fusion Module (CAFM), and the calibrated features are then fed to a decoder for feature stitching. Within the decoder, Dynamic Task Awareness Module (DTAM) is employed to further extract task-specific features. Finally, different task heads generate semantic understanding results based on these features at various levels of granularity.
![模型结构修改3.png](https://github.com/BUCT-IUSRC/Research__DSMT/blob/main/models/framework_DSMT.png)
## Install
### Environment
```bash
$ pip install -r requirements.txt    
```
### Dataset Prepare
Detection, segmentation, and classification datasets are stored under data/det, data/seg, and data/cls, respectively, and RGB images and depth images share the same annotation. If you need to modify the reading method, you can implement it in utils/datasets.py and SegmentationDataset.py.	
## Run
### Evaluation
```bash
$ python detect.py --weights ./weight.pt --source data/images --conf 0.25 --img-size 640  
```
Resulting image in runs/detect.
### Train
```bash
$ python train.py --data data.yaml --cfg cfg.yaml --batch-size 18 --epochs 300 --weights ./weight.pt --workers 8 --label-smoothing 0.1 --img-size 640 --noautoanchor
```
## Contact Us
If you have any problem about this work, please feel free to reach us out at 2022200763@buct.edu.cn.

