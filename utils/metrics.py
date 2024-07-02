# Model validation metrics

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import general


def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)

def fitness2(x, mIoU):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.2, 0.7]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95, mIoU]
    # print(x)
    x_m = np.expand_dims(np.append(x[:, :4], mIoU), 0)  # ｘ　在train.py有reshape，确定只有１行
    return (x_m * w).sum(1)

# 原始best.pt选择函数
def fitness3(x, mIoU, accuracy):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.2, 0.6, 0.1]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95, mIoU, acc]
    # print(x)
    x_m = np.expand_dims(np.append(x[:, :4], [mIoU, accuracy]), 0)  # ｘ　在train.py有reshape，确定只有１行
    return (x_m * w).sum(1)

# def fitness3(x, mIoU, accuracy):
#     # Model fitness as a weighted combination of metrics
#     w = [0.25, 0.25, 0.0, 0.0, 0.4, 0.1]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95, mIoU, acc]
#     # print(x)
#     x_m = np.expand_dims(np.append(x[:, :4], [mIoU, accuracy]), 0)  # ｘ　在train.py有reshape，确定只有１行
#     return (x_m * w).sum(1)

# def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
#     """ Compute the average precision, given the recall and precision curves.
#     Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
#     # Arguments
#         tp:  True positives (nparray, nx1 or nx10).
#         conf:  Objectness value from 0-1 (nparray).
#         pred_cls:  Predicted object classes (nparray).
#         target_cls:  True object classes (nparray).
#         plot:  Plot precision-recall curve at mAP@0.5
#         save_dir:  Plot save directory
#     # Returns
#         The average precision as computed in py-faster-rcnn.
#     """
#
#     # Sort by objectness
#     i = np.argsort(-conf)
#     tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
#
#     # Find unique classes
#     unique_classes = np.unique(target_cls)
#     nc = unique_classes.shape[0]  # number of classes, number of detections
#
#     # Create Precision-Recall curve and compute AP for each class
#     px, py = np.linspace(0, 1, 1000), []  # for plotting
#     ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
#     for ci, c in enumerate(unique_classes):
#         i = pred_cls == c
#         n_l = (target_cls == c).sum()  # number of labels
#         n_p = i.sum()  # number of predictions
#
#         if n_p == 0 or n_l == 0:
#             continue
#         else:
#             # Accumulate FPs and TPs
#             fpc = (1 - tp[i]).cumsum(0)
#             tpc = tp[i].cumsum(0)
#
#             # Recall
#             recall = tpc / (n_l + 1e-16)  # recall curve
#             r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases
#
#             # Precision
#             precision = tpc / (tpc + fpc)  # precision curve
#             p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score
#
#             # AP from recall-precision curve
#             for j in range(tp.shape[1]):
#                 ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
#                 if plot and j == 0:
#                     py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5
#
#     # Compute F1 (harmonic mean of precision and recall)
#     f1 = 2 * p * r / (p + r + 1e-16)
#     if plot:
#         plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
#         plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
#         plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
#         plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')
#
#     i = f1.mean(0).argmax()  # max F1 index
#     return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')

#解耦头精度计算
def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed

def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16, prefix=""):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / f'{prefix}PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / f'{prefix}F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / f'{prefix}P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / f'{prefix}R_curve.png', names, ylabel='Recall')

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


# def compute_ap(recall, precision):
#     """ Compute the average precision, given the recall and precision curves
#     # Arguments
#         recall:    The recall curve (list)
#         precision: The precision curve (list)
#     # Returns
#         Average precision, precision curve, recall curve
#     """
#
#     # Append sentinel values to beginning and end
#     mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
#     mpre = np.concatenate(([1.], precision, [0.]))
#
#     # Compute the precision envelope
#     mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
#
#     # Integrate area under curve
#     method = 'interp'  # methods: 'continuous', 'interp'
#     if method == 'interp':
#         x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
#         ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
#     else:  # 'continuous'
#         i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
#         ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve
#
#     return ap, mpre, mrec

#解耦头计算ap
def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

# class ConfusionMatrix:
#     # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
#     def __init__(self, nc, conf=0.25, iou_thres=0.45):
#         self.matrix = np.zeros((nc + 1, nc + 1))
#         self.nc = nc  # number of classes
#         self.conf = conf
#         self.iou_thres = iou_thres
#
#     def process_batch(self, detections, labels):
#         """
#         Return intersection-over-union (Jaccard index) of boxes.
#         Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
#         Arguments:
#             detections (Array[N, 6]), x1, y1, x2, y2, conf, class
#             labels (Array[M, 5]), class, x1, y1, x2, y2
#         Returns:
#             None, updates confusion matrix accordingly
#         """
#         detections = detections[detections[:, 4] > self.conf]
#         gt_classes = labels[:, 0].int()
#         detection_classes = detections[:, 5].int()
#         iou = general.box_iou(labels[:, 1:], detections[:, :4])
#
#         x = torch.where(iou > self.iou_thres)
#         if x[0].shape[0]:
#             matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
#             if x[0].shape[0] > 1:
#                 matches = matches[matches[:, 2].argsort()[::-1]]
#                 matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
#                 matches = matches[matches[:, 2].argsort()[::-1]]
#                 matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
#         else:
#             matches = np.zeros((0, 3))
#
#         n = matches.shape[0] > 0
#         m0, m1, _ = matches.transpose().astype(np.int16)
#         for i, gc in enumerate(gt_classes):
#             j = m0 == i
#             if n and sum(j) == 1:
#                 self.matrix[gc, detection_classes[m1[j]]] += 1  # correct
#             else:
#                 self.matrix[self.nc, gc] += 1  # background FP
#
#         if n:
#             for i, dc in enumerate(detection_classes):
#                 if not any(m1 == i):
#                     self.matrix[dc, self.nc] += 1  # background FN
#
#     def matrix(self):
#         return self.matrix
#
#     def plot(self, save_dir='', names=()):
#         try:
#             import seaborn as sn
#
#             array = self.matrix / (self.matrix.sum(0).reshape(1, self.nc + 1) + 1E-6)  # normalize
#             array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)
#
#             fig = plt.figure(figsize=(12, 9), tight_layout=True)
#             sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
#             labels = (0 < len(names) < 99) and len(names) == self.nc  # apply names to ticklabels
#             sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
#                        xticklabels=names + ['background FP'] if labels else "auto",
#                        yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
#             fig.axes[0].set_xlabel('True')
#             fig.axes[0].set_ylabel('Predicted')
#             fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
#         except Exception as e:
#             pass
#
#     def print(self):
#         for i in range(self.nc + 1):
#             print(' '.join(map(str, self.matrix[i])))
#

#解耦头新加混淆矩阵

import contextlib
import platform
import warnings

def emojis(str=''):
    # Return platform-dependent emoji-safe version of string
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str

class TryExcept(contextlib.ContextDecorator):
    # YOLOv5 TryExcept class. Usage: @TryExcept() decorator or 'with TryExcept():' context manager
    def __init__(self, msg=''):
        self.msg = msg

    def __enter__(self):
        pass

    def __exit__(self, exc_type, value, traceback):
        if value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True

def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        if detections is None:
            gt_classes = labels.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # background FN
            return

        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # true background

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # predicted background

    def matrix(self):
        return self.matrix

    def tp_fp(self):
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return tp[:-1], fp[:-1]  # remove background class

    @TryExcept('WARNING ⚠️ ConfusionMatrix plot failure')
    def plot(self, normalize=True, save_dir='', names=()):
        import seaborn as sn

        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)  # number of classes, names
        sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (names + ['background']) if labels else "auto"
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(array,
                       ax=ax,
                       annot=nc < 30,
                       annot_kws={
                           "size": 8},
                       cmap='Blues',
                       fmt='.2f',
                       square=True,
                       vmin=0.0,
                       xticklabels=ticklabels,
                       yticklabels=ticklabels).set_facecolor((1, 1, 1))
        ax.set_ylabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title('Confusion Matrix')
        fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        plt.close(fig)

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))




# Plots ----------------------------------------------------------------------------------------------------------------

def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)


# semantic segmentation ------------------------------------------------------------------------------------------------

def batch_pix_accuracy(output, target):
    """Batch Pixel Accuracy
    Args:
        predict: input 4D tensor
        target: label 3D tensor
    """
    _, predict = torch.max(output, 1)

    predict = predict.cpu().numpy().astype('int64') + 1
    target = target.cpu().numpy().astype('int64') + 1

    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target)*(target > 0))
    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """
    _, predict = torch.max(output, 1)
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = predict.cpu().numpy().astype('int64') + 1
    target = target.cpu().numpy().astype('int64') + 1

    predict = predict * (target > 0).astype(predict.dtype)
    intersection = predict * (predict == target)
    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union

#------------------------------检测双模态-----------------------------------
def bbox_ioa(box1, box2, eps=1E-7):
    """ Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """

    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area


