from torchvision.ops import nms

def nms_2D(boxes, scores, iou_threshold):
    return nms(boxes, scores, iou_threshold)

def nms_3D(boxes, scores, iou_threshold):
    raise NotImplementedError