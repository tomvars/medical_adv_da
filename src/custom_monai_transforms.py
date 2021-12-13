import numpy as np
from typing import Callable, Optional, Union, Sequence, Tuple, List, Mapping, Hashable, Dict

import torch
from monai.transforms import MapTransform, Transform
from monai.transforms.utils import generate_spatial_bounding_box
from monai.transforms.utils import is_positive, optional_import, min_version
from monai.config import KeysCollection, IndexSelection
from scipy.ndimage.measurements import label as lb
import cc3d

## This would live in monai/transforms/utils.py

measure, _ = optional_import("skimage.measure", "0.14.2", min_version)

def convert_bbox_coordinate_format(original_bbox):
    """
    Simple function which expects bounding box coordinate data in the form:
    [[spatial dim 1 start, spatial dim 2 start], [spatial dim 1 end, spatial dim 2 end], ...]
    returns: [[y1, x1, y2, x2, z1, z2]] medicaldetectiontoolkit format
    """
    if original_bbox.shape[-1] == 2:
        new_bbox = original_bbox.reshape(original_bbox.shape[0], 4)[:, [1, 0, 3, 2]]
    elif original_bbox.shape[-1] == 3:
        new_bbox = original_bbox.reshape(original_bbox.shape[0], 6)[:, [1, 0, 4, 3, 2, 5]]
    else:
        return original_bbox
    return new_bbox

def get_cc_mask(img: np.array, connectivity: Optional[int] = None) -> np.array:
    """
    Gets the largest connected component mask of an image.

    Args:
        img: Image to get largest connected component from. Shape is (spatial_dim1 [, spatial_dim2, ...])
        connectivity: Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
            Accepted values are ranging from  1 to input.ndim. If ``None``, a full
            connectivity of ``input.ndim`` is used.
    """
    return measure.label(img, connectivity=connectivity)

def get_cc_mask_3d(img: np.array, connectivity: Optional[int] = None) -> np.array:
    """
    Gets the largest connected component mask of an image.

    Args:
        img: Image to get largest connected component from. Shape is (spatial_dim1 [, spatial_dim2, ...])
        connectivity: Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
            Accepted values are ranging from  1 to input.ndim. If ``None``, a full
            connectivity of ``input.ndim`` is used.
    """
    return np.expand_dims(cc3d.connected_components(img[0, ...].astype(int)), axis=0)

def pad_bboxes(bboxes, img_shape, pad_size=0):
    """
    bboxes is of shape N x 4 (6 for 3D) in medicaldetectiontoolkit format
    [y1, x1, y2, x2, z1, z2]
    img_shape is CxHxWxD
    """
    
    if len(bboxes.shape) == 2:
        if bboxes.shape[1] == 6:
            bboxes[:, 0] = np.maximum(bboxes[:, 0] - pad_size, 0)
            bboxes[:, 1] = np.maximum(bboxes[:, 1] - pad_size, 0)
            bboxes[:, 2] = np.minimum(bboxes[:, 2] + pad_size, img_shape[2])
            bboxes[:, 3] = np.minimum(bboxes[:, 3] + pad_size, img_shape[1])
            bboxes[:, 4] = np.maximum(bboxes[:, 4] - pad_size, 0)
            bboxes[:, 5] = np.minimum(bboxes[:, 5] + pad_size, img_shape[3])
        elif bboxes.shape[1] == 4:
            bboxes[:, 0] = np.maximum(bboxes[:, 0] - pad_size, 0)
            bboxes[:, 1] = np.maximum(bboxes[:, 1] - pad_size, 0)
            bboxes[:, 2] = np.minimum(bboxes[:, 2] + pad_size, img_shape[2])
            bboxes[:, 3] = np.minimum(bboxes[:, 3] + pad_size, img_shape[1])
    return bboxes

def convert_seg_to_bounding_box_coordinates(
    img: np.array,
    foreground_classes: list = None,
    pad_bbox: int = 0
) -> Tuple[List[int], List[int]]:
    """
        Args:
        img: source image to generate bounding box from.
        foreground_classes: a list of foreground class values
        returns:
        List of tuples of lists of coordinates
    """
    connected_component_func = get_cc_mask_3d if len(img.shape) == 4 else get_cc_mask
    if foreground_classes is None:
        cc_mask = connected_component_func(img)
        bboxes = [list(generate_spatial_bounding_box(cc_mask, select_fn=lambda x: x==c)) for c in range(1, cc_mask.max()+1)]
        return pad_bboxes(convert_bbox_coordinate_format(np.array(bboxes)), img.shape, pad_bbox), [1] * len(bboxes)
    else:
        target_output, bboxes = [], []
        for class_idx in foreground_classes:
            cc_mask = connected_component_func((img==class_idx).astype(np.int32))
            class_specific_bboxes = [list(generate_spatial_bounding_box(cc_mask,
                                                                   select_fn=lambda x: x==c))
                                     for c in range(1, cc_mask.max()+1)]
            bboxes += class_specific_bboxes
            target_output += [class_idx] * len(class_specific_bboxes)
        return pad_bboxes(convert_bbox_coordinate_format(np.array(bboxes)), img.shape, pad_bbox), target_output



## This would live in monai/transforms/utility/array.py
class BoundingBoxes(Transform):
    """
    Compute coordinates of axis-aligned bounding rectangles from input image `img`.
    The output format of the coordinates is (shape is [channel, 2 * spatial dims]):

        [[1st_spatial_dim_start, 1st_spatial_dim_end,
         2nd_spatial_dim_start, 2nd_spatial_dim_end,
         ...,
         Nth_spatial_dim_start, Nth_spatial_dim_end],

         ...

         [1st_spatial_dim_start, 1st_spatial_dim_end,
         2nd_spatial_dim_start, 2nd_spatial_dim_end,
         ...,
         Nth_spatial_dim_start, Nth_spatial_dim_end]]

    The bounding boxes edges are aligned with the input image edges.
    This function returns [-1, -1, ...] if there's no positive intensity.

    Args:
        select_fn: function to select expected foreground, default is to select values > 0.
    """
    def __init__(self, foreground_classes, pad_bbox=0):
        super().__init__()
        self.foreground_classes = foreground_classes
        self.pad_bbox = pad_bbox
        
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        See also: :py:class:`monai.transforms.utils.generate_spatial_bounding_box`.
        """
        return convert_seg_to_bounding_box_coordinates(img,
                                                       foreground_classes=self.foreground_classes,
                                                       pad_bbox=self.pad_bbox
                                                      )
    
## This would live in monai/transforms/utility/dictionary.py
class BoundingBoxesd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.BoundingRectangles`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        bbox_key_postfix: the output bounding box coordinates will be
            written to the value of `{key}_{bbox_key_postfix}`.
        classes: expects a tuple of class indices.
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        keys: KeysCollection,
        bbox_key_postfix: str = "bbox",
        class_target_postfix: str = "class_target",
        foreground_classes: list = None,
        allow_missing_keys: bool = False,
        pad_bbox: int = 0
    ):
        super().__init__(keys, allow_missing_keys)
        self.bbox = BoundingBoxes(foreground_classes, pad_bbox)
        self.bbox_key_postfix = bbox_key_postfix
        self.class_target_postfix = class_target_postfix
        self.pad_bbox = pad_bbox

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        """
        See also: :py:class:`monai.transforms.utils.generate_spatial_bounding_box`.
        """
        d = dict(data)
        for key in self.key_iterator(d):
            bbox, class_target = self.bbox(d[key])
            bbox_key_to_add = f"{key}_{self.bbox_key_postfix}"
            class_target_key_to_add = f"{key}_{self.class_target_postfix}"
            if bbox_key_to_add in d:
                raise KeyError(f"Bounding box data with key {bbox_key_to_add} already exists.")
            if class_target_key_to_add in d:
                raise KeyError(f"Class target data with key {class_target_key_to_add} already exists.")
            d[bbox_key_to_add] = bbox
            d[class_target_key_to_add] = class_target
            
        return d