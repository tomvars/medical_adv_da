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

def convert_seg_to_bounding_box_coordinates(
    img: np.array,
    foreground_classes: list = None
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
        return convert_bbox_coordinate_format(np.array(bboxes)), [1] * len(bboxes)
    else:
        target_output, bboxes = [], []
        for class_idx in foreground_classes:
            cc_mask = connected_component_func((img==class_idx).astype(np.int32))
            class_specific_bboxes = [list(generate_spatial_bounding_box(cc_mask,
                                                                   select_fn=lambda x: x==c))
                                     for c in range(1, cc_mask.max()+1)]
            bboxes += class_specific_bboxes
            target_output += [class_idx] * len(class_specific_bboxes)
        return convert_bbox_coordinate_format(np.array(bboxes)), target_output

# def convert_seg_to_bounding_box_coordinates(img: np.ndarray,
#     select_fn: Callable = is_positive,
#     channel_indices: Optional[IndexSelection] = None,
#     get_rois_from_seg_flag: bool = False,
#     margin: Union[Sequence[int], int] = 0,
# ) -> Tuple[List[int], List[int]]:
#     """
#         Args:
#         img: source image to generate bounding box from.
#         select_fn: function to select expected foreground, default is to select values > 0.
#         channel_indices: if defined, select foreground only on the specified channels
#             of image. if None, select foreground on the whole image.
#         margin: add margin value to spatial dims of the bounding box, if only 1 value provided, use it for all dims.
#     """

#     bb_target = []
#     roi_masks = []
#     roi_labels = []
#     out_seg = np.copy(img)
#     for b in range(img.shape[0]):

#         p_coords_list = []
#         p_roi_masks_list = []
#         p_roi_labels_list = []

#         if np.sum(img[b]!=0) > 0:
# #             if get_rois_from_seg_flag:
# #                 clusters, n_cands = lb(img[b])
# #                 data_dict['class_target'][b] = [data_dict['class_target'][b]] * n_cands
# #             else:
#             n_cands = int(np.max(img[b]))
#             clusters = img[b]

#             rois = np.array([(clusters == ii) * 1 for ii in range(1, n_cands + 1)])  # separate clusters and concat
#             for rix, r in enumerate(rois):
#                 if np.sum(r !=0) > 0: #check if the lesion survived data augmentation
#                     seg_ixs = np.argwhere(r != 0)
#                     coord_list = [np.min(seg_ixs[:, 1])-1, np.min(seg_ixs[:, 2])-1, np.max(seg_ixs[:, 1])+1,
#                                      np.max(seg_ixs[:, 2])+1]
#                     if dim == 3:

#                         coord_list.extend([np.min(seg_ixs[:, 3])-1, np.max(seg_ixs[:, 3])+1])

#                     p_coords_list.append(coord_list)
#                     p_roi_masks_list.append(r)
#                     # add background class = 0. rix is a patient wide index of lesions. since 'class_target' is
#                     # also patient wide, this assignment is not dependent on patch occurrances.
#                     p_roi_labels_list.append(data_dict['class_target'][b][rix] + 1)

#                 if class_specific_seg_flag:
#                     out_seg[b][img[b] == rix + 1] = data_dict['class_target'][b][rix] + 1

#             if not class_specific_seg_flag:
#                 out_seg[b][img[b] > 0] = 1

#             bb_target.append(np.array(p_coords_list))
#             roi_masks.append(np.array(p_roi_masks_list).astype('uint8'))
#             roi_labels.append(np.array(p_roi_labels_list))


#         else:
#             bb_target.append([])
#             roi_masks.append(np.zeros_like(img[b])[None])
#             roi_labels.append(np.array([-1]))

#     if get_rois_from_seg_flag:
#         data_dict.pop('class_target', None)

#     bb_target = np.array(bb_target)
#     roi_masks = np.array(roi_masks)
#     class_target = np.array(roi_labels)
#     seg = out_seg

#     return bb_target, roi_masks, class_target, seg

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
    def __init__(self, foreground_classes):
        super().__init__()
        self.foreground_classes = foreground_classes
        
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        See also: :py:class:`monai.transforms.utils.generate_spatial_bounding_box`.
        """
        return convert_seg_to_bounding_box_coordinates(img,
                                                       foreground_classes=self.foreground_classes)
    
    
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
        allow_missing_keys: bool = False
    ):
        super().__init__(keys, allow_missing_keys)
        self.bbox = BoundingBoxes(foreground_classes)
        self.bbox_key_postfix = bbox_key_postfix
        self.class_target_postfix = class_target_postfix

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