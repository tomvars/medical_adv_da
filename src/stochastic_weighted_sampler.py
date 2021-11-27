import contextlib
from copy import deepcopy
from enum import Enum
from itertools import chain
from math import ceil, floor
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
from monai.config.type_definitions import NdarrayOrTensor
from monai.config import IndexSelection, KeysCollection
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import MapTransform, Randomizable
from monai.transforms.utils import (
    allow_missing_keys_mode,
    generate_label_classes_crop_centers,
    generate_pos_neg_label_crop_centers,
    is_positive,
    map_binary_to_indices,
    map_classes_to_indices,
    weighted_patch_samples,
)
from monai.utils import ImageMetaKey as Key
from monai.utils import Method, NumpyPadMode, PytorchPadMode, ensure_tuple, ensure_tuple_rep, fall_back_tuple, first
from monai.transforms.croppad.array import (
    BorderPad,
    SpatialCrop,
)
from monai.utils.enums import TraceKeys
import random

class RandStochaticWeightedCropd(Randomizable, MapTransform, InvertibleTransform):
    """
    Samples a list of `num_samples` image patches according to the provided `weight_map`.
    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        w_key: key for the weight map. The corresponding value will be used as the sampling weights,
            it should be a single-channel array in size, for example, `(1, spatial_dim_0, spatial_dim_1, ...)`
        spatial_size: the spatial size of the image patch e.g. [224, 224, 128].
            If its components have non-positive values, the corresponding size of `img` will be used.
        num_samples: number of samples (image patches) to take in the returned list.
        center_coord_key: if specified, the actual sampling location will be stored with the corresponding key.
        meta_keys: explicitly indicate the key of the corresponding meta data dictionary.
            used to add `patch_index` to the meta dict.
            for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
            the meta data is a dictionary object which contains: filename, original_shape, etc.
            it can be a sequence of string, map to the `keys`.
            if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
        meta_key_postfix: if meta_keys is None, use `key_{postfix}` to to fetch the meta data according
            to the key data, default is `meta_dict`, the meta data is a dictionary object.
            used to add `patch_index` to the meta dict.
        allow_missing_keys: don't raise exception if key is missing.
    See Also:
        :py:class:`monai.transforms.RandWeightedCrop`
    """

    backend = SpatialCrop.backend

    def __init__(
        self,
        keys: KeysCollection,
        w_key: str,
        spatial_size: Union[Sequence[int], int],
        num_samples: int = 1,
        center_coord_key: Optional[str] = None,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = "meta_dict",
        allow_missing_keys: bool = False,
        prob: float = 1.0
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.spatial_size = ensure_tuple(spatial_size)
        self.w_key = w_key
        self.prob = prob
        self.num_samples = int(num_samples)
        self.center_coord_key = center_coord_key
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.centers: List[np.ndarray] = []

    def randomize(self, weight_map: NdarrayOrTensor) -> None:
        self.centers = weighted_patch_samples(
            spatial_size=self.spatial_size, w=weight_map[0], n_samples=self.num_samples, r_state=self.R
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> List[Dict[Hashable, NdarrayOrTensor]]:
        d = dict(data)
        if random.random() < self.prob:
            self.randomize(d[self.w_key])
        else:
            if isinstance(d[self.w_key], np.array):                
                self.randomize(np.ones_like(d[self.w_key]))
            elif isinstance(d[self.w_key], torch.tensor):
                self.randomize(torch.ones_like(d[self.w_key]))
        _spatial_size = fall_back_tuple(self.spatial_size, d[self.w_key].shape[1:])

        # initialize returned list with shallow copy to preserve key ordering
        results: List[Dict[Hashable, NdarrayOrTensor]] = [dict(data) for _ in range(self.num_samples)]
        # fill in the extra keys with unmodified data
        for i in range(self.num_samples):
            for key in set(data.keys()).difference(set(self.keys)):
                results[i][key] = deepcopy(data[key])
        for key in self.key_iterator(d):
            img = d[key]
            if img.shape[1:] != d[self.w_key].shape[1:]:
                raise ValueError(
                    f"data {key} and weight map {self.w_key} spatial shape mismatch: "
                    f"{img.shape[1:]} vs {d[self.w_key].shape[1:]}."
                )
            for i, center in enumerate(self.centers):
                cropper = SpatialCrop(roi_center=center, roi_size=_spatial_size)
                orig_size = img.shape[1:]
                results[i][key] = cropper(img)
                self.push_transform(results[i], key, extra_info={"center": center}, orig_size=orig_size)
                if self.center_coord_key:
                    results[i][self.center_coord_key] = center
        # fill in the extra keys with unmodified data
        for i in range(self.num_samples):
            # add `patch_index` to the meta data
            for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key not in results[i]:
                    results[i][meta_key] = {}  # type: ignore
                results[i][meta_key][Key.PATCH_INDEX] = i  # type: ignore

        return results

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Create inverse transform
            orig_size = np.asarray(transform[TraceKeys.ORIG_SIZE])
            current_size = np.asarray(d[key].shape[1:])
            center = transform[TraceKeys.EXTRA_INFO]["center"]
            cropper = SpatialCrop(roi_center=center, roi_size=self.spatial_size)
            # get required pad to start and end
            pad_to_start = np.array([s.indices(o)[0] for s, o in zip(cropper.slices, orig_size)])
            pad_to_end = orig_size - current_size - pad_to_start
            # interleave mins and maxes
            pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
            inverse_transform = BorderPad(pad)
            # Apply inverse transform
            d[key] = inverse_transform(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)

        return d