# Copyright (c) OpenMMLab. All rights reserved.
import json
import mmcv
import numpy as np
import os.path as osp
import rawpy
from imageio import imread

from .transforms import Resize, RandomCrop
from ..builder import PIPELINES

try:
    from panopticapi.utils import rgb2id
except ImportError:
    rgb2id = None


@PIPELINES.register_module()
class LoadRAWFromFile:
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        if filename.endswith('TIF'):
            img = imread(filename)
        else:
            with rawpy.imread(filename) as f:
                img = f.raw_image_visible.copy()

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img.astype(np.float32)
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']

        # todo move meta_path to dataloader
        dir_path = results['img_prefix'].replace('raw', 'metainfo')
        base_name = osp.basename(results['img_info']['filename']).split('.')[0]
        meta_path = osp.join(dir_path, f'{base_name}.json')
        # with open(results['img_info']['meta_path'], 'r') as f:
        with open(meta_path, 'r') as f:
            results = {**results, **json.load(f)}
        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}()'
        return repr_str


@PIPELINES.register_module()
class Rearrange:
    """Rearrange RAW to four channels.
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        img = results['img']
        h, w = img.shape
        rearrange = np.zeros([h // 2, w // 2, 4], dtype=img.dtype)
        if results['bayer_pattern'] == 'rggb':
            img = img
        elif results['bayer_pattern'] == 'byyr':
            img = np.pad(img, (1, 0), (1, 0), mode='reflect')[:-1, :-1]
        else:
            raise NotImplementedError
        rearrange[..., 0] = img[0::2, 0::2]
        rearrange[..., 1] = img[0::2, 1::2]
        rearrange[..., 2] = img[1::2, 0::2]
        rearrange[..., 3] = img[1::2, 1::2]

        results['img'] = rearrange
        results['img_shape'] = rearrange.shape
        results['ori_shape'] = rearrange.shape
        """Resize seg,emtatopm with 0.5."""
        for key in results.get('seg_fields', []):
            results[key] = mmcv.imrescale(results[key], 0.5, interpolation='nearest')
        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}()'
        return repr_str


@PIPELINES.register_module()
class HDRSplit:
    """ Split HDR into two sub image
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        img = results['img']
        h, w, c = img.shape
        rearrange = np.zeros([h, w, 6], dtype=img.dtype)
        blc, saturate = results['black_level'], results['white_level']
        rearrange[..., :3] = np.clip(img, blc, blc + 255) - blc
        rearrange[..., 3:] = (np.clip(img, blc + 255, None) + 1 - blc) / 256 - 1
        max_v = (saturate + 1 - blc) / 256 - 1
        max_v = np.array([255., 255., 255., max_v, max_v, max_v], dtype=img.dtype).reshape([1, 1, 6])
        rearrange /= max_v
        results['img'] = rearrange
        results['img_shape'] = rearrange.shape
        results['ori_shape'] = rearrange.shape
        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}()'
        return repr_str


@PIPELINES.register_module()
class Demosaic:
    """ Demosaic RAW image.
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        assert results['bayer_pattern'] == 'rggb', 'Only support rggb pattern'
        img = results['img']
        if len(img.shape) == 3:
            assert img.shape[2] == 1
            img = img[..., 0]
        h, w = img.shape
        rgb = np.zeros((h, w, 3), dtype=img.dtype)

        img = np.pad(img, ((2, 2), (2, 2)), mode='reflect')
        r, gb, gr, b = img[0::2, 0::2], img[0::2, 1::2], img[1::2, 0::2], img[1::2, 1::2]

        rgb[0::2, 0::2, 0] = r[1:-1, 1:-1]
        rgb[0::2, 0::2, 1] = (gr[1:-1, 1:-1] + gr[:-2, 1:-1] + gb[1:-1, 1:-1] + gb[1:-1, :-2]) / 4
        rgb[0::2, 0::2, 2] = (b[1:-1, 1:-1] + b[:-2, :-2] + b[1:-1, :-2] + b[:-2, 1:-1]) / 4

        rgb[1::2, 0::2, 0] = (r[1:-1, 1:-1] + r[2:, 1:-1]) / 2
        rgb[1::2, 0::2, 1] = gr[1:-1, 1:-1]
        rgb[1::2, 0::2, 2] = (b[1:-1, 1:-1] + b[1:-1, :-2]) / 2

        rgb[0::2, 1::2, 0] = (r[1:-1, 1:-1] + r[1:-1, 2:]) / 2
        rgb[0::2, 1::2, 1] = gb[1:-1, 1:-1]
        rgb[0::2, 1::2, 2] = (b[1:-1, 1:-1] + b[:-2, 1:-1]) / 2

        rgb[1::2, 1::2, 0] = (r[1:-1, 1:-1] + r[2:, 2:] + r[1:-1, 2:] + r[2:, 1:-1]) / 4
        rgb[1::2, 1::2, 1] = (gr[1:-1, 1:-1] + gr[1:-1, 2:] + gb[1:-1, 1:-1] + gb[2:, 1:-1]) / 4
        rgb[1::2, 1::2, 2] = b[1:-1, 1:-1]

        results['img'] = rgb
        results['img_shape'] = rgb.shape
        results['ori_shape'] = rgb.shape
        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}()'
        return repr_str


@PIPELINES.register_module()
class RAWNormalize:
    """Rearrange RAW to four channels.
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        img = results['img']
        blc, saturate = results['black_level'], results['white_level']
        img = (img - blc) / (saturate - blc)
        img = np.clip(img, 0, 1)
        results['img'] = img
        from imageio import imwrite
        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}()'
        return repr_str

    
@PIPELINES.register_module()
class Gamma:
    """Rearrange RAW to four channels.
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        img = results['img']
        assert img.max() <= 1 and img.min() >= 0
        results['img'] = np.where(img <= 0.0031308, 12.92*img, 1.055*np.power(img, 1/2.4)-0.055)
        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}()'
        return repr_str

@PIPELINES.register_module()
class CombineVegetationTerrainAnno(object):
    """Combine vegetation and terrain annotations.
    """

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        gt_semantic_seg = results['gt_semantic_seg']
        for i in range(6, 12):
            gt_semantic_seg[gt_semantic_seg == i] = i - 1
        results['gt_semantic_seg'] = gt_semantic_seg
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + '()'
        return repr_str

@PIPELINES.register_module()
class MosaicResize(Resize):
    def __init__(self, black_level=528, saturation=4095, **kwargs):
        self.black_level = black_level
        self.saturation = saturation
        self._ori_img_scale = kwargs['img_scale']
        kwargs['img_scale'] = (kwargs['img_scale'][0] * 2, kwargs['img_scale'][1] * 2)
        super().__init__(**kwargs)

    def _mosaic_img(self, img):
        # image is bgr
        assert len(img.shape) == 3
        h, w = img.shape[:2]
        mosaic = np.zeros((h//2, w//2, 4), dtype=img.dtype)
        mosaic[..., 0] = img[0::2, 0::2, 2]
        mosaic[..., 1] = img[0::2, 1::2, 1]
        mosaic[..., 2] = img[1::2, 0::2, 1]
        mosaic[..., 3] = img[1::2, 1::2, 0]
        mosaic = mosaic.astype(np.float32)
        mosaic = (mosaic - self.black_level) / (self.saturation - self.black_level)
        return mosaic

    def __call__(self, results):
        if 'scale' not in results:
            self._random_scale(results)
        results['scale'] = tuple([i//2*2 for i in results['scale']])
        self._resize_img(results)
        assert len(results['img_shape']) == 3

        # mosaic image
        results['img'] = self._mosaic_img(results['img'])
        results['img_shape'] = results['img'].shape
        results['pad_shape'] = results['img_shape']
        results['scale_factor'] = results['scale_factor'] / 2

        # resize seg
        results['scale'] = tuple([i // 2 for i in results['scale']])
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(black_level={self.black_level}, '
                     f'saturation={self.saturation}, '
                     f'img_scale={self._ori_img_scale}, '
                     f'multiscale_mode={self.multiscale_mode}, '
                     f'ratio_range={self.ratio_range}, '
                     f'keep_ratio={self.keep_ratio})')
        return repr_str

@PIPELINES.register_module()
class ResizeGT():
    def __call__(self, results):
        for key in results.get('seg_fields', []):
            gt = results[key]
            h, w = gt.shape[:2]
            gt = mmcv.imresize(gt, (w//2, h//2), interpolation='nearest')
            results[key] = gt
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + '()'
        return repr_str
