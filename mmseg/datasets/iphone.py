# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .custom import CustomDataset
import numpy as np

@DATASETS.register_module()
class IphoneDataset(CustomDataset):
    """Iphone dataset.
    """

    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'plant', 'sky', 'person',
               'rider', 'car', 'truck', 'bus', 'tricycle', 'motorcycle',
               'bicycle')

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [70, 130, 180], [220, 20, 60], [255, 0, 0],
               [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230],
               [119, 11, 32]]

    def __init__(self, crop_pseudo_margins=None, **kwargs):
        kwargs['classes'] = ['road', 'building', 'fence', 'traffic light',
                             'traffic sign', 'plant', 'sky', 'person', 'car',
                             'truck', 'bus', 'bicycle']
        if crop_pseudo_margins is not None:
            assert kwargs['pipeline'][-1]['type'] == 'Collect'
            kwargs['pipeline'][-1]['keys'].append('valid_pseudo_mask')
        super(IphoneDataset, self).__init__(**kwargs)

        self.pseudo_margins = crop_pseudo_margins
        self.valid_mask_size = [1024, 2048]

    def pre_pipeline(self, results):
        super(IphoneDataset, self).pre_pipeline(results)
        if self.pseudo_margins is not None:
            results['valid_pseudo_mask'] = np.ones(
                self.valid_mask_size, dtype=np.uint8)
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            if self.pseudo_margins[0] > 0:
                results['valid_pseudo_mask'][:self.pseudo_margins[0], :] = 0
            # Here, the if statement is absolutely necessary
            if self.pseudo_margins[1] > 0:
                results['valid_pseudo_mask'][-self.pseudo_margins[1]:, :] = 0
            if self.pseudo_margins[2] > 0:
                results['valid_pseudo_mask'][:, :self.pseudo_margins[2]] = 0
            # Here, the if statement is absolutely necessary
            if self.pseudo_margins[3] > 0:
                results['valid_pseudo_mask'][:, -self.pseudo_margins[3]:] = 0
            results['seg_fields'].append('valid_pseudo_mask')
