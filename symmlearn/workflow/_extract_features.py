import numpy as np


class ExperimentWorkflow:

    def __init__(self, img, patch_size, num_patches, lbs_img=None):
        self.img = img
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.lbs_img = lbs_img

        self.pts = None
        self.patches = None
        self.lbs = None

    def process(self, seed=None):
        # get the points, assign to self.pts
        # extract patches, assign to self.patches
        # if self.img_lbs is None, self.lbs is None;
        # else self.lbs is not None, it is assigned as the dominant labels of the corresponding patch.
        pass
