import numpy as np
from ..sampling import stratified_sampling


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
        # if self.lbs_img is None, self.lbs is None;
        # else self.lbs is not None, it is assigned as the dominant labels of the corresponding patch.
        h, w = self.img.shape[:2]
        half = self.patch_size // 2

        # Generate candidate points via stratified sampling; pts are (x, y)
        pts = stratified_sampling(
            size=(h, w),
            n_samples=self.num_patches,
            seed=seed,
        )
        pts = np.round(pts).astype(np.int64)

        # Keep only points whose patch stays fully within image bounds
        valid = (
            (pts[:, 0] >= half) &
            (pts[:, 0] + half < w) &
            (pts[:, 1] >= half) &
            (pts[:, 1] + half < h)
        )
        pts = pts[valid]

        # Randomly down-select to num_patches if we have more than needed
        if len(pts) > self.num_patches:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(pts), size=self.num_patches, replace=False)
            pts = pts[idx]

        self.pts = pts

        # Extract patches centered at each point; pts are (x, y), arrays are [y, x]
        self.patches = np.array([
            self.img[y - half:y + half, x - half:x + half]
            for x, y in pts
        ])

        # Compute dominant label per patch when a label image is provided
        if self.lbs_img is None:
            self.lbs = None
        else:
            lbs = []
            for x, y in pts:
                region = self.lbs_img[y - half:y + half, x - half:x + half].ravel()
                values, counts = np.unique(region, return_counts=True)
                lbs.append(values[np.argmax(counts)])
            self.lbs = np.array(lbs)
