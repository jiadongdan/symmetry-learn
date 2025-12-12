import warnings
import numpy as np
from typing import Optional, Union, Tuple
from ..sampling import stratified_sampling, sobol_sampling


class SymmMeasurement:
    """
    Patch-based measurement tool for symmetry analysis in multi-channel images.

    This class handles patch extraction from multi-channel image data using various
    sampling strategies. It maintains an (x, y) coordinate convention where x is the
    horizontal axis (width/columns) and y is the vertical axis (height/rows).

    Note on coordinate convention:
        - Points are stored as (x, y) tuples
        - When indexing numpy arrays, use data[y, x] (row, col ordering)
        - x ranges from [0, width), y ranges from [0, height)

    Attributes
    ----------
    data : np.ndarray
        Image data with shape (height, width, num_channels)
    num_channels : int
        Number of channels in the image
    height : int
        Image height in pixels
    width : int
        Image width in pixels
    patch_size : int or tuple of (int, int)
        Size of patches to extract (height, width)
    pts : np.ndarray or None
        Sampling points as (x, y) coordinates, shape (n_points, 2)
    ps : np.ndarray or None
        Extracted patches, shape (n_patches, patch_h, patch_w, num_channels)

    Examples
    --------
    >>> data = np.random.rand(256, 256, 8)  # 256x256 image with 8 channels
    >>> measurer = SymmMeasurement(data, num_channels=8, patch_size=32)
    >>> pts, patches = measurer.sample(overlap=0.5, method='sobol', seed=42)
    >>> measurer.show()
    """

    def __init__(
            self,
            data: np.ndarray,
            num_channels: int,
            patch_size: Union[int, Tuple[int, int]]
    ):
        """
        Initialize SymmMeasurement for patch-based analysis.

        Parameters
        ----------
        data : np.ndarray
            3D array containing image data. Can have channels in any axis position.
            Will be automatically reordered to (height, width, channels).
        num_channels : int
            Number of channels in the data. Must match one dimension of data.shape.
        patch_size : int or tuple of (int, int)
            Size of patches to extract. If int, patches will be square.
            If tuple, specifies (patch_height, patch_width).

        Raises
        ------
        ValueError
            If data is not 3D
            If num_channels not found in data.shape
            If patch_size is invalid

        Examples
        --------
        >>> data = np.random.rand(8, 256, 256)  # Channels first
        >>> measurer = SymmMeasurement(data, num_channels=8, patch_size=32)
        >>> measurer.data.shape
        (256, 256, 8)  # Automatically reordered to (H, W, C)

        >>> # Rectangular images are supported
        >>> data = np.random.rand(512, 256, 8)
        >>> measurer = SymmMeasurement(data, num_channels=8, patch_size=(64, 32))
        """
        # Validate data dimensionality
        if data.ndim != 3:
            raise ValueError(f"Expected 3D data, got {data.ndim}D")

        # Validate num_channels exists in data
        if num_channels not in data.shape:
            raise ValueError(
                f"num_channels={num_channels} not found in data.shape={data.shape}"
            )

        # Reorder data to (height, width, channels) format
        channel_axis = data.shape.index(num_channels)
        if channel_axis != 2:
            data = np.moveaxis(data, channel_axis, -1)

        self.data = data
        self.num_channels = num_channels
        self.height, self.width = data.shape[:2]

        # Validate patch_size
        if isinstance(patch_size, int):
            if patch_size <= 0:
                raise ValueError(f"patch_size must be positive, got {patch_size}")
            if patch_size > min(self.height, self.width):
                raise ValueError(
                    f"patch_size={patch_size} exceeds minimum image dimension "
                    f"(min({self.height}, {self.width}) = {min(self.height, self.width)})"
                )
        else:
            patch_h, patch_w = patch_size
            if patch_h <= 0 or patch_w <= 0:
                raise ValueError(f"patch_size dimensions must be positive, got {patch_size}")
            if patch_h > self.height or patch_w > self.width:
                raise ValueError(
                    f"patch_size={patch_size} exceeds image dimensions "
                    f"({self.height}, {self.width})"
                )

        self.patch_size = patch_size
        self.pts = None  # Will store (x, y) coordinates
        self.ps = None   # Will store extracted patches

    def __repr__(self) -> str:
        """
        Return string representation of SymmMeasurement instance.

        Returns
        -------
        str
            String describing the instance state

        Examples
        --------
        >>> measurer = SymmMeasurement(data, num_channels=8, patch_size=32)
        >>> print(measurer)
        SymmMeasurement(shape=(256, 256, 8), channels=8, patch_size=32, no points, no patches)

        >>> measurer.generate_pts(overlap=0.5, seed=42)
        >>> measurer.extract_patches()
        >>> print(measurer)
        SymmMeasurement(shape=(256, 256, 8), channels=8, patch_size=32, 100 points, 100 patches)
        """
        pts_str = f"{len(self.pts)} points" if self.pts is not None else "no points"
        ps_str = f"{len(self.ps)} patches" if self.ps is not None else "no patches"
        return (
            f"SymmMeasurement(shape={self.data.shape}, "
            f"channels={self.num_channels}, "
            f"patch_size={self.patch_size}, "
            f"{pts_str}, {ps_str})"
        )

    def _get_patch_dims(self) -> Tuple[int, int]:
        """
        Get patch dimensions as (height, width).

        Returns
        -------
        tuple of (int, int)
            Patch height and width
        """
        if isinstance(self.patch_size, int):
            return self.patch_size, self.patch_size
        else:
            return self.patch_size

    def generate_pts(
            self,
            patch_size: Optional[Union[int, Tuple[int, int]]] = None,
            overlap: float = 0.5,
            method: str = 'stratified',
            seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate sampling points for patch extraction.

        Points are generated such that patches centered at these points will not
        extend beyond image boundaries. Coordinates are returned as (x, y) where
        x is horizontal position and y is vertical position.

        Parameters
        ----------
        patch_size : int or tuple of (int, int), optional
            Size of patches. If None, uses self.patch_size.
            If provided and differs from self.patch_size, updates self.patch_size
            and issues a warning.
        overlap : float, default=0.5
            Overlap between adjacent patches as a fraction (0.0 to <1.0).
            - 0.0 = no overlap (patches are adjacent)
            - 0.5 = 50% overlap (patches overlap by half their size)
            - 0.75 = 75% overlap
        method : {'stratified', 'sobol'}, default='stratified'
            Sampling method to use:
            - 'stratified': Regular grid with optional jittering
            - 'sobol': Quasi-random low-discrepancy sampling
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        np.ndarray
            Array of shape (n_points, 2) containing (x, y) coordinates

        Raises
        ------
        ValueError
            If method is not 'stratified' or 'sobol'
            If overlap is invalid (< 0 or >= 1)
            If patch_size is invalid
            If no valid points remain after boundary filtering

        Notes
        -----
        The number of points generated depends on overlap:
        - stride = patch_size * (1 - overlap)
        - n_patches_per_axis = ceil(image_size / stride)
        - Total points ≈ n_patches_per_axis²

        Points at edges where patches would extend beyond boundaries are
        automatically filtered out.

        Examples
        --------
        >>> measurer = SymmMeasurement(data, num_channels=8, patch_size=32)
        >>> pts = measurer.generate_pts(overlap=0.5, method='sobol', seed=42)
        >>> pts.shape
        (100, 2)  # 100 points, each with (x, y) coordinates

        >>> # High overlap for dense sampling
        >>> pts = measurer.generate_pts(overlap=0.9, method='stratified', seed=42)
        >>> pts.shape
        (900, 2)  # Many more points due to high overlap
        """
        # Validate overlap
        if not (0.0 <= overlap < 1.0):
            raise ValueError(
                f"overlap must be in [0.0, 1.0), got {overlap}. "
                f"Use fractional overlap (e.g., 0.5 for 50% overlap)."
            )

        size = (self.height, self.width)

        # Handle patch_size parameter
        if patch_size is None:
            patch_size = self.patch_size
        elif patch_size != self.patch_size:
            warnings.warn(
                f"patch_size={patch_size} differs from initialized "
                f"patch_size={self.patch_size}. Using {patch_size} for sampling.",
                UserWarning
            )
            # Validate new patch_size
            if isinstance(patch_size, int):
                if patch_size <= 0 or patch_size > min(self.height, self.width):
                    raise ValueError(
                        f"Invalid patch_size={patch_size} for image dimensions "
                        f"({self.height}, {self.width})"
                    )
            else:
                patch_h, patch_w = patch_size
                if patch_h <= 0 or patch_w <= 0:
                    raise ValueError(f"Invalid patch_size={patch_size}")
                if patch_h > self.height or patch_w > self.width:
                    raise ValueError(
                        f"patch_size={patch_size} exceeds image dimensions "
                        f"({self.height}, {self.width})"
                    )
            self.patch_size = patch_size

        # Generate sampling points using selected method
        if method == 'stratified':
            pts = stratified_sampling(
                size, seed=seed, patch_size=patch_size, overlap=overlap
            )
        elif method == 'sobol':
            pts = sobol_sampling(
                size, seed=seed, patch_size=patch_size, overlap=overlap
            )
        else:
            raise ValueError(
                f"method must be 'stratified' or 'sobol', got '{method}'"
            )

        # Convert to integer coordinates (rounding to nearest pixel)
        pts = np.round(pts).astype(np.int64)

        # Filter out points where patches would extend beyond boundaries
        # Note: pts are (x, y), so pts[:, 0] is x, pts[:, 1] is y
        patch_h, patch_w = self._get_patch_dims()
        half_h = patch_h // 2
        half_w = patch_w // 2

        valid_mask = (
                (pts[:, 0] >= half_w) &              # x >= half_width (left boundary)
                (pts[:, 0] + half_w < self.width) &  # x + half_width < width (right boundary)
                (pts[:, 1] >= half_h) &              # y >= half_height (top boundary)
                (pts[:, 1] + half_h < self.height)   # y + half_height < height (bottom boundary)
        )

        self.pts = pts[valid_mask]
        self.ps = None  # Invalidate cached patches when points change

        # Warn if too many points were filtered out
        n_filtered = len(pts) - len(self.pts)
        if n_filtered > 0:
            pct_filtered = 100 * n_filtered / len(pts)
            if pct_filtered > 10:
                warnings.warn(
                    f"Filtered out {n_filtered}/{len(pts)} points ({pct_filtered:.1f}%) "
                    f"due to boundary constraints. Consider using smaller patch_size "
                    f"or less overlap.",
                    UserWarning
                )

        # Check if we have any valid points
        if len(self.pts) == 0:
            raise ValueError(
                f"No valid sampling points after boundary filtering. "
                f"patch_size={patch_size} may be too large for image size "
                f"({self.height}, {self.width})."
            )

        return self.pts

    def extract_patches(self) -> np.ndarray:
        """
        Extract image patches centered at sampling points.

        Patches are extracted from self.data centered at each point in self.pts.
        Must call generate_pts() first to define sampling locations.

        Returns
        -------
        np.ndarray
            Array of shape (n_patches, patch_h, patch_w, num_channels)
            containing extracted patches

        Raises
        ------
        ValueError
            If generate_pts() has not been called yet (self.pts is None/empty)

        Notes
        -----
        For a point (x, y) and patch size (patch_h, patch_w):
        - Patch is extracted from data[y-half_h:y+half_h, x-half_w:x+half_w]
        - Remember: numpy uses [row, col] = [y, x] indexing

        Examples
        --------
        >>> measurer.generate_pts(overlap=0.5, seed=42)
        >>> patches = measurer.extract_patches()
        >>> patches.shape
        (100, 32, 32, 8)  # 100 patches of 32x32 with 8 channels
        """
        if self.pts is None or len(self.pts) == 0:
            raise ValueError(
                "No sampling points available. Call generate_pts() first."
            )

        patch_h, patch_w = self._get_patch_dims()
        half_h = patch_h // 2
        half_w = patch_w // 2

        patches = []
        # Iterate through points (x, y)
        for x, y in self.pts:
            # Extract patch using numpy [row, col] = [y, x] indexing
            patch = self.data[y-half_h:y+half_h, x-half_w:x+half_w]
            patches.append(patch)

        self.ps = np.array(patches)
        return self.ps

    def sample(
            self,
            patch_size: Optional[Union[int, Tuple[int, int]]] = None,
            overlap: float = 0.5,
            method: str = 'stratified',
            seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate sampling points and extract patches in one call.

        Convenience method that calls generate_pts() followed by extract_patches().

        Parameters
        ----------
        patch_size : int or tuple of (int, int), optional
            Size of patches. If None, uses self.patch_size.
        overlap : float, default=0.5
            Overlap between patches as fraction (0.5 = 50% overlap)
        method : {'stratified', 'sobol'}, default='stratified'
            Sampling method
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        pts : np.ndarray
            Sampling points with shape (n_points, 2) as (x, y) coordinates
        patches : np.ndarray
            Extracted patches with shape (n_patches, patch_h, patch_w, num_channels)

        Examples
        --------
        >>> measurer = SymmMeasurement(data, num_channels=8, patch_size=32)
        >>> pts, patches = measurer.sample(overlap=0.5, method='sobol', seed=42)
        >>> pts.shape, patches.shape
        ((100, 2), (100, 32, 32, 8))
        """
        # Generate sampling points
        self.generate_pts(
            patch_size=patch_size, overlap=overlap, method=method, seed=seed
        )

        # Extract patches at those points
        self.extract_patches()

        return self.pts, self.ps

    def show(
            self,
            ax: Optional[Tuple] = None,
            show_patch_idx: Optional[int] = None
    ) -> Optional[Tuple]:
        """
        Visualize the data, sampling points, and a sample patch.

        Creates a 3-panel visualization:
        1. First channel of the image
        2. Sampling points overlaid on image with patch rectangles
        3. A randomly selected patch (or specified by show_patch_idx)

        Parameters
        ----------
        ax : tuple of 3 matplotlib axes, optional
            If provided, plot into these axes. Otherwise creates new figure.
        show_patch_idx : int, optional
            Index of patch to display in panel 3. If None, selects randomly.

        Returns
        -------
        tuple of 3 axes or None
            Returns axes if created internally (ax=None), otherwise None

        Notes
        -----
        - Red dots and rectangles show all sampling points
        - Yellow star and rectangle highlight the displayed patch
        - Works even if extract_patches() hasn't been called (extracts on-the-fly)

        Examples
        --------
        >>> measurer.generate_pts(overlap=0.5, seed=42)
        >>> measurer.show()  # Show with random patch
        >>> measurer.show(show_patch_idx=5)  # Show specific patch
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        # Setup axes
        if ax is None:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        else:
            if len(ax) != 3:
                raise ValueError("ax must be a list/tuple of 3 axes")
            ax1, ax2, ax3 = ax

        # Panel 1: Show first channel
        ax1.imshow(self.data[:, :, 0], cmap='gray')
        ax1.set_title('First Channel')
        ax1.axis('off')

        # Check if points have been generated
        if self.pts is None or len(self.pts) == 0:
            ax2.imshow(self.data[:, :, 0], cmap='gray', alpha=0.5)
            ax2.text(
                0.5, 0.5, 'Call generate_pts() first',
                transform=ax2.transAxes, ha='center', va='center',
                fontsize=12, color='red'
            )
            ax2.set_title('Sampling Points')
            ax2.axis('off')

            ax3.text(
                0.5, 0.5, 'No patch to show',
                transform=ax3.transAxes, ha='center', va='center',
                fontsize=12, color='red'
            )
            ax3.axis('off')

            plt.tight_layout()
            return (ax1, ax2, ax3) if ax is None else None

        # Panel 2: Show sampling points with patch rectangles
        ax2.imshow(self.data[:, :, 0], cmap='gray', alpha=0.5)

        # Scatter all points (remember: pts are (x, y))
        ax2.scatter(self.pts[:, 0], self.pts[:, 1], color='red', s=10, alpha=0.6)

        # Get patch dimensions
        patch_h, patch_w = self._get_patch_dims()
        half_h = patch_h // 2
        half_w = patch_w // 2

        # Draw rectangles for all patches
        # Rectangle takes (x, y) for bottom-left corner, then (width, height)
        for x, y in self.pts:
            rect = Rectangle(
                (x - half_w, y - half_h),  # Bottom-left corner (x, y)
                patch_w, patch_h,          # Width, height
                linewidth=1, edgecolor='red', facecolor='none', alpha=0.3
            )
            ax2.add_patch(rect)

        # Select and highlight a specific patch
        if show_patch_idx is None:
            random_idx = np.random.randint(0, len(self.pts))
        else:
            if show_patch_idx < 0 or show_patch_idx >= len(self.pts):
                raise ValueError(
                    f"show_patch_idx={show_patch_idx} out of range "
                    f"[0, {len(self.pts)})"
                )
            random_idx = show_patch_idx

        x_rand, y_rand = self.pts[random_idx]

        # Highlight the selected point with yellow star
        ax2.scatter(
            x_rand, y_rand, color='yellow', s=50, marker='*',
            edgecolor='black', linewidth=1, zorder=10
        )

        # Draw highlighted rectangle in yellow
        rect_highlight = Rectangle(
            (x_rand - half_w, y_rand - half_h),
            patch_w, patch_h,
            linewidth=2, edgecolor='yellow', facecolor='none'
        )
        ax2.add_patch(rect_highlight)

        # Panel 3: Show the selected patch
        if self.ps is not None and len(self.ps) > random_idx:
            # Use pre-extracted patches if available
            ax3.imshow(self.ps[random_idx, :, :, 0], cmap='gray')
        else:
            # Extract on-the-fly if patches haven't been extracted yet
            # Remember: numpy indexing is [y, x] = [row, col]
            patch = self.data[
                    y_rand-half_h:y_rand+half_h,
                    x_rand-half_w:x_rand+half_w,
                    0
                    ]
            ax3.imshow(patch, cmap='gray')

        ax3.set_title(f'Patch at (x={x_rand}, y={y_rand}), index={random_idx}')

        ax2.set_title(f'Sampling Points (n={len(self.pts)})')
        ax2.axis('off')
        ax3.axis('off')

        plt.tight_layout()
        return (ax1, ax2, ax3) if ax is None else None

