import numpy as np


def sub2ind(shape, i, j):
    """Custom sub2ind function for linear indexing."""
    return i * shape[1] + j


def seg2edge(seg, radius, label_ignore=None, edge_type="regular"):
    """This function computes edge pixels based on segment labels.

    Args:
        seg: A numpy array of shape (height, width, channels) representing the segmentation mask.
        radius: An integer specifying the neighborhood radius for considering edges.
        label_ignore: A numpy array of shape (num_ignore_labels, channels) containing labels to ignore
                        when detecting edges. Defaults to None.
        edge_type: A string specifying the type of edge to detect. Can be 'regular' (any contrast),
                    'inner' (object boundary going from inside to outside), or 'outer' (object boundary
                    going from outside to inside). Defaults to 'regular'.

    Returns:
        A numpy array of shape (height, width) where values indicate the intensity of the edge.
    """

    height, width, chn = seg.shape
    if label_ignore is not None and chn != label_ignore.shape[1]:
        raise ValueError(
            "Channel dimension of segmentation and label_ignore must match."
        )

    # Set neighborhood area
    radius_search = max(int(np.ceil(radius)), 1)
    x, y = np.meshgrid(
        np.arange(-radius_search, radius_search + 1),
        np.arange(-radius_search, radius_search + 1),
    )

    # Reshape data
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    X = X.flatten()
    Y = Y.flatten()
    x = x.flatten()
    y = y.flatten()

    # Initialize edge map
    edge_map = np.zeros((height, width), dtype=float)

    # Loop through each neighbor
    for i in range(len(x)):
        # Get neighboring pixel coordinates
        neighbor_x = X + x[i]
        neighbor_y = Y + y[i]

        # Ignore out-of-bounds indices
        valid_idx = (
            (neighbor_x >= 0)
            & (neighbor_x < width)
            & (neighbor_y >= 0)
            & (neighbor_y < height)
        )

        # Extract center and neighbor labels
        center_labels = seg[Y[valid_idx], X[valid_idx], :]
        neighbor_labels = seg[neighbor_y[valid_idx], neighbor_x[valid_idx], :]

        # Identify different edge types based on edge_type argument
        if edge_type == "regular":
            diff_idx = np.any(center_labels != neighbor_labels, axis=1)
        elif edge_type == "inner":
            diff_idx = np.logical_and(
                np.any(center_labels != neighbor_labels, axis=1),
                np.any(center_labels != 0, axis=1),
                np.all(neighbor_labels == 0, axis=1),
            )
        elif edge_type == "outer":
            diff_idx = np.logical_and(
                np.any(center_labels != neighbor_labels, axis=1),
                np.all(center_labels == 0, axis=1),
                np.any(neighbor_labels != 0, axis=1),
            )
        else:
            raise ValueError(
                "Invalid edge_type. Must be 'regular', 'inner', or 'outer'."
            )

        # Exclude edges with labels to ignore (if label_ignore provided)
        if label_ignore is not None:
            edge_labels = center_labels[diff_idx]
            neighbor_labels = neighbor_labels[diff_idx]
            ignore_mask = np.zeros(len(diff_idx), dtype=bool)
            for j in range(label_ignore.shape[0]):
                ignore_mask |= np.all(
                    np.logical_or(
                        edge_labels == label_ignore[j, :],
                        neighbor_labels == label_ignore[j, :],
                    ),
                    axis=1,
                )
            diff_idx = diff_idx[
                ~ignore_mask
            ]  # Update diff_idx to exclude ignored edges

        # Update edge map based on loop results
        edge_map[Y[valid_idx][diff_idx], X[valid_idx][diff_idx]] = (
            1  # Set to 1 at valid edge indices
        )

    # Return the edge map
    return edge_map
