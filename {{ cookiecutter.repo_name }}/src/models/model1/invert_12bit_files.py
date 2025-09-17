import json

import cv2
import numpy as np
import tifffile as tif
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

from src.models.swrd_model.utils import (
    label_to_color_mapping,
    symbol_to_label_mapping,
    unicode_to_label_mapping,
)


def apply_weld_filter(
    image: np.ndarray,
    cutoff_pct: float = 0.05,
    alpha: float = 1.0,
    clahe_clip: float = 2.0,
    clahe_grid: tuple = (8, 8),
    gamma: float = 1.0,
) -> np.ndarray:
    """Full pipeline: high-boost → CLAHE → gamma."""
    # 1) High-boost freq
    hb = high_boost_smooth(image, cutoff_pct=cutoff_pct, alpha=alpha)
    # 2) CLAHE
    cl = clahe_enhance(hb, clip_limit=clahe_clip, tile_grid_size=clahe_grid)
    # 3) Gamma
    out = gamma_correction(cl, gamma=gamma)
    return out


def high_boost_smooth(image, cutoff_pct=0.05, alpha=1.0, order=2):
    # –– mirror-pad 50% on each side
    ph, pw = image.shape[0] // 2, image.shape[1] // 2
    img = cv2.copyMakeBorder(image, ph, ph, pw, pw, cv2.BORDER_REFLECT)

    # –– FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    # –– radial map
    Y, X = np.ogrid[:rows, :cols]
    D = np.hypot(X - ccol, Y - crow)
    max_rad = np.hypot(crow, ccol)
    r0 = cutoff_pct * max_rad

    # –– smooth Butterworth high-pass
    H_hp = 1.0 / (1.0 + (r0 / (D + 1e-6)) ** (2 * order))
    H = 1.0 + alpha * H_hp

    # –– apply & iFFT
    fshift *= H
    img_back = np.fft.ifft2(np.fft.ifftshift(fshift))
    img_back = np.abs(img_back)

    # –– crop back to original
    img_back = img_back[ph:-ph, pw:-pw]

    # –– normalize
    img_back -= img_back.min()
    img_back *= 255.0 / img_back.max()
    return img_back.astype(np.uint8)


def clahe_enhance(
    image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)
) -> np.ndarray:
    """Apply CLAHE (adaptive histogram equalization)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Gamma correction: output = 255 * (image/255)**(1/gamma)."""
    inv_gamma = 1.0 / gamma
    # build a lookup table mapping each pixel [0,255] to its gamma'd value
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype(
        "uint8"
    )
    return cv2.LUT(image, table)


if __name__ == "__main__":
    # img_path = "/media/luke/Extreme SSD/SWRD_Data/Raw_data/images/DJ-RT-20220623-22.tif"

    # img = tif.imread(img_path)
    # # scale to 8bit
    # img = img / 65535 * 255
    # img = img.astype(np.uint8)
    # img_filtered = apply_weld_filter(img)

    # cv2.imwrite(
    #     "/home/luke/deeplify/repos/weld_defect_detection/img_filtered.png", img_filtered
    # )

    img_path = "/home/luke/deeplify/repos/weld_defect_detection/img_filtered.png"
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    mask_path = "/media/luke/Extreme SSD/SWRD_Data/Raw_data/json/DJ-RT-20220623-22.json"
    with open(mask_path, "r") as f:
        labels = json.load(f)

    annotations = labels["shapes"]
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    defect_info = []  # Store defect information for analysis

    for anno in annotations:
        if "label" not in anno.keys() or anno["label"] is None:
            continue

        label = anno["label"]
        if not label.startswith("\\u"):
            if label not in unicode_to_label_mapping.keys():
                continue
            label = unicode_to_label_mapping[label]
        else:
            if label not in symbol_to_label_mapping.keys():
                continue
            label = symbol_to_label_mapping[label]

        color = label_to_color_mapping[label]
        points = np.array(anno["points"], dtype=np.int32)

        # Create a temporary mask for this defect to calculate area
        temp_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        temp_mask = cv2.fillPoly(temp_mask, [points], 255)

        # Calculate area and bounding box
        area = np.sum(temp_mask == 255)
        x, y, w, h = cv2.boundingRect(points)

        defect_info.append(
            {
                "points": points,
                "color": color,
                "label": label,
                "area": area,
                "bbox": (x, y, w, h),
                "temp_mask": temp_mask,
            }
        )

        mask = cv2.fillPoly(mask, [points], color)

    # Create a custom colormap for the mask
    # Define colors for each defect type (RGB values)
    defect_colors = [
        [0, 0, 0],  # Background (black)
        [1, 0, 0],  # Porosity (red)
        [0, 1, 0],  # Inclusion (green)
        [0, 0, 1],  # Crack (blue)
        [1, 1, 0],  # Undercut (yellow)
        [1, 0, 1],  # Lack of fusion (magenta)
        [0, 1, 1],  # Lack of penetration (cyan)
    ]

    # Create colormap
    cmap_defects = ListedColormap(defect_colors)

    # Create legend elements
    legend_elements = []
    label_names = {
        1: "Porosity",
        2: "Inclusion",
        3: "Crack",
        4: "Undercut",
        5: "Lack of Fusion",
        6: "Lack of Penetration",
    }

    # Find the second largest defect by area
    second_largest_defect = None
    if len(defect_info) >= 2:
        sorted_defects = sorted(defect_info, key=lambda x: x["area"], reverse=True)
        second_largest_defect = sorted_defects[1]  # Second largest
    elif len(defect_info) == 1:
        second_largest_defect = defect_info[
            0
        ]  # Use the only defect if there's just one

    # Create mask overlay only for the second largest defect
    second_largest_defect_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    if second_largest_defect:
        second_largest_defect_mask = cv2.fillPoly(
            second_largest_defect_mask,
            [second_largest_defect["points"]],
            second_largest_defect["color"],
        )

    # Only add legend entries for defects that are actually present in the mask
    unique_values = np.unique(mask)
    for val in unique_values:
        if val > 0 and val in label_names:  # Skip background (0)
            color = defect_colors[val]
            legend_elements.append(Patch(facecolor=color, label=label_names[val]))

    # Create subplot layout
    # Left subplot: New visualization with bounding boxes and largest defect mask
    plt.imshow(img, cmap="gray")

    # Show mask overlay only for the second largest defect
    if second_largest_defect:
        mask_overlay = np.ma.masked_where(
            second_largest_defect_mask == 0, second_largest_defect_mask
        )
        plt.imshow(mask_overlay, cmap=cmap_defects, alpha=0.6, vmin=0, vmax=6)

    # Identify leftmost defect (by x-coordinate)
    leftmost_defect = None
    leftmost_defect_idx = None
    third_leftmost_idx = None
    if defect_info:
        # Sort defects by x-coordinate to find leftmost and third leftmost
        sorted_by_x = sorted(
            range(len(defect_info)), key=lambda i: defect_info[i]["bbox"][0]
        )
        leftmost_defect_idx = sorted_by_x[0]
        leftmost_defect = defect_info[leftmost_defect_idx]
        if len(sorted_by_x) >= 3:
            third_leftmost_idx = sorted_by_x[2]  # Third from the left

    # Identify four smallest defects by area (excluding leftmost)
    remaining_indices = [i for i in range(len(defect_info)) if i != leftmost_defect_idx]
    smallest_indices = sorted(remaining_indices, key=lambda i: defect_info[i]["area"])[
        :4
    ]

    # Draw colored bounding boxes around all defects (made slightly wider)
    for i, defect in enumerate(defect_info):
        x, y, w, h = defect["bbox"]
        # Make bounding box slightly wider by adding padding
        padding = 3
        x_padded = max(0, x - padding)
        y_padded = max(0, y - padding)
        w_padded = min(img.shape[1] - x_padded, w + 2 * padding)
        h_padded = min(img.shape[0] - y_padded, h + 2 * padding)

        # Determine bounding box color using index-based comparison
        if i == leftmost_defect_idx:
            edge_color = "lightgray"
        elif i in smallest_indices:
            edge_color = "orange"  # light yellow/orange-ish
        else:
            edge_color = "indianred"

        rect = Rectangle(
            (x_padded, y_padded),
            w_padded,
            h_padded,
            linewidth=0.5,
            edgecolor=edge_color,
            facecolor="none",
        )
        plt.gca().add_patch(rect)

        # Add defect label text above the bounding box
        label_names = {
            1: "Porosity",
            2: "Inclusion",
            3: "Crack",
            4: "Undercut",
            5: "Lack of Fusion",
            6: "Lack of Penetration",
        }

        defect_color = defect["color"]
        if defect_color in label_names:
            label_text = label_names[defect_color]

            # Position text below bounding box for third leftmost defect, above for others
            if i == third_leftmost_idx:
                text_y = y_padded + h_padded + 5  # Below the bounding box
                v_align = "top"
            else:
                text_y = y_padded - 5  # Above the bounding box
                v_align = "bottom"

            plt.text(
                x_padded,
                text_y,
                label_text,
                fontsize=10,
                color=edge_color,
                fontweight="bold",
                verticalalignment=v_align,
            )

    # plt.suptitle(
    #     "Filtered Image with Defect Bounding Boxes and Second Largest Defect Mask"
    # )
    plt.axis("off")

    # # Right subplot: Original visualization with full colored segmentation mask
    # ax2.imshow(img, cmap="gray")
    # mask_overlay_full = np.ma.masked_where(mask == 0, mask)
    # ax2.imshow(mask_overlay_full, cmap=cmap_defects, alpha=0.6, vmin=0, vmax=6)
    # ax2.set_title("Reference: Original Image with Full Segmentation Mask")
    # ax2.axis("off")

    # # Add legend if there are defects present
    # if legend_elements:
    #     ax2.legend(
    #         handles=legend_elements,
    #         loc="upper right",
    #         bbox_to_anchor=(1, 1),
    #         frameon=True,
    #         fancybox=True,
    #         shadow=True,
    #         framealpha=0.9,
    #     )

    plt.show()

    # plt.savefig(
    #     f"/home/luke/deeplify/repos/weld_defect_detection/examples_with_defect/{img_path.split('/')[-1].replace('.tif', '.png')}",
    #     dpi=300,
    #     bbox_inches="tight",
    # )
