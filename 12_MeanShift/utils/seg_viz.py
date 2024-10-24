import numpy as np
from skimage.filters import threshold_multiotsu

def label_channel(image: np.ndarray, num_labels: int) -> tuple[int, np.ndarray]:
    """Labels the input image into specified number of classes using multi-Otsu thresholding.

    Args:
        image (np.ndarray): The input image to be labeled.
        num_labels (int): The number of labels/classes to segment the image into.

    Returns:
        tuple[int, np.ndarray]: A tuple containing the number of labels and the labeled image.
    """
    thresholds = threshold_multiotsu(image, classes=num_labels)
    labeled_image = np.zeros_like(image, dtype=np.int32)
    for i, threshold in enumerate(thresholds):
        labeled_image[image < threshold] = i
    return num_labels, labeled_image

def label_segments(image: np.ndarray, num_labels: int) -> tuple[int, np.ndarray]:
    """Labels the input 3-channel segmented image into specified number of classes.

    Args:
        image (np.ndarray): 3-channel segmented image (M, N, 3)

    Returns:
        tuple[int, np.ndarray]: Number of labels and Labelled image (np.ndarray): (M, N)
    """
    labeled_list: list[np.ndarray] = []
    num_labels_per_channel = round(num_labels ** (1/3))
    for channel in range(3):
        _, labels = label_channel(image[:, :, channel], num_labels=num_labels_per_channel)
        labeled_list.append(labels)

    # 将三个通道的标签矩阵堆叠在一起，形成一个 (M, N, 3) 的矩阵
    labeled_image = np.stack(labeled_list, axis=-1).squeeze()

    # 用于存储三通道标签元组 (r_label, g_label, b_label) 和它们的唯一标签
    labels2idx: dict[tuple, int] = {}
    label_result = np.zeros_like(labeled_image[:, :, 0], dtype=np.int32)  # 创建输出标签图像

    # 遍历每个像素并为每个唯一的 (r, g, b) 标签组合分配一个新的标签
    for x, y in np.ndindex(label_result.shape):
        label_tuple = tuple(labeled_image[x, y])  # 获取当前像素的 (r_label, g_label, b_label)
        
        # 如果这个组合没有出现在字典中，分配一个新的标签
        if label_tuple not in labels2idx:
            labels2idx[label_tuple] = len(labels2idx)  # 从 0 开始
        
        # 将唯一标签赋给结果图像
        label_result[x, y] = labels2idx[label_tuple]
    
    # 返回总标签数和标记结果
    return len(labels2idx), label_result

def color_labels(image: np.ndarray) -> np.ndarray:
    """Colorizes the labeled image.

    Args:
        image (np.ndarray): The labeled image.
        num_labels (int): The number of labels.

    Returns:
        np.ndarray: The colorized image.
    """
    num_labels = np.unique(image).shape[0]
    colors = np.random.randint(0, 255, (num_labels, 3), dtype=np.uint8)
    return colors[image]