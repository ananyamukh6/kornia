import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


import torch
import kornia
import cv2
import numpy as np
import pdb  # todo remove
import matplotlib.pyplot as plt


def dilation(img: torch.Tensor, structuring_element: torch.Tensor):
    r"""Function that computes dilated image given a structuring element.
    See :class:`~kornia.morphology.Dilation` for details.
    """
    if not torch.is_tensor(img):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(img)}")
    if not torch.is_tensor(structuring_element):
        raise TypeError(f"Structuring element type is not a torch.Tensor. Got {type(structuring_element)}")
    img_shape = img.shape
    if not (len(img_shape) == 3 or len(img_shape) == 4):
        raise ValueError(f"Expected input tensor to be of ndim 3 or 4, but got {len(img_shape)}")
    if len(img_shape) == 3:
        # unsqueeze introduces a batch dimension
        img = img.unsqueeze(0)
    else:
        if(img_shape[1] != 1):
            raise ValueError(f"Expected a single channel image, but got {img_shape[1]} channels")
    if len(structuring_element.shape) != 2:
        raise ValueError(
            f"Expected structuring element tensor to be of ndim=2, but got {len(structuring_element.shape)}")

    # Check if the input image is a binary containing only 0, 1
    unique_vals = torch.unique(img)

    if len(unique_vals) > 2:
        raise ValueError(
            f"Expected only 2 unique values in the tensor, since it should be binary, but got {len(torch.unique(img))}")
    if not ((unique_vals == 0.0) + (unique_vals == 1.0)).all():
        raise ValueError("Expected image to contain only 1's and 0's since it should be a binary image")

    # Convert structuring_element from shape [a, b] to [1, 1, a, b]
    structuring_element = structuring_element.unsqueeze(0).unsqueeze(0)

    se_shape = structuring_element.shape
    conv1 = F.conv2d(img, structuring_element, padding=(se_shape[2] // 2, se_shape[2] // 2))

    convert_to_binary = (conv1 > 0).float()

    if len(img_shape) == 3:
        # If the input ndim was 3, then remove the fake batch dim introduced to do conv
        return torch.squeeze(convert_to_binary, 0)
    else:
        return convert_to_binary


from torch.nn import functional as f

def dilation2(img: torch.Tensor, structuring_element: torch.Tensor):
    r"""Function that computes dilated image given a structuring element.
    See :class:`~kornia.morphology.Dilation` for details.
    """
    if not torch.is_tensor(img):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(img)}")
    if not torch.is_tensor(structuring_element):
        raise TypeError(f"Structuring element type is not a torch.Tensor. Got {type(structuring_element)}")
    img_shape = img.shape
    if not (len(img_shape) == 3 or len(img_shape) == 4):
        raise ValueError(f"Expected input tensor to be of ndim 3 or 4, but got {len(img_shape)}")
    if len(img_shape) == 3:
        # unsqueeze introduces a batch dimension
        img = img.unsqueeze(0)
    else:
        if(img_shape[1] != 1):
            raise ValueError(f"Expected a single channel image, but got {img_shape[1]} channels")
    if len(structuring_element.shape) != 2:
        raise ValueError(
            f"Expected structuring element tensor to be of ndim=2, but got {len(structuring_element.shape)}")

    img_pad = torch.nn.ConstantPad2d((structuring_element.shape[0]//2, structuring_element.shape[0]//2, structuring_element.shape[1]//2, structuring_element.shape[1]//2), 0)(img)
    windows = f.unfold(img_pad, kernel_size=structuring_element.shape)
    #pdb.set_trace()
    # st_elem_tmp of shape [1, 9, 1] (assuming structuring_element was 3x3)
    st_elem_tmp = structuring_element.flatten().unsqueeze(0).unsqueeze(-1)
    processed = windows.add(st_elem_tmp).max(dim=1, keepdims=True)[0]
    out = f.fold(processed, img.shape[-2:], kernel_size=1)

    #pdb.set_trace()

    if len(img_shape) == 3:
        # If the input ndim was 3, then remove the fake batch dim introduced to do conv
        return torch.squeeze(out, 0)
    else:
        return out


def dilation3(img: torch.Tensor, structuring_element: torch.Tensor):
    r"""Function that computes dilated image given a structuring element.
    See :class:`~kornia.morphology.Dilation` for details.
    """
    if not torch.is_tensor(img):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(img)}")
    if not torch.is_tensor(structuring_element):
        raise TypeError(f"Structuring element type is not a torch.Tensor. Got {type(structuring_element)}")
    img_shape = img.shape
    if not (len(img_shape) == 3 or len(img_shape) == 4):
        raise ValueError(f"Expected input tensor to be of ndim 3 or 4, but got {len(img_shape)}")
    if len(img_shape) == 3:
        # unsqueeze introduces a batch dimension
        img = img.unsqueeze(0)
    else:
        if(img_shape[1] != 1):
            raise ValueError(f"Expected a single channel image, but got {img_shape[1]} channels")
    if len(structuring_element.shape) != 2:
        raise ValueError(
            f"Expected structuring element tensor to be of ndim=2, but got {len(structuring_element.shape)}")

    img_pad = torch.nn.ConstantPad2d((structuring_element.shape[0]//2, structuring_element.shape[0]//2, structuring_element.shape[1]//2, structuring_element.shape[1]//2), 0)(img)
    windows = f.unfold(img_pad, kernel_size=structuring_element.shape)
    #pdb.set_trace()
    # st_elem_tmp of shape [1, 9, 1] (assuming structuring_element was 3x3)
    st_elem_tmp = structuring_element.flatten().unsqueeze(0).unsqueeze(-1)
    max_kernel = structuring_element.max()
    processed = windows.add(st_elem_tmp).max(dim=1, keepdims=True)[0] - max_kernel
    out = f.fold(processed, img.shape[-2:], kernel_size=1)

    #pdb.set_trace()

    if len(img_shape) == 3:
        # If the input ndim was 3, then remove the fake batch dim introduced to do conv
        return torch.squeeze(out, 0)
    else:
        return out


# create an image
img = np.zeros([1, 10, 10], dtype=float)
img[:, 3:6, 3:6] = 1.0
img[:, 3, 3] = 0.0
img[:, 4:5, 4:5] = 0.0
img[:, 6:8, 6] = 1.0

# convert to torch tensor
bin_image: torch.tensor = torch.tensor(img, dtype=torch.float32)

# structuring_element is a torch.tensor of ndims 2 containing only 1's and 0's
np_structuring_element = np.ones([3, 3])
np_structuring_element[0, 0] = 0.0
structuring_element = torch.tensor(np_structuring_element).float()
# The structuring element is:
# 0 1 1
# 1 1 1
# 1 1 1

dilated_image = dilation3(bin_image, structuring_element)

# convert back to numpy
dilated_image: np.array = kornia.tensor_to_image(dilated_image)

# Create the plot
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
axs = axs.ravel()

axs[0].axis('off')
axs[0].set_title('Original image')
axs[0].imshow(img.squeeze() * 0.9, cmap='gray', vmin=0, vmax=1.0)

axs[1].axis('off')
axs[1].set_title('Structuring element')
axs[1].imshow(np_structuring_element * 0.9, cmap='gray', vmin=0, vmax=1.0)

axs[2].axis('off')
axs[2].set_title('Dilated image')
axs[2].imshow(dilated_image * 0.9, cmap='gray', vmin=0, vmax=1.0)

axs[3].axis('off')
axs[3].set_title('Superimposed')
axs[3].imshow(0.9 * (img.squeeze() * 0.5 + dilated_image * 0.5), cmap='gray', vmin=0, vmax=1.0)

plt.grid(True)
plt.show()
