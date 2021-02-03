import argparse
import glob
import importlib
import numpy as np
import os
import PIL.Image
import random
import torch
from skimage.io import imsave
from skimage.transform import resize

from training.augment import AugmentPipe


TORCH_IS_AVAILABLE = False  #importlib.util.find_spec('torch') is not None
NP_IS_AVAILABLE = True #importlib.util.find_spec('numpy') is not None

def tensor_info(t, name='', print_stats=False, detailed=False):
    """
    Convenience method to format diagnostic tensor information, including
    shape, type, and optional attributes if specified as string.
    This information can then be logged as:
    logger.debug(tensor_info(my_tensor, 'my tensor'))
    Log output:
    my_tensor: [10, 2, 100, 100] (torch.float32)
    Args:
        t: input pytorch tensor or numpy array or None
        name: human readable name of the tensor (optional)
        print_stats: if True, includes mean/max/min statistics (takes compute time)
        detailed: if True, includes details about tensor properties
    Returns:
        formatted string
    """
    def _get_stats_str():
        if NP_IS_AVAILABLE and type(t) == np.ndarray:
            t_min = np.min(t)
            t_max = np.max(t)
            t_mean = np.mean(t)
        elif TORCH_IS_AVAILABLE and torch.is_tensor(t):
            t_min = torch.min(t).item()
            t_max = torch.max(t).item()
            t_mean = torch.mean(t).item()
        else:
            raise RuntimeError('Not implemented for {}'.format(type(t)))
        return ' - [min %0.4f, max %0.4f, mean %0.4f]' % (t_min, t_max, t_mean)

    def _get_details_str():
        if TORCH_IS_AVAILABLE and torch.is_tensor(t):
            return ' - req_grad={}, is_leaf={}, device={}, layout={}'.format(
                t.requires_grad, t.is_leaf, t.device, t.layout)

    if t is None:
        return '%s: None' % name

    shape_str = ''
    if hasattr(t, 'shape'):
        shape_str = '%s ' % str(t.shape)

    if hasattr(t, 'dtype'):
        type_str = '%s' % str(t.dtype)
    else:
        type_str = '{}'.format(type(t))

    name_str = ''
    if name is not None and len(name) > 0:
        name_str = '%s: ' % name

    return ('%s%s(%s) %s %s' %
            (name_str, shape_str, type_str,
             (_get_stats_str() if print_stats else ''),
             (_get_details_str() if detailed else '')))


def resize_square_rgb(img, new_width, nchannels=3):
    if img.shape[0] == new_width and img.shape[1] == new_width:
        return img[:, :, 0:nchannels]
    else:
        return resize(img[:, :, 0:nchannels], (new_width, new_width, img.shape[2]), preserve_range=True)


class RandomPatchGenerator:
    '''
    Returns random patches from an image.
    '''

    def __init__(self, patch_width, patch_range=(0.2, 1.0), center_bias=False):
        """
        patch_width: output patch width
        patch_range: tuple of min/max in fraction of the original image
        center_bias: to bias patch selection to central pixels
        """
        self.patch_width = patch_width
        if patch_range is None:
            self.patch_range = (patch_width, patch_width)
        else:
            self.patch_range = patch_range
        self.center_bias = center_bias

    def get_random_pos(self, rwidth, img_width):
        if not self.center_bias:
            start_row = random.randint(0, img_width - rwidth)
            start_col = random.randint(0, img_width - rwidth)
        else:
            pos = np.random.normal([img_width / 2.0, img_width / 2.0],
                                   [img_width * 0.3, img_width * 0.3]) - rwidth / 2.0
            start_row = int(max(0, min(img_width - rwidth, pos[0])))
            start_col = int(max(0, min(img_width - rwidth, pos[1])))
        return start_row, start_col

    def random_patch(self, img, return_ind=None):
        assert img.shape[0] == img.shape[1]
        img_width = img.shape[0]
        rwidth = random.randint(
            min(img_width, int(self.patch_range[0] * img_width)),
            min(img_width, int(self.patch_range[1] * img_width)))
        start_row, start_col = self.get_random_pos(rwidth, img_width)

        patch = img[start_row:start_row + rwidth, start_col:start_col + rwidth, :]
        patch = resize_square_rgb(patch, self.patch_width)

        if return_ind:
            return start_col, start_row, rwidth, rwidth, patch
        else:
            return patch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main training script.')
    # RUN FLAGS ---------------------------------------------------------------
    parser.add_argument(
        '--input_dir', action='store', type=str, required=True,
        help='Directory with input images.')
    parser.add_argument(
        '--output_dir', action='store', type=str, required=True,
        help='Directory for output augmented images and patches')
    parser.add_argument(
        '--width', action='store', type=int, default=256,
        help='Width to which to resize the image patches.')
    parser.add_argument(
        '--naugment', action='store', type=int, default=30,
        help='Number of augmentatations per image.')
    parser.add_argument(
        '--npatches', action='store', type=int, default=3,
        help='Number of patches per augmented images.')
    args = parser.parse_args()

    default_settings = dict(#strength=0.9,
        xflip=1, # --> good
        scale=1, #1,  --> ok, but probably not needed for random patches
        rotate=1, #1, --> great
        brightness=1, #1,  --> good
        contrast=1, #1,  --> good (occasional gray canvas)
        lumaflip=0, #1,  --> NO (black canvass)
        hue=1, #1,   --> good (but not enough variation)
        saturation=1, #1,  --> good
        imgfilter=0) #1)  --> bad with standard settings, and even std=0.2
    patch_generator = RandomPatchGenerator(args.width)
    augmentor = AugmentPipe(**default_settings)

    image_filenames = sorted(glob.glob(os.path.join(args.input_dir, '*')))
    print('Found {} filenames'.format(len(image_filenames)))

    # img = np.asarray(PIL.Image.open(image_filenames[0])).astype(np.float32) / 255.0
    # patches = tf.convert_to_tensor(
    #     np.stack([patch_generator.random_patch(img).transpose([2, 0, 1]) * 2 / 1.0
    #               for x in range(args.npatches_per_image)]))
    # augmented, _ = augment_pipeline(patches, None, **default_settings)

    pidx = 0
    for fname in image_filenames:
        img = np.asarray(PIL.Image.open(fname))
        img = img.transpose([2, 0, 1]).astype(np.float32) / 255.0 * 2 - 1  # HWC => CHW
        print(tensor_info(img, 'img', print_stats=True))
        images = torch.from_numpy(np.stack([img for x in range(50)]))
        print(tensor_info(images, 'images orig'))
        augmented = augmentor(images)
        print(tensor_info(augmented, 'images augmented'))

        augmented_images = np.clip((augmented.detach().numpy() + 1) / 2.0, 0, 1.0).transpose([0, 2, 3, 1])
        patches = np.stack(
            [patch_generator.random_patch(augmented_images[i, ...])
             for x in range(args.npatches)
             for i in range(augmented_images.shape[0])]) * 255

        for i in range(patches.shape[0]):
            patch = patches[i, ...]
            patch = patch.astype(np.uint8)
            imsave(os.path.join(args.output_dir, 'patch_%05d.png' % pidx), patch)
            pidx += 1

        # img = result[i, ...]
        # print(tensor_info(img, 'result img', print_stats=True))
        # img = img.transpose([1, 2, 0]).astype(np.uint8)
        # imsave(os.path.join(args.output_dir, 'image_%02d.png' % i), img)


# installed requests, click, tqdm, psutil, tensorboard