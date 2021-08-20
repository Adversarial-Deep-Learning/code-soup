import random
from typing import List, Tuple, Union

import cv2
import numpy as np
import scipy
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def seed(value=42):
    """Set random seed for everything.
    Args:
        value (int): Seed
    """
    np.random.seed(value)
    torch.manual_seed(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(value)


class ZooAttackConfig:
    def __init__(
        self,
        binary_search_steps=1,
        max_iterations=10000,
        learning_rate=2e-3,
        abort_early=True,
        targeted=True,
        confidence=0,
        initial_const=0.5,
        use_log=False,
        use_tanh=True,
        reset_adam_after_found=False,
        batch_size=128,
        const=0.5,
        early_stop_iters=0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        use_importance=True,
        use_resize=False,
        init_size=32,
        adam_eps=1e-8,
    ):
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.abort_early = abort_early
        self.targeted = targeted
        self.confidence = confidence
        self.initial_const = initial_const
        self.use_log = use_log
        self.use_tanh = use_tanh
        self.reset_adam_after_found = reset_adam_after_found
        self.batch_size = batch_size
        self.const = const
        self.confidence = confidence
        self.early_stop_iters = early_stop_iters
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.use_importance = use_importance
        self.use_resize = use_resize
        self.init_size = init_size
        self.adam_eps = adam_eps


class ZooAttack:
    def __init__(
        self,
        model: torch.nn.Module,
        config: ZooAttackConfig,
        input_image_shape: List[int],
        device: str,
    ):

        assert len(input_image_shape) == 3, "`input_image_shape` must be of length 3"

        self.config = config
        self.device = device
        self.input_image_shape = input_image_shape

        # Put model in eval mode
        self.model = model.to(device)
        self.model.eval()

        # DUMMIES - Values will be reset during attack
        var_size = np.prod(input_image_shape)  # width * height * num_channels
        self.var_list = np.array(range(0, var_size), dtype=np.int32)

        # Initialize Adam optimizer values
        self.mt_arr = np.zeros(var_size, dtype=np.float32)
        self.vt_arr = np.zeros(var_size, dtype=np.float32)
        self.adam_epochs = np.ones(var_size, dtype=np.int64)

        # Sampling Probabilities
        self.sample_prob = np.ones(var_size, dtype=np.float32) / var_size

    def get_perturbed_image(self, orig_img: torch.tensor, modifier: torch.tensor):
        if self.config.use_tanh:
            return torch.tanh(orig_img + modifier) / 2
        else:
            return orig_img + modifier

    def l2_distance_loss(self, orig_img: torch.tensor, new_img: torch.tensor):

        # assert orig_img.shape == new_img.shape, "Images must be the same shape"

        if new_img.ndim == 3:
            dim = (0, 1, 2)
        else:
            dim = (1, 2, 3)

        if self.config.use_tanh:
            return (
                torch.sum(torch.square(new_img - torch.tanh(orig_img) / 2), dim=dim)
                .detach()
                .cpu()
                .numpy()
            )
        else:
            return (
                torch.sum(torch.square(new_img - orig_img), dim=dim)
                .detach()
                .cpu()
                .numpy()
            )

    def confidence_loss(self, new_img: torch.tensor, target: torch.tensor):

        assert (
            new_img.ndim == 3 or new_img.ndim == 4
        ), "`new_img` must be of shape (N, H, W, C) or (H, W, C)"
        assert (
            target.ndim == 1 or target.ndim == 2
        ), "`target` must be of shape (N,L)  or (L,) where L is number of classes"

        # Reshape to 4D input/output pairs if the input is for a single image
        if new_img.ndim == 3:
            assert (
                target.ndim == 1
            ), "`target` must be of shape (L,) where L is number of classes when single image is passed."
            new_img = new_img.unsqueeze(0)
            target = target.unsqueeze(0)

        new_img = new_img.permute(0, 3, 1, 2)

        model_output = self.model(new_img)
        if self.config.use_log:
            model_output = F.softmax(model_output, dim=1)

        real = torch.sum(target * model_output, dim=1)
        other = torch.max((1 - target) * model_output - (target * 10000), dim=1)[0]

        if self.config.use_log:
            real = torch.log(real + 1e-30)
            other = torch.log(other + 1e-30)

        confidence = torch.tensor(self.config.confidence, device=self.device).type(
            torch.float64
        )

        if self.config.targeted:
            # If targetted, optimize for making the other class most likely
            output = (
                torch.max(torch.zeros_like(real), other - real + confidence)
                .detach()
                .cpu()
                .numpy()
            )
        else:
            # If untargetted, optimize for making this class least likely.
            output = (
                torch.max(torch.zeros_like(real), real - other + confidence)
                .detach()
                .cpu()
                .numpy()
            )

        if new_img.ndim == 3:
            return output.squeeze(0)
        else:
            return output

    def total_loss(
        self, orig_img: torch.tensor, new_img: torch.tensor, target: torch.tensor
    ):
        l2_loss = self.l2_distance_loss(orig_img, new_img)
        confidence_loss = self.confidence_loss(new_img, target)
        return l2_loss + self.config.const * confidence_loss

    # Adapted from original code
    def max_pooling(self, orig_img: np.ndarray, patch_size: int):
        img_pool = np.copy(orig_img)
        img_x = orig_img.shape[0]
        img_y = orig_img.shape[1]
        for i in range(0, img_x, patch_size):
            for j in range(0, img_y, patch_size):
                img_pool[i : i + patch_size, j : j + patch_size] = np.max(
                    orig_img[i : i + patch_size, j : j + patch_size]
                )
        return img_pool

    def zero_order_gradients(self, losses: np.ndarray):
        grad = np.zeros(self.config.batch_size)
        for i in range(self.config.batch_size):
            grad[i] = (losses[i * 2 + 1] - losses[i * 2 + 2]) / 0.0002
        return grad

    def coordinate_adam(
        self, indices: np.ndarray, grad: np.ndarray, modifier: np.ndarray, proj: bool
    ):
        # First moment
        mt = self.mt_arr[indices]
        mt = self.config.adam_beta1 * mt + (1 - self.config.adam_beta1) * grad

        self.mt_arr[indices] = mt

        # Second moment
        vt = self.vt_arr[indices]
        vt = self.config.adam_beta2 * vt + (1 - self.config.adam_beta2) * (grad * grad)

        self.vt_arr[indices] = vt

        epochs = self.adam_epochs[indices]

        # Bias Correction
        mt_hat = mt / (1 - np.power(self.config.adam_beta1, epochs))
        vt_hat = vt / (1 - np.power(self.config.adam_beta2, epochs))

        m = modifier.reshape(-1)
        old_val = m[indices]
        old_val -= (
            self.config.learning_rate
            * mt_hat
            / (np.sqrt(vt_hat) + self.config.adam_eps)
        )
        if proj:
            old_val = np.maximum(
                np.minimum(old_val, self.up[indices]), self.down[indices]
            )
        m[indices] = old_val
        self.adam_epochs[indices] = epochs + 1

    # Adapted from original code
    def get_new_prob(self, modifier, max_pooling_ratio=8, gen_double=False):
        modifier = np.squeeze(modifier)
        old_shape = modifier.shape
        if gen_double:
            new_shape = (old_shape[0] * 2, old_shape[1] * 2, old_shape[2])
        else:
            new_shape = old_shape
        prob = np.empty(shape=new_shape, dtype=np.float32)
        for i in range(modifier.shape[2]):
            image = np.abs(modifier[:, :, i])
            image_pool = self.max_pooling(image, old_shape[0] // max_pooling_ratio)
            if gen_double:
                prob[:, :, i] = np.array(
                    Image.fromarray(image_pool).resize(
                        (new_shape[0], new_shape[1]), Image.NEAREST
                    )
                )
            else:
                prob[:, :, i] = image_pool
        prob /= np.sum(prob)
        return prob

    # Adapted from original code
    def resize_img(
        self,
        small_x,
        small_y,
        num_channels,
        modifier,
        max_pooling_ratio=8,
        reset_only=False,
    ):
        small_single_shape = (small_x, small_y, num_channels)
        if reset_only:
            modifier = np.zeros((1,) + small_single_shape, dtype=np.float32)
        else:
            # run the resize_op once to get the scaled image
            prev_modifier = np.copy(modifier)
            modifier = cv2.resize(
                modifier, (small_x, small_y), interpolation=cv2.INTER_LINEAR
            )

        # prepare the list of all valid variables
        var_size = np.prod(small_single_shape)
        self.var_list = np.array(range(0, var_size), dtype=np.int32)
        # ADAM status
        self.mt_arr = np.zeros(var_size, dtype=np.float32)
        self.vt_arr = np.zeros(var_size, dtype=np.float32)
        self.adam_epochs = np.ones(var_size, dtype=np.int32)
        # update sample probability
        if reset_only:
            self.sample_prob = np.ones(var_size, dtype=np.float32) / var_size
        else:
            self.sample_prob = self.get_new_prob(prev_modifier, max_pooling_ratio, True)
            self.sample_prob = self.sample_prob.reshape(var_size)

        return modifier

    # def single_step(self, iter, modifier, orig_img, target):
    #     var = np.repeat(modifier, self.config.batch_size * 2 + 1, axis=0)
    #     print(var.shape)
    #     var_size = modifier.size

    #     print(self.var_list.size)
    #     # Select indices for current iteration
    #     if self.config.use_importance:
    #         var_indice = np.random.choice(
    #             self.var_list.size,
    #             self.config.batch_size,
    #             replace=False,
    #             p=self.sample_prob,
    #         )
    #     else:
    #         var_indice = np.random.choice(
    #             self.var_list.size, self.config.batch_size, replace=False
    #         )
    #     indice = self.var_list[var_indice]

    #     for i in range(self.config.batch_size):
    #         var[i * 2 + 1].reshape(-1)[indice[i]] += 0.0001
    #         var[i * 2 + 2].reshape(-1)[indice[i]] -= 0.0001

    #     new_img = self.get_perturbed_image(orig_img, var)
    #     losses = self.total_loss(orig_img, new_img, target)
