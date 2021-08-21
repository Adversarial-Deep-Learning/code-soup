import random
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


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
        resize_iter_1=2000,
        resize_iter_2=10000,
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
        self.early_stop_iters = (
            early_stop_iters if early_stop_iters != 0 else max_iterations // 10
        )
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.use_importance = use_importance
        self.use_resize = use_resize
        self.init_size = init_size
        self.adam_eps = adam_eps
        self.resize_iter_1 = resize_iter_1
        self.resize_iter_2 = resize_iter_2


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

        if self.config.use_tanh is False and self.config.use_resize is True:
            # NOTE: self.up and self.down need to be updated dynamically to match the modifier shape.
            # Original Implementation is possibly flawed in this aspect.
            raise NotImplementedError(
                "Current implementation does not support `use_tanh` as `False` and `use_resize` as `True` at the same time."
            )

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

    def get_perturbed_image(self, orig_img: torch.tensor, modifier: np.ndarray):

        assert orig_img.ndim == 4, "`orig_img` must be a 4D tensor"
        assert modifier.ndim == 4, "`modifier` must be a 4D tensor"

        b = modifier.shape[0]
        x = orig_img.shape[1]
        y = orig_img.shape[2]
        z = orig_img.shape[3]

        new_modifier = np.zeros((b, x, y, z), dtype=np.float32)

        if x != modifier.shape[1] or y != modifier.shape[2]:
            for k, v in enumerate(modifier):
                new_modifier[k, :, :, :] = cv2.resize(
                    modifier[k, :, :, :],
                    (x, y),
                    interpolation=cv2.INTER_LINEAR,
                )
        else:
            new_modifier = modifier

        if self.config.use_tanh:
            return torch.tanh(orig_img + new_modifier) / 2
        else:
            return orig_img + new_modifier

    def l2_distance_loss(self, orig_img: torch.tensor, new_img: torch.tensor):

        # assert orig_img.shape == new_img.shape, "Images must be the same shape"

        assert new_img.ndim == 4, "`new_img` must be a 4D tensor"
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
        assert new_img.ndim == 4, "`new_img` must be of shape (N, H, W, C)"
        assert (
            target.ndim == 2
        ), "`target` must be of shape (N,L) where L is number of classes"

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

        return output, model_output

    def total_loss(
        self,
        orig_img: torch.tensor,
        new_img: torch.tensor,
        target: torch.tensor,
        const: int,
    ):
        l2_loss = self.l2_distance_loss(orig_img, new_img)
        confidence_loss, model_output = self.confidence_loss(new_img, target)
        return (
            l2_loss + const * confidence_loss,
            l2_loss,
            confidence_loss,
            model_output,
        )

    # Adapted from original code
    def max_pooling(self, modifier: np.ndarray, patch_size: int):

        assert modifier.ndim == 2, "`modifier` must be a 2D array"
        img_pool = np.copy(modifier)
        img_x = modifier.shape[0]
        img_y = modifier.shape[1]
        for i in range(0, img_x, patch_size):
            for j in range(0, img_y, patch_size):
                img_pool[i : i + patch_size, j : j + patch_size] = np.max(
                    modifier[i : i + patch_size, j : j + patch_size]
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

        return m.reshape(modifier.shape)

    # Adapted from original code
    def get_new_prob(
        self, modifier: np.ndarray, max_pooling_ratio: int = 8, gen_double: bool = False
    ):
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

        # NOTE: This is here to handle all zeros input
        if np.sum(prob) != 0:
            prob /= np.sum(prob)
        else:
            prob = np.ones(shape=new_shape, dtype=np.float32)
            prob /= np.sum(prob)

        return prob

    # Adapted from original code
    def resize_img(
        self,
        small_x: int,
        small_y: int,
        num_channels: int,
        modifier: np.ndarray,
        max_pooling_ratio: int = 8,
        reset_only: bool = False,
    ):

        small_single_shape = (small_x, small_y, num_channels)

        new_modifier = np.zeros((1,) + small_single_shape, dtype=np.float32)
        if not reset_only:
            # run the resize_op once to get the scaled image
            assert modifier.ndim == 4, "Expected 4D array as modifier"
            prev_modifier = np.copy(modifier)
            for k, v in enumerate(modifier):
                new_modifier[k, :, :, :] = cv2.resize(
                    modifier[k, :, :, :],
                    (small_x, small_y),
                    interpolation=cv2.INTER_LINEAR,
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

        return new_modifier

    def single_step(
        self,
        modifier: np.ndarray,
        orig_img: torch.tensor,
        target: torch.tensor,
        const: int,
        max_pooling_ratio: int = 8,
        var_indice: list = None,
    ):

        assert modifier.ndim == 4, "Expected 4D array as modifier"
        assert modifier.shape[0] == 1, "Expected 1 batch for modifier"
        assert target.ndim == 2, "Expected 2D tensor as target"

        var = np.repeat(modifier, self.config.batch_size * 2 + 1, axis=0)
        var_size = modifier.size

        # Select indices for current iteration

        if var_indice is None:
            if self.config.use_importance:
                var_indice = np.random.choice(
                    self.var_list.size,
                    self.config.batch_size,
                    replace=False,
                    p=self.sample_prob,
                )
            else:
                var_indice = np.random.choice(
                    self.var_list.size, self.config.batch_size, replace=False
                )
        indices = self.var_list[var_indice]

        for i in range(self.config.batch_size):
            var[i * 2 + 1].reshape(-1)[indices[i]] += 0.0001
            var[i * 2 + 2].reshape(-1)[indices[i]] -= 0.0001

        new_img = self.get_perturbed_image(orig_img, var)
        losses, l2_losses, confidence_losses, model_output = self.total_loss(
            orig_img, new_img, target, const
        )

        if modifier.shape[1] > self.config.init_size:
            self.sample_prob = self.get_new_prob(
                modifier, max_pooling_ratio=max_pooling_ratio
            )
            self.sample_prob = self.sample_prob.reshape(var_size)

        grad = self.zero_order_gradients(losses)

        # Modifier is updated here, so is adam epochs, mt_arr, and vt_arr
        modifier = self.coordinate_adam(
            indices, grad, modifier, not self.config.use_tanh
        )

        return (
            losses[0],
            l2_losses[0],
            confidence_losses[0],
            model_output[0].detach().numpy(),
            new_img[0],
            modifier,
        )

    def attack(
        self,
        orig_img: np.ndarray,
        target: np.ndarray,
        modifier_init: np.ndarray = None,
        max_pooling_ratio: int = 8,
    ):
        def compare(x, y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                if self.config.targeted:
                    x[y] -= self.config.confidence
                else:
                    x[y] += self.config.confidence
                x = np.argmax(x)
            if self.config.targeted:
                return x == y
            else:
                return x != y

        assert orig_img.ndim == 3, "Expected 3D array as image"
        assert target.ndim == 1, "Expected 1D array as target"

        if modifier_init is not None:
            assert modifier_init.ndim == 3, "Expected 3D array as modifier"
            modifier = modifier_init.copy()
        else:
            if self.config.use_resize:
                modifier = self.resize_img(
                    self.config.init_size,
                    self.config.init_size,
                    3,
                    modifier_init,
                    max_pooling_ratio,
                    reset_only=True,
                )
            else:
                modifier = np.zeros(orig_img.shape, dtype=np.float32)

        if self.config.use_tanh:
            orig_img = np.arctanh(orig_img * 1.999999)

        var_size = np.prod(orig_img.shape)  # width * height * num_channels
        self.var_list = np.array(range(0, var_size), dtype=np.int32)

        # Initialize Adam optimizer values
        self.mt_arr = np.zeros(var_size, dtype=np.float32)
        self.vt_arr = np.zeros(var_size, dtype=np.float32)
        self.adam_epochs = np.ones(var_size, dtype=np.int64)
        self.up = np.zeros(var_size, dtype=np.float32)
        self.down = np.zeros(var_size, dtype=np.float32)

        # Sampling Probabilities
        self.sample_prob = np.ones(var_size, dtype=np.float32) / var_size

        low = 0.0
        mid = self.config.initial_const
        high = 1e10

        if not self.config.use_tanh:
            self.up = 0.5 - orig_img.reshape(-1)
            self.down = -0.5 - orig_img.reshape(-1)

        outer_best_const = mid
        outer_best_l2 = 1e10
        outer_best_score = -1
        outer_best_adv = orig_img

        # Make Everything 4D and Tensorize
        orig_img = torch.from_numpy(orig_img).unsqueeze(0).to(self.device)
        target = torch.from_numpy(target).unsqueeze(0).to(self.device)
        modifier = modifier.reshape((-1,) + modifier.shape)

        for outer_step in range(self.config.binary_search_steps):

            best_l2 = 1e10
            best_score = -1

            # NOTE: In the original implemenation there is a step to move mid to high
            # at last step with some condition

            prev = 1e6
            last_confidence_loss = 1.0

            if modifier_init is not None:
                assert modifier_init.ndim == 3, "Expected 3D array as modifier"
                modifier = modifier_init.copy()
                modifier = modifier.reshape((-1,) + modifier.shape)
            else:
                if self.config.use_resize:
                    modifier = self.resize_img(
                        self.config.init_size,
                        self.config.init_size,
                        3,
                        modifier_init,
                        max_pooling_ratio,
                        reset_only=True,
                    )

                else:
                    modifier = np.zeros(orig_img.shape, dtype=np.float32)

            self.mt_arr.fill(0.0)
            self.vt_arr.fill(0.0)
            self.adam_epochs.fill(1)
            stage = 0
            eval_costs = 0

            # NOTE: Original code allows for a custom start point in iterations
            for iter in range(0, self.config.max_iterations):
                if self.config.use_resize:
                    if iter == self.config.resize_iter_1:
                        modifier = self.resize_img(
                            self.config.init_size * 2,
                            self.config.init_size * 2,
                            3,
                            modifier,
                            max_pooling_ratio,
                        )
                    if iter == self.config.resize_iter_2:
                        modifier = self.resize_img(
                            self.config.init_size * 4,
                            self.config.init_size * 4,
                            3,
                            modifier,
                            max_pooling_ratio,
                        )
                if iter % (self.config.max_iterations // 10) == 0:
                    new_img = self.get_perturbed_image(orig_img, modifier)
                    (
                        total_losses,
                        l2_losses,
                        confidence_losses,
                        model_output,
                    ) = self.total_loss(orig_img, new_img, target, mid)
                    print(
                        f"iter = {iter}, cost = {eval_costs},  size = {modifier.shape}, "
                        f"total_loss = {total_losses[0]:.5g}, l2_loss = {l2_losses[0]:.5g}, "
                        f"confidence_loss = {confidence_losses[0]:.5g}"
                    )

                (
                    total_loss,
                    l2_loss,
                    confidence_loss,
                    model_output,
                    adv_img,
                    modifier,
                ) = self.single_step(
                    modifier, orig_img, target, mid, max_pooling_ratio=max_pooling_ratio
                )

                eval_costs += self.config.batch_size

                if (
                    confidence_loss == 0.0
                    and last_confidence_loss != 0.0
                    and stage == 0
                ):

                    if self.config.reset_adam_after_found:
                        print("Resetting Adam")
                        self.mt_arr.fill(0.0)
                        self.vt_arr.fill(0.0)
                        self.adam_epochs.fill(1)
                    print("Setting Stage to 1")
                    stage = 1

                last_confidence_loss = confidence_loss

                if self.config.abort_early and iter % self.config.early_stop_iters == 0:
                    if total_loss > prev * 0.9999:
                        print("Early stopping because there is no improvement")
                        break
                    prev = total_loss

                if l2_loss < best_l2 and compare(model_output, np.argmax(target[0])):
                    best_l2 = l2_loss
                    best_score = np.argmax(model_output)

                if l2_loss < outer_best_l2 and compare(
                    model_output, np.argmax(target[0])
                ):
                    outer_best_l2 = l2_loss
                    outer_best_score = np.argmax(model_output)
                    outer_best_adv = adv_img
                    outer_best_const = mid

            if compare(best_score, np.argmax(target[0])) and best_score != -1:

                print("Old Constant: ", mid)
                high = min(high, mid)
                if high < 1e9:
                    mid = (low + high) / 2
                print("New Constant: ", mid)
            else:
                print("Old Constant: ", mid)
                low = max(low, mid)
                if high < 1e9:
                    mid = (low + high) / 2
                else:
                    mid *= 10
                print("new constant: ", mid)

        return outer_best_adv, outer_best_const
