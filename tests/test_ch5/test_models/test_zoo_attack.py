import unittest

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

from code_soup.ch5.models.zoo_attack import ZooAttack, ZooAttackConfig


class TestZooAttackConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = ZooAttackConfig(learning_rate=1e-2, batch_size=12, init_size=32)

    def test_learning_rate(self):
        self.assertEqual(self.config.learning_rate, 1e-2)

    def test_update(self):
        self.config.learning_rate = 1e-3
        self.assertEqual(self.config.learning_rate, 1e-3)

    def test_equality(self):
        self.assertEqual(self.config.learning_rate, 1e-2)
        self.assertEqual(self.config.batch_size, 12)
        self.assertEqual(self.config.init_size, 32)


class TestZooAttackBasic(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:

        cls.orig_img = torch.tensor(
            [
                [
                    [-1.6505998, -1.0305759, 1.0229983],
                    [-0.49261865, 1.0394262, -2.0290275],
                    [0.21951008, -2.1673787, -0.38990623],
                    [-0.2866124, 1.0799991, -0.11442444],
                ],
                [
                    [-0.7052935, -0.5529446, 0.26524046],
                    [-1.0540642, 0.6887131, 1.6723113],
                    [1.1097006, 2.1335971, 0.9231482],
                    [0.37910375, -0.12366215, -0.25093704],
                ],
                [
                    [-1.9404864, -1.3078933, 0.88476175],
                    [0.35099706, -1.254437, 0.05408821],
                    [0.7342985, -0.43663985, 0.11520719],
                    [-0.07479854, -2.5859993, 1.4102333],
                ],
                [
                    [0.21304935, -0.3496548, -0.19856042],
                    [-0.434919, -0.27774376, 1.1471609],
                    [1.4504786, 0.67261624, -0.23560882],
                    [1.0592173, 0.6655428, 1.1890292],
                ],
            ],
            dtype=torch.float32,
        )

        cls.modifier = np.array(
            [
                [
                    [-0.21563086, 0.54629284, 1.0879989],
                    [-0.17234534, 0.37302095, 1.5072422],
                    [-0.14709516, -0.08446954, -1.0199878],
                    [-0.46581882, 0.41346493, -1.6357177],
                ],
                [
                    [0.97039294, -0.46038368, -0.5377948],
                    [-0.08285582, -1.4017423, -0.6447743],
                    [-0.6031785, -2.003339, -0.01103557],
                    [0.41714168, -1.94303, 0.6685426],
                ],
                [
                    [-0.83851266, 0.79823476, 0.2532903],
                    [-0.76351106, 0.90984505, 1.331635],
                    [-1.1300149, -0.8444777, -2.2185612],
                    [1.0166003, 0.9233805, 0.98315567],
                ],
                [
                    [-0.88205546, -0.3438152, -0.36559045],
                    [0.56274384, 1.5836877, -1.2370849],
                    [1.4234338, -0.5929535, -1.3011148],
                    [0.84160084, 0.90161383, 0.80880517],
                ],
            ],
            dtype=np.float32,
        )

        cls.labels = torch.tensor([0, 1])

        cls.config = ZooAttackConfig(batch_size=cls.orig_img.shape[0])
        cls.model = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=1, kernel_size=2, padding=0, bias=False
            ),
            nn.Flatten(),
            nn.Linear(4 * 4 * 3, 2, bias=False),
        )

        with torch.no_grad():
            cls.model[0].weight = nn.Parameter(
                torch.tensor(
                    [
                        [
                            [[0.18992287], [-0.6111586], [-0.41560256]],
                            [[0.19819254], [0.06157357], [-0.29873127]],
                        ],
                        [
                            [[0.08528781], [-0.4988662], [0.51414317]],
                            [[0.5520558], [0.35638297], [0.29052997]],
                        ],
                    ]
                ).permute(3, 2, 0, 1)
            )
            cls.model[2].weight = nn.Parameter(
                torch.tensor(
                    [
                        [0.26311237, 0.7238547],
                        [-0.2869757, -0.6140047],
                        [-0.11846703, -0.57517225],
                        [-0.72543985, 0.6393444],
                        [0.45188862, 0.35718697],
                        [-0.7197881, 0.17988789],
                        [0.18161213, 0.32464463],
                        [0.37511164, 0.07291293],
                        [-0.27989575, -0.37013885],
                    ]
                ).T
            )
        cls.attack = ZooAttack(
            model=cls.model,
            config=cls.config,
            input_image_shape=cls.orig_img.shape,
            device="cpu:0",
        )

    def test_get_perturbed_image(self):
        perturbed_image = self.attack.get_perturbed_image(self.orig_img, self.modifier)
        self.assertEqual(perturbed_image.shape, self.orig_img.shape)

        output = torch.tanh(self.orig_img + self.modifier) / 2
        self.assertTrue(torch.allclose(perturbed_image, output, atol=1e-5))

        # Integration Test
        self.assertTrue(
            torch.allclose(
                perturbed_image,
                torch.tensor(
                    [
                        [
                            [-0.47662562, -0.22483358, 0.4855427],
                            [-0.2908287, 0.44400635, -0.23953833],
                            [0.0361443, -0.4890532, -0.44373578],
                            [-0.3182986, 0.45198008, -0.47069582],
                        ],
                        [
                            [0.12952949, -0.38356757, -0.13300003],
                            [-0.4066872, -0.30628642, 0.38645932],
                            [0.23361546, 0.06476317, 0.36107236],
                            [0.33096626, -0.48422426, 0.19745564],
                        ],
                        [
                            [-0.49615836, -0.23483951, 0.40687856],
                            [-0.19530259, -0.16578534, 0.44111317],
                            [-0.18813889, -0.42839125, -0.4853233],
                            [0.3680245, -0.46528453, 0.49172965],
                        ],
                        [
                            [-0.2921629, -0.3001033, -0.25552535],
                            [0.06356658, 0.43162277, -0.04484118],
                            [0.49682045, 0.03974732, -0.4557841],
                            [0.4781537, 0.4582861, 0.4819371],
                        ],
                    ],
                ),
            )
        )

    def test_l2_distance_loss(self):
        new_img = self.attack.get_perturbed_image(self.orig_img, self.modifier)
        temp_orig_img = self.orig_img.unsqueeze(0)
        temp_new_img = new_img.unsqueeze(0)
        loss = self.attack.l2_distance_loss(temp_orig_img, temp_new_img)
        self.assertEqual(loss.shape[0], temp_orig_img.shape[0])

        # Integration Test
        self.assertTrue(np.allclose(np.array([3.7336116]), loss, atol=1e-5))

    def test_confidence_loss(self):
        new_img = self.attack.get_perturbed_image(self.orig_img, self.modifier)
        temp_new_img = new_img.unsqueeze(0)
        labels = self.labels.unsqueeze(0)
        loss = self.attack.confidence_loss(temp_new_img, labels)

        self.assertEqual(loss.shape[0], temp_new_img.shape[0])

        # Integration Test
        self.assertTrue(np.allclose(np.array([0.2148518]), loss, atol=1e-5))

    def test_zero_order_gradients(self):
        losses = np.random.randn(2 * self.config.batch_size + 1)
        grads = self.attack.zero_order_gradients(losses)
        self.assertEqual(grads.shape, (self.config.batch_size,))

    def test_coordinate_adam(self):
        # Integration Test
        indices = np.array([15, 24, 32, 45])

        grad = np.array([2000.0, 3500.0, -1000.0, -1500.0])

        proj = not self.config.use_tanh

        modifier = deepcopy(self.modifier)

        self.attack.coordinate_adam(indices, grad, modifier, proj)

        self.assertTrue(
            np.allclose(
                self.attack.mt_arr,
                np.array(
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        200.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        350.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        -100.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        -150.0,
                        0.0,
                        0.0,
                    ],
                ),
                atol=1e-5,
            )
        )

        self.assertTrue(
            np.allclose(
                self.attack.vt_arr,
                np.array(
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4000.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        12250.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1000.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        2250.0,
                        0.0,
                        0.0,
                    ],
                ),
                atol=1e-5,
            )
        )

        self.assertTrue(
            np.allclose(
                modifier,
                np.array(
                    [
                        [
                            [-0.21563086, 0.54629284, 1.0879989],
                            [-0.17234534, 0.37302095, 1.5072422],
                            [-0.14709516, -0.08446954, -1.0199878],
                            [-0.46581882, 0.41346493, -1.6357177],
                        ],
                        [
                            [0.97039294, -0.46038368, -0.5377948],
                            [-0.08485582, -1.4017423, -0.6447743],
                            [-0.6031785, -2.003339, -0.01103557],
                            [0.41714168, -1.94303, 0.6685426],
                        ],
                        [
                            [-0.84051266, 0.79823476, 0.2532903],
                            [-0.76351106, 0.90984505, 1.331635],
                            [-1.1300149, -0.8444777, -2.2165612],
                            [1.0166003, 0.9233805, 0.98315567],
                        ],
                        [
                            [-0.88205546, -0.3438152, -0.36559045],
                            [0.56274384, 1.5836877, -1.2370849],
                            [1.4234338, -0.5929535, -1.3011148],
                            [0.84360084, 0.90161383, 0.80880517],
                        ],
                    ],
                ),
                atol=1e-5,
            )
        )

        self.assertTrue(
            (
                self.attack.adam_epochs
                == np.array(
                    [
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        2,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        2,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        2,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        2,
                        1,
                        1,
                    ],
                )
            ).all(),
        )
