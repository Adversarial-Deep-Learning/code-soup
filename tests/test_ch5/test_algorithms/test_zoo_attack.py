import unittest
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torchvision

from code_soup.ch5 import ZooAttack


class TestZooAttack(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:

        cls.orig_img = torch.tensor(
            [
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
                ]
            ],
            dtype=torch.float32,
        )

        cls.modifier = np.array(
            [
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
                ]
            ],
            dtype=np.float32,
        )

        cls.labels = torch.tensor([[0, 1]])

        cls.config = {
            "binary_search_steps": 1,
            "max_iterations": 100,
            "learning_rate": 2e-3,
            "abort_early": True,
            "targeted": True,
            "confidence": 0,
            "initial_const": 0.5,
            "use_log": False,
            "use_tanh": True,
            "reset_adam_after_found": True,
            "batch_size": 4,
            "const": 0.5,
            "early_stop_iters": 0,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "use_importance": True,
            "use_resize": False,
            "init_size": 4,
            "adam_eps": 1e-8,
            "resize_iter_1": 2000,
            "resize_iter_2": 10000,
        }
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
            input_image_shape=cls.orig_img.shape[1:],
            device="cpu:0",
        )

    def test_get_perturbed_image(self):
        perturbed_image = self.attack.get_perturbed_image(self.orig_img, self.modifier)
        self.assertEqual(perturbed_image.shape, self.orig_img.shape)

        output = torch.tanh(self.orig_img + self.modifier) / 2
        self.assertTrue(torch.allclose(perturbed_image, output, atol=1e-5))

        # Without Tanh
        attack = deepcopy(self.attack)
        attack.config["use_tanh"] = False

        perturbed_image_2 = attack.get_perturbed_image(self.orig_img, self.modifier)
        self.assertEqual(perturbed_image_2.shape, self.orig_img.shape)

        output_2 = self.orig_img + torch.from_numpy(self.modifier)
        self.assertTrue(torch.allclose(perturbed_image_2, output_2, atol=1e-5))

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
        loss = self.attack.l2_distance_loss(self.orig_img, new_img)
        self.assertEqual(loss.shape[0], self.orig_img.shape[0])

        # Without Tanh
        attack = deepcopy(self.attack)
        attack.config["use_tanh"] = False

        new_img_2 = attack.get_perturbed_image(self.orig_img, self.modifier)
        loss_2 = attack.l2_distance_loss(self.orig_img, new_img_2)
        self.assertEqual(loss_2.shape[0], self.orig_img.shape[0])

        # Integration Test
        self.assertTrue(np.allclose(np.array([3.7336116]), loss, atol=1e-5))

    def test_confidence_loss(self):
        new_img = self.attack.get_perturbed_image(self.orig_img, self.modifier)
        loss, model_output = self.attack.confidence_loss(new_img, self.labels)

        self.assertEqual(loss.shape[0], new_img.shape[0])

        # With Log and Untargeted
        attack = deepcopy(self.attack)
        attack.config["use_log"] = True
        attack.config["targeted"] = False

        new_img_2 = attack.get_perturbed_image(self.orig_img, self.modifier)
        loss_2, model_output = attack.confidence_loss(new_img_2, self.labels)

        self.assertEqual(loss_2.shape[0], new_img_2.shape[0])

        # Integration Test
        self.assertTrue(np.allclose(np.array([0.2148518]), loss, atol=1e-5))

    def test_zero_order_gradients(self):
        losses = np.random.randn(2 * self.config["batch_size"] + 1)
        grads = self.attack.zero_order_gradients(losses)
        self.assertEqual(grads.shape, (self.config["batch_size"],))

    def test_total_loss(self):
        new_img = self.attack.get_perturbed_image(self.orig_img, self.modifier)

        loss, l2_loss, confidence_loss, model_output = self.attack.total_loss(
            self.orig_img, new_img, self.labels, self.config["initial_const"]
        )
        self.assertEqual(loss.shape[0], self.orig_img.shape[0])

        self.assertEqual(confidence_loss.shape[0], self.orig_img.shape[0])

        self.assertEqual(l2_loss.shape[0], self.orig_img.shape[0])

        self.assertEqual(model_output.shape, self.labels.shape)

    def test_max_pooling(self):
        modifier = self.modifier[0][:, :, 0]
        pooled_output = self.attack.max_pooling(modifier, 2)
        self.assertEqual(pooled_output.shape, modifier.shape)

        # Integration Test
        self.assertTrue(
            np.allclose(
                pooled_output,
                np.array(
                    [
                        [
                            0.97039294,
                            0.97039294,
                            0.41714168,
                            0.41714168,
                        ],
                        [
                            0.97039294,
                            0.97039294,
                            0.41714168,
                            0.41714168,
                        ],
                        [
                            0.56274384,
                            0.56274384,
                            1.4234338,
                            1.4234338,
                        ],
                        [
                            0.56274384,
                            0.56274384,
                            1.4234338,
                            1.4234338,
                        ],
                    ]
                ),
                atol=1e-5,
            )
        )

    def test_coordinate_adam(self):

        # With Proj True
        attack = deepcopy(self.attack)
        attack.config["use_tanh"] = False
        attack.up = 0.5 - self.orig_img.numpy().reshape(-1)
        attack.down = -0.5 - self.orig_img.numpy().reshape(-1)
        indices = np.array([15, 24, 32, 45])

        grad = np.array([2000.0, 3500.0, -1000.0, -1500.0])

        proj = not attack.config["use_tanh"]

        modifier = deepcopy(self.modifier)

        attack.coordinate_adam(indices, grad, modifier, proj)

        self.assertTrue(
            np.allclose(
                attack.mt_arr,
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
                attack.vt_arr,
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
                            [0.55406415, -1.4017423, -0.6447743],
                            [-0.6031785, -2.003339, -0.01103557],
                            [0.41714168, -1.94303, 0.6685426],
                        ],
                        [
                            [1.4404864, 0.79823476, 0.2532903],
                            [-0.76351106, 0.90984505, 1.331635],
                            [-1.1300149, -0.8444777, -0.6152072],
                            [1.0166003, 0.9233805, 0.98315567],
                        ],
                        [
                            [-0.88205546, -0.3438152, -0.36559045],
                            [0.56274384, 1.5836877, -1.2370849],
                            [1.4234338, -0.5929535, -1.3011148],
                            [-0.55921733, 0.90161383, 0.80880517],
                        ],
                    ]
                ),
                atol=1e-5,
            )
        )

        self.assertTrue(
            (
                attack.adam_epochs
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

        # Integration Test
        # Without Proj True
        attack = deepcopy(self.attack)
        indices = np.array([15, 24, 32, 45])

        grad = np.array([2000.0, 3500.0, -1000.0, -1500.0])

        proj = not attack.config["use_tanh"]

        modifier = deepcopy(self.modifier)

        attack.coordinate_adam(indices, grad, modifier, proj)

        self.assertTrue(
            np.allclose(
                attack.mt_arr,
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
                attack.vt_arr,
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
                attack.adam_epochs
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

    def test_get_new_prob(self):
        probs = self.attack.get_new_prob(self.modifier, 2)
        self.assertEqual(probs.shape, self.modifier.shape[1:])

        # Integration Test
        self.assertTrue(
            np.allclose(
                probs,
                np.array(
                    [
                        [
                            [0.01471687, 0.02125866, 0.02285866],
                            [0.01471687, 0.02125866, 0.02285866],
                            [0.00914774, 0.03038241, 0.0248071],
                            [0.00914774, 0.03038241, 0.0248071],
                        ],
                        [
                            [0.01471687, 0.02125866, 0.02285866],
                            [0.01471687, 0.02125866, 0.02285866],
                            [0.00914774, 0.03038241, 0.0248071],
                            [0.00914774, 0.03038241, 0.0248071],
                        ],
                        [
                            [0.01337715, 0.02401802, 0.02019542],
                            [0.01337715, 0.02401802, 0.02019542],
                            [0.02158763, 0.01400388, 0.03364644],
                            [0.02158763, 0.01400388, 0.03364644],
                        ],
                        [
                            [0.01337715, 0.02401802, 0.02019542],
                            [0.01337715, 0.02401802, 0.02019542],
                            [0.02158763, 0.01400388, 0.03364644],
                            [0.02158763, 0.01400388, 0.03364644],
                        ],
                    ]
                ),
                atol=1e-5,
            )
        )

    def test_resize_img(self):
        # Reset Only True

        attack = deepcopy(self.attack)
        new_modifier = attack.resize_img(8, 8, 3, self.modifier, 2, reset_only=True)

        self.assertEqual(new_modifier.shape, (1, 8, 8, 3))

        self.assertEqual(attack.sample_prob.shape, np.prod(8 * 8 * 3))

        # Reset Only False
        attack = deepcopy(self.attack)
        new_modifier = attack.resize_img(8, 8, 3, self.modifier, 2)

        self.assertEqual(new_modifier.shape, (1, 8, 8, 3))

        self.assertEqual(attack.sample_prob.shape, np.prod(8 * 8 * 3))

        # Integration Test
        self.assertTrue(
            np.allclose(
                new_modifier,
                np.array(
                    [
                        [
                            [
                                [-0.21563086, 0.54629284, 1.0879989],
                                [-0.20480949, 0.50297487, 1.1928097],
                                [-0.18316671, 0.41633892, 1.4024314],
                                [-0.16603279, 0.25864834, 0.8754347],
                                [-0.15340771, 0.02990307, -0.38818032],
                                [-0.22677608, 0.04001407, -1.1739203],
                                [-0.3861379, 0.28898132, -1.4817853],
                                [-0.46581882, 0.41346493, -1.6357177],
                            ],
                            [
                                [0.0808751, 0.2946237, 0.68155044],
                                [0.02316307, 0.2033003, 0.7534723],
                                [-0.09226094, 0.02065352, 0.89731616],
                                [-0.17775872, -0.19404912, 0.53499115],
                                [-0.23333023, -0.44080764, -0.3335028],
                                [-0.25710666, -0.4670549, -0.8407254],
                                [-0.24908802, -0.2727908, -0.98667693],
                                [-0.2450787, -0.17565879, -1.0596527],
                            ],
                            [
                                [0.673887, -0.20871457, -0.1313464],
                                [0.47910815, -0.39604884, -0.12520233],
                                [0.0895506, -0.77071726, -0.11291426],
                                [-0.20121056, -1.099444, -0.14589605],
                                [-0.3931753, -1.3822291, -0.22414777],
                                [-0.31776786, -1.481193, -0.17433581],
                                [0.02501175, -1.396335, 0.00353971],
                                [0.19640155, -1.3539063, 0.09247744],
                            ],
                            [
                                [0.51816654, -0.14572906, -0.34002355],
                                [0.32536998, -0.3152582, -0.29268563],
                                [-0.0602231, -0.6543164, -0.19800991],
                                [-0.3734866, -1.04629, -0.25373322],
                                [-0.61442065, -1.4911791, -0.45985574],
                                [-0.4094141, -1.5918247, -0.23538877],
                                [0.24153282, -1.3482264, 0.4196677],
                                [0.56700635, -1.2264273, 0.7471959],
                            ],
                            [
                                [-0.38628626, 0.48358017, 0.05551901],
                                [-0.4380515, 0.4456721, 0.25102246],
                                [-0.54158205, 0.36985612, 0.6420292],
                                [-0.6945869, -0.03458703, 0.21147956],
                                [-0.89706624, -0.76765776, -1.0406268],
                                [-0.5320455, -0.7989503, -1.0238843],
                                [0.4004752, -0.12846482, 0.2617069],
                                [0.8667356, 0.20677787, 0.9045024],
                            ],
                            [
                                [-0.8493984, 0.51272225, 0.09857011],
                                [-0.7450356, 0.6541181, 0.24629137],
                                [-0.5363101, 0.93690985, 0.54173374],
                                [-0.44687366, 0.6133301, 0.01979139],
                                [-0.4767264, -0.31662107, -1.319536],
                                [-0.12552696, -0.35671276, -1.2570076],
                                [0.60672456, 0.493055, 0.20737618],
                                [0.9728504, 0.9179388, 0.93956804],
                            ],
                            [
                                [-0.87116975, -0.05830276, -0.21087027],
                                [-0.5955823, 0.3100797, -0.3068789],
                                [-0.04440734, 1.0468445, -0.49889636],
                                [0.369653, 0.89746165, -0.8287977],
                                [0.6465987, -0.1380692, -1.2965835],
                                [0.8101414, -0.26511204, -0.9347591],
                                [0.8602809, 0.51633304, 0.25667554],
                                [0.8853507, 0.9070555, 0.8523928],
                            ],
                            [
                                [-0.88205546, -0.3438152, -0.36559045],
                                [-0.52085567, 0.13806051, -0.583464],
                                [0.20154402, 1.1018119, -1.0192113],
                                [0.7779163, 1.0395274, -1.2530923],
                                [1.2082613, -0.04879326, -1.2851074],
                                [1.2779756, -0.21931165, -0.7736348],
                                [0.98705906, 0.52797204, 0.28132522],
                                [0.84160084, 0.90161383, 0.80880517],
                            ],
                        ]
                    ]
                ),
                atol=1e-5,
            )
        )

        self.assertTrue(
            np.allclose(
                attack.sample_prob,
                np.array(
                    [
                        0.00367922,
                        0.00531467,
                        0.00571467,
                        0.00367922,
                        0.00531467,
                        0.00571467,
                        0.00367922,
                        0.00531467,
                        0.00571467,
                        0.00367922,
                        0.00531467,
                        0.00571467,
                        0.00228693,
                        0.0075956,
                        0.00620178,
                        0.00228693,
                        0.0075956,
                        0.00620178,
                        0.00228693,
                        0.0075956,
                        0.00620178,
                        0.00228693,
                        0.0075956,
                        0.00620178,
                        0.00367922,
                        0.00531467,
                        0.00571467,
                        0.00367922,
                        0.00531467,
                        0.00571467,
                        0.00367922,
                        0.00531467,
                        0.00571467,
                        0.00367922,
                        0.00531467,
                        0.00571467,
                        0.00228693,
                        0.0075956,
                        0.00620178,
                        0.00228693,
                        0.0075956,
                        0.00620178,
                        0.00228693,
                        0.0075956,
                        0.00620178,
                        0.00228693,
                        0.0075956,
                        0.00620178,
                        0.00367922,
                        0.00531467,
                        0.00571467,
                        0.00367922,
                        0.00531467,
                        0.00571467,
                        0.00367922,
                        0.00531467,
                        0.00571467,
                        0.00367922,
                        0.00531467,
                        0.00571467,
                        0.00228693,
                        0.0075956,
                        0.00620178,
                        0.00228693,
                        0.0075956,
                        0.00620178,
                        0.00228693,
                        0.0075956,
                        0.00620178,
                        0.00228693,
                        0.0075956,
                        0.00620178,
                        0.00367922,
                        0.00531467,
                        0.00571467,
                        0.00367922,
                        0.00531467,
                        0.00571467,
                        0.00367922,
                        0.00531467,
                        0.00571467,
                        0.00367922,
                        0.00531467,
                        0.00571467,
                        0.00228693,
                        0.0075956,
                        0.00620178,
                        0.00228693,
                        0.0075956,
                        0.00620178,
                        0.00228693,
                        0.0075956,
                        0.00620178,
                        0.00228693,
                        0.0075956,
                        0.00620178,
                        0.00334429,
                        0.00600451,
                        0.00504886,
                        0.00334429,
                        0.00600451,
                        0.00504886,
                        0.00334429,
                        0.00600451,
                        0.00504886,
                        0.00334429,
                        0.00600451,
                        0.00504886,
                        0.00539691,
                        0.00350097,
                        0.00841161,
                        0.00539691,
                        0.00350097,
                        0.00841161,
                        0.00539691,
                        0.00350097,
                        0.00841161,
                        0.00539691,
                        0.00350097,
                        0.00841161,
                        0.00334429,
                        0.00600451,
                        0.00504886,
                        0.00334429,
                        0.00600451,
                        0.00504886,
                        0.00334429,
                        0.00600451,
                        0.00504886,
                        0.00334429,
                        0.00600451,
                        0.00504886,
                        0.00539691,
                        0.00350097,
                        0.00841161,
                        0.00539691,
                        0.00350097,
                        0.00841161,
                        0.00539691,
                        0.00350097,
                        0.00841161,
                        0.00539691,
                        0.00350097,
                        0.00841161,
                        0.00334429,
                        0.00600451,
                        0.00504886,
                        0.00334429,
                        0.00600451,
                        0.00504886,
                        0.00334429,
                        0.00600451,
                        0.00504886,
                        0.00334429,
                        0.00600451,
                        0.00504886,
                        0.00539691,
                        0.00350097,
                        0.00841161,
                        0.00539691,
                        0.00350097,
                        0.00841161,
                        0.00539691,
                        0.00350097,
                        0.00841161,
                        0.00539691,
                        0.00350097,
                        0.00841161,
                        0.00334429,
                        0.00600451,
                        0.00504886,
                        0.00334429,
                        0.00600451,
                        0.00504886,
                        0.00334429,
                        0.00600451,
                        0.00504886,
                        0.00334429,
                        0.00600451,
                        0.00504886,
                        0.00539691,
                        0.00350097,
                        0.00841161,
                        0.00539691,
                        0.00350097,
                        0.00841161,
                        0.00539691,
                        0.00350097,
                        0.00841161,
                        0.00539691,
                        0.00350097,
                        0.00841161,
                    ]
                ),
                atol=1e-5,
            )
        )

    def test_single_step(self):

        # Random Without Importance and init size reduce
        attack = ZooAttack(
            model=self.model,
            config=self.config,
            input_image_shape=self.orig_img.shape[1:],
            device="cpu:0",
        )
        attack.config["use_importance"] = False
        attack.config["init_size"] = 2
        modifier = deepcopy(self.modifier)

        (
            total_loss,
            l2_loss,
            confidence_loss,
            model_output,
            new_img,
        ) = attack.single_step(
            modifier,
            self.orig_img,
            self.labels,
            self.config["initial_const"],
            max_pooling_ratio=2,
        )

        self.assertFalse(np.allclose(modifier, self.modifier, atol=1e-5))

        self.assertEqual(new_img.shape, self.modifier.shape[1:])

        # With Custom Indices
        attack = deepcopy(self.attack)
        modifier = deepcopy(self.modifier)
        indices = [15, 24, 32, 45]
        (
            total_loss,
            l2_loss,
            confidence_loss,
            model_output,
            new_img,
        ) = attack.single_step(
            modifier,
            self.orig_img,
            self.labels,
            self.config["initial_const"],
            var_indice=indices,
            max_pooling_ratio=2,
        )

        self.assertFalse(np.allclose(modifier, self.modifier, atol=1e-5))

        self.assertEqual(new_img.shape, self.modifier.shape[1:])

    def test_attack(self):
        attack = deepcopy(self.attack)
        orig_img = deepcopy(self.orig_img[0].numpy())
        orig_img /= 10 * np.max(orig_img)
        labels = self.labels[0].numpy()
        outer_best_adv, outer_best_const = attack.attack(
            orig_img, labels, max_pooling_ratio=2
        )

        self.assertEqual(outer_best_adv.shape, self.modifier.shape[1:])

        # Without x10
        attack = deepcopy(self.attack)
        orig_img = deepcopy(self.orig_img[0].numpy())
        orig_img /= 100 * np.max(orig_img)
        outer_best_adv, outer_best_const = attack.attack(
            orig_img, labels, max_pooling_ratio=2
        )

        self.assertEqual(outer_best_adv.shape, self.modifier.shape[1:])

        # With modifier init
        attack = deepcopy(self.attack)
        outer_best_adv, outer_best_const = attack.attack(
            orig_img,
            labels,
            modifier_init=self.modifier[0],
            max_pooling_ratio=2,
        )

        self.assertEqual(outer_best_adv.shape, self.modifier.shape[1:])

        # With use resize and untargeted and max iterations 10k
        attack = deepcopy(self.attack)
        attack.config["use_resize"] = True
        attack.config["resize_iter_1"] = 20
        attack.config["resize_iter_2"] = 80
        attack.config["abort_early"] = False
        attack.config["targeted"] = False

        orig_img = deepcopy(self.orig_img[0].numpy())
        orig_img /= 10 * np.max(orig_img)
        outer_best_adv, outer_best_const = attack.attack(
            orig_img, labels, max_pooling_ratio=2
        )
        self.assertEqual(outer_best_adv.shape, self.modifier.shape[1:])

        # Without tanh
        attack = deepcopy(self.attack)
        attack.config["use_tanh"] = False
        outer_best_adv, outer_best_const = attack.attack(
            orig_img, labels, max_pooling_ratio=2
        )
        self.assertEqual(outer_best_adv.shape, self.modifier.shape[1:])
