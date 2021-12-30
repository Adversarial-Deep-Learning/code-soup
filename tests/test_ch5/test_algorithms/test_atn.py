import unittest

import torch
import torch.nn as nn

from code_soup.ch5.algorithms.atn import ATNBase, SimpleAAE, SimplePATN


class TestATNBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.classifier_model = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 10), nn.Softmax(dim=1)
        )
        cls.model = ATNBase(
            classifier_model=cls.classifier_model,
            target_idx=2,
            device=torch.device("cpu"),
        )

    def test_init_with_alpha_less_than_one(self):
        with self.assertRaises(ValueError):
            model = ATNBase(classifier_model=nn.Linear(1, 1), target_idx=0, alpha=0.5)

    def test_rerank(self):
        softmax_logits = torch.tensor(
            [
                [
                    0.11319724,
                    0.02375807,
                    0.17357929,
                    0.31361626,
                    0.01670836,
                    0.04249263,
                    0.05608005,
                    0.11300851,
                    0.09807534,
                    0.04948426,
                ],
                [
                    0.01500716,
                    0.01716916,
                    0.04945158,
                    0.07586802,
                    0.0117808,
                    0.09239875,
                    0.13248007,
                    0.15326169,
                    0.17005756,
                    0.2825252,
                ],
            ]
        )

        rerank_logits = self.model.rerank(softmax_logits)
        print(rerank_logits)
        self.assertEqual(rerank_logits.shape, (2, 10))

    def test_forward(self):

        with self.assertRaises(NotImplementedError):
            out = self.model.forward(torch.tensor([]))

    def test_compute_loss(self):
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        x_hat = torch.tensor([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]])

        y = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
        y_hat = torch.tensor([[2.1, 3.1, 4.1], [5.1, 6.1, 7.1]])

        loss = self.model.compute_loss(x, x_hat, y, y_hat)

        self.assertTrue(isinstance(loss, torch.Tensor))
        self.assertEqual(tuple(loss.shape), ())

    def test_step(self):
        with self.assertRaises(NotImplementedError):
            adv_out, adv_logits = self.model.step((torch.tensor([]), torch.tensor([])))


class TestSimpleAAE(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        classifier_model = nn.Sequential(
            nn.Flatten(), nn.Linear(9, 10), nn.Softmax(dim=1)
        )
        cls.model_a = SimpleAAE(
            classifier_model=classifier_model,
            target_idx=2,
            device=torch.device("cpu"),
            input_shape=(1, 3, 3),
        )

        cls.model_b = SimpleAAE(
            classifier_model=classifier_model,
            target_idx=2,
            device=torch.device("cpu"),
            input_shape=(1, 3, 3),
            typ="b",
        )

        cls.model_c = SimpleAAE(
            classifier_model=classifier_model,
            target_idx=2,
            device=torch.device("cpu"),
            input_shape=(1, 3, 3),
            typ="c",
        )

    def test_forward(self):
        x = torch.tensor(
            [
                [
                    [
                        [0.1677, 1.3509, -0.8152],
                        [-0.6369, -1.2858, 0.4709],
                        [0.8874, 0.0070, 0.1990],
                    ]
                ],
                [
                    [
                        [-1.4228, 0.6089, -0.2605],
                        [-0.3259, -0.1384, -0.8231],
                        [-2.6140, 0.3131, -0.4660],
                    ]
                ],
            ]
        )

        adv_out, adv_logits = self.model_a(x)

        self.assertTrue(isinstance(adv_out, torch.Tensor))
        self.assertEqual(tuple(adv_out.shape), (2, 1, 3, 3))

        self.assertTrue(isinstance(adv_logits, torch.Tensor))
        self.assertEqual(tuple(adv_logits.shape), (2, 10))

        adv_out, adv_logits = self.model_b(x)

        self.assertTrue(isinstance(adv_out, torch.Tensor))
        self.assertEqual(tuple(adv_out.shape), (2, 1, 3, 3))

        self.assertTrue(isinstance(adv_logits, torch.Tensor))
        self.assertEqual(tuple(adv_logits.shape), (2, 10))

        adv_out, adv_logits = self.model_c(x)

        self.assertTrue(isinstance(adv_out, torch.Tensor))
        self.assertEqual(tuple(adv_out.shape), (2, 1, 3, 3))

        self.assertTrue(isinstance(adv_logits, torch.Tensor))
        self.assertEqual(tuple(adv_logits.shape), (2, 10))

    def test_step(self):
        x = torch.tensor(
            [
                [
                    [
                        [0.1677, 1.3509, -0.8152],
                        [-0.6369, -1.2858, 0.4709],
                        [0.8874, 0.0070, 0.1990],
                    ]
                ],
                [
                    [
                        [-1.4228, 0.6089, -0.2605],
                        [-0.3259, -0.1384, -0.8231],
                        [-2.6140, 0.3131, -0.4660],
                    ]
                ],
            ]
        )

        y = torch.tensor([])

        adv_out, adv_logits, loss_item = self.model_a.step((x, y))

        self.assertTrue(isinstance(adv_out, torch.Tensor))
        self.assertEqual(tuple(adv_out.shape), (2, 1, 3, 3))

        self.assertTrue(isinstance(adv_logits, torch.Tensor))
        self.assertEqual(tuple(adv_logits.shape), (2, 10))

        self.assertTrue(isinstance(loss_item, float))


class TestSimplePATN(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        classifier_model = nn.Sequential(
            nn.Flatten(), nn.Linear(9, 10), nn.Softmax(dim=1)
        )
        cls.model = SimplePATN(
            classifier_model=classifier_model,
            target_idx=2,
            device=torch.device("cpu"),
            input_shape=(1, 3, 3),
        )

    def test_forward(self):
        x = torch.tensor(
            [
                [
                    [
                        [0.1677, 1.3509, -0.8152],
                        [-0.6369, -1.2858, 0.4709],
                        [0.8874, 0.0070, 0.1990],
                    ]
                ],
                [
                    [
                        [-1.4228, 0.6089, -0.2605],
                        [-0.3259, -0.1384, -0.8231],
                        [-2.6140, 0.3131, -0.4660],
                    ]
                ],
            ]
        )

        adv_out, adv_logits = self.model(x)

        self.assertTrue(isinstance(adv_out, torch.Tensor))
        self.assertEqual(tuple(adv_out.shape), (2, 1, 3, 3))

        self.assertTrue(isinstance(adv_logits, torch.Tensor))
        self.assertEqual(tuple(adv_logits.shape), (2, 10))

    def test_step(self):
        x = torch.tensor(
            [
                [
                    [
                        [0.1677, 1.3509, -0.8152],
                        [-0.6369, -1.2858, 0.4709],
                        [0.8874, 0.0070, 0.1990],
                    ]
                ],
                [
                    [
                        [-1.4228, 0.6089, -0.2605],
                        [-0.3259, -0.1384, -0.8231],
                        [-2.6140, 0.3131, -0.4660],
                    ]
                ],
            ]
        )

        y = torch.tensor([])

        adv_out, adv_logits, loss_item = self.model.step((x, y))

        self.assertTrue(isinstance(adv_out, torch.Tensor))
        self.assertEqual(tuple(adv_out.shape), (2, 1, 3, 3))

        self.assertTrue(isinstance(adv_logits, torch.Tensor))
        self.assertEqual(tuple(adv_logits.shape), (2, 10))

        self.assertTrue(isinstance(loss_item, float))
