# code-soup
[![codecov](https://codecov.io/gh/Adversarial-Deep-Learning/code-soup/branch/main/graph/badge.svg?token=OQIJCADZF0)](https://codecov.io/gh/Adversarial-Deep-Learning/code-soup)
[![Tests](https://github.com/Adversarial-Deep-Learning/code-soup/actions/workflows/pytest.yml/badge.svg)](https://github.com/Adversarial-Deep-Learning/code-soup/actions/workflows/pytest.yml)
[![Lint](https://github.com/Adversarial-Deep-Learning/code-soup/actions/workflows/lint.yml/badge.svg)](https://github.com/Adversarial-Deep-Learning/code-soup/actions/workflows/lint.yml)

**code-soup is the python code for the book "Adversarial Deep Learning" and its tutorials. You can use this in conjunction with a course on Adversarial Deep Learning, or for study on your own. We're looking for solid contributors to help.**

Despite the great success of deep neural networks in a wide range of applications, they have been repeatedly shown to be vulnerable to adversarial attacks. *Adversarial Deep Learning* is a book being written by [Dr. Di Jin](https://scholar.google.com/citations?user=x5QTK9YAAAAJ&hl=en), [Dr. Yifang Yin](https://yifangyin.github.io/), [Yaman Kumar](https://sites.google.com/view/yaman-kumar/), and [Dr. Rajiv Ratn Shah](https://www.iiitd.ac.in/rajivratn), which gives the reader an introduction to the progress made in this field. At code-soup we are building the codebase of these algorithms in a *clean, simple and minimal* manner . We strive to give the reader a smooth experience while reading the book and understanding the code in parallel with a minimal set of dependencies and library. Contact of the core developers can be seen in [AUTHORS](./AUTHORS.md).

## Structure of the project
When complete, this project will have Python implementations for all the pseudocode algorithms in the book, as well as tests and examples of use. You can check the exact repository structure here [Repository Structure Docs](./REPO_STRUCTURE.md).
The overall idea is to let the user read the algorithm and understand the attack in the `code-soup/ch{ch_num}/models/{topic}.py` and the demonstration in the tutorial.

## Requirements
The requirements are stored in `requirements.txt` you can install them using
```
pip install -r requirements.txt
```
We recommend to use a virtual environment, the exported yaml is available at `environment.yml`.

## Tutorials
The tutorial to each algorithm is available in the Tutorials folder.

## Index
Index for tutorials and test suite for each algorithm.

| Topic, Chapter | Tutorial |
|--|--|
| Generative Adversarial Networks (Chapter 5) | [Tutorial](./Tutorials/ch5/GAN/GAN_Tutorial.ipynb) |


## Contribution
Please take a look the [CONTRIBUTING.md](https://github.com/Adversarial-Deep-Learning/code-soup/blob/main/CONTRIBUTING.md) for details, :star: us if you liked the work.
