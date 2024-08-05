# BananaGan

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Description

BananaGan is a Python library for generating synthetic images of banana diseases using Generative Adversarial Networks (GANs). It provides a simple API to access the [Pix2PixHd](https://github.com/NVIDIA/pix2pixHD) pre-trained models for different banana parts and disease types.

## Installation

To install BananaGan, you can use pip:

```bash
pip install bananagan
```

## Usage

Here are some examples of how to use BananaGan:

**1. Generate an image of a pseudostem with Xanthomonas wilt:**

```python
from bananagan import *
from PIL import Image
from matplotlib import pyplot as plt

if __name__ == '__main__':
    image_path = "./../images/pseudostem-healthy.png"
    image_color = Image.open(image_path)
    model = BananaGan.get_model(PseudostemModels.xanthomonas_wilt)
    generated_image = model(input_image=image_color, block_size=13, c=1)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(image_color)
    ax.axis('off')
    ax.set_title('Original Image')

    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(generated_image)
    ax.axis('off')
    ax.set_title('Generated Image')

    plt.tight_layout()
    plt.show()
```

![Pseudostem with Xanthomonas wilt](https://raw.githubusercontent.com/haruiz/bananagan/main/images/plot1.png)


**2. Generate images with different parameters:**

```python
from bananagan import *
from PIL import Image
from matplotlib import pyplot as plt

if __name__ == '__main__':
    image_path = "./../images/pseudostem-healthy.png"
    image_color = Image.open(image_path)
    model = BananaGan.get_model(PseudostemModels.xanthomonas_wilt)

    images = []
    block_size = [3, 5, 7, 9, 11, 13]
    for bsz in block_size:
        generated_image = model(input_image=image_color, block_size=bsz, c=1)
        images.append(generated_image)

    fig = plt.figure(figsize=(10, 5))
    for i, generated_image in enumerate(images):
        ax = fig.add_subplot(1, len(images), i+1)
        ax.imshow(generated_image)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
```

![Pseudostem with Xanthomonas wilt](https://raw.githubusercontent.com/haruiz/bananagan/main/images/plot2.png)

## Available Models

**Pseudostem Models:**

* `PseudostemModels.healthy`: Generates healthy pseudostems.
* `PseudostemModels.xanthomonas_wilt`: Generates pseudostems with Xanthomonas wilt.
* `PseudostemModels.fusarium_wilt`: Generates pseudostems with Fusarium wilt.

**Rachis Models:**

* `RachisModels.healthy`: Generates healthy rachis.
* `RachisModels.banana_blood_disease`: Generates rachis with banana blood disease.

## License

This project is licensed under the MIT License.
