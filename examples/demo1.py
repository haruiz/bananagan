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
