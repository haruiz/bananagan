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