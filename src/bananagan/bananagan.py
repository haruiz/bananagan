from abc import ABC, abstractmethod
from PIL import Image
from bananagan.models.pix2pixHD import Pix2PixHDModelInference
from bananagan.models.pix2pixHD import TestOptions
from pathlib import Path
from bananagan.utils import SquarePad, create_label_image
from torchvision import transforms
from bananagan.models.pix2pixHD.util import misc_util, data_utils
import itertools
from PIL import Image as PILImage
import random
import typing
from enum import Enum, auto
import torch
import gdown


class ModelCheckPoint(ABC):
    def __init__(self, checkpoint_file: typing.Union[str, Path]):

        self._model_loading_opts = TestOptions().parse(save=False)
        self._model_loading_opts.nThreads = 1  # test code only supports nThreads = 1
        self._model_loading_opts.batchSize = 1  # test code only supports batchSize = 1
        self._model_loading_opts.serial_batches = True  # no shuffle
        self._model_loading_opts.no_flip = True  # no flip
        self._model_loading_opts.label_nc = 0
        self._model_loading_opts.checkpoint_file = str(checkpoint_file)
        self._model_loading_opts.no_instance = True
        self._model_loading_opts.verbose = False
        self._model_loading_opts.resize_or_crop = "scale_width"

    def load_checkpoint(self):
        """
        Load a model checkpoint
        :param checkpoint_file:
        :return:
        """
        model = Pix2PixHDModelInference()
        model.initialize(self._model_loading_opts)
        return model

    def process_image(self, image: PILImage, model_loading_opts: TestOptions, block_size=13, c=1):
        """
        process an image for a model
        :param image:
        :param model_loading_opts:
        :param block_size:
        :param c:
        :return:
        """
        image_gray = create_label_image(image, block_size, c)
        processing_transform = transforms.Compose([SquarePad(), transforms.Resize(1024)])
        image_gray = processing_transform(image_gray)
        model_tranform_params = data_utils.get_params(model_loading_opts, image_gray.size)
        model_transform = data_utils.get_transform(model_loading_opts, model_tranform_params)
        model_input = model_transform(image_gray.convert('RGB'))
        return model_input

    def generate_image(self, input_image: PILImage.Image, block_size=13, c=1):
        """
        Process an image
        :param input_image:
        :param block_size:
        :param c:
        """
        model_checkpoint = self.load_checkpoint()
        model_input = self.process_image(input_image, self._model_loading_opts, block_size, c)
        generated_image = model_checkpoint.inference(model_input, torch.tensor([0]))
        generated_image = misc_util.tensor2im(generated_image.data)
        generated_image = Image.fromarray(generated_image)
        return generated_image

    def __call__(self, *args, **kwargs):
        return self.generate_image(*args, **kwargs)

    def generate(self, input_image: PILImage.Image):
        """
        Generate an image
        :param input_image:
        :param block_size:
        :param c:
        :return:
        """
        c_values = [1]
        block_sizes = [3, 5, 7, 9, 11, 13]
        runs = itertools.product(c_values, block_sizes)
        runs = list(runs)
        random.shuffle(runs)

        for c, block_size in runs:
            yield self.generate_image(input_image, block_size, c)




class PseudostemModels(Enum):
    healthy = auto()
    xanthomonas_wilt = auto()
    fusarium_wilt = auto()


class RachisModels(Enum):
    healthy = auto()
    banana_blood_disease = auto()


GDRIVE_FILE_IDS = {
    PseudostemModels.healthy: "1GoXyCmWB6amIE-CDL7HWQpQfwoR_1a5Q",
    PseudostemModels.xanthomonas_wilt: "1ZmP1azGq1O74sQsu3bhU5iK2L69OlNt1",
    PseudostemModels.fusarium_wilt: "1r_s9t141mkG30b25XNyw_wUmqJkfFO5r",
    RachisModels.healthy: "1-Mmyi7Afqyx6OeGrlaExGufSeT6hEwPg",
    RachisModels.banana_blood_disease: "1kTidkMlVcRZg8d8WSS1UT-DhX8bsHyZN"
}

CHECKPOINT_FILES = {
    PseudostemModels.healthy: "pseudostem_healthy.pth",
    PseudostemModels.xanthomonas_wilt: "pseudostem_bxw.pth",
    PseudostemModels.fusarium_wilt: "pseudostem_fwb.pth",
    RachisModels.healthy: "rachis_healthy.pth",
    RachisModels.banana_blood_disease: "rachis_bbd.pth"
}


class BananaGan:
    @staticmethod
    def get_model(model: PseudostemModels or RachisModels):
        """
        Get a model by name
        :param model:
        :return:
        """
        gdrive_file_id = GDRIVE_FILE_IDS[model]
        checkpoint_file = CHECKPOINT_FILES[model]

        checkpoints_dir = "checkpoints"
        Path(checkpoints_dir).mkdir(exist_ok=True)
        checkpoint_path = Path(checkpoints_dir + "/" + checkpoint_file)

        if not checkpoint_path.exists():
            print(f"Downloading {checkpoint_file} from google drive")
            gdown.download(id=gdrive_file_id, output=str(checkpoint_path), quiet=False)

        return ModelCheckPoint(checkpoint_path)
