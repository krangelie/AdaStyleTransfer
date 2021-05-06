import os

import imageio
from PIL import Image
import torch
import torchvision.transforms as transforms


def image_loader(image_name, loader):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return image.to(device, torch.float)


def load_style_content(style_path, content_path):
    imsize = (
        512 if torch.cuda.is_available() else 128
    )  # use small size if no gpu
    loader = transforms.Compose(
        [
            transforms.Resize(imsize),  # scale imported image
            transforms.ToTensor(),
        ]
    )  # transform it into a torch tensor
    style_img = image_loader(style_path, loader)
    content_img = image_loader(content_path, loader)
    print(content_img.size())
    if content_img.size() != style_img.size():
        crop = transforms.CenterCrop(imsize)
        style_img = crop(style_img)
        content_img = crop(content_img)

    assert (
        style_img.size() == content_img.size()
    ), "we need to import style and content images of the same size"
    return style_img, content_img


def store_video_as_frames(content_video_path, frame_path):
    video_path = os.path.join(
        frame_path, os.path.basename(content_video_path).split(".")[0]
    )
    if not os.path.isdir(video_path) or len(os.listdir(video_path)) < 10:
        reader = imageio.get_reader(content_video_path)
        os.makedirs(video_path, exist_ok=True)
        for i, im in enumerate(reader):
            imageio.imwrite(
                os.path.join(video_path, f"{str(i).zfill(4)}.png"),
                im,
                format="png",
            )

        print("Finished writing frames.")
    return video_path
