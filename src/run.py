import os
import argparse

import torch
import torchvision.models as models
import matplotlib.pyplot as plt

from utils import load_style_content, store_video_as_frames
from neural_style import run_style_transfer


def load_data_and_transfer(
    style_image,
    content_image,
    cnn,
    cnn_normalization_mean,
    cnn_normalization_std,
    output_path,
):
    style_img, content_img = load_style_content(style_image, content_image)
    input_img = content_img.clone()
    output = run_style_transfer(
        cnn,
        cnn_normalization_mean,
        cnn_normalization_std,
        content_img,
        style_img,
        input_img,
        output_path,
    )
    return output


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    output_subpath = os.path.join(
        args.output_path, os.path.basename(args.content_image).split(".")[0]
    )
    os.makedirs(output_subpath, exist_ok=True)
    if args.transfer_video:
        video_path = store_video_as_frames(args.content_image, args.frame_path)
        for i, frame in enumerate(sorted(os.listdir(video_path))):
            frame_output_path = os.path.join(output_subpath, str(i).zfill(4))
            os.makedirs(frame_output_path, exist_ok=True)
            output = load_data_and_transfer(
                args.style_image,
                os.path.join(video_path, frame),
                cnn,
                cnn_normalization_mean,
                cnn_normalization_std,
                frame_output_path,
            )

    else:
        output = load_data_and_transfer(
            args.style_image,
            args.content_image,
            cnn,
            cnn_normalization_mean,
            cnn_normalization_std,
            output_subpath,
        )

    plt.figure()
    imshow(output, title="Output image")

    # sphinx_gallery_thumbnail_number = 4
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run neural style transfer.")
    parser.add_argument("style_image", type=str, help="Path to style image")
    parser.add_argument(
        "content_image",
        type=str,
        help="Path to content image. This should be a video if flag --transfer_video is used.",
    )
    parser.add_argument(
        "--output_path", type=str, default="./output", help="Set output path."
    )
    parser.add_argument(
        "--transfer_video",
        action="store_true",
        help="Transfer images of a video. Default=False",
    )
    parser.add_argument(
        "--frame_path",
        type=str,
        default="./input_video_frames",
        help="Directory where input video frames are stored.",
    )
    args = parser.parse_args()
    main(args)
