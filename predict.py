import os
import pickle
import argparse
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from method import model, utils
from feature_extraction import ffmpeg_utils


def main(args):
    if not os.path.exists(args.model_path):
        print(f"File {args.model_path} does not exist, set --model_path correctly.")
        exit(1)

    out_file, _ = os.path.splitext(args.features)
    out_file += f".{args.category}.csv"
    if args.export_dir is not None:
        os.makedirs(args.export_dir, exist_ok=True)
        out_file = os.path.join(args.export_dir, os.path.basename(out_file))
    if os.path.exists(out_file):
        print(f"File {out_file} exists.")
        return

    state_dict = torch.load(args.model_path)
    if args.category not in state_dict:
        print(f"Category {args.category} not in model, select one of:", ", ".join(state_dict.keys()))
        exit(1)

    network = model.MultiClassMLP(layers=[4096, 512])
    network.load_state_dict(state_dict[args.category])
    network.eval().cuda()

    with open(args.features, "rb") as f:
        video_feats = pickle.load(f)
    vector1 = video_feats["video"]["resnext"]
    vector2 = video_feats["video"]["tsm"]
    video_feats = torch.from_numpy(np.concatenate([vector1[0], vector2[0]], 1).astype(np.float32))

    with torch.no_grad():
        pred = network(video_feats.cuda())
        pred_action = torch.sigmoid(pred["action"]).cpu().numpy()
        pred_state = torch.softmax(pred["state"], -1).cpu().numpy()

    with open(out_file, "w") as f:
        f.write(f"TIME[s],STATE1,STATE2,ACTION\n")
        for i in range(len(pred_state)):
            f.write(f"{i:7d},{pred_state[i][0]:.4f},{pred_state[i][1]:.4f},{pred_action[i][0]:.4f}\n")

    if not args.visualize:
        return
    if not os.path.exists(args.video):
        print(f"File {args.video} does not exist, set --video correctly if using --visualize.")
        exit(1)

    colors = [(223, 71, 60), (38, 119, 120), (246, 192, 60)]
    label_texts = ["INITIAL ST.", "ACTION", "END ST."]
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        font = None

    s1_loc, s2_loc, ac_loc = utils.constrained_argmax(pred_action, pred_state)
    frames01fps = ffmpeg_utils.extract_frames(args.video, fps=1, size=(320, 180)).copy()

    triplet_image = Image.fromarray(np.concatenate(frames01fps[[s1_loc, ac_loc, s2_loc]], 1))
    triplet_draw = ImageDraw.Draw(triplet_image)
    for i, (color, text) in enumerate(zip(colors, label_texts)):
        triplet_draw.rectangle([i * 320, 0, (i + 1) * 320, 180], outline=color, width=4)
        if font is not None:
            triplet_draw.rectangle([0 + i * 320, 0, 80 + i * 320, 20], fill=color)
            triplet_draw.text((40 + i * 320, 10), text, fill="white", anchor="mm", font=font)
    triplet_image.save(out_file[:-3] + "triplet.png")

    reordered_pred = np.concatenate([pred_state[:, :1], pred_action, pred_state[:, 1:]], 1)
    for i in range(len(frames01fps)):
        frames01fps[i][3:5 + 7 * len(reordered_pred[i]), 3:37] = 0
        for j, (p, c) in enumerate(zip(reordered_pred[i], colors)):
            frames01fps[i][7 * j + 5:7 * j + 10, 5:5 + int(30 * p)] = c

    if len(frames01fps) % 10 != 0:
        frames01fps = np.concatenate([frames01fps] + [np.zeros_like(frames01fps[:1])] * (10 - len(frames01fps) % 10), 0)
    frames01fps = np.concatenate([
        np.concatenate(frames01fps[start:start + 10], 1) for start in range(0, len(frames01fps), 10)], 0)
    Image.fromarray(frames01fps).save(out_file[:-3] + "frames.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("category", type=str)
    parser.add_argument("features", type=str)
    parser.add_argument("--model_path", type=str, default="./weights/look-for-the-change.pth")
    parser.add_argument("--export_dir", type=str, default=None)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--video", type=str, default=None)

    main(parser.parse_args())
