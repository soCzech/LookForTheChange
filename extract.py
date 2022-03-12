import os
import pickle
import argparse
import numpy as np
import scipy.ndimage

from feature_extraction import ffmpeg_utils, resnext_model, transforms, tsm_model


def main(args):
    if not os.path.exists(f"{args.resnext_path}-0040.params"):
        print(f"File {args.resnext_path}-0040.params does not exist, set --resnext_path correctly.")
        exit(1)

    tsm = tsm_model.get_tsm_model()
    resnext = resnext_model.get_resnext_fc(args.resnext_path, 40)

    for video_file in args.videos:
        print(f"Processing {video_file}.")

        out_file, _ = os.path.splitext(video_file)
        out_file += ".pickle"
        if args.export_dir is not None:
            os.makedirs(args.export_dir, exist_ok=True)
            out_file = os.path.join(args.export_dir, os.path.basename(out_file))

        if os.path.exists(out_file):
            print(f"File {out_file} exists, skipping.")
            continue

        size = (480, 270)
        crop = None
        if args.eval_center_crop:
            size = (398, 224)
            crop = (398 - 224, 0)

        frames01fps = ffmpeg_utils.extract_frames(video_file, fps= 1, size=size, crop=crop)
        frames25fps = ffmpeg_utils.extract_frames(video_file, fps=25, size=size, crop=crop)

        feats = []
        for i in range(args.n_augmentations + 1):
            transform = transforms.get_transform(frames01fps.shape[1:3][::-1], identity=i == 0)

            resnext_feats = resnext(transform(frames01fps, end_size=(224, 224))).astype(np.float16)
            tsm_feats = tsm(transform(frames25fps, end_size=(200, 200)))
            tsm_feats_resized = scipy.ndimage.zoom(tsm_feats, (len(resnext_feats) / len(tsm_feats), 1), order=1).astype(np.float16)

            feats.append([resnext_feats, tsm_feats_resized])

        with open(out_file, "wb") as f:
            obj = {"video": {
                "resnext": [f1 for f1, f2 in feats],
                "tsm": [f2 for f1, f2 in feats]
            }}
            pickle.dump(obj, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--resnext_path", type=str, default="./weights/resnext-101-1")
    parser.add_argument("--export_dir", type=str, default=None)
    parser.add_argument("--n_augmentations", type=int, default=0)
    parser.add_argument("--eval_center_crop", action="store_true")

    main(parser.parse_args())
