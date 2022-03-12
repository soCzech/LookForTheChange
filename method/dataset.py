import os
import glob
import torch
import pickle
import random
import itertools
import numpy as np
from torch.utils.data import Dataset


class ChangeItDataset(Dataset):

    def __init__(self,
                 pickle_roots,
                 single_class=None,
                 annotation_root=None,
                 file_mode="unannotated",  # "unannotated", "annotated", "all"
                 noise_adapt_weight_root=None,
                 noise_adapt_weight_threshold_file=None,
                 deterministic=False):

        self.classes = {x: i for i, x in enumerate(sorted(set([os.path.basename(fn) for fn in itertools.chain(*[
            glob.glob(os.path.join(root, "*")) for root in pickle_roots
        ]) if os.path.isdir(fn)])))}
        if single_class is not None:
            self.classes = {single_class: 0}

        self.files = {key: sorted(itertools.chain(*[
            glob.glob(os.path.join(root, key, "*.pickle")) for root in pickle_roots
        ])) for key in self.classes.keys()}

        self.annotations = {key: {
            os.path.basename(fn).split(".")[0]: np.uint8([int(line.strip().split(",")[1]) for line in open(fn).readlines()])
            for fn in glob.glob(os.path.join(annotation_root, key, "*.csv"))
        } for key in self.classes.keys()} if annotation_root is not None else None

        if file_mode == "unannotated":
            for key in self.classes.keys():
                for fn in self.files[key].copy():
                    if os.path.basename(fn).split(".")[0] in self.annotations[key]:
                        self.files[key].remove(fn)
        elif file_mode == "annotated":
            for key in self.classes.keys():
                for fn in self.files[key].copy():
                    if os.path.basename(fn).split(".")[0] not in self.annotations[key]:
                        self.files[key].remove(fn)
        elif file_mode == "all":
            pass
        else:
            raise NotImplementedError()

        self.flattened_files = []
        for key in self.classes.keys():
            self.flattened_files.extend([(key, fn) for fn in self.files[key]])

        self.deterministic = deterministic

        # Noise adaptive weighting
        if noise_adapt_weight_root is None:
            return

        self.noise_adapt_weight = {}
        for key in self.classes.keys():
            with open(os.path.join(noise_adapt_weight_root, f"{key}.csv"), "r") as f:
                for line in f.readlines():
                    vid_id, score = line.strip().split(",")
                    self.noise_adapt_weight[vid_id] = float(score)

        self.noise_adapt_weight_thr = {line.split(",")[0]: float(line.split(",")[2].strip())
                                       for line in open(noise_adapt_weight_threshold_file, "r").readlines()[1:]}

    def __getitem__(self, idx):
        class_name, pickle_fn = self.flattened_files[idx]
        file_id = os.path.basename(pickle_fn).split(".")[0]

        with open(pickle_fn, "rb") as f:
            data = pickle.load(f)
        vector1 = data["video"]["resnext"]
        vector2 = data["video"]["tsm"]
        video_features = torch.from_numpy(np.concatenate([
            vector1[0 if self.deterministic else random.randint(0, len(vector1) - 1)],
            vector2[0 if self.deterministic else random.randint(0, len(vector2) - 1)]
        ], 1).astype(np.float32))

        annotation = self.annotations[class_name][file_id] \
            if self.annotations is not None and file_id in self.annotations[class_name] else None
        video_level_score = self.noise_adapt_weight[file_id] - self.noise_adapt_weight_thr[class_name] \
            if hasattr(self, "noise_adapt_weight") else None

        return class_name + "/" + file_id, self.classes[class_name], video_features, annotation, video_level_score

    @property
    def n_classes(self):
        return len(self.classes)

    def __len__(self):
        return len(self.flattened_files)

    def __repr__(self):
        string = f"ChangeItDataset(n_classes: {self.n_classes}, n_samples: {self.__len__()}, " \
                 f"deterministic: {self.deterministic})"
        for key in sorted(self.classes.keys()):
            string += f"\n> {key:20} {len(self.files[key]):4d}"
            if hasattr(self, "noise_adapt_weight_thr"):
                string += f" (above threshold {self.noise_adapt_weight_thr[key]:.3f}: " \
                          f"{len([fn for fn in self.files[key] if self.noise_adapt_weight[os.path.basename(fn).split('.')[0]] > self.noise_adapt_weight_thr[key]]):4d})"
        return string


def changeit_collate_fn(items):
    lens = [len(item[2]) for item in items]
    max_len = max(lens)
    lens = torch.tensor(lens, dtype=torch.int32)

    feats = []
    for _, _, frame_feats, _, _ in items:
        add = max_len - len(frame_feats)
        if add > 0:
            frame_feats = torch.cat([frame_feats, torch.zeros((add, frame_feats.shape[1]), dtype=torch.float32)], 0)
        feats.append(frame_feats)

    file_ids = [item[0] for item in items]
    classes = torch.tensor([item[1] for item in items], dtype=torch.int32)
    video_level_scores = torch.tensor([item[-1] for item in items], dtype=torch.float32) \
        if items[0][-1] is not None else None
    feats = torch.stack(feats, 0)
    annotations = [item[3] for item in items]

    return {"features": feats, "lens": lens, "classes": classes, "file_ids": file_ids,
            "annotations": annotations, "video_level_scores": video_level_scores}
