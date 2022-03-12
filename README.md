# Look for the Change: Learning Object States and State-Modifying Actions from Untrimmed Web Videos

### [[Project Website :dart:]](https://data.ciirc.cvut.cz/public/projects/2022LookForTheChange/)&nbsp;&nbsp;&nbsp;[[Paper :page_with_curl:]](https://arxiv.org/abs/2203.11637)&nbsp;&nbsp;&nbsp;[Code :octocat:]&nbsp;&nbsp;&nbsp;[[ChangeIt Dataset :octocat:]](https://github.com/soCzech/ChangeIt)

This repository contrains code for the CVPR'22 paper [Look for the Change: Learning Object States and State-Modifying Actions from Untrimmed Web Videos](https://arxiv.org/abs/2203.11637).

<img src="https://data.ciirc.cvut.cz/public/projects/2022LookForTheChange/resources/overview.svg" style="width:100%">


## Run the model on your video
0. **Prerequisites**
   - Nvidia docker (i.e. only linux is supported, you can install environment from docker file manually or not use GPU and then you do not have to use docker / linux).
   - Tested with at least 4GB VRAM GPU.

1. **Download model weights**
   - ```
     mkdir weights; cd weights
     wget https://data.ciirc.cvut.cz/public/projects/2022LookForTheChange/look-for-the-change.pth
     wget https://isis-data.science.uva.nl/mettes/imagenet-shuffle/mxnet/resnext101_bottomup_12988/resnext-101-1-0040.params
     wget https://isis-data.science.uva.nl/mettes/imagenet-shuffle/mxnet/resnext101_bottomup_12988/resnext-101-symbol.json
     mv resnext-101-symbol.json resnext-101-1-symbol.json
     ```

2. **Setup the environment**
   - Our code can be run in a docker container. Build it by running the following command.
     Note that by default, we compile custom CUDA code for architectures 6.1, 7.0, 7.5, and 8.0.
     You may need to update the Dockerfile with your GPU architecture. 
     ```
     docker build -t look-for-the-change .
     ```
   - Go into the docker image.
     ```
     docker run -it --rm --gpus 1 -v $(pwd):$(pwd) -w $(pwd) look-for-the-change bash
     ```

3. **Extract video features**
   - Our model runs with preextracted features, run the following command for the extraction.
     ```
     python extract.py path/to/video.mp4
     ```
     The script creates `path/to/video.pickle` file with the extracted features.
   - Note you may need to edit `memory_limit` of `tensorflow` in `feature_extraction/tsm_model.py` if you have less than 6 GB of VRAM.

4. **Get predictions**
   - Run the following command to get predictions for your video.
     ```
     python predict.py category path/to/video.pickle [--visualize --video path/to/video.mp4]
     ```
     where `category` is id of a dataset category such as `bacon` for *Bacon Frying*.
     See [ChangeIt](https://github.com/soCzech/ChangeIt) dataset categories for all options.
   - The script creates `path/to/video.category.csv` with raw model predictions for each second of the original video.
   - If a path to the original video is provided, the script also generates visualization of the predictions.


## Replicate our experiments
1. **Prerequisites**
   - Set up the docker environment and download the ResNeXT model weights as in points 0., 1., and 2. of the previous chapter.
   - Note that for training the GPU is required due to the custom CUDA op.

2. **Dataset preparation**
   - Download [ChangeIt](https://github.com/soCzech/ChangeIt) dataset videos.
     Note it is not necessary to download the videos in the best resolution available as only 224-by-224 px resolution is needed for feature extraction.
   - Extract features from the videos.
     ```
     python extract.py path/to/video1.mp4 path/to/video2.mp4 ... --n_augmentations 10 --export_dir path/to/dataset_root/category_name
     ```
     This script will create `path/to/dataset_root/category_name/video1.pickle` and `path/to/dataset_root/category_name/video2.pickle` files with extracted features.
     It is important to have some `dataset_root` folder containing `category_name` sub-folders with individual video feature files.

3. **Train a model**
   - Run the following command to train on the preextracted features. Note that for every category a separate training needs to be run.
     Also keep in mind that due to the unsupervised nature of the algorithm, you may end up in bad local minima. We recommend to run the training multiple times to get the best results.
     ```
     python train.py --pickle_roots path/to/dataset_root
                     --category category_name
                     --annotation_root path/to/annotation_root
                     --noise_adapt_weight_root path/to/video_csv_files
                     --noise_adapt_weight_threshold_file path/to/categories.csv
     ```
   - `--annotation_root` is the location of `annotations` folder of [ChangeIt](https://github.com/soCzech/ChangeIt) dataset,
     `--noise_adapt_weight_root` is the location of `videos` folder of the dataset, and
     `--noise_adapt_weight_threshold_file` points to `categories.csv` file of the dataset.


## References
Tomáš Souček, Jean-Baptiste Alayrac, Antoine Miech, Ivan Laptev, and Josef Sivic.
[Look for the Change: Learning Object States and State-Modifying Actions from Untrimmed Web Videos](https://arxiv.org/abs/2203.11637).
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

```bibtex
@inproceedings{soucek2022lookforthechange,
    title={Look for the Change: Learning Object States and State-Modifying Actions from Untrimmed Web Videos},
    author={Sou\v{c}ek, Tom\'{a}\v{s} and Alayrac, Jean-Baptiste and Miech, Antoine and Laptev, Ivan and Sivic, Josef},
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2022}
}
```


## Acknowledgements
The project was supported by the European Regional Development Fund under the project IMPACT (reg. no. CZ.02.1.01/0.0/0.0/15_003/0000468) and by the Ministry of Education, Youth and Sports of the Czech Republic through the e-INFRA CZ (ID:90140), the French government under management of Agence Nationale de la Recherche as part of the "Investissements d'avenir" program, reference ANR19-P3IA-0001 (PRAIRIE 3IA Institute), and Louis Vuitton ENS Chair on Artificial Intelligence. We would like to also thank Kateřina Součková and Lukáš Kořínek for their help with the dataset.


