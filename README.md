# Video face tracker

![](samples/ffiw-sample.gif)

*Sample video from the [FFIW dataset](https://github.com/tfzhou/FFIW).*

This is a tool to perform simple tracking of multiple faces in video files.
The main focus is to annotate video datasets with face bounding box information
to crop faces in a posterior step.

## Installation

### Run locally

Install python >= 3.10 and run ```pip install -r requirements.txt```.
You need CUDA 12.x and cuDNN 9.x to run on GPU.

### Run on docker

Simply run ```./run.sh``` to launch the docker container.
The first time it is executed, the script will build the docker image.
To use CUDA inside the docker container, install the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Usage

This project includes a python script to process video files, directories and
directory trees. Here are some examples:

```bash
# Annotate a video file
python scripts/detect_faces.py <VIDEO_FILE>
# with docker
./run.sh <VIDEO_FILE>
```

This creates a JSON annotation file with the same name as ```<VIDEO_FILE>```.
The annotations follow this structure:

```json
{
    "0": {  # Face with ID 0
        "0": {  # First frame of the video
            "bbox": [x1, y1, x2, y2],
            "prob": <detector score between 0 and 1>,
            "landmarks": [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]
        },
        "1": {  # Second frame of the video
            ...
        },
        ...
    },
    "1": {  # Face with ID 1
        ...
    },
    ...
}
```

The script also works with directories:

```bash
# Annotate all videos inside a directory
python scripts/detect_faces.py <DIR_PATH>
# with docker
./run.sh <DIR_PATH>


# Annotate all videos inside a directory tree
python scripts/detect_faces.py <DIR_PATH> --recursive
# with docker
./run.sh <DIR_PATH> --recursive
```

For more usage information, run the script with the ```--help``` flag.

## Try the demo

If you want to visualize the detections in a video file before generating
annotation files, you can run ```python scripts/demo.py <VIDEO_FILE>```

## Acknowledgements

This repo uses pre-trained face detection and recognition models provided by
[insightface](https://github.com/deepinsight/insightface).
