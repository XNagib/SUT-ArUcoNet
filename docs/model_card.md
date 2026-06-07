# Model Card

## Model

SUT-ArUcoNet v1 public inference release.

## Intended Use

Detect and decode `DICT_4X4_1000` ArUco markers in camera images for robotics perception.

The optional MIP36h12 path is an extension only. It is not core SUT-ArUcoNet v1 evidence.

## Inputs

RGB or BGR camera images as NumPy arrays. The default detector input size is 416.

## Outputs

Detected marker ID, decoded status, confidence fields, four marker corners, Hamming distance, and optional pose when camera intrinsics and marker size are supplied through the Python API.

## Architecture

The pipeline uses a `YOLO26n` marker proposal detector, a `CornerRefiner64` crop model on 64x64 proposal crops, and a deterministic rectified-bit `DICT_4X4_1000` decoder.

The optional MIP36h12 extension uses the same detector and corner refiner with a deterministic MIP36h12 rectified-bit decoder backed by the local OpenCV ArUco dictionary.

## Training Data Summary

The detector was trained on a size and angle balanced marker proposal dataset with 4,174 training images and 12,973 marker instances. The validation split contained 762 reviewed camera-wall images with 3,121 marker instances.

The detector training mixture included generated `DICT_4X4_1000` images, augmented reviewed camera-wall images, random multi-tag generated scenes, and public synthetic marker data for proposal geometry. Dataset files are not included in this public release.

The corner refiner was trained on 265,488 crop samples and validated on 9,605 crop samples. Raw datasets and training files are not included in this public release.

## Limitations

Partial markers may be detected but ID decoding is only trusted when deterministic decoding passes the Hamming threshold. Strong glare, severe occlusion, very large close-up crops, and extreme viewing angles may reduce reliability. Runtime depends on hardware, backend, camera resolution, and detector input size.

Core support is `DICT_4X4_1000`. MIP36h12 is an optional extension and should be labelled separately when used.

## License

This public inference release is distributed under AGPL-3.0 because it uses the Ultralytics detector runtime and checkpoint format.