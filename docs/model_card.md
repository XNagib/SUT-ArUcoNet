# Model Card

## Model

SUT-ArUcoNet v1.0 public inference release.

## Intended Use

Detect and decode `DICT_4X4_1000` ArUco markers in camera images for robotics perception.

## Inputs

RGB/BGR camera images. The default detector input size is 416.

## Outputs

Detected marker ID, decoded status, confidence fields, 4 marker corners, Hamming distance, and optional pose when camera intrinsics and marker size are supplied through the Python API.

## Training Data Summary

The model was trained using generated and reviewed camera datasets containing 4x4_1000 marker observations with variations in size, viewpoint, lighting, blur, and camera-captured scenes. Raw datasets are not included in this public release.

## Approved Local Evaluation Summary

Reviewed camera validation, strict gate: recall 0.8537, precision 0.8659, mean corner error 2.3739 px, p95 corner error 5.5702 px, mean runtime 19.1137 ms.

Reviewed camera validation, lenient gate: recall 0.8745, precision 0.8870, mean corner error 2.5600 px, p95 corner error 6.1355 px, mean runtime 18.2409 ms.

## Limitations

Partial markers may be detected but ID decoding is only trusted when deterministic decoding passes the Hamming threshold. Strong glare, severe occlusion, very large close-up crops, and extreme viewing angles may reduce reliability. Runtime depends on hardware, backend, camera resolution, and detector input size.

## License

This public inference release is distributed under AGPL-3.0 because it uses the Ultralytics detector runtime and checkpoint format.
