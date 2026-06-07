# Architecture

SUT-ArUcoNet v1 is a staged ArUco marker inference pipeline for `DICT_4X4_1000`.

## Pipeline

1. A `YOLO26n` one-class detector proposes marker bounding boxes at 416 px input resolution.
2. `CornerRefiner64` receives a 64x64 crop around each proposal and regresses four marker corners.
3. The marker is rectified from the refined corners.
4. A deterministic `DICT_4X4_1000` decoder reads the marker ID and accepts it only when the configured Hamming threshold is satisfied.

## Optional MIP36h12 Extension

The release also contains a deterministic MIP36h12 decoder extension using OpenCV's local ArUco dictionary. This path uses the same detector and corner refiner but must be labelled as `SUT-ArUcoNet v1 MIP extension`. It is not part of the core SUT-ArUcoNet v1 evidence.

## Rationale

The detector is intentionally lightweight so it can be used in robot camera workflows. The separate corner-refinement stage keeps geometric localization measurable instead of relying only on the detector box. The deterministic core decoder gives an interpretable ID acceptance rule through the Hamming threshold.

## Scope

The core release supports `DICT_4X4_1000`. MIP36h12 support is an optional extension only and should not be mixed with core SUT-ArUcoNet v1 results.