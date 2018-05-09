# CSE252C_Project

TODO:
- Faster R-CNN
    - Dataset [x]. NOTE: Should probably store the data as encoded jpegs + split data into smaller .tfrecords
    - Network
        - ResNet50 backbone [x]
        - Anchors
        - RoIPool
        - Classification and BB regression heads
    - Training
        - Training Loop
        - MS COCO
        - Validation
    - Inference

- Mask R-CNN
    - Network
        - Mask prediction head
    - Train and test on MS COCO keypoints

- SkeleNet
    - Predict 3D locations
    - Integrate constraints