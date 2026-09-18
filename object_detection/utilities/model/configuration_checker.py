
# verify consistency of thresholds for nms and foreground and background settings
# Make sure the choice of the detections are coherent across anchors, and detection information.
# Make sure image size is an exponent such of 2. So that we do not have issues with the pooling layers in terms of keeping track of downsampling.
# Make sure scales are integers