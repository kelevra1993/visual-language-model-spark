# Serving Benchmark Plan

- **Drivers:** `scripts/deployment/benchmark_serving_formats.py` (exports, speed and memory) and `scripts/deployment/evaluate_served_formats.py` (test metric), their settings in their main blocks, run from the repository root with `.venv/bin/python -m scripts.deployment.<name>`, one after the other; each resumes from its results file, so a stopped sweep restarts where it stopped.
- **Outputs:** `~/Data/serving_benchmark/`, outside the repository: the Triton model repositories of every precision setting, the TensorRT engine caches, the inputs sent to the models, the raw file of every load, the logs, `results.csv` and `test_metrics.csv`. Only this plan and the copies next to it are committed: the two results files of the sweep (`serving_benchmark_results.csv`, `serving_benchmark_test_metrics.csv`), the listing of the model repositories (`serving_benchmark_model_repository_files.txt`) and the results of the exported NMS fix (`serving_benchmark_nms_fix_test_metrics.csv`, `serving_benchmark_nms_fix_loads/`).
- **Machine:** DGX Spark (GB10, unified memory), Triton `nvcr.io/nvidia/tritonserver:26.01-py3` (Triton 2.65.0, ONNX Runtime 1.23.2, TensorRT 10.14.1), `perf_analyzer` of `nvcr.io/nvidia/tritonserver:26.01-py3-sdk`, no training running.

## Table Of Content
- [Rows](#rows)
- [Models](#models)
- [Server Settings And Loads](#server-settings-and-loads)
- [Protocol](#protocol)
- [Outputs](#outputs)
- [Results](#results)
- [Status](#status)

## Rows
Before reading a result, find its row: every way of running an exported model that the repository writes today, the reference being the eager model in the process. INT8 and FP8 are left out (user decision, 2026-11-17), and BF16 cannot be served by the ONNX Runtime backend of this Triton image (row 7).

| Row | Where it runs | Format | Precision | Produced by |
|---|---|---|---|---|
| 1 | In the process | Eager PyTorch (checkpoint) | FP32, convolutions in TF32 (PyTorch default) | `load_local_inference_model` |
| 2 | In the process | TorchScript `model.pt` (`torch.jit.load`) | FP32 | `export_experiment`, `formats=["torchscript"]` |
| 3 | Triton | TorchScript `model.pt` (`pytorch_libtorch`) | FP32 | the same file |
| 4 | Triton | ONNX `model.onnx`, ONNX Runtime CUDA | FP32 | `tensorrt_precision=None` |
| 5 | Triton | ONNX with TensorRT | FP32, TF32 allowed by TensorRT | `tensorrt_precision="FP32"` |
| 6 | Triton | ONNX with TensorRT | FP16 where TensorRT picks it | `tensorrt_precision="FP16"` |
| 7 | Triton | ONNX with TensorRT | BF16 | not measured: the backend refuses the load with `precision_mode` "BF16" ("unsupported precision mode 'BF16' is requested"), takes only the TensorRT parameters of its own list (`trt_bf16_enable` is not one), and passes its options to ONNX Runtime, which then ignores `ORT_TENSORRT_BF16_ENABLE=1` (its log reads `trt_bf16_enable: 0`) |
| 8 | Triton | Native TensorRT engine `model.plan` | FP16 | not implemented, not measured |

## Models
To cover the three kinds of graphs TensorRT meets in the repository (a plain classifier, a detector with its NMS inside the model, a recurrent sequence model), the rows run on three trained runs, read from their copies outside the repository (`~/Data/tutorial_runs`, the same checkpoints as the copies in `experiments/` and `hyperparameter_experiments/`):

| Model | Run | Checkpoint | Input sent by `perf_analyzer` | Test split |
|---|---|---|---|---|
| CLS, 512 pixels, 9 classes | `readme_screenshots/CLS_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_HFLAT.128-HLN-CWB_MC9` (the README classifier) | 1500 | the first test image | `classifier_data/synthetic_shapes/test`, 50,000 images |
| DET, 1024 pixels, 35 classes, NMS inside the model | `hyperparameter_experiments/object_detection/run_9/DET_IN-KAR-1024_..._MC35` (reference of the detection benchmark) | 15,000 | the first test image (8.5 objects per image on average) | `object_detection_data/synthetic_hard_shapes/test_10000`, 10,000 images |
| M2M, 256 pixels, bidirectional LSTM, 71 classes | `tutorials/M2M_IN-KAR-256_SE-RES-BN-B1x16.2x16.2x16.4x32.5x32.5x32-PT_T50_LSTM2x96-BI_HFLAT.96.96-HLN-CWB_MC71` | 2000 | the first test sequence of 7 frames, the median length (2 to 12 frames in the test split, 50 at most in training) | `sequence_data/synthetic_changing_shapes/test_10000`, 10,000 sequences |

The inputs are real test samples preprocessed by `load_inference_inputs` (the resize and padding of the data loaders), so that the NMS of the detector works on real detections. For M2M, the Images Per Second count frames.

The ONNX file of M2M could not be served as exported: the dynamo export of its bidirectional LSTM writes the 50 frames of its trace into every shape after the recurrent layer (output `class_probabilities` declared [-1, 50, 71] in the file, [-1, -1, 71] in `config.pbtxt`), and Triton refuses the load ("the model expects 3 dimensions (shape [-1,50,71]) but the model configuration specifies ... [-1,-1,71]"); the graph itself computes one row per input frame (ONNX Runtime on the CPU only warns, which is why the export check, on the inputs only, passed). The driver repaired the copies of the benchmark (the output frame dimension made the one of the input, the intermediate shapes dropped), and the test metric checked that the repaired model gives the eager results. Since 2026-11-18 the export makes that change itself (`restore_dynamic_output_frames` of `vision_training/deployment/export_formats.py`, for the sequence labeller and the frame selector), its dimension check covers the outputs, and the drivers no longer repair.

## Server Settings And Loads
To keep the sweep to about an hour and a half, every Triton row (3 to 6) is measured with the configuration of its export over the whole grid, and the other server settings only for TensorRT FP16 at batch 1 (focused scope, user decision 2026-11-17, against 864 loads and about 5 hours for the full grid). The server keeps only the measured model loaded (`--model-control-mode=explicit`), its settings changed by a configuration given at load time:

| Rows | Server settings (dynamic batching, `instance_group` count) | Transports | Client batch sizes | Concurrencies (requests in flight) | Loads |
|---|---|---|---|---|---|
| Every row of every model | on (`max_queue_delay_microseconds` 1000, as exported), 1 | gRPC, system shared memory | 1, 4, 16 | 1, 4, 16 | 216 |
| TensorRT FP16 of every model | off, 1; on, 2; off, 2 | gRPC, system shared memory | 1 | 1, 4, 16 | 54 |

Every served model keeps the `max_batch_size` 16 of its export. The in-process rows (1 and 2) run one call at a time at the batch sizes 1, 4 and 16.

## Protocol
Whenever a number of a result is read, it was taken as follows:
- **Latency and throughput, shared memory (Triton):** `perf_analyzer` with the inputs in system shared memory, measurement windows of 2 seconds until three consecutive windows agree within 10 % (10 then 30 seconds when perf_analyzer stops without a measurement, as for the detector at batch 16, whose requests last several seconds), p50 and p99 of the client latency, throughput in inferences per second (`perf_analyzer` counts every sample of a batch, checked at batch 16: 16 / 37.1 milliseconds = 431 against 404 reported), times 7 frames for M2M; the GPU use and power of the server metrics; `perf_analyzer` runs in a container limited to 16 GB.
- **Latency and throughput, gRPC (Triton):** the tritonclient client of the repository (`run_triton_inference`, the client of `run_inference.py`), one process per request in flight (a client of its own, like a client of another pod, so that the clients share no Python interpreter lock) sending the same request again and again, 2 seconds of untimed requests then a window of 10 seconds; the requests ending within the window give p50, p99 and the throughput, two readings of the statistics of the server their queue and compute times, and the server metrics sampled every 0.5 second the GPU use and power. `perf_analyzer` is not used for gRPC: version 2.59.1 keeps about 10 MB per 512-pixel request sent without shared memory (gRPC or HTTP) until the end of its run (629 MiB after 100 requests, 3,624 MiB after 400, 35 MB flat with shared memory), which filled the 128 GB of the machine twice in the pilot.
- **Latency and throughput (in the process):** 20 untimed calls, then 200 timed calls with a CUDA synchronization after each, p50 and p99 of the call times, images per second over the timed calls.
- **Engine build:** the TensorRT engine cache of the precision setting (mounted at `/tmp/tensorrt_engine_cache`) is emptied before the first load of each model, then the first request at batch 1, 4 and 16 is timed: ONNX Runtime builds an engine at the first request and again at each larger batch. The same three first requests are timed for the rows without TensorRT.
- **Memory:** the drop of `MemAvailable` (`/proc/meminfo`) between the server with no model loaded and the server after the warm-up requests of the model, the GB10 sharing one memory between the CPU and the GPU (`nvidia-smi` gives no used memory); in the process, the drop from before the model is loaded to after its last timed call, and the peak reserved by the PyTorch allocator.
- **Test metric against FP32:** `scripts/deployment/evaluate_served_formats.py` sends the test split of every model through the eager inference model in the process (the reference) and through each served format (export configuration: dynamic batching on, one instance), in requests of at most 16 samples, and scores the outputs with the test evaluation of the trainer (`predict_batch` replaced by the requests); the gap is in points against the reference. Every row receives the same requests: the real frames of each sequence alone (the served sequence model takes no sequence mask), and for DET only the detections at the deployment score threshold (`Detector.inference.score_threshold` 0.5, written into the export) instead of the 0.05 of the training evaluation, so the DET mAP here is lower than the one of the run, the gap staying a gap of format; the proposal recall is not compared (the served detector returns no proposals).
- **Idle machine:** before each measurement the driver checks that no other process uses the GPU, waits while one does, and records any it saw.

## Outputs
Should a result need checking, its raw files are under `~/Data/serving_benchmark/`:

| Folder | Content |
|---|---|
| `model_repositories/<setting>/` | the exports of the three models for `onnxruntime_cuda` (TorchScript and ONNX), `tensorrt_fp32` and `tensorrt_fp16` (ONNX); the file list of the README classifier and detector folders of `onnxruntime_cuda` is copied next to this plan as `serving_benchmark_model_repository_files.txt`, drawn by the deployment tutorials |
| `tensorrt_engine_cache/<setting>/` | the engines built by ONNX Runtime, mounted into the Triton container |
| `inputs/<model>/` | the preprocessed test sample sent by `perf_analyzer` (one raw float32 file per input) and its source images |
| `loads/<model>/` | one file per load: the `perf_analyzer` CSV report and the log of its runs for shared memory, the JSON measure and latencies of the client for gRPC |
| `logs/` | the Triton server logs |
| `results.csv` | one line per measurement, copied next to this plan as `serving_benchmark_results.csv` |
| `test_metric/<model>/` | the evaluation files of every row of the test metric |
| `test_metrics.csv` | one line per scalar of every evaluated row, copied next to this plan as `serving_benchmark_test_metrics.csv` |

## Results
Should you need the numbers behind a serving choice, they come from `serving_benchmark_results.csv` (sweep of 2026-11-17 from 13:44 to 15:29, 324 measurements, none failed, no other process on the GPU, the memory never under 20 GB available) and `serving_benchmark_test_metrics.csv`. Throughputs count images, frames for M2M (7 per sequence); "shared memory" is `perf_analyzer` with system shared memory, "gRPC" the tritonclient client.

### One Client
In case a single caller sends one request at a time (concurrency 1), the exported configuration (dynamic batching on, 1 instance) gives:

| Model | Row | Batch 1 p50 / p99 (ms), gRPC or in-process call | Batch 1 p50 / p99 (ms), shared memory | Batch 16 images/s, gRPC or in-process call | Batch 16 images/s, shared memory | Memory (GB) |
|---|---|---|---|---|---|---|
| CLS | In process, eager | 2.2 / 4.7 | | 349 | | 1.74 |
| CLS | In process, TorchScript | 3.3 / 5.5 | | 355 | | 1.42 |
| CLS | Triton, TorchScript | 6.5 / 8.4 | 4.5 / 6.1 | 86 | 320 | 2.55 |
| CLS | Triton, ONNX Runtime CUDA | 5.1 / 6.5 | 3.6 / 5.0 | 93 | 435 | 3.93 |
| CLS | Triton, TensorRT FP32 | 4.9 / 6.0 | 3.0 / 4.2 | 99 | 646 | 4.08 |
| CLS | Triton, TensorRT FP16 | 3.9 / 5.1 | 2.5 / 3.6 | 109 | 1,215 | 3.98 |
| DET | In process, eager | 16.0 / 20.3 | | 11 | | 6.73 |
| DET | In process, TorchScript | 15.6 / 21.0 | | 11 | | 6.61 |
| DET | Triton, TorchScript | 31.4 / 49.2 | 24.3 / 28.5 | 8 | 11 | 8.72 |
| DET | Triton, ONNX Runtime CUDA | 23.3 / 28.0 | 17.0 / 22.0 | 14 | 30 | 13.66 |
| DET | Triton, TensorRT FP32 | 31.6 / 51.2 | 23.8 / 27.7 | 24 | 153 | 7.26 |
| DET | Triton, TensorRT FP16 | 38.4 / 44.5 | 17.6 / 22.6 | 27 | 246 | 5.99 |
| M2M | In process, eager | 3.6 / 4.8 | | 1,841 | | 2.23 |
| M2M | In process, TorchScript | 3.4 / 4.6 | | 2,013 | | 2.05 |
| M2M | Triton, TorchScript | 14.3 / 21.1 | 5.9 / 7.6 | 392 | 1,788 | 2.41 |
| M2M | Triton, ONNX Runtime CUDA | 16.5 / 18.7 | 5.3 / 7.0 | 426 | 2,207 | 5.49 |
| M2M | Triton, TensorRT FP32 | 14.6 / 16.2 | 4.2 / 5.7 | 437 | 3,357 | 2.73 |
| M2M | Triton, TensorRT FP16 | 13.9 / 17.4 | 3.1 / 4.3 | 470 | 6,249 | 2.47 |

### Highest Throughput Of The Exported Configuration
Suppose many callers share the server: the best throughput over the batch sizes 1, 4 and 16 and the concurrencies 1, 4 and 16, with the load reaching it:

| Model | Row | Best images/s, gRPC (batch, concurrency) | Best images/s, shared memory (batch, concurrency) | GPU use at best, shared memory |
|---|---|---|---|---|
| CLS | Triton, TorchScript | 344 (1, 4) | 394 (1, 4) | 89 % |
| CLS | Triton, ONNX Runtime CUDA | 385 (1, 16) | 498 (1, 4) | 82 % |
| CLS | Triton, TensorRT FP32 | 522 (1, 16) | 677 (16, 16) | 96 % |
| CLS | Triton, TensorRT FP16 | 860 (1, 16) | 1,326 (16, 16) | 71 % |
| DET | Triton, TorchScript | 40 (1, 4) | 41 (1, 1) | 62 % |
| DET | Triton, ONNX Runtime CUDA | 50 (4, 4) | 58 (1, 1) | 70 % |
| DET | Triton, TensorRT FP32 | 122 (16, 16) | 158 (16, 16) | 96 % |
| DET | Triton, TensorRT FP16 | 149 (16, 16) | 253 (16, 16) | 96 % |
| M2M | Triton, TorchScript | 1,717 (1, 16) | 1,988 (1, 16) | 87 % |
| M2M | Triton, ONNX Runtime CUDA | 1,862 (1, 16) | 2,269 (16, 16) | 95 % |
| M2M | Triton, TensorRT FP32 | 2,460 (1, 16) | 3,483 (16, 4) | 96 % |
| M2M | Triton, TensorRT FP16 | 3,661 (1, 16) | 6,623 (16, 4) | 95 % |

### First Requests After A Load
Before a model serves at full speed, its first request at every larger batch builds a TensorRT engine (cache emptied before the first load):

| Model | Row | First request batch 1 (s) | then batch 4 (s) | then batch 16 (s) |
|---|---|---|---|---|
| CLS | Triton, TorchScript | 0.26 | 0.34 | 0.52 |
| CLS | Triton, ONNX Runtime CUDA | 0.55 | 0.07 | 0.23 |
| CLS | Triton, TensorRT FP32 | 5.11 | 6.40 | 9.97 |
| CLS | Triton, TensorRT FP16 | 5.64 | 7.92 | 9.63 |
| DET | Triton, TorchScript | 0.35 | 1.76 | 3.77 |
| DET | Triton, ONNX Runtime CUDA | 0.12 | 0.28 | 1.50 |
| DET | Triton, TensorRT FP32 | 11.94 | 17.06 | 34.40 |
| DET | Triton, TensorRT FP16 | 13.24 | 19.47 | 27.48 |
| M2M | Triton, TorchScript | 0.07 | 0.56 | 0.79 |
| M2M | Triton, ONNX Runtime CUDA | 0.03 | 0.10 | 0.38 |
| M2M | Triton, TensorRT FP32 | 7.10 | 8.85 | 13.00 |
| M2M | Triton, TensorRT FP16 | 10.12 | 10.77 | 13.75 |

### Server Settings Of TensorRT FP16 At Batch 1
If one wishes to change the configuration the export writes, these are the images per second of TensorRT FP16 at batch 1 under the four server settings:

| Model | Transport | Server setting | Concurrency 1 | Concurrency 4 | Concurrency 16 | p50 at 16 (ms) |
|---|---|---|---|---|---|---|
| CLS | gRPC | batching on, 1 instance (exported) | 250 | 586 | 860 | 19.2 |
| CLS | gRPC | batching off, 1 instance | 318 | 1,022 | 1,295 | 12.2 |
| CLS | gRPC | batching on, 2 instances | 240 | 574 | 1,022 | 15.9 |
| CLS | gRPC | batching off, 2 instances | 313 | 943 | 1,385 | 11.5 |
| CLS | shared memory | batching on, 1 instance (exported) | 390 | 784 | 1,168 | 13.6 |
| CLS | shared memory | batching off, 1 instance | 835 | 1,395 | 1,391 | 11.3 |
| CLS | shared memory | batching on, 2 instances | 388 | 788 | 1,068 | 14.9 |
| CLS | shared memory | batching off, 2 instances | 692 | 1,737 | 1,747 | 8.9 |
| DET | gRPC | batching on, 1 instance (exported) | 30 | 89 | 131 | 126.1 |
| DET | gRPC | batching off, 1 instance | 42 | 64 | 63 | 251.7 |
| DET | gRPC | batching on, 2 instances | 38 | 69 | 135 | 116.7 |
| DET | gRPC | batching off, 2 instances | 42 | 58 | 61 | 265.1 |
| DET | shared memory | batching on, 1 instance (exported) | 56 | 134 | 171 | 93.2 |
| DET | shared memory | batching off, 1 instance | 60 | 64 | 64 | 248.2 |
| DET | shared memory | batching on, 2 instances | 55 | 99 | 188 | 85.1 |
| DET | shared memory | batching off, 2 instances | 58 | 58 | 57 | 278.5 |
| M2M | gRPC | batching on, 1 instance (exported) | 501 | 2,415 | 3,661 | 30.5 |
| M2M | gRPC | batching off, 1 instance | 530 | 2,940 | 4,871 | 20.8 |
| M2M | gRPC | batching on, 2 instances | 513 | 1,793 | 4,162 | 23.8 |
| M2M | gRPC | batching off, 2 instances | 767 | 2,377 | 5,106 | 17.7 |
| M2M | shared memory | batching on, 1 instance (exported) | 2,192 | 4,000 | 5,436 | 20.3 |
| M2M | shared memory | batching off, 1 instance | 3,511 | 5,263 | 5,262 | 21.0 |
| M2M | shared memory | batching on, 2 instances | 2,149 | 3,974 | 4,441 | 25.0 |
| M2M | shared memory | batching off, 2 instances | 3,319 | 6,516 | 6,594 | 16.9 |

At one client, the queue delay of the dynamic batcher (1,000 microseconds) is paid by every request: the server queue time reads about 1.2 milliseconds with batching on against 0.1 to 0.2 without, for the same compute (CLS 0.8 milliseconds, DET 16 milliseconds, M2M 1.4 milliseconds), so the CLS p50 over shared memory falls from 2.50 to 1.13 milliseconds without batching.

### Test Metric Against The Eager Model
So that a faster row is not bought with wrong outputs, the test split of each model went through every row (15:29 to 17:31), its headline metric compared in points with the eager model:

| Model | Metric | Row | Value | Against eager (points) | Evaluated | Duration (seconds) |
|---|---|---|---|---|---|---|
| CLS | Accuracy | In the process, eager PyTorch | 82.73 % | reference | 50000 Test Images | 195 |
| CLS | Accuracy | Triton, TorchScript | 82.73 % | 0.00 | 50000 Test Images | 627 |
| CLS | Accuracy | Triton, ONNX Runtime CUDA | 82.72 % | 0.00 | 50000 Test Images | 582 |
| CLS | Accuracy | Triton, ONNX with TensorRT FP32 | 82.72 % | -0.01 | 50000 Test Images | 551 |
| CLS | Accuracy | Triton, ONNX with TensorRT FP16 | 82.70 % | -0.03 | 50000 Test Images | 511 |
| DET | mAP@0.5/area | In the process, eager PyTorch | 86.77 % | reference | 10000 Test Images | 924 |
| DET | mAP@0.5/area | Triton, TorchScript | 86.77 % | 0.00 | 10000 Test Images | 1,295 |
| DET | mAP@0.5/area | Triton, ONNX Runtime CUDA | 86.77 % | +0.01 | 10000 Test Images | 679 |
| DET | mAP@0.5/area | Triton, ONNX with TensorRT FP32 | 86.31 % | -0.46 | 10000 Test Images | 453 |
| DET | mAP@0.5/area | Triton, ONNX with TensorRT FP16 | 78.89 % | -7.88 | 10000 Test Images | 425 |
| M2M | Accuracy | In the process, eager PyTorch | 77.15 % | reference | 10000 Test Sequences, 69918 Labelled Frames | 146 |
| M2M | Accuracy | Triton, TorchScript | 77.16 % | +0.01 | 10000 Test Sequences, 69918 Labelled Frames | 171 |
| M2M | Accuracy | Triton, ONNX Runtime CUDA | 77.16 % | +0.01 | 10000 Test Sequences, 69918 Labelled Frames | 182 |
| M2M | Accuracy | Triton, ONNX with TensorRT FP32 | 77.16 % | 0.00 | 10000 Test Sequences, 69918 Labelled Frames | 203 |
| M2M | Accuracy | Triton, ONNX with TensorRT FP16 | 77.17 % | +0.02 | 10000 Test Sequences, 69918 Labelled Frames | 214 |

The gap of DET is one of precision, not of recall: TensorRT FP16 finds the same objects (mAR@0.5 -0.22 points) but returns extra detections, large objects losing most:

| Row | mAP@0.5/area | mAP@[0.5:0.95]/area | mAR@0.5/all | mAP@0.5 By Size/small | mAP@0.5 By Size/medium | mAP@0.5 By Size/large |
|---|---|---|---|---|---|---|
| In the process, eager PyTorch | 86.77 % | 68.27 % | 87.53 % | 73.85 % | 93.04 % | 87.52 % |
| Triton, TorchScript | 86.77 % (0.00) | 68.27 % (0.00) | 87.53 % (0.00) | 73.87 % (+0.03) | 93.04 % (0.00) | 87.51 % (-0.01) |
| Triton, ONNX Runtime CUDA | 86.77 % (+0.01) | 68.28 % (+0.01) | 87.54 % (+0.01) | 73.88 % (+0.03) | 93.04 % (0.00) | 87.53 % (+0.01) |
| Triton, ONNX with TensorRT FP32 | 86.31 % (-0.46) | 68.03 % (-0.24) | 87.03 % (-0.49) | 72.35 % (-1.49) | 92.85 % (-0.19) | 87.44 % (-0.08) |
| Triton, ONNX with TensorRT FP16 | 78.89 % (-7.88) | 62.54 % (-5.73) | 87.31 % (-0.22) | 66.88 % (-6.97) | 89.05 % (-4.00) | 78.29 % (-9.23) |

In order to find the cause, the first 200 test images were sent again to the eager model and to the TensorRT engines, one image per request then 16, and scored against their ground truth at IoU 0.5 (pooled over the classes, so not the mAP of the evaluation):

| Row | Request | True positives | False positives | False positives at 0.99 or more | Pooled AP@0.5 |
|---|---|---|---|---|---|
| In the process, eager PyTorch | 1 image | 1,531 | 168 | 0 | 86.81 % |
| Triton, ONNX with TensorRT FP16 | 1 image | 1,530 | 170 | 2 | 86.62 % |
| Triton, ONNX with TensorRT FP16 | 16 images | 1,529 | 752 | 60 | 76.59 % |
| Triton, ONNX with TensorRT FP32 | 16 images | 1,524 | 163 | 0 | 86.42 % |

The FP16 engine is right one image at a time and wrong in batches: by position in the request, the false positives read 7, 22, 41, 51, 47, 32, 54, 68, 63, 49, 72, 30, 59, 32, 56 and 69 (eager 7, 16, 14, 14, 6, 9, 12, 8, 11, 15, 11, 6, 10, 6, 9 and 14). The NMS inside the model explains this pattern: `suppress_groups_apart` moves the boxes of every image and class apart by their group index (image times 35 plus class) times the largest coordinate plus one, about 1,025 pixels, so the moved coordinates pass 65,504, the largest FP16 value, from about the 64th group, that is the last classes of the second image and every class of the later ones, whose duplicates are then no longer suppressed. Two other causes were ruled out: the box areas (FP16 keeps the 414 boxes of 65,504 square pixels or more of the eager model, with 4 same-class pairs above the NMS threshold in 200 images, all small) and the score resolution (the eager detector with its scores rounded to FP16 loses 0.00 points of mAP@0.5 over the test split). With dynamic batching on, the server forms such batches from requests of one image, so a client sending single images is not protected.

### Detector NMS Without Offsets
Once the cause was found, the exported NMS of the detectors stopped moving the groups apart (user decision, first of the follow-ups): `suppress_stacked_groups` of `vision_training/models/detection/postprocessing.py` is a custom operator that the ONNX export writes as one `NonMaxSuppression` whose batches are the groups (the images for the proposals, the images and classes for the detections), every coordinate staying in the image, TorchScript keeping the offsets in float32. run_9 exported again with it reads, on the test split through Triton and over shared memory with one client:

| Row | mAP@0.5 | mAP@[0.5:0.95] | mAR@0.5 | Batch 1 p50 (milliseconds) | Batch 16 (images per second) |
|---|---|---|---|---|---|
| In the process, eager PyTorch | 86.77 % | 68.27 % | 87.53 % | 16.0 | 11 |
| Triton, ONNX with TensorRT FP32, before | 86.31 % (-0.46) | 68.03 % (-0.24) | 87.03 % | 23.8 | 153 |
| Triton, ONNX with TensorRT FP32, after | 86.77 % (+0.01) | 68.28 % (+0.01) | 87.53 % | 7.5 | 189 |
| Triton, ONNX with TensorRT FP16, before | 78.89 % (-7.88) | 62.54 % (-5.73) | 87.31 % | 17.6 | 246 |
| Triton, ONNX with TensorRT FP16, after | 86.97 % (+0.21) | 68.32 % (+0.05) | 87.72 % | 7.7 | 189 |

These rows come from `~/Data/serving_benchmark_nms_fix/`, whose `test_metrics.csv` and `perf_analyzer` reports of `loads/` are copied next to this plan as `serving_benchmark_nms_fix_test_metrics.csv` and `serving_benchmark_nms_fix_loads/`.

Both engines now match the eager model, the small gap of FP32 having come from the same offsets. One image takes half the time, likely because every suppression now compares the 250 boxes of one class instead of the 8,500 pairs of the image; at batch 16 both precisions take the same 83 milliseconds of server compute, so FP16 no longer gains there (its 246 images per second before were measured with the broken suppression). The dynamic batcher no longer helps DET either: with single images over shared memory at 1, 4 and 16 requests in flight, the FP16 engine serves 125, 160 and 153 images per second with it and 149, 166 and 166 without (171 against 64 before the fix).

DIoU-NMS, the `Detector.nms_criterion` of the detection configurations since stage 5b, had the other FP16 overflow: `suppress_every_pair_by_distance` computed the areas in pixels inside the graph, infinite in FP16 from 256 x 256 pixels. On a copy of run_9 with DIoU-NMS (exponent 0.6), its FP16 engine gave 718 false positives on the first 200 test images sent one at a time against 174 for the eager model (pooled AP@0.5 78.39 % against 86.84 %); with the boxes in fractions of the image, which leave the IoU and the normalized centre distance unchanged, it gives the 174 false positives of the eager model (86.87 %).

### Findings
To read the tables in one pass:
- **TensorRT FP16 is accurate enough for CLS and M2M, not for DET.** CLS and M2M stay within 0.03 points of the eager model with every row; DET loses 7.88 points of mAP@0.5 with TensorRT FP16 (0.46 with TensorRT FP32) for the same recall: the group offsets of the NMS inside the model overflow in FP16 from the second image of a batch, so the engine keeps duplicates in every image of a request but the first (752 false positives against 170 one image at a time on 200 test images). Since the NMS without offsets, DET reads +0.21 points in FP16 and +0.01 in FP32 ([Detector NMS Without Offsets](#detector-nms-without-offsets)).
- **TensorRT FP16 is the fastest row for every model** (DET measured with the broken NMS, 189 images per second in FP16 and FP32 since its fix). With one client at batch 16 over shared memory: CLS 1,215 images per second (3.5 times the eager model, 1.9 times TensorRT FP32), DET 246 (22 times the eager model, 1.6 times TensorRT FP32), M2M 6,249 frames per second (3.4 times the eager model, 1.9 times TensorRT FP32). At batch 1 the gain is small or none: DET takes 17.6 milliseconds against 16.0 in the process, its 16 milliseconds of compute being the same.
- **TorchScript through Triton is not faster than the eager model**, at batch 16 (CLS 320 against 349 images per second, DET 11 against 11) and at batch 1 (CLS 4.5 against 2.2 milliseconds); ONNX Runtime CUDA lies between them and takes the most memory (DET 13.66 GB against 5.99 for TensorRT FP16).
- **TensorRT builds its engines while serving:** the first request of DET at batch 1, 4 and 16 waits 11.94, 17.06 and 34.40 seconds with FP32 (13.24, 19.47 and 27.48 with FP16), about a minute before the three batch sizes are served, against at most 3.77 seconds without TensorRT; `docker-compose.yml` does not mount the engine cache, so every recreated container builds them again.
- **The exported dynamic batching suits none of the three models.** At concurrency 16 and batch 1, TensorRT FP16 reached 171 DET images per second with it against 64 without, but with the broken NMS (153 against 166 since its fix), and CLS and M2M are faster without it and with two instances (CLS 1,747 against 1,168 images per second, M2M 6,594 against 5,436 frames per second over shared memory), and at one client batching off doubles CLS (835 against 390).
- **A gRPC request of large images costs more than the model.** One Python client sending batches of 16 over gRPC gets CLS 109 images per second against 1,215 over shared memory with TensorRT FP16 (DET 27 against 246, M2M 470 against 6,249): building and sending 50 MB (CLS) to 201 MB (DET) requests dominates. Over gRPC the best is many clients sending single images (CLS 860 images per second at batch 1 and concurrency 16, M2M 3,661 frames per second), the server batching them.
- **The eager and TorchScript DET lose from batches:** in the process a batch takes 16.0, 130.7 and 1,424.1 milliseconds at 1, 4 and 16 images (62, 30 and 11 images per second), about the square of the batch size, while the TensorRT FP16 engine takes 17.6, 27.5 and 64.0 milliseconds (56, 144 and 246 images per second). The PyTorch profiler of the eager model puts the cause in its two torchvision `nms` calls (proposals and detections), each given every box of the whole batch with the images moved apart (`suppress_groups_apart`): 4.2 milliseconds of GPU time at batch 1 and 1,313 at batch 16 (89 % of the 1,479), while the backbone and neck grow from 9.1 to 140.7 milliseconds, in proportion to the batch.

## Status
In order to follow the benchmark, this section keeps its steps.

- 2026-11-17: plan written, exports and pilot next.
- 2026-11-17, until 12:37: the three models exported for the three precision settings, every export matching the eager model; BF16 refused by the backend (row 7), INT8 and FP8 left out.
- 2026-11-17, about 12:44: pilot stopped when `perf_analyzer` over gRPC filled the memory of the machine; gRPC moved to the tritonclient client, `perf_analyzer` kept for shared memory in a container limited to 16 GB, the M2M ONNX outputs repaired.
- 2026-11-17, 13:44 to 15:29: speed sweep of the focused scope, 324 measurements, none failed.
- 2026-11-17, 15:29 to 17:31: test metric of the 15 rows (the eager model and the four served formats of each model), DET TensorRT FP16 7.88 points below the eager model.
- 2026-11-17, 17:34 to 18:00: the eager DET profiled (the two `nms` calls take 89 % of a batch of 16), and the FP16 gap traced to the group offsets of the NMS overflowing in batches (200 test images sent one and 16 at a time, eager scores rounded to FP16 over the test split). Benchmark done; the follow-ups are in the TODO of the README.
- 2026-11-17, 18:10 to 19:10: the exported NMS of the detectors without offsets (`suppress_stacked_groups`) and DIoU-NMS on boxes in fractions of the image; run_9 exported again (test metric of TensorRT FP16 and FP32, speed at batch 1 and 16, batching on and off) and a DIoU copy of run_9 checked on 200 test images.
- 2026-11-18: the ONNX export restores the output frames of the sequence labeller and frame selector itself (`restore_dynamic_output_frames`) and checks them, the repair of the drivers removed.
