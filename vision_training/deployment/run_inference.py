from pathlib import Path

from vision_training.deployment.deployment_utilities import (load_inference_images, load_local_detector, measure_local_throughput,
                                             measure_triton_throughput)

if __name__ == "__main__":
    # Trained experiment giving the input size and resizing mode (and the weights of the local run)
    experiment_folder = Path(__file__).absolute().parents[2] / "experiments" / "poseidon_experiment_4"
    # Folder of the images sent to the models, one image per request
    image_folder = Path(__file__).absolute().parent / "inference_data"
    # Triton gRPC address of the server started with deployment/triton/docker-compose.yml
    url = "localhost:8001"
    # Send the images to Triton through system shared memory instead of inside the gRPC requests, which needs the
    # 'ipc: host' setting of deployment/triton/docker-compose.yml
    use_shared_memory = False
    # Number of timed requests per run, and of untimed requests made first while TensorRT and CUDA initialise
    number_iterations = 1000
    number_warmup_iterations = 20
    # Checkpoint iteration used by the local run, None for the last saved one
    iteration = None

    images = load_inference_images(image_folder=image_folder, experiment_folder=experiment_folder)

    # Local run, the detector in this process without Triton
    measure_local_throughput(images=images,
                             detector=load_local_detector(experiment_folder=experiment_folder, iteration=iteration),
                             number_iterations=number_iterations, number_warmup_iterations=number_warmup_iterations)

    # Triton runs, the TorchScript model and the ONNX model accelerated by TensorRT
    for model_name in ["object_detector_torchscript", "object_detector_onnx"]:
        measure_triton_throughput(images=images, model_name=model_name, url=url, use_shared_memory=use_shared_memory,
                                  number_iterations=number_iterations,
                                  number_warmup_iterations=number_warmup_iterations)
