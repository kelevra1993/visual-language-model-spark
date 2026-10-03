"""
Measures the inference throughput of a trained experiment of any task, run locally in this process and served by
Triton as TorchScript and ONNX (TensorRT), on the images of deployment/inference_data. Run as a script once the
models are exported by deployment/export_model.py and the Triton server is started.
"""

from pathlib import Path

from vision_training.deployment.deployment_utilities import (load_inference_inputs, load_local_inference_model,
                                                             measure_local_throughput, measure_triton_throughput)

if __name__ == "__main__":
    # Trained experiment of any task, giving the input size, resizing mode and sequence length (and the weights of the
    # local run and the names of its served models)
    experiment_folder = Path(__file__).absolute().parents[2] / "experiments" / "poseidon_experiment_4"
    # Folder of the images sent to the models, one image per request (one window of consecutive images for the
    # sequence models, and two windows, the after one starting where the before one ends, for the comparison model)
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

    inputs = load_inference_inputs(image_folder=image_folder, experiment_folder=experiment_folder)

    # Local run, the inference model in this process without Triton
    inference_model, specification = load_local_inference_model(experiment_folder=experiment_folder,
                                                                iteration=iteration)
    measure_local_throughput(inputs=inputs, inference_model=inference_model, number_iterations=number_iterations,
                             number_warmup_iterations=number_warmup_iterations)

    # Triton runs, the TorchScript model and the ONNX model accelerated by TensorRT
    output_names = [output["name"] for output in specification["outputs"]]
    for export_format in ["torchscript", "onnx"]:
        measure_triton_throughput(inputs=inputs, model_name=f"{specification['model_name']}_{export_format}",
                                  output_names=output_names, url=url, use_shared_memory=use_shared_memory,
                                  number_iterations=number_iterations,
                                  number_warmup_iterations=number_warmup_iterations)
