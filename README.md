# RMS Norm CUDA

This repository contains a CUDA implementation of RMS norm.

## Prerequisites

Ensure that Docker is installed on your system.

## Running the Code

Follow these steps to run the code:

1. Clone the repository to your local machine:

    ```bash
    git clone <repository-url>
    ```

2. Navigate to the `docker` folder in the repository directory:

    ```bash
    cd docker/
    ```

3. Build the Docker image:

    ```bash
    ./build.sh
    ```

4. Run the Docker container:

    ```bash
    ./run.sh
    ```

The correctness tests will be executed inside the Docker container and the results will be displayed.
Then, the code will run the kernels to check outputs and time to run through a larger number of batches.
You will have to wait a couple minutes for torch to build the cuda kernels

## Configuration

You can change the `.env` file in the `docker` folder to set `CUDA_VISIBLE_DEVICES` or the Docker image name.

## Tests

The `pytest` command in the `entrypoint.sh` runs tests in `test_rms.py` to check correctness of the kernels