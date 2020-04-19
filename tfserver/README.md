# tf Server

Tensorflow server is an implementation of KFServing for serving Tensorflow models, and provides a Tensorflow model implementation for prediction, pre and post processing.

To start the server locally for development needs, run the following command under this folder in your github repository.

```
pip install -e .
```

Once tf server is up and running, you can check for successful installation by running the following command

```
python3 -m tfserver
usage: __main__.py [-h] [--http_port HTTP_PORT] [--grpc_port GRPC_PORT]
                   [--workers WORKERS] --model_dir MODEL_DIR
                   [--model_name MODEL_NAME] --input_name INPUT_NAME
                   --output_name OUTPUT_NAME

optional arguments:
  -h, --help            show this help message and exit
  --http_port HTTP_PORT
                        The HTTP Port listened to by the model server.
  --grpc_port GRPC_PORT
                        The GRPC Port listened to by the model server.
  --workers WORKERS     The number of works to fork
  --model_dir MODEL_DIR
                        The path of the model directory
  --model_name MODEL_NAME
                        The name that the model is served under.
  --input_name INPUT_NAME
                        The model of input layer name.
  --output_name OUTPUT_NAME
                        The name of output layer name.
```

You can now point to your `tf` model directory and use the server to load the model and test for prediction. Model and associaed model class file can be on local filesystem, S3 compatible object storage, Azure Blob Storage, or Google Cloud Storage. Please follow [this sample](https://github.com/kubeflow/kfserving/tree/master/docs/samples/tf) to test your server by generating your own model.

## Development

Install the development dependencies with:

```bash
pip install -e .[test]
```

To run tests, please change the test file to point to your model.pt file. Then run the following command:

```bash
make test
```

To run static type checks:

```bash
mypy --ignore-missing-imports tfserver
```

An empty result will indicate success.

## Building your own tf server Docker Image

You can build and publish your own image for development needs. Please ensure that you modify the inferenceservice files for tf in the api directory to point to your own image.

To build your own image, navigate up one directory level to the `python` directory and run:

```bash
docker build -t docker_user_name/tfserver -f tf.Dockerfile .
```

To push your image to your dockerhub repo,

```bash
docker push docker_user_name/tfserver:latest
```
