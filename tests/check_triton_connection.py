import numpy as np
import tritonclient.http as httpclient


def test_triton_connection():
    try:
        # Create Triton HTTP client (Triton is listening on HTTP port 8000)
        triton_client = httpclient.InferenceServerClient(url="localhost:8000")

        # Verify server is live and ready
        if not triton_client.is_server_live():
            print("Error: Triton Server is not live.")
            return False
        if not triton_client.is_server_ready():
            print("Error: Triton Server is not ready.")
            return False

        print("Triton Server is Live and Ready!")

        # Verify model is ready
        model_name = "yolov8_onnx"
        if not triton_client.is_model_ready(model_name):
            print(f"Error: Model '{model_name}' is not ready.")
            return False

        print(f"Model '{model_name}' is Ready!")

        # Get model metadata
        metadata = triton_client.get_model_metadata(model_name)
        print("Model Metadata:")
        print(metadata)

        # Perform a dummy inference request
        input_data = np.random.randn(1, 3, 640, 640).astype(np.float32)
        inputs = [httpclient.InferInput("images", input_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)

        outputs = [httpclient.InferRequestedOutput("output0")]

        response = triton_client.infer(model_name, inputs, outputs=outputs)
        output_data = response.as_numpy("output0")

        print(f"Inference Success! Output shape: {output_data.shape}")
        return True

    except Exception as e:
        print(f"Connection/Inference failed: {e}")
        return False


if __name__ == "__main__":
    success = test_triton_connection()
    if success:
        exit(0)
    else:
        exit(1)
