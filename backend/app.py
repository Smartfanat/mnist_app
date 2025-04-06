import os
import numpy as np
import cv2
import base64
import io
from flask import Flask, request, jsonify
from flask_cors import CORS

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image

import time
import traceback

MODEL_FILENAME = 'mnist_pytorch_cnn_v2.pth'
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)
IMAGE_DIM = 28
COMPILE_INFERENCE = True

app = Flask(__name__)
CORS(app)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input: 1x28x28
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # Output: 10x24x24
        self.pool1 = nn.MaxPool2d(2)                 # Output: 10x12x12
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)# Output: 20x8x8
        self.pool2 = nn.MaxPool2d(2)                 # Output: 20x4x4
        self.fc1 = nn.Linear(20 * 4 * 4, 50)         # 320 features -> 50
        self.fc2 = nn.Linear(50, 10)                 # 50 features -> 10 (classes 0-9)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = x.view(-1, 20 * 4 * 4) # Flatten the tensor (320 features)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Return raw logits
        return x

predict_device = torch.device("cpu")

print(f"--- PyTorch Information ---")
print(f"Torch Version: {torch.__version__}")
if hasattr(torchvision, '__version__'):
    print(f"Torchvision Version: {torchvision.__version__}")
print(f"* Using device for prediction: {predict_device}")
if predict_device.type == 'cuda':
    print(f"* GPU Name: {torch.cuda.get_device_name(0)}")
print(f"---")

model = None
model_instance = None
inference_compiled_success = False
try:
    print(f"* Attempting to load model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at specified path: {MODEL_PATH}")

    model_instance = Net().to(predict_device)

    model_instance.load_state_dict(torch.load(MODEL_PATH, map_location=predict_device))

    model_instance.eval()
    print(f"* Model architecture instantiated and state_dict loaded successfully.")

    if COMPILE_INFERENCE and hasattr(torch, 'compile') and torch.__version__.startswith('2.'):
        print("* Attempting to compile model for inference with torch.compile()...")
        compile_mode = "reduce-overhead"
        try:
            start_time = time.time()
            compiled_model = torch.compile(model_instance, mode=compile_mode, dynamic=True)
            end_time = time.time()
            model = compiled_model
            inference_compiled_success = True
            print(f"* Inference model compiled successfully with mode='{compile_mode}' in {end_time - start_time:.2f}s.")
        except Exception as compile_error:
            print(f"* WARNING: Inference model compilation failed (using uncompiled model): {compile_error}")
            model = model_instance
    else:
        if COMPILE_INFERENCE:
             print("* torch.compile() disabled or not available. Using uncompiled model for inference.")
        model = model_instance
    # ---------------------------------------

    print(f"* Model ready for predictions. (Compiled: {inference_compiled_success})")

except FileNotFoundError as fnf_error:
    print(f"!!! FATAL ERROR: Model file not found !!!")
    print(f"    {fnf_error}")
    print(f"    Please ensure '{MODEL_FILENAME}' exists in the backend directory and train_model.py ran successfully.")
    model = None
except Exception as load_error:
    print(f"!!! FATAL ERROR: Failed to load PyTorch model !!!")
    print(f"    Error Type: {type(load_error).__name__}")
    print(f"    Error Details: {load_error}")
    print(f"    Traceback:")
    traceback.print_exc()
    print(f"    Possible causes: Mismatched model definition (Net class) between train_model.py and app.py, corrupted model file, PyTorch installation issues.")
    model = None

preprocess_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_DIM, IMAGE_DIM), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def preprocess_image_pytorch(image_data_url):
    try:
        header, encoded_data = image_data_url.split(',', 1)
        decoded_data = base64.b64decode(encoded_data)
    except Exception as e:
        raise ValueError(f"Could not decode base64 image data: {e}")

    nparr = np.frombuffer(decoded_data, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if img_cv is None:
        raise ValueError("Could not decode image bytes using OpenCV.")

    if img_cv.shape[2] == 4:
        alpha = img_cv[:, :, 3]
        rgb = img_cv[:, :, :3]
        bg = np.ones_like(rgb, dtype=np.uint8) * 255
        alpha_normalized = alpha.astype(float) / 255.0
        blended_rgb = (rgb * alpha_normalized[..., np.newaxis] +
                       bg * (1.0 - alpha_normalized[..., np.newaxis]))
        img_cv_blended = blended_rgb.astype(np.uint8)
        img_gray = cv2.cvtColor(img_cv_blended, cv2.COLOR_BGR2GRAY)
    elif img_cv.shape[2] == 3:
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    elif img_cv.shape[2] == 1:
        img_gray = img_cv
    else:
        raise ValueError(f"Unsupported number of image channels: {img_cv.shape[2]}")


    img_inverted = cv2.bitwise_not(img_gray)

    image_pil = Image.fromarray(img_inverted)

    tensor = preprocess_transform(image_pil)

    tensor = tensor.unsqueeze(0)

    return tensor


#API for predictions
@app.route('/predict', methods=['POST'])
def predict_digit():
    if model is None:
        print("Prediction attempt failed: Model is not loaded.")
        return jsonify({"error": "Model not loaded or failed to load. Check backend logs."}), 503

    if not request.is_json:
         print("Prediction failed: Request is not JSON.")
         return jsonify({"error": "Request must be JSON"}), 400

    req_data = request.get_json()
    if not req_data or 'imageDataUrl' not in req_data:
        print("Prediction failed: Missing 'imageDataUrl' in JSON body.")
        return jsonify({"error": "Missing 'imageDataUrl' in request body"}), 400

    image_data_url = req_data['imageDataUrl']

    try:
        print("Received image data, starting preprocessing...")
        input_tensor = preprocess_image_pytorch(image_data_url)
        input_tensor = input_tensor.to(predict_device) # Move tensor to the prediction device
        print(f"Preprocessing complete. Tensor shape: {input_tensor.shape}, Device: {input_tensor.device}")

        print("Performing inference...")
        start_inf = time.time()
        with torch.no_grad():
            outputs = model(input_tensor)

        end_inf = time.time()
        print(f"Inference complete in {end_inf - start_inf:.4f} seconds.")

        probabilities_tensor = F.softmax(outputs, dim=1)
        _, predicted_idx = torch.max(probabilities_tensor, 1)

        predicted_digit = int(predicted_idx.item())
        probabilities_list = probabilities_tensor.cpu().numpy()[0].tolist()

        probabilities_dict = {str(i): prob for i, prob in enumerate(probabilities_list)}

        print(f"Prediction successful: Digit = {predicted_digit}")

        return jsonify({
            "prediction": predicted_digit,
            "probabilities": probabilities_dict
        })

    except ValueError as ve:
         print(f"! Image Processing Error: {ve}")
         return jsonify({"error": f"Image Processing Error: {ve}"}), 400
    except Exception as e:
        print(f"!!! Prediction Error: {type(e).__name__} - {e}")
        traceback.print_exc()
        return jsonify({"error": f"An internal error occurred during prediction."}), 500

# --- Health check ---
@app.route('/', methods=['GET'])
def health_check():
    model_status = "Loaded and Ready" if model is not None else "Not Loaded or Error"
    return jsonify({
        "status": "Backend is running!",
        "model_status": model_status,
        "model_file_checked": MODEL_PATH,
        "predict_device": str(predict_device),
        "inference_compiled": inference_compiled_success
        })

if __name__ == '__main__':
    port = int(os.environ.get('FLASK_PORT', 5001))
    use_threading = not (predict_device.type == 'cuda')
    print(f"* Starting Flask server on host 0.0.0.0, port {port}")
    print(f"* Debug mode: {app.debug}")
    print(f"* Threading enabled: {use_threading}")
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=True, threaded=use_threading)
