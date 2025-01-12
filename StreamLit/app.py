import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import comet_ml
from comet_ml import Experiment

# Set up the CometML experiment
experiment = Experiment(
    api_key="your_cometml_api_key",
    project_name="digit-classification",
    workspace="your_workspace_name"
)

# Define the CNN model class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# Load the model and weights
def load_model(weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

# Preprocess the uploaded image
def preprocess_image(image):
    # Ensure the image is converted to grayscale and resized correctly
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((28, 28)),                # Resize to 28x28
        transforms.ToTensor(),                      # Convert to Tensor
        transforms.Normalize((0.5,), (0.5,))        # Normalize with mean=0.5 and std=0.5
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Predict the digit
def predict(model, image_tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)
    with torch.no_grad():  # Disable gradient calculation
        output = model(image_tensor)
    prediction = torch.argmax(output, dim=1).item()  # Get the class index with the highest score
    return prediction

# Streamlit App
st.title("Digit Classification with CNN")
st.write("Upload an image of a handwritten digit (0-9), and the model will classify it.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Log the uploaded image to CometML
    experiment.log_image(image)

    # Log the image type and size
    experiment.log_text(f"Uploaded image type: {uploaded_file.type}")
    experiment.log_text(f"Uploaded image size: {uploaded_file.size} bytes")

    # Ensure the image is compressed and processed to 28x28
    st.write("Processing the image...")

    # Load the model
    model = load_model("mnist_cnn_model.pth")

    # Log the model loading
    experiment.log_text("Model loaded successfully.")

    # Preprocess the image
    image_tensor = preprocess_image(image)

    # Log the shape of the preprocessed image tensor
    experiment.log_text(f"Shape of preprocessed image tensor: {image_tensor.shape}")

    # Predict the digit
    prediction = predict(model, image_tensor)

    # Log the prediction result
    experiment.log_metric("predicted_digit", prediction)

    # Display the prediction
    st.write(f"Predicted Digit: {prediction}")

    # Debugging information
    st.write("Shape of preprocessed image tensor:", image_tensor.shape)

    # Log additional metrics and information as needed
    experiment.log_text(f"Prediction completed for the image with predicted digit: {prediction}")
