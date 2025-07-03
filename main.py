from GradCAM import GRADCAM
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from utils import get_image_net_single_image, get_image_net_transform


# Load a pre-trained model (e.g., ResNet18)
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# Choose the target layer (e.g., the last convolutional layer in the backbone)
# For ResNet, this is often the last layer in layer4.
target_layer_name = 'layer4' 

# Create a Grad-CAM object
grad_cam = GRADCAM(model, target_layer_name)

# Load an example image and preprocess it
# You would replace 'path/to/your/image.jpg' with a real image path
try:
    input_image = get_image_net_single_image()
    input_image_tensor = get_image_net_transform()(input_image)

    # Add a batch dimension
    input_batch = input_image_tensor.unsqueeze(0)  # Shape: [1, C, H, W]

    # Ensure the input tensor requires gradients
    input_batch.requires_grad_(True)

    # Compute the Grad-CAM heatmap
    # You can specify class_idx=... or leave it as None to use the top prediction
    # For example, to get the heatmap for 'tiger cat' (ImageNet class 292): class_idx=292
    heatmap = grad_cam(input_batch, class_idx=None) # heatmap shape: [1, H_orig, W_orig]

    # Convert tensors to numpy arrays for visualization
    input_np = input_image # Keep as PIL image for display
    heatmap_np = heatmap.squeeze().cpu().numpy() # Remove batch dim and convert

    # Overlay heatmap on the image
    # Resize the heatmap to match the image dimensions if not already done in forward
    # (The forward method does this resizing and returns [B, H_orig, W_orig])

    # Use matplotlib to visualize
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(input_np)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(input_np)
    # Apply a colormap (like 'jet') to the heatmap and overlay it
    plt.imshow(heatmap_np, cmap='jet', alpha=0.5) 
    plt.title(f"Grad-CAM ({target_layer_name})")
    plt.axis('off')
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")
    # Handle specific exceptions if needed, e.g., file not found, model loading issues, etc.
