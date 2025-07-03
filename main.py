from GradCAM import GRADCAM
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from utils import get_image_net_single_image, get_image_net_transform

print("🚀 Starting Grad-CAM visualization script...")

# Load a pre-trained model (e.g., ResNet18)
print("📦 Loading pre-trained ResNet18 model...")
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
print("✅ Model loaded successfully.")

# Choose the target layer (e.g., the last convolutional layer in the backbone)
# For ResNet, this is often the last layer in layer4.
target_layer_name = 'layer4'
print(f"🎯 Target layer set to: '{target_layer_name}'")

# Create a Grad-CAM object
print("🛠️ Creating Grad-CAM object...")
grad_cam = GRADCAM(model, target_layer_name)
print("✅ Grad-CAM object created.")

# Load an example image and preprocess it
print("🖼️ Loading and preprocessing input image...")
input_image = get_image_net_single_image()
input_image_tensor = get_image_net_transform()(input_image)
print("✅ Image loaded and preprocessed.")

# Add a batch dimension
input_batch = input_image_tensor.unsqueeze(0)  # Shape: [1, C, H, W]
print("➕ Added batch dimension to input tensor.")

# Ensure the input tensor requires gradients
# input_batch.requires_grad_(True)

# Compute the Grad-CAM heatmap
# You can specify class_idx=... or leave it as None to use the top prediction
# For example, to get the heatmap for 'tiger cat' (ImageNet class 292): class_idx=292
print("🔥 Computing Grad-CAM heatmap...")
heatmap = grad_cam(input_batch, class_idx=None) # heatmap shape: [1, H_orig, W_orig]
print("✅ Heatmap computed.")


# resize the heatmap to match the input image size
print("🔄 Resizing heatmap to match input image size...")
heatmap = grad_cam.resize_heatmap(heatmap, input_image.size)  # Resize heatmap to original image size
print("✅ Heatmap resized.")

# Convert tensors to numpy arrays for visualization
input_np = input_image # Keep as PIL image for display
heatmap_np = heatmap.cpu().detach().numpy() # Heatmap is already [B, H_orig, W_orig] and normalized [0,1]
print("🔄 Converted heatmap tensor to numpy array.")
# Convert to [0, 255] range for visualization
heatmap_np = (heatmap_np * 255).astype(np.uint8)
print("✅ Heatmap converted to uint8.")
# Remove the batch dimension for visualization
heatmap_np = np.squeeze(heatmap_np)
\
# Use matplotlib to visualize
print("📊 Generating visualization...")
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(input_np)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(input_np)
# Apply a colormap (like 'jet') to the heatmap and overlay it
# Use the resized heatmap for overlay
plt.imshow(heatmap_np, cmap='jet', alpha=0.5)
plt.title(f"Grad-CAM ({target_layer_name})")
plt.axis('off')
print("✅ Visualization generated.")

# save
print("💾 Saving output image...")
plt.savefig('grad_cam_output.png', bbox_inches='tight', dpi=300)
print("✅ Grad-CAM output saved as 'grad_cam_output.png'.")

print("👀 Displaying visualization...")
plt.show()
print("✅ Script finished.")
