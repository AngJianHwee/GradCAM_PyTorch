from GradCAM import GRADCAM
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch # Added for denormalization

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
if input_image is None:
    print("❌ Failed to load input image. Exiting.")
    exit()
# Compute the Grad-CAM heatmap and get the preprocessed image tensor
# You can specify class_idx=... or leave it as None to use the top prediction
# For example, to get the heatmap for 'tiger cat' (ImageNet class 292): class_idx=292
print("🔥 Computing Grad-CAM heatmap and retrieving preprocessed image...")
heatmap, pre_processed_image_tensor = grad_cam(get_image_net_transform()(input_image).unsqueeze(0), class_idx=None)
print("✅ Heatmap and preprocessed image retrieved.")


# resize the heatmap to match the input image size manually
print("🔄 Resizing heatmap to match input image size...")
# Get the original size of the input image
# input_image.size returns (width, height), but resize expects (height, width)
target_heatmap_size = (input_image.size[1], input_image.size[0]) 
heatmap = torchvision.transforms.functional.resize(
    heatmap,
    size=target_heatmap_size,
    interpolation=torchvision.transforms.InterpolationMode.BILINEAR
)
print(f"✅ Heatmap resized to: {target_heatmap_size}.")

# Convert tensors to numpy arrays for visualization
pre_processed_image = denormalize_image(pre_processed_image_tensor) # Denormalize the preprocessed tensor for display
heatmap_np = heatmap.cpu().detach().numpy() # Heatmap is already [B, 1, H, W]
print("🔄 Converted heatmap tensor to numpy array.")
# Convert to [0, 255] range for visualization
heatmap_np = (heatmap_np * 255).astype(np.uint8)
print("✅ Heatmap converted to uint8.")
# Remove the batch dimension for visualization
heatmap_np = np.squeeze(heatmap_np)

# Use matplotlib to visualize
print("📊 Generating visualization...")
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(pre_processed_image)
plt.title("Pre-processed Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(pre_processed_image)
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
