from GradCAM import GRADCAM
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch # Added for denormalization

from utils import get_image_net_single_image, get_image_net_transform

def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalizes a tensor image and converts it to a NumPy array."""
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    denormalized_tensor = tensor * std + mean
    # Clamp values to [0, 1] and convert to HWC format for matplotlib
    return denormalized_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)

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
input_np = denormalize_image(input_image_tensor.unsqueeze(0)) # Denormalize the input tensor for display
heatmap_np = heatmap.cpu().detach().numpy() # Heatmap is already [B, H_orig, W_orig] and normalized [0,1]
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
