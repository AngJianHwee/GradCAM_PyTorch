import torch
import torch.nn.functional as F

class GRADCAM:
    def __init__(self, model, target_layer):
        self.model = model
        # Ensure the model is in evaluation mode (important for consistent behavior)
        self.model.eval() 
        self.target_layer = target_layer
        
        # Lists to store the captured gradients and activations
        # These lists will be cleared before each forward pass
        self.gradients = []
        self.activations = []

        # Register hooks to capture gradients flowing back through and 
        # activations flowing forward from the target layer.
        self.hook_layers()

    def hook_layers(self):
        # Define the forward hook function
        def forward_hook(_module, _input, output):
            # The output of the module is the activation map Ak (or batch of Ak's)
            self.activations.append(output)

        # Define the backward hook function
        def backward_hook(_module, _grad_input, grad_output):
            # grad_output[0] contains the gradient of the loss/target score 
            # with respect to the output of the module. This is ∂yc / ∂Ak.
            self.gradients.append(grad_output[0])

        # Find the target layer by its name and register the hooks
        # We assume target_layer is a string representing the layer name 
        # in the model's named_modules dictionary.
        try:
            target_module = dict(self.model.named_modules())[self.target_layer]
        except KeyError:
            raise AttributeError(f"Target layer '{self.target_layer}' not found in model.")

        # Register the hooks
        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)

    def forward(self, input, class_idx=None):
        """
        Computes the Grad-CAM heatmap for the given input and target class.

        Args:
            input (torch.Tensor): The input image tensor (e.g., [B, C, H, W]).
                                  Must have requires_grad_(True) set.
            class_idx (int, optional): The index of the target class c. 
                                       If None, the predicted class (argmax of logits)
                                       for the first image in the batch is used.

        Returns:
            torch.Tensor: The computed Grad-CAM heatmap(s), 
                          typically resized to the original image size. 
                          Shape is usually [B, 1, H_orig, W_orig] or [B, H_orig, W_orig].
        """
        # Ensure input requires gradients, which is necessary for backpropagation
        # through the activation maps.
        if not input.requires_grad:
             input.requires_grad_(True)

        # Clear lists from previous computations
        self.gradients = []
        self.activations = []

        # --- Step 1: Forward pass ---
        # Run the forward pass through the model. This calculates all intermediate
        # activations and the final output scores.
        # The forward hook will capture the activations Ak of the target layer.
        output = self.model(input)

        # Get the predicted class if class_idx is not specified.
        # We assume batch processing, so get the predicted class for each item
        # in the batch if class_idx is None. If a single class_idx is given,
        # we target that class for all items in the batch.
        if class_idx is None:
            # Get the index of the class with the highest score for each item in the batch
            target_classes = output.argmax(dim=1)
        else:
            # Use the specified class index for all items in the batch
            # Ensure target_classes is a tensor of shape [B]
            target_classes = torch.tensor([class_idx] * input.size(0), device=input.device)

        # Get the score for the target class(es).
        # This is yc (before softmax). We need to gather these scores for the backward pass.
        # Using index_select would select a whole row/column, we need specific elements.
        # torch.gather is suitable here: output shape [B, N_classes], target_classes shape [B]
        # We want output[i, target_classes[i]] for each i in the batch.
        target_scores = torch.gather(output, dim=1, index=target_classes.unsqueeze(1)).squeeze() # Shape [B]

        # --- Step 2: Backward pass ---
        # Zero out any previously accumulated gradients in the model's parameters
        # and intermediate tensors.
        self.model.zero_grad()

        # Perform the backward pass starting from the target score.
        # This computes the gradient of yc with respect to all intermediate tensors,
        # including the activations Ak of the target layer.
        # The backward hook will capture these gradients ∂yc / ∂Ak.
        # We need to pass a gradient tensor to backward if the tensor is not a scalar.
        # target_scores is shape [B]. To backpropagate, we need a gradient tensor of the same shape.
        # A tensor of ones means we're calculating ∂(sum of target_scores) / ∂...
        # Since target_scores are independent for each item in the batch, this correctly
        # calculates ∂(target_score_i) / ∂... for each item i.
        target_scores.backward(gradient=torch.ones_like(target_scores), retain_graph=True) # retain_graph=True may be needed if you want to call .backward() multiple times or use parts of the graph later. For standard single-pass Grad-CAM, False is often fine.

        # --- Step 3: Compute importance weights αc_k (Equation 1) ---
        # Retrieve the stored gradients ∂yc / ∂Ak. The last item is the most recent.
        # gradients shape: [B, C, H, W] (where C is number of channels in target layer)
        gradients = self.gradients[-1]
        print(f"Gradients shape: {gradients.shape}")  # Debugging line to check gradients shape
        
        # Global average pooling of the gradients over the width and height dimensions.
        # This averages the gradients for each feature map k across all spatial locations.
        # The result is a tensor of shape [B, C], where each element is αc_k for a specific batch item and feature map k.
        # We use keepdim=True to maintain dimensions for easier multiplication later (will be [B, C, 1, 1]).
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True) # Shape [B, C, 1, 1]
        print(f"Weights shape: {weights.shape}")  # Debugging line to check weights shape

        # Retrieve the stored activations Ak. The last item is the most recent.
        # activations shape: [B, C, H, W]
        activations = self.activations[-1]
        print(f"Activations shape: {activations.shape}")  # Debugging line to check activations shape

        # --- Step 4: Compute the weighted combination and apply ReLU (Equation 2) ---
        # Perform a weighted combination of the activation maps Ak using the weights αc_k.
        # We multiply the weights (shape [B, C, 1, 1]) by the activations (shape [B, C, H, W]).
        # Due to broadcasting, this multiplies each feature map Ak by its corresponding weight αc_k.
        # Then, sum along the channel dimension (dim=1) to get the final linear combination.
        # The result is a tensor of shape [B, 1, H, W].
        cam = torch.sum(weights * activations, dim=1, keepdim=True) # Shape [B, 1, H, W]
        print(f"CAM shape before ReLU: {cam.shape}")  # Debugging line to check CAM shape before ReLU

        # Apply the ReLU function. This ensures we only keep feature regions that
        # positively contribute to the target class score.
        # Negative values are set to zero, as they either suppress the target class
        # or activate for other classes.
        heatmap = F.relu(cam) # Shape [B, 1, H, W]
        print(f"Heatmap shape after ReLU: {heatmap.shape}")

        # # --- Optional: Resize heatmap to original input size ---
        # # The heatmap is typically the size of the target convolutional layer's output.
        # # For visualization, it's usually resized to the original image dimensions.
        # # Use bilinear interpolation for smoothing.
        # original_h, original_w = input.size(2), input.size(3)
        # heatmap_resized = F.interpolate(heatmap, size=(original_h, original_w), 
        #                                 mode='bilinear', align_corners=False) # Shape [B, 1, H_orig, W_orig]

        # # Normalize the heatmap to values between 0 and 1 for visualization purposes.
        # # This makes it easier to overlay on the image.
        # # Avoid division by zero if the heatmap is all zeros.
        # heatmap_max = heatmap_resized.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        # heatmap_normalized = heatmap_resized / (heatmap_max + 1e-8) # Shape [B, 1, H_orig, W_orig]

        # # Squeeze the channel dimension if it's 1, common for heatmaps
        # heatmap_final = heatmap_normalized.squeeze(1) # Shape [B, H_orig, W_orig]

        return heatmap

    def __call__(self, input, class_idx=None):
        return self.forward(input, class_idx)
