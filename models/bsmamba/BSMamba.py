import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from models.bsmamba.BrightnessMamba import BrightnessMamba
from models.bsmamba.SemanticMamba import SemanticMamba
from models.bsmamba.ffc import FFCResnetBlock

class BrightnessStateSpaceBlock(nn.Module):
    def __init__(self, dim, d_state, mlp_ratio=2.0, scale=1.0):
        """
        Single Attentive State Space Block
        Args:
            dim: Input and output feature dimension.
            d_state: State dimension for ASSM.
            mlp_ratio: Expansion ratio for the MLP.
            scale: Scaling factor for residual connections.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.brightmamba = BrightnessMamba(dim, d_state, mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.SiLU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.scale = scale
    def forward(self, x, x_size, brightness_score):
        # First normalization and ASSM
        residual = x
        x = self.norm1(x)
        x = self.brightmamba(x, x_size, brightness_score)
        x = residual + self.scale * x  # Residual connection with scaling
        # Second normalization and MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + self.scale * x  # Residual connection with scaling

        return x

class SemanticStateSpaceBlock(nn.Module):
    def __init__(self, dim, d_state, mlp_ratio=2.0, scale=1.0):
        """
        Single Attentive State Space Block
        Args:
            dim: Input and output feature dimension.
            d_state: State dimension for ASSM.
            mlp_ratio: Expansion ratio for the MLP.
            scale: Scaling factor for residual connections.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.semanticmamba = SemanticMamba(dim, d_state, mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.SiLU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.scale = scale
    def forward(self, x, x_size, semantic_score):
        # First normalization and ASSM
        residual = x
        x = self.norm1(x)
        x = self.semanticmamba(x, x_size, semantic_score)
        x = residual + self.scale * x  # Residual connection with scaling
        # Second normalization and MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + self.scale * x  # Residual connection with scaling
        return x

class SecondProcessModel(nn.Module):
    def __init__(self, dim=16, num_blocks=3, in_channels=3):
        """
        Second Stage --- Spatial and Texture Reconstruction Stage.
        """
        super(SecondProcessModel, self).__init__()

        # Initial convolution layers (downsampling)
        self.conv1 = nn.Conv2d(in_channels, dim, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(dim, dim, 3, 2, 1, bias=True)
        self.conv3 = nn.Conv2d(dim, dim, 3, 2, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # Dual-branch blocks
        self.fft_blocks = nn.ModuleList([FFCResnetBlock(dim) for _ in range(num_blocks)])
        # Upsample convolution layers
        self.upconv1 = nn.Conv2d(dim * 2, dim * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(dim * 2, dim * 4, 3, 1, 1, bias=True)
        self.upconv3 = nn.Conv2d(dim * 2, dim, 3, 1, 1, bias=True)
        self.upconv_last = nn.Conv2d(dim, 3, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def _upsample(self, x, skip, upconv):
        x = self.lrelu(self.pixel_shuffle(upconv(torch.cat((x, skip), dim=1))))
        return x

    def forward(self, x):
        # Downsampling with intermediate feature extraction
        x1 = self.lrelu(self.conv1(x))  # First downsampling
        x2 = self.lrelu(self.conv2(x1))  # Second downsampling
        x3 = self.lrelu(self.conv3(x2))  # Third downsampling
        # Dual-branch processing
        fft_features = x3
        for fft_block in self.fft_blocks:
            fft_features = fft_block(fft_features)
        # Reconstruction and upsampling
        out_noise = self._upsample(fft_features, x3, self.upconv1)
        out_noise = self._upsample(out_noise, x2, self.upconv2)
        out_noise = self.upconv3(torch.cat((out_noise, x1), dim=1))
        out_noise = self.upconv_last(out_noise)
        # Residual connection
        out_noise = out_noise + x
        # Ensure output size matches input size
        B, C, H, W = x.size()
        out_noise = out_noise[:, :, :H, :W]
        return out_noise

class BSMamba(nn.Module):
    def __init__(self, in_channels, dim, d_state, num_blocks, mlp_ratio=2.0, scale=1.0):
        """
        Main Model with initial convolution and multiple ASSM blocks.
        Args:
            in_channels: Number of input channels (e.g., 3 for RGB images).
            dim: Feature dimension for ASSM and intermediate layers.
            d_state: State dimension for ASSM.
            num_blocks: Number of Attentive State Space Blocks.
            mlp_ratio: Expansion ratio for the MLP in each block.
            scale: Scaling factor for residual connections.
        """
        super().__init__()
        self.initial_conv = nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1)
        self.lsblocks = nn.ModuleList([
            nn.Sequential(
                BrightnessStateSpaceBlock(dim, d_state, mlp_ratio, scale),
                SemanticStateSpaceBlock(dim, d_state, mlp_ratio, scale)
            )
            for _ in range(num_blocks)
        ])
        # 加载预训练的 Faster R-CNN 模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_list = ['low-light']
        self.spatial_projection = nn.Linear(768, 512)
        self.conv_last = nn.Conv2d(dim, 3, 3, 1, 1)
        self.refinenet = SecondProcessModel(dim=dim, num_blocks=3, in_channels=in_channels)

    def compute_similarity_map(self, img, text_list, daclip):
        spatial_features = self.get_spatial_features(img, daclip)  # Shape: (B, C, H', W')
        # Step 2: Generate text features
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        text_tokens = tokenizer(text_list).to(self.device)
        with torch.no_grad():
            text_features = daclip.encode_text(text_tokens)  # Shape: (B, 512)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Normalize
        # Step 3: Normalize spatial features
        spatial_features = spatial_features / spatial_features.norm(dim=1, keepdim=True)  # Shape: (B, C, H', W')
        # Step 4: Adjust spatial features for similarity computation
        B, C, H_prime, W_prime = spatial_features.shape  # H' and W' are patch-level spatial dimensions
        spatial_features = spatial_features.view(B, C, -1).permute(0, 2, 1)  # Shape: (B, H'*W', C)
        spatial_features = self.spatial_projection(spatial_features)  # Shape: (B, H'*W', 512)
        spatial_features = spatial_features / spatial_features.norm(dim=-1, keepdim=True)  # Normalize
        # Step 5: Compute similarity between spatial features and text features
        text_features = text_features.unsqueeze(-1)  # Shape: (B, 512, 1)
        similarity_scores = torch.bmm(spatial_features, text_features).squeeze(-1)  # Shape: (B, H'*W')
        # Step 6: Reshape similarity scores to spatial dimensions
        similarity_map = similarity_scores.view(B, H_prime, W_prime)  # Shape: (B, H', W')
        # Step 7: Interpolate to match input image resolution
        H, W = img.shape[2], img.shape[3]  # Original input image resolution
        similarity_map = F.interpolate(similarity_map.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False)
        similarity_map = similarity_map.squeeze(1)  # Shape: (B, H, W)
        return similarity_map

    def get_spatial_features(self, img, daclip):
        with torch.no_grad():
            # Step 1: Pass through the initial layers (conv1 and ln_pre)
            x = daclip.visual.conv1(img)  # Shape: (B, 768, H/32, W/32)
            x = x.flatten(2).permute(0, 2, 1)  # Shape: (B, N, 768), where N = (H/32) * (W/32)
            x = daclip.visual.ln_pre(x)  # Shape: (B, N, 768)
            # Step 2: Pass through the transformer and extract patch features
            for i, block in enumerate(daclip.visual.transformer.resblocks):
                x = block(x)  # Shape: (B, N, 768)
            # Step 3: Reshape patch features into a spatial feature map
            B, N, C = x.shape
            patch_size_h = daclip.visual.patch_size[0]  # Patch height
            patch_size_w = daclip.visual.patch_size[1]  # Patch width
            H = img.shape[2] // patch_size_h  # Compute H dynamically
            W = img.shape[3] // patch_size_w  # Compute W dynamically
            assert H * W == N, f"Mismatch in patch dimensions: H*W={H * W}, N={N}"
            spatial_features = x.view(B, H, W, C).permute(0, 3, 1, 2)  # Shape: (B, C, H, W)
        return spatial_features

    def semantic_score(self, images,segmentation_model, image_size):
        """
        Generate semantic scores using a segmentation model with dynamic grading.

        Args:
            images (torch.Tensor): Input images, shape (B, C, H, W).
            image_size (tuple): Target image size (H, W).

        Returns:
            torch.Tensor: Attention map of shape (B, H, W).
        """
        B, H, W = image_size
        attention_map = torch.zeros((B, H, W), device=images.device)  # Initialize attention map
        # Pass images through the segmentation model
        with torch.no_grad():
            outputs = segmentation_model(images)  # List of dicts, one per image
        for b in range(B):
            masks = outputs[b]['masks']  # Shape: (num_instances, 1, H, W)
            scores = outputs[b]['scores']  # Shape: (num_instances,)
            num_instances = len(scores)
            if num_instances == 0:
                background_mask = torch.ones((H, W), device=images.device)
                attention_map[b] = background_mask  # 背景直接占据整个图像
                continue
                # Sort instances by confidence scores in descending order
            sorted_indices = torch.argsort(scores, descending=True)
            masks = masks[sorted_indices]  # Reorder masks
            scores = scores[sorted_indices]  # Reorder scores
            # Define dynamic grading ranges
            levels = num_instances + 1  # Include background
            ranges = [(i / levels, (i + 1) / levels) for i in range(levels)]
            # Create a mask to track background pixels
            background_mask = torch.ones((H, W), device=images.device)
            # Process each instance
            for i, (mask, score) in enumerate(zip(masks, scores)):
                min_val, max_val = ranges[i + 1]  # Get the range for this instance (skip background)
                mask = mask.squeeze(0)  # Shape: (H, W)
                # Normalize mask values to [0, 1]
                normalized_mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
                # Map normalized mask values to [min_val, max_val]
                mapped_mask = normalized_mask * (max_val - min_val) + min_val
                # Add the mapped mask to the attention map
                attention_map[b] += mapped_mask * score  # Weighted by instance confidence score
                # Update the background mask (exclude current instance)
                background_mask = background_mask * (1 - mask)
                # Map background to the lowest range
            background_min, background_max = ranges[0]  # Background is mapped to the lowest range
            normalized_background = (background_mask - background_mask.min()) / (
                        background_mask.max() - background_mask.min() + 1e-8)
            background_attention = normalized_background * (background_max - background_min) + background_min
            attention_map[b] += background_attention
            # Normalize the attention map to [0, 1]
            attention_map[b] = (attention_map[b] - attention_map[b].min()) / (
                        attention_map[b].max() - attention_map[b].min() + 1e-8)

        return attention_map

    def forward(self, x, daclip, segmentation_model):
        x_short = x
        B, C, H, W = x.shape
        #####################   light score  ############################
        light_score = self.compute_similarity_map(x, self.text_list * x.size(0), daclip)
        ##################### semantic score ############################
        semantic_score = self.semantic_score(x, segmentation_model, image_size=(B, H, W))
        # Initial convolution
        x = self.initial_conv(x)  # (B, dim, H, W)
        # Flatten spatial dimensions for ASSM
        x = x.view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, dim)
        # Pass through each ASSM block
        for light_block, semantic_block in self.lsblocks:
            x = light_block(x, (H, W), light_score)
            x = semantic_block(x, (H, W), semantic_score)
        # Reshape back to image format
        x = x.permute(0, 2, 1).view(B, -1, H, W)  # (B, dim, H, W)
        x_out = self.conv_last(x) + x_short
        x_out_refine =self.refinenet(x_out)
        return x_out, x_out_refine

