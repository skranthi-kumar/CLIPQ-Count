
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from collections import OrderedDict
import math
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import numpy as np
import random
from torch.optim import AdamW
import argparse


# ===========================
# BACKBONE MODULE
# ===========================
class HCTBackbone(nn.Module):
    """
    A wrapper for the Vision Transformer (ViT) backbone.
    This module loads a pre-trained ViT model and prepares it for feature extraction.
    """
    def __init__(self, model_name='vit_base_patch16_384', checkpoint_path=None):
        """
        Initializes the Vision Transformer backbone.
        
        Args:
            model_name (str): The name of the ViT model that matches the architecture.
            checkpoint_path (str, optional): Path to the .pth checkpoint file.
        """
        super().__init__()
        
        # Create the ViT architecture from the timm library
        self.backbone = timm.create_model(model_name, pretrained=True if checkpoint_path is None else False)
        
        # Store the feature dimension (e.g., 768 for ViT-Base)
        self.feature_dim = self.backbone.embed_dim
        
        # Load custom weights if provided
        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)
            
    def load_from_checkpoint(self, path):
        """
        Loads encoder-specific weights from a CounTR checkpoint file.
        """
        try:
            checkpoint = torch.load(path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                full_state_dict = checkpoint['model']
            else:
                full_state_dict = checkpoint
            
            backbone_state_dict = OrderedDict()
            
            # Filter relevant keys for the backbone
            for key, value in full_state_dict.items():
                if key.startswith('patch_embed') or key.startswith('pos_embed') or \
                   key.startswith('blocks') or key.startswith('norm'):
                    backbone_state_dict[key] = value
            
            msg = self.backbone.load_state_dict(backbone_state_dict, strict=False)
            print(f"Backbone weights loaded from {path}.")
            print(f"Missing keys: {msg.missing_keys}")
            
        except FileNotFoundError:
            print(f"Warning: Checkpoint file not found at {path}. Using pretrained ImageNet weights.")
        except Exception as e:
            print(f"Error loading weights: {e}. Using pretrained ImageNet weights.")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patch features from input image.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W) or (B*N, C, H, W) for exemplars.
            
        Returns:
            torch.Tensor: Patch features of shape (B, num_patches, embed_dim) or (B*N, num_patches, embed_dim).
        """
        features = self.backbone.forward_features(x)
        return features


# ===========================
# LOCA FUSION MODULE
# ===========================
class LOCAFusionModule(nn.Module):
    """
    Generates a high-quality object prototype from exemplar features and masks.
    """
    def __init__(self, in_dim: int = 768, out_dim: int = 256, iterative_steps: int = 3):
        """
        Args:
            in_dim (int): Input feature dimension from ViT backbone.
            out_dim (int): Output prototype dimension.
            iterative_steps (int): Number of refinement iterations.
        """
        super().__init__()
        self.iterative_steps = iterative_steps
        
        self.projection = nn.Linear(in_dim, out_dim)
        self.refinement_layer = nn.Linear(out_dim, out_dim)
    
    def forward(self, exemplar_feats: torch.Tensor, exemplar_masks: torch.Tensor) -> torch.Tensor:
        """
        Generate prototype from exemplars.
        
        Args:
            exemplar_feats (torch.Tensor): Shape (B, num_exemplars, num_patches, in_dim).
            exemplar_masks (torch.Tensor): Shape (B, num_exemplars, num_patches, 1).
        
        Returns:
            torch.Tensor: Prototype of shape (B, out_dim).
        """
        B, num_exemplars, num_patches, in_dim = exemplar_feats.shape
        
        # Reshape to process all exemplars together
        exemplar_feats = exemplar_feats.view(B * num_exemplars, num_patches, in_dim)
        exemplar_masks = exemplar_masks.view(B * num_exemplars, num_patches, 1)
        
        # Project features
        projected_feats = self.projection(exemplar_feats)
        
        # Initial prototype via masked averaging
        masked_feats = projected_feats * exemplar_masks
        prototype = masked_feats.sum(dim=1) / exemplar_masks.sum(dim=1).clamp(min=1)
        
        # Iterative refinement
        for _ in range(self.iterative_steps):
            # Calculate similarity scores
            similarity_scores = torch.bmm(projected_feats, prototype.unsqueeze(-1))
            
            # Get attention weights
            attention_weights = F.softmax(similarity_scores, dim=1)
            
            # Update prototype
            refined_prototype = torch.bmm(projected_feats.transpose(1, 2), attention_weights).squeeze(-1)
            prototype = self.refinement_layer(refined_prototype)
        
        # Average across exemplars
        prototype = prototype.view(B, num_exemplars, -1).mean(dim=1)
        
        return prototype


# ===========================
# DECODER MODULE
# ===========================
class HCTDecoder(nn.Module):
    """
    Transformer decoder that uses cross-attention to find object instances.
    """
    def __init__(self, encoder_dim: int = 768, decoder_dim: int = 512, nhead: int = 8, num_layers: int = 2):
        super().__init__()
        self.decoder_dim = decoder_dim

        self.input_proj = nn.Linear(encoder_dim, decoder_dim)
        self.prototype_proj = nn.Linear(decoder_dim, decoder_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=nhead,
            dim_feedforward=decoder_dim * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
    def forward(self, image_feats: torch.Tensor, prototype: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_feats (torch.Tensor): Shape (B, num_patches, encoder_dim).
            prototype (torch.Tensor): Shape (B, decoder_dim).
        
        Returns:
            torch.Tensor: Shape (B, num_patches, decoder_dim).
        """
        # Project features
        image_feats_proj = self.input_proj(image_feats)
        prototype_proj = self.prototype_proj(prototype)

        # Use prototype as query and image features as both target and memory
        tgt = prototype_proj.unsqueeze(1).expand(-1, image_feats_proj.size(1), -1)
        memory = image_feats_proj
        
        output = self.decoder(tgt=tgt, memory=memory)
        
        return output


# ===========================
# DENSITY HEAD
# ===========================
class DensityHead(nn.Module):
    """
    Converts decoder output to density map.
    """
    def __init__(self, in_dim: int = 512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, decoder_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            decoder_output (torch.Tensor): Shape (B, num_patches, in_dim).
        
        Returns:
            torch.Tensor: Density map of shape (B, 1, H, W).
        """
        num_patches = decoder_output.shape[1]
        height = width = int(math.sqrt(num_patches))
        
        # Reshape to 2D feature map
        feature_map = decoder_output.permute(0, 2, 1).contiguous().view(
            -1, decoder_output.size(-1), height, width)
        
        # Apply convolutions
        density_map = self.head(feature_map)
        
        # Upsample to larger resolution
        density_map = F.interpolate(density_map, scale_factor=4, mode='bilinear', align_corners=False)
        
        return density_map


# ===========================
# MAIN HCT MODEL
# ===========================
class HCT(nn.Module):
    """
    The complete Hybrid Counting Transformer model.
    """
    def __init__(self, config: dict):
        super().__init__()
        
        # Initialize backbone
        self.backbone = HCTBackbone(
            model_name=config.get('backbone_model_name', 'vit_base_patch16_384'),
            checkpoint_path=config.get('checkpoint_path', None)
        )
        
        backbone_dim = self.backbone.feature_dim
        
        # Initialize LOCA fusion
        self.loca_fusion_module = LOCAFusionModule(
            in_dim=backbone_dim,
            out_dim=config.get('loca_fusion_dim', 256),
            iterative_steps=config.get('loca_iterative_steps', 3)
        )
        
        fusion_dim = config.get('loca_fusion_dim', 256)
        
        # Initialize decoder
        self.decoder = HCTDecoder(
            encoder_dim=backbone_dim,
            decoder_dim=fusion_dim,
            nhead=config.get('decoder_nhead', 8),
            num_layers=config.get('decoder_num_layers', 2)
        )
        
        # Initialize density head
        self.density_head = DensityHead(in_dim=fusion_dim)
    
    def forward(self, image: torch.Tensor, exemplar_patches: torch.Tensor, exemplar_masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image (torch.Tensor): Shape (B, C, H, W).
            exemplar_patches (torch.Tensor): Shape (B, num_exemplars, C, H, W).
            exemplar_masks (torch.Tensor): Shape (B, num_exemplars, num_patches, 1).
        
        Returns:
            torch.Tensor: Density map of shape (B, 1, H, W).
        """
        # Extract image features
        image_feats = self.backbone(image)
        
        # Extract exemplar features
        B, num_exemplars, C, H, W = exemplar_patches.shape
        exemplar_patches_flat = exemplar_patches.view(B * num_exemplars, C, H, W)
        exemplar_feats_flat = self.backbone(exemplar_patches_flat)
        exemplar_feats = exemplar_feats_flat.view(B, num_exemplars, exemplar_feats_flat.size(1), exemplar_feats_flat.size(2))
        
        # Generate prototype
        loca_prototype = self.loca_fusion_module(exemplar_feats, exemplar_masks)
        
        # Process through decoder
        processed_feats = self.decoder(image_feats, loca_prototype)
        
        # Generate density map
        density_map = self.density_head(processed_feats)
        
        return density_map


# ===========================
# DATASET
# ===========================
def get_transforms(image_size):
    """Standard image transformations."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class FSC147Dataset(Dataset):
    """
    FSC147 dataset for few-shot counting.
    """
    def __init__(self, data_dir, annotation_file, split_file, split='train', image_size=384, num_exemplars=3):
        """
        Args:
            data_dir (str): Path to images directory.
            annotation_file (str): Path to annotation JSON file.
            split_file (str): Path to train/val/test split JSON file.
            split (str): Dataset split ('train', 'val', or 'test').
            image_size (int): Image resize dimension.
            num_exemplars (int): Number of exemplar patches per image.
        """
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.num_exemplars = num_exemplars
        self.transforms = get_transforms(image_size)
        
        # Load annotations
        with open(annotation_file) as f:
            self.annotations = json.load(f)
        with open(split_file) as f:
            split_data = json.load(f)
            
        self.image_ids = split_data[split]
        
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.annotations[image_id]
        
        # Load main image
        # The correct line based on your annotation file
        image_path = os.path.join(self.data_dir, image_info['img_path'])
        if not os.path.exists(image_path):
            # Try alternative filename format
            image_path = os.path.join(self.data_dir, image_id)
        
        image = Image.open(image_path).convert('RGB')
        
        # Get exemplar bounding boxes
        all_bboxes = image_info['box_examples_coordinates']
        if len(all_bboxes) < self.num_exemplars:
            selected_bboxes = random.choices(all_bboxes, k=self.num_exemplars)
        else:
            selected_bboxes = random.sample(all_bboxes, self.num_exemplars)

        # Extract exemplar patches
        exemplar_patches = []
        # Corrected lines
        for bbox in selected_bboxes:
            coords = bbox[0]
            x1, y1, w, h = coords
            patch = image.crop((x1, y1, x1 + w, y1 + h))

        # Apply transformations
        image_tensor = self.transforms(image)
        exemplar_tensors = torch.stack([self.transforms(p) for p in exemplar_patches])
        
        # Create simple exemplar masks (assume entire patch is object)
        num_patches = (self.image_size // 16) ** 2  # For ViT patch size 16
        exemplar_masks = torch.ones(self.num_exemplars, num_patches, 1)

        # Ground truth count
        gt_count = len(image_info['points'])
        
        return {
            'image': image_tensor,
            'exemplar_patches': exemplar_tensors,
            'exemplar_masks': exemplar_masks,
            'gt_count': torch.tensor(gt_count, dtype=torch.float32)
        }


def create_dataloader(data_dir, annotation_file, split_file, split, image_size, batch_size, num_workers=4):
    """Create DataLoader for FSC147 dataset."""
    dataset = FSC147Dataset(data_dir, annotation_file, split_file, split=split, image_size=image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


# ===========================
# TRAINING FUNCTIONS
# ===========================
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for i, batch in enumerate(dataloader):
        # Move to device
        image = batch['image'].to(device)
        exemplar_patches = batch['exemplar_patches'].to(device)
        exemplar_masks = batch['exemplar_masks'].to(device)
        gt_count = batch['gt_count'].to(device)

        optimizer.zero_grad()

        # Forward pass
        pred_density_map = model(image, exemplar_patches, exemplar_masks)
        pred_count = pred_density_map.sum(dim=(-1, -2, -3))

        # Calculate loss
        loss = criterion(pred_count, gt_count)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        if i % 50 == 0:
            print(f"  Batch {i}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0

    for batch in dataloader:
        image = batch['image'].to(device)
        exemplar_patches = batch['exemplar_patches'].to(device)
        exemplar_masks = batch['exemplar_masks'].to(device)
        gt_count = batch['gt_count'].to(device)

        pred_density_map = model(image, exemplar_patches, exemplar_masks)
        pred_count = pred_density_map.sum(dim=(-1, -2, -3))

        loss = criterion(pred_count, gt_count)
        mae = torch.abs(pred_count - gt_count).sum()

        total_loss += loss.item()
        total_mae += mae.item()

    avg_loss = total_loss / len(dataloader.dataset)
    avg_mae = total_mae / len(dataloader.dataset)
    return avg_loss, avg_mae


def main(args):
    """Main training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataloaders
    train_loader = create_dataloader(
        args.data_dir, args.annotation_file, args.split_file, 
        'train', args.image_size, args.batch_size
    )
    val_loader = create_dataloader(
        args.data_dir, args.annotation_file, args.split_file,
        'val', args.image_size, args.batch_size
    )

    # Model configuration
    config = {
        'backbone_model_name': 'vit_base_patch16_384',
        'checkpoint_path': args.checkpoint_path,
        'loca_fusion_dim': 256,
        'loca_iterative_steps': 3,
        'decoder_nhead': 8,
        'decoder_num_layers': 2,
    }

    # Initialize model
    model = HCT(config).to(device)

    # Setup optimizer and loss
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    # Training loop
    print("Starting training...")
    best_mae = float('inf')

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Training Loss: {train_loss:.4f}")

        val_loss, val_mae = evaluate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, MAE: {val_mae:.2f}")

        # Save best model
        if val_mae < best_mae:
            best_mae = val_mae
            save_path = os.path.join(args.output_path, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with MAE: {best_mae:.2f}")

    print("Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train HCT model')
    
    # Dataset paths (updated for your directory structure)
    parser.add_argument('--data_dir', type=str, default='FSC147/images_384_VarV2',
                        help='Path to images directory')
    parser.add_argument('--annotation_file', type=str, default='FSC147/annotation_FSC147_384.json',
                        help='Path to annotation file')
    parser.add_argument('--split_file', type=str, default='FSC147/Train_Test_Val_FSC_147.json',
                        help='Path to split file')
    
    # Model and training parameters
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to pre-trained CounTR weights')
    parser.add_argument('--output_path', type=str, default='outputs/checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--image_size', type=int, default=384, help='Image size')

    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    main(args)
