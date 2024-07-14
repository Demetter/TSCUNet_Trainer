import argparse
import cv2
import os
from config import Config
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from config import Config
from models.network_tscunet import TSCUNet as net
import signal
import sys
import traceback
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

if not torch.cuda.is_available():
    print('CUDA is not available. Using CPU...')
    device = torch.device('cpu')
else:
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True


current_epoch = 0
current_iteration = 0
best_val_loss = float('inf')
model = None
optimizer = None
config = None

def save_checkpoint(epoch, iteration, model, optimizer, train_loss, val_loss):
    checkpoint_path = os.path.join(config.output, f'checkpoint_epoch_{epoch + 1}.pth')
    torch.save({
        'epoch': epoch,
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def signal_handler(sig, frame):
    global current_epoch, current_iteration, model, optimizer
    save_checkpoint(current_epoch, current_iteration, model, optimizer, None, None)
    print("Training interrupted. Checkpoint saved. Exiting...")
    sys.exit(0)

class VideoSuperResolutionDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, clip_size, scale, crop_size):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.clip_size = clip_size
        self.scale = scale
        self.crop_size = crop_size
        self.frames = {}

        for f in sorted(os.listdir(lr_dir)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                show_prefix = f.split('_')[0]
                lr_path = os.path.join(lr_dir, f)
                hr_path = os.path.join(hr_dir, f)

                if os.path.exists(hr_path):
                    if show_prefix not in self.frames:
                        self.frames[show_prefix] = []
                    self.frames[show_prefix].append((lr_path, hr_path))
                else:
                    print(f"Warning: No matching HR file for {f}")

        print(f"Found {sum(len(v) for v in self.frames.values())} valid file pairs across {len(self.frames)} shows")

    def __len__(self):
        return sum(max(0, len(v) - self.clip_size + 1) for v in self.frames.values())

    def __getitem__(self, idx):
        for show_prefix, clips in self.frames.items():
            if idx < len(clips) - self.clip_size + 1:
                lr_clip = []
                hr_clip = []
                
                for i in range(self.clip_size):
                    lr_path, hr_path = clips[idx + i]
                    hr_img, lr_img = self._load_and_crop_imgs(hr_path, lr_path, self.crop_size, self.scale)
                    
                    lr_clip.append(lr_img)
                    hr_clip.append(hr_img)

                return torch.stack(lr_clip), torch.stack(hr_clip), hr_clip[self.clip_size // 2]
            idx -= len(clips) - self.clip_size + 1

        raise IndexError("Index out of range.")

    def _load_and_crop_imgs(self, hr_path, lr_path, crop_size, scale):
        hr_img = cv2.imread(hr_path, cv2.IMREAD_UNCHANGED)
        lr_img = cv2.imread(lr_path, cv2.IMREAD_UNCHANGED)

        if hr_img is None:
            raise ValueError(f"Failed to read image: {hr_path}")
        if lr_img is None:
            raise ValueError(f"Failed to read image: {lr_path}")

        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)

        h, w = hr_img.shape[:2]
        top = np.random.randint(0, h - crop_size + 1)
        left = np.random.randint(0, w - crop_size + 1)
        hr_img_cropped = hr_img[top:top+crop_size, left:left+crop_size]

        lr_crop_size = crop_size // scale
        lr_top = top // scale
        lr_left = left // scale
        lr_img_cropped = lr_img[lr_top:lr_top+lr_crop_size, lr_left:lr_left+lr_crop_size]

        hr_img_cropped = torch.from_numpy(hr_img_cropped.transpose((2, 0, 1))).float() / 255.0
        lr_img_cropped = torch.from_numpy(lr_img_cropped.transpose((2, 0, 1))).float() / 255.0

        return hr_img_cropped, lr_img_cropped

def custom_loss(output, target):
    return nn.L1Loss()(output, target)

config = Config()

def color_balance_loss(output, target):
    if output.dim() == 3:  # [channels, height, width]
        output_mean = output.mean(dim=[1, 2])
        target_mean = target.mean(dim=[1, 2])
    elif output.dim() == 4:  # [batch, channels, height, width]
        output_mean = output.mean(dim=[2, 3])
        target_mean = target.mean(dim=[2, 3])
    else:
        raise ValueError(f"Unexpected tensor dimension: {output.dim()}")
    return nn.MSELoss()(output_mean, target_mean)

def combined_loss(output_clip, hr_center):
    reconstruction_loss = custom_loss(output_clip, hr_center)
    color_loss = color_balance_loss(output_clip, hr_center)
    
    total_loss = reconstruction_loss + 0.1 * color_loss  # Adjust the weight as needed
    return total_loss, (reconstruction_loss, color_loss)

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, gradient_accumulation_steps, epoch_iteration=0):
    global current_iteration
    model.train()
    total_loss = 0
    num_batches = 0
    optimizer.zero_grad()
    total_iterations = len(dataloader)

    dataloader_iterator = iter(dataloader)
    for _ in range(epoch_iteration):
        next(dataloader_iterator, None)

    for i, (lr_batch, hr_batch, hr_center) in enumerate(dataloader_iterator, start=epoch_iteration):
        with autocast():
            output_clip = model(lr_batch.to(device))
            output_clip = torch.clamp(output_clip, 0, 1)
            
            loss, (recon_loss, color_loss) = combined_loss(output_clip, hr_center.to(device))

            if torch.isnan(loss):
                print(f"NaN loss detected at iteration {current_iteration}")
                continue

            scaler.scale(loss).backward()

            total_loss += loss.item()

            if (i + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            num_batches += 1
            current_iteration += 1

        if (i + 1) % 20 == 0:
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Iteration [{i + 1}/{total_iterations}], Running Avg Loss: {avg_loss:.4f}, Recon Loss: {recon_loss.item():.4f}, Color Loss: {color_loss.item():.4f}")

    if num_batches == 0:
        print("Warning: No batches were processed in this epoch.")
        return 0, current_iteration
    
    return total_loss / num_batches, current_iteration

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for lr_batch, hr_batch, hr_center in val_loader:
            lr_batch, hr_center = lr_batch.to(device), hr_center.to(device)

            output_clip = model(lr_batch)
            output_clip = torch.clamp(output_clip, 0, 1)
            loss, _ = combined_loss(output_clip, hr_center)

            total_loss += loss.item()
            num_batches += 1

    if num_batches == 0:
        print("Warning: No batches were processed in validation.")
        return 0

    return total_loss / num_batches

def freeze_layers(model, num_layers):
    if num_layers <= 0:
        return
    
    for name, param in list(model.named_parameters())[:num_layers]:
        param.requires_grad = False
        print(f"Frozen layer: {name}")

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def save_validation_image(model, val_dataset, device, epoch, save_path):
    model.eval()
    with torch.no_grad():
        lr_clip, hr_clip, hr_center = val_dataset[0]
        
        lr_clip = lr_clip.unsqueeze(0).to(device)
        hr_center = hr_center.unsqueeze(0).to(device)
        
        sr_image = model(lr_clip)
        
        lr_image = lr_clip[:, lr_clip.size(1)//2].cpu().squeeze()
        hr_image = hr_center.cpu().squeeze()
        sr_image = sr_image.cpu().squeeze()

        lr_image_resized = torch.nn.functional.interpolate(
            lr_image.unsqueeze(0),
            size=(hr_image.shape[1], hr_image.shape[2]), 
            mode='bicubic', 
            align_corners=False
        ).squeeze(0)

        sr_image_resized = torch.nn.functional.interpolate(
            sr_image.unsqueeze(0),
            size=(hr_image.shape[1], hr_image.shape[2]), 
            mode='bicubic', 
            align_corners=False
        ).squeeze(0)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(lr_image_resized.permute(1, 2, 0))
        ax[0].set_title('LR (upscaled)')
        ax[1].imshow(sr_image_resized.permute(1, 2, 0))
        ax[1].set_title('SR')
        ax[2].imshow(hr_image.permute(1, 2, 0))
        ax[2].set_title('HR')
        
        plt.savefig(os.path.join(save_path, f'val_image_epoch_{epoch}.png'), bbox_inches='tight', pad_inches=0.1)
        plt.close()

    model.train()

def main():
    global current_epoch, current_iteration, best_val_loss, model, optimizer, config
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_size', type=int, choices=[128, 256], default=Config.crop_size, help='Crop size for training')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file to resume training')
    args = parser.parse_args()

    config = Config()
    config.crop_size = args.crop_size

    print("Configuration:")
    for attr, value in vars(config).items():
        if not isinstance(value, str) or not value.startswith('$'):
            print(f"{attr}: {value}")

    model = net(scale=config.scale)
    print(f"Model clip size: {model.clip_size}")
    
   # if config.pretrained:
     #   print(f"Loading pre-trained model from {config.pretrained}")
    #    checkpoint = torch.load(config.pretrained, map_location=device)
    #    model_dict = model.state_dict()
    #    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and model_dict[k].shape == v.shape}
    #    model_dict.update(pretrained_dict)
    #    model.load_state_dict(model_dict, strict=False)
    #    print(f"Pre-trained model partially loaded. {len(pretrained_dict)} layers were updated.")

    model = model.to(device)

   # if config.freeze_layers > 0:
    #    freeze_layers(model, config.freeze_layers)

    criterion = combined_loss
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    scaler = GradScaler()

    start_epoch = 0
    current_iteration = 0
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        current_iteration = checkpoint.get('iteration', 0)
        print(f"Resuming training from epoch {start_epoch}, iteration {current_iteration}")

    train_dataset = VideoSuperResolutionDataset(config.train_lr_dir, config.train_hr_dir, model.clip_size, config.scale, config.crop_size)
    val_dataset = VideoSuperResolutionDataset(config.val_lr_dir, config.val_hr_dir, model.clip_size, config.scale, config.crop_size)
    
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Val dataset length: {len(val_dataset)}")

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: Dataset is empty. Please check the data directories.")
        return

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if not os.path.exists(config.val_image_save_path):
        os.makedirs(config.val_image_save_path)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        for current_epoch in range(start_epoch, config.epochs):
            print(f"Starting epoch {current_epoch + 1}")
            print(f"Number of batches in train loader: {len(train_loader)}")
            print(f"Number of batches in val loader: {len(val_loader)}")
            
            epoch_iteration = current_iteration % len(train_loader)
            print(f"Starting from iteration {epoch_iteration} in this epoch")
            
            print("Starting training for this epoch")
            train_loss, current_iteration = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, config.gradient_accumulation_steps, epoch_iteration=epoch_iteration)
            print(f"Finished training for epoch {current_epoch + 1}")
            
            print("Starting validation")
            val_loss = validate(model, val_loader, criterion, device)
            print(f"Finished validation for epoch {current_epoch + 1}")
            
            scheduler.step()

            print(f"Epoch [{current_epoch + 1}/{config.epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if (current_epoch + 1) % config.val_image_save_epoch == 0:
                save_validation_image(model, val_dataset, device, current_epoch + 1, config.val_image_save_path)
                print(f"Validation image saved for epoch {current_epoch + 1}")
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(config.output, 'best_model.pth'))
                print("Best model saved.")

            if (current_epoch + 1) % config.model_save_interval == 0:
                torch.save(model.state_dict(), os.path.join(config.output, f'model_epoch_{current_epoch + 1}.pth'))
                print(f"Model saved at epoch {current_epoch + 1}.")

            if (current_epoch + 1) % 10 == 0:
                save_checkpoint(current_epoch, current_iteration, model, optimizer, train_loss, val_loss)

            print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        print(f"Current epoch: {current_epoch}, Current iteration: {current_iteration}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
