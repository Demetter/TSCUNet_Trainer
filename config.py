# config.py

class Config:
    train_lr_dir = "Dataset/LR"
    train_hr_dir = "Dataset/HR"
    val_lr_dir = "Dataset/VAL_LR" # Must have consecutive frames in multiples of 5
    val_hr_dir = "Dataset/VAL_HR" # Same as above. Make sure the LR is 50% smaller than the HR
    val_image_save_path = "Dataset/VAL_IMAGES"
    output = "Dataset/Models"
    epochs = 200
    batch_size = 1
    lr = 0.0001
    scale = 2 # Don't change, its bugged
#pretrained = "pretrained_models/2x_eula_anifilm_vsr.pth"
#Uncomment pre-train in here and in Train_TSCUNet.py to use one
    freeze_layers = 10
    gradient_accumulation_steps = 1
    crop_size = 128
    model_save_interval = 5
    val_image_save_epoch = 1
