import torch
import os
from pathlib import Path
import numpy as np
import torchvision
from torch.utils.tensorboard import SummaryWriter
from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    RandSpatialCrop,
    SpatialPadd,
    Spacingd,
    ToTensord,
    RandFlipd,
    RandAffined,
    ResizeWithPadOrCropd,
    RandSpatialCropd,
    NormalizeIntensityd
)
from monai.networks.nets import UNet, BasicUNet
from monai.networks.layers import Norm
from monai.metrics import compute_meandice
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset
from monai.data.utils import pad_list_data_collate
from monai.visualize.img2tensorboard import add_animated_gif_no_channels, add_animated_gif

from networks.nets.unet2d5_spvPA import UNet2d5_spvPA
from losses.dice_spvPA import Dice_spvPA, compute_dice_score
from utils import get_center_of_mass_slice

pad_crop_shape = [128, 128, 32]
batch_size = 24
max_epochs = 600
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epochs_with_const_lr = 100
lr_divisor = 2.0
weight_decay = 1e-7
learning_rate = 1e-3
debug = True
val_size = 24
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys=["image"]),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=pad_crop_shape),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandSpatialCropd(
            keys=["image", "label"], roi_size=pad_crop_shape, random_center=True, random_size=False
        ),
        ToTensord(keys=["image", "label"]),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys=["image"]),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=pad_crop_shape),
        RandSpatialCropd(
            keys=["image", "label"], roi_size=pad_crop_shape, random_center=True, random_size=False,
        ),
        ToTensord(keys=["image", "label"]),
    ]
)
root_dir = Path('/data2/tom/crossmoda/source_training')
train_images = [str(root_dir / f) for f in root_dir.iterdir() if str(f).endswith('ceT1.nii.gz')]
train_labels = [f.replace('ceT1', 'Label') for f in train_images]
data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]
train_files, val_files = data_dicts[:-val_size], data_dicts[-val_size:]
train_ds = CacheDataset(
    data=train_files, transform=train_transforms, num_workers=12)

# use batch_size=2 to load images and use RandCropByPosNegLabeld
# to generate 2 x 4 images for network training


train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=12, collate_fn=pad_list_data_collate)

val_ds = CacheDataset(
    data=val_files, transform=val_transforms, num_workers=12)
val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=12, collate_fn=pad_list_data_collate)

device = torch.device("cuda:0")
model = UNet2d5_spvPA(
                dimensions=3,
                in_channels=1,
                out_channels=3,
                channels=(16, 32, 48, 64, 80, 96),
                strides=(
                    (2, 2, 1),
                    (2, 2, 1),
                    (2, 2, 2),
                    (2, 2, 2),
                    (2, 2, 2),
                ),
                kernel_sizes=(
                    (3, 3, 1),
                    (3, 3, 1),
                    (3, 3, 3),
                    (3, 3, 3),
                    (3, 3, 3),
                    (3, 3, 3),
                ),
                sample_kernel_sizes=(
                    (3, 3, 1),
                    (3, 3, 1),
                    (3, 3, 3),
                    (3, 3, 3),
                    (3, 3, 3),
                ),
                num_res_units=2,
                norm=Norm.BATCH,
                dropout=0.1,
                attention_module=True,
            ).to(device)
loss_function = Dice_spvPA(
            to_onehot_y=True, softmax=True, supervised_attention=True, hardness_weighting=False
        )
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,  weight_decay=weight_decay)
epoch_loss_values = []
metric_values = []
post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=3)
post_label = AsDiscrete(to_onehot=True, n_classes=3)
tb_writer = SummaryWriter(f'/data2/tom/domain_adaptation_journal/runs/working_labels,lr={learning_rate}')
model_path = '/data2/tom/domain_adaptation_journal/models/'

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(
            f"{step}/{len(train_ds) // train_loader.batch_size}, "
            f"train_loss: {loss.item():.4f}")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    if debug:
        images_for_grid = []
        for batch_data in train_loader:
            images, labels = batch_data["image"], batch_data["label"]
            for image, label in zip(images, labels):
                central_slice_number = get_center_of_mass_slice(np.squeeze(label[0, :, :, :]))
                images_for_grid.append(image[..., central_slice_number])
                images_for_grid.append(label[..., central_slice_number])
        image_grid = torchvision.utils.make_grid(images_for_grid, normalize=True, scale_each=True)
        tb_writer.add_image("images and preds", image_grid, 0)
    # validation
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():  # turns of PyTorch's auto grad for better performance
            metric_sum = 0.0
            metric_count = 0  # counts number of images
            epoch_loss_val = 0
            step = 0  # counts number of batches
            for val_data in val_loader:  # loop over images in validation set
                step += 1
                val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                val_outputs = model(val_inputs)
                dice_score = compute_dice_score(val_outputs[0], val_labels, device=device)
                loss = loss_function(val_outputs, val_labels)
                metric_count += len(dice_score)
                metric_sum += dice_score.sum().item()
                epoch_loss_val += loss.item()
                metric_count += len(dice_score)
                metric_sum += dice_score.sum().item()
                epoch_loss_val += loss.item()

            metric = metric_sum / metric_count  # calculate mean Dice score of current epoch for validation set
            metric_values.append(metric)
            epoch_loss_val /= step  # calculate mean loss over current epoch

            tb_writer.add_scalars("Loss Train/Val", {"train": epoch_loss, "val": epoch_loss_val}, epoch)
            tb_writer.add_scalar("Dice Score Val", metric, epoch)
            image_grids = []
            for slice_idx in range(0, val_inputs.shape[-1], 1):
                images_for_grid = []
                for image, label, pred in zip(val_inputs, val_labels, val_outputs[0]):
                    # central_slice_number = get_center_of_mass_slice(np.squeeze(label[0, :, :, :]))
                    images_for_grid.append(image[..., slice_idx])
                    images_for_grid.append(label[..., slice_idx])
                    images_for_grid.append(pred[0, ..., slice_idx].unsqueeze(0))
                    images_for_grid.append(pred[1, ..., slice_idx].unsqueeze(0))
                    images_for_grid.append(pred[2, ..., slice_idx].unsqueeze(0))
                image_grid = torchvision.utils.make_grid(images_for_grid, nrow=5, normalize=True, scale_each=True)
                image_grids.append(image_grid)
            image_stack = torch.stack(image_grids, dim=-1).cpu().detach().numpy()
            print(image_stack.shape)
            add_animated_gif(writer=tb_writer, tag='image stack',
                             image_tensor=image_stack, max_out=32, scale_factor=255)
            if metric > best_metric:  # if it's the best Dice score so far, proceed to save
                best_metric = metric
                best_metric_epoch = epoch + 1
                # save the current best model weights
                torch.save(model.state_dict(), os.path.join(model_path, "best_metric_model.pth"))
                print("saved new best metric model")
            print(
                "current epoch {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                    epoch + 1, metric, best_metric, best_metric_epoch
                )
            )

    # learning rate update
    if (epoch + 1) % epochs_with_const_lr == 0 and epoch < 40:
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] / lr_divisor
            print(
                "Dividing learning rate by {}. "
                "New learning rate is: lr = {}".format(lr_divisor, param_group["lr"])
            )