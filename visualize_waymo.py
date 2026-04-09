import os
import glob
import random
import matplotlib.pyplot as plt
from PIL import Image

train_dir = "/scratch/rbaskar5/Dataset/waymo_front/training"
val_dir = "/scratch/rbaskar5/Dataset/waymo_front/validation"

# Get all images
train_images = glob.glob(os.path.join(train_dir, "**/*.jpg"), recursive=True)
val_images = glob.glob(os.path.join(val_dir, "**/*.jpg"), recursive=True)

print(f"Total extracted training images: {len(train_images)}")
print(f"Total extracted validation images: {len(val_images)}")
print(f"Total dataset: {len(train_images) + len(val_images)} images")

all_images = train_images + val_images

if not all_images:
    print("No images found to visualize.")
    exit(0)

# Select random 6 images or fewer if we don't have enough
num_samples = min(6, len(all_images))
sample_images = random.sample(all_images, num_samples)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Waymo Dataset FRONT Camera Samples", fontsize=16)

for ax, img_path in zip(axes.flatten(), sample_images):
    img = Image.open(img_path)
    ax.imshow(img)
    ax.set_title(os.path.basename(img_path)[:20] + "...")
    ax.axis("off")

# Hide empty subplots
for i in range(num_samples, 6):
    axes.flatten()[i].axis("off")

plt.tight_layout()
output_path = "/scratch/rbaskar5/GLANCE/waymo_front_samples.png"
plt.savefig(output_path, dpi=100)
print(f"\nSaved visualization to: {output_path}")
