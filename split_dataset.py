{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "60e3b91e-7ed0-4dad-b5ab-cc145bd580f6",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import shutil\n",
    "import random"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "b67f839f-2660-4d58-ac32-5c5ba78c865e",
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset_dir = r\"C:\\Users\\Risha H M\\AI_Crop_System\\crop_disease_detection\\data\\PlantVillage\"\n",
    "\n",
    "# Paths for new dataset folders\n",
    "base_dir = r\"C:\\Users\\Risha H M\\AI_Crop_System\\crop_disease_detection\\data\"\n",
    "train_dir = os.path.join(base_dir, \"train\")\n",
    "val_dir = os.path.join(base_dir, \"val\")\n",
    "test_dir = os.path.join(base_dir, \"test\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "5086a7ad-5b93-4cf1-941f-594772565d36",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_split = 0.7\n",
    "val_split = 0.2\n",
    "test_split = 0.1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "38649679-187e-42d4-975f-c736b9955f5d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Dataset split into train/val/test successfully!\n"
     ]
    }
   ],
   "source": [
    "for split_dir in [train_dir, val_dir, test_dir]:\n",
    "    if not os.path.exists(split_dir):\n",
    "        os.makedirs(split_dir)\n",
    "\n",
    "# ====== SPLIT DATA ======\n",
    "for class_name in os.listdir(dataset_dir):\n",
    "    class_path = os.path.join(dataset_dir, class_name)\n",
    "\n",
    "    if not os.path.isdir(class_path):\n",
    "        continue  # Skip if not a folder\n",
    "\n",
    "    # Create subfolders for each split\n",
    "    for split_dir in [train_dir, val_dir, test_dir]:\n",
    "        split_class_dir = os.path.join(split_dir, class_name)\n",
    "        if not os.path.exists(split_class_dir):\n",
    "            os.makedirs(split_class_dir)\n",
    "\n",
    "    # Get all files in class folder\n",
    "    images = [img for img in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, img))]\n",
    "    random.shuffle(images)\n",
    "\n",
    "    # Calculate split sizes\n",
    "    total_images = len(images)\n",
    "    train_count = int(total_images * train_split)\n",
    "    val_count = int(total_images * val_split)\n",
    "\n",
    "    # Split into train, val, test\n",
    "    train_files = images[:train_count]\n",
    "    val_files = images[train_count:train_count + val_count]\n",
    "    test_files = images[train_count + val_count:]\n",
    "\n",
    "    # Copy files\n",
    "    for img in train_files:\n",
    "        shutil.copyfile(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))\n",
    "\n",
    "    for img in val_files:\n",
    "        shutil.copyfile(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))\n",
    "\n",
    "    for img in test_files:\n",
    "        shutil.copyfile(os.path.join(class_path, img), os.path.join(test_dir, class_name, img))\n",
    "\n",
    "print(\"✅ Dataset split into train/val/test successfully!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "abe8235d-c007-4249-a88e-477eb890cb60",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
