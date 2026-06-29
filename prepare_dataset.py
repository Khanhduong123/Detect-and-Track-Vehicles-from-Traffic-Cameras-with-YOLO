import glob
import os
import shutil
from collections import Counter

from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Define directories
ROOT_DIR = "/media/khanhduong/1TB/master/25MCS-ML_2025-2026_S2_Final-Project"  # pragma: allowlist secret
HUTECH_DIR = os.path.join(ROOT_DIR, "data/640 Hutech Vehicle.v2-fix-label.yolov8")
DETRAC_DIR = os.path.join(ROOT_DIR, "data/DETRAC-YOLO")
OUTPUT_DIR = os.path.join(ROOT_DIR, "dataset")

# Class mapping configurations
# Hutech classes: ['xe buyt':0, 'xe container':1, 'xe cuu hoa':2, 'xe dap':3, 'xe hoi':4, 'xe may':5, 'xe tai':6, 'xe van':7]
hutech_class_map = {
    0: 3,  # xe buyt -> bus
    1: 2,  # xe container -> truck
    2: 2,  # xe cuu hoa -> truck
    3: -1,  # xe dap -> bicycle (Omit)
    4: 1,  # xe hoi -> car
    5: 0,  # xe may -> motorbike
    6: 2,  # xe tai -> truck
    7: 1,  # xe van -> car
}

# Target classes names
target_classes = {0: "motorbike", 1: "car", 2: "truck", 3: "bus"}


def clean_output_dir():
    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)


def collect_hutech_samples():
    print("Collecting HUTECH samples...")
    samples = []
    splits = ["train", "valid", "test"]

    for split in splits:
        img_dir = os.path.join(HUTECH_DIR, split, "images")
        lbl_dir = os.path.join(HUTECH_DIR, split, "labels")

        img_files = glob.glob(os.path.join(img_dir, "*"))
        for img_path in img_files:
            basename = os.path.basename(img_path)
            name_no_ext, ext = os.path.splitext(basename)
            lbl_path = os.path.join(lbl_dir, f"{name_no_ext}.txt")

            if os.path.exists(lbl_path):
                samples.append(
                    {
                        "img_path": img_path,
                        "lbl_path": lbl_path,
                        "basename": basename,
                        "name_no_ext": name_no_ext,
                    }
                )
    print(f"Total HUTECH samples collected: {len(samples)}")
    return samples


def determine_stratification_label(samples):
    print("Analyzing label rarity for stratification...")
    strat_labels = []

    # Rarity rank based on HUTECH distribution: Bus (3) > Truck (2) > Car (1) > Motorbike (0)
    # Rarest-first heuristic for multi-label stratification
    for sample in samples:
        mapped_classes = set()
        with open(sample["lbl_path"], "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    mapped = hutech_class_map.get(cls_id, -1)
                    if mapped != -1:
                        mapped_classes.add(mapped)

        if 3 in mapped_classes:
            strat_labels.append(3)
        elif 2 in mapped_classes:
            strat_labels.append(2)
        elif 1 in mapped_classes:
            strat_labels.append(1)
        elif 0 in mapped_classes:
            strat_labels.append(0)
        else:
            strat_labels.append(-1)

    print(f"Stratification labels distribution: {Counter(strat_labels)}")
    return strat_labels


def split_hutech_data(samples, strat_labels):
    print("Performing stratified split of HUTECH dataset (70/15/15)...")

    # Step 1: 70% Train, 30% Temp (Valid + Test)
    train_samples, temp_samples, train_strat, temp_strat = train_test_split(
        samples, strat_labels, test_size=0.30, stratify=strat_labels, random_state=42
    )

    # Step 2: Split Temp into 50% Valid, 50% Test (15% each of total)
    valid_samples, test_samples = train_test_split(
        temp_samples, test_size=0.50, stratify=temp_strat, random_state=42
    )

    print("HUTECH Split Results:")
    print(f"  Train: {len(train_samples)} images")
    print(f"  Valid: {len(valid_samples)} images")
    print(f"  Test:  {len(test_samples)} images")

    return train_samples, valid_samples, test_samples


def process_and_copy_hutech(samples, split_name):
    print(f"Processing and copying HUTECH samples for {split_name}...")
    dest_img_dir = os.path.join(OUTPUT_DIR, split_name, "images")
    dest_lbl_dir = os.path.join(OUTPUT_DIR, split_name, "labels")

    for sample in tqdm(samples):
        # Destination file names (prefixed to avoid potential name collisions)
        new_basename = f"hutech_{sample['basename']}"
        new_lblname = f"hutech_{sample['name_no_ext']}.txt"

        # 1. Copy image file
        shutil.copy2(sample["img_path"], os.path.join(dest_img_dir, new_basename))

        # 2. Map and filter label file
        mapped_lines = []
        with open(sample["lbl_path"], "r") as f_in:
            for line in f_in:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    mapped_cls = hutech_class_map.get(cls_id, -1)
                    if mapped_cls != -1:
                        mapped_lines.append(
                            f"{mapped_cls} {parts[1]} {parts[2]} {parts[3]} {parts[4]}"
                        )

        with open(os.path.join(dest_lbl_dir, new_lblname), "w") as f_out:
            f_out.write("\n".join(mapped_lines) + "\n")


def filter_and_copy_detrac():
    print("Filtering and extracting additional data from UA-DETRAC...")

    detrac_img_dir = os.path.join(DETRAC_DIR, "images", "train")
    detrac_lbl_dir = os.path.join(DETRAC_DIR, "labels", "train")

    dest_img_dir = os.path.join(OUTPUT_DIR, "train", "images")
    dest_lbl_dir = os.path.join(OUTPUT_DIR, "train", "labels")

    label_files = glob.glob(os.path.join(detrac_lbl_dir, "*.txt"))
    copied_count = 0

    # We only process files that do NOT start with hutech_ (the original UA-DETRAC files)
    for lbl_path in tqdm(label_files):
        basename = os.path.basename(lbl_path)
        if basename.startswith("hutech_"):
            continue

        name_no_ext, _ = os.path.splitext(basename)
        img_name = f"{name_no_ext}.jpg"
        img_path = os.path.join(detrac_img_dir, img_name)

        if not os.path.exists(img_path):
            continue

        # Check label content for Class 2 (truck) or Class 3 (bus)
        contains_target = False
        mapped_lines = []

        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    # UA-DETRAC classes: 0=motorcycle, 1=car, 2=truck, 3=bus
                    if cls_id in (2, 3):
                        contains_target = True
                    # Keep all classes in target format
                    mapped_lines.append(line.strip())

        # Copy only if it contains truck or bus objects
        if contains_target:
            # 1. Copy image file (resolves symlinks automatically)
            shutil.copy2(img_path, os.path.join(dest_img_dir, f"detrac_{img_name}"))

            # 2. Write labels file
            with open(os.path.join(dest_lbl_dir, f"detrac_{basename}"), "w") as f_out:
                f_out.write("\n".join(mapped_lines) + "\n")

            copied_count += 1

    print(f"Total filtered UA-DETRAC samples merged into Train: {copied_count}")


def create_dataset_yaml():
    yaml_path = os.path.join(OUTPUT_DIR, "dataset.yaml")
    print(f"Creating dataset configuration YAML: {yaml_path}")

    yaml_content = f"""# Joint HUTECH and UA-DETRAC Unified Dataset
path: {OUTPUT_DIR}
train: train/images
val: valid/images
test: test/images

names:
  0: motorbike
  1: car
  2: truck
  3: bus
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print("YAML config written successfully.")


def main():
    print("=== Starting Dataset Preprocessing Pipeline ===")

    # 0. Clean and prepare output directories
    clean_output_dir()

    # 1. Collect HUTECH images and labels
    hutech_samples = collect_hutech_samples()

    # 2. Analyze label distribution & stratify HUTECH
    strat_labels = determine_stratification_label(hutech_samples)
    train_hutech, valid_hutech, test_hutech = split_hutech_data(
        hutech_samples, strat_labels
    )

    # 3. Process HUTECH splits
    process_and_copy_hutech(train_hutech, "train")
    process_and_copy_hutech(valid_hutech, "valid")
    process_and_copy_hutech(test_hutech, "test")

    # 4. Filter and merge UA-DETRAC data
    filter_and_copy_detrac()

    # 5. Create dataset.yaml configuration file
    create_dataset_yaml()

    print("\n=== Dataset Preprocessing and Merging completed successfully! ===")
    print(f"Output location: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
