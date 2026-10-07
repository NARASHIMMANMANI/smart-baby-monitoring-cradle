
import os
import hashlib
from collections import defaultdict, Counter

dataset_path = r"D:\Baby_Cry_Project\Baby Cry Set"

classes = [
    "belly pain",
    "burping",
    "cold_hot",
    "discomfort",
    "hungry",
    "laugh",
    "noise",
    "silence",
    "tired"
]

supported_extensions = (".wav", ".ogg", ".mp3")

hash_records = defaultdict(list)

print("Reading Original dataset...")

for class_name in classes:

    class_path = os.path.join(dataset_path, class_name)

    if not os.path.exists(class_path):
        continue

    for root, folders, files in os.walk(class_path):

        for file_name in files:

            if not file_name.lower().endswith(supported_extensions):
                continue

            file_path = os.path.join(root, file_name)

            sha256 = hashlib.sha256()

            with open(file_path, "rb") as file:

                while True:

                    data = file.read(1024 * 1024)

                    if not data:
                        break

                    sha256.update(data)

            file_hash = sha256.hexdigest()

            hash_records[file_hash].append({
                "class": class_name,
                "file": file_path
            })


usable_counts = Counter()
conflicting_files = 0
usable_files = 0

for file_hash, records in hash_records.items():

    labels = set(record["class"] for record in records)

    if len(labels) > 1:

        conflicting_files += len(records)

    else:

        for record in records:

            usable_counts[record["class"]] += 1
            usable_files += 1


print()
print("=" * 60)
print("ORIGINAL DATASET USABLE FILE COUNTS")
print("=" * 60)

print("Files in conflicting groups:", conflicting_files)
print("Files in non-conflicting groups:", usable_files)

print()
print("Usable files by class:")

for class_name in classes:

    print(class_name, ":", usable_counts[class_name])