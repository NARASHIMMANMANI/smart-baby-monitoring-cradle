
import os
import csv
import hashlib
from collections import defaultdict

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


output_file = "original_conflict_report.csv"

with open(output_file, "w", newline="", encoding="utf-8") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Group_ID",
        "Audio_Hash",
        "Labels",
        "Class",
        "File_Path"
    ])

    group_id = 1

    for file_hash, records in hash_records.items():

        labels = sorted(set(record["class"] for record in records))

        if len(labels) <= 1:
            continue

        label_text = " | ".join(labels)

        for record in records:

            writer.writerow([
                group_id,
                file_hash,
                label_text,
                record["class"],
                record["file"]
            ])

        group_id += 1


print()
print("Conflict CSV created successfully!")
print("File:", output_file)
print("Total conflict groups:", group_id - 1)