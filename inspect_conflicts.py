
import os
import hashlib
from collections import defaultdict, Counter

dataset_paths = {
    "Original": r"D:\Baby_Cry_Project\Baby Cry Set",
    "External": r"D:\Baby_Cry_Project\External Dataset\Baby Cry Dataset"
}

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

print("Reading audio files...")

for source_name, dataset_path in dataset_paths.items():

    print("Reading:", source_name)

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

                try:
                    with open(file_path, "rb") as file:
                        while True:
                            data = file.read(1024 * 1024)

                            if not data:
                                break

                            sha256.update(data)

                    file_hash = sha256.hexdigest()

                    hash_records[file_hash].append({
                        "source": source_name,
                        "class": class_name,
                        "file": file_path
                    })

                except Exception as error:
                    print("Error:", file_path)
                    print(error)


conflicting_groups = []

for file_hash, records in hash_records.items():

    labels = set(record["class"] for record in records)

    if len(labels) > 1:
        conflicting_groups.append(records)


label_pairs = Counter()

for records in conflicting_groups:

    labels = sorted(set(record["class"] for record in records))

    for index in range(len(labels)):

        for second_index in range(index + 1, len(labels)):

            pair = (labels[index], labels[second_index])
            label_pairs[pair] += 1


print()
print("=" * 60)
print("CONFLICT REPORT")
print("=" * 60)

print("Total conflicting groups:", len(conflicting_groups))

print()
print("Conflicting label pairs:")

for pair, count in label_pairs.most_common():

    print(pair[0], "<-->", pair[1], ":", count)


print()
print("=" * 60)
print("EXAMPLE CONFLICTS")
print("=" * 60)

for group_number, records in enumerate(conflicting_groups[:10], start=1):

    print()
    print("Conflict group:", group_number)

    for record in records:

        print("Source:", record["source"])
        print("Class:", record["class"])
        print("File:", record["file"])