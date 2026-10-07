
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

            hash_records[file_hash].append(class_name)


conflicting_groups = 0
label_pairs = Counter()

for records in hash_records.values():

    labels = sorted(set(records))

    if len(labels) > 1:

        conflicting_groups += 1

        for index in range(len(labels)):

            for second_index in range(index + 1, len(labels)):

                pair = (labels[index], labels[second_index])
                label_pairs[pair] += 1


print()
print("=" * 60)
print("ORIGINAL DATASET CONFLICT REPORT")
print("=" * 60)

print("Total conflicting groups:", conflicting_groups)

print()
print("Conflicting label pairs:")

for pair, count in label_pairs.most_common():

    print(pair[0], "<-->", pair[1], ":", count)