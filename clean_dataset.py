
import os
import shutil
import hashlib
from collections import defaultdict


# ==================================================
# DATASET PATHS
# ==================================================

dataset_paths = {

    "Original": r"D:\Baby_Cry_Project\Baby Cry Set",

    "External": r"D:\Baby_Cry_Project\External Dataset\Baby Cry Dataset"

}


clean_dataset_path = r"D:\Baby_Cry_Project\Clean Baby Cry Dataset"


# ==================================================
# CLASS NAMES
# ==================================================

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


# ==================================================
# SUPPORTED FILE TYPES
# ==================================================

supported_extensions = [

    ".wav",
    ".ogg",
    ".mp3"

]


# ==================================================
# CALCULATE FILE HASH
# ==================================================

def calculate_file_hash(file_path):

    hash_object = hashlib.sha256()


    with open(
        file_path,
        "rb"
    ) as file:

        while True:

            data = file.read(
                1024 * 1024
            )


            if not data:

                break


            hash_object.update(
                data
            )


    return hash_object.hexdigest()


# ==================================================
# COLLECT AUDIO FILES BY HASH
# ==================================================

hash_records = defaultdict(list)


total_files = 0


for dataset_name, dataset_path in dataset_paths.items():

    print("\nReading:", dataset_name)


    for class_name in classes:

        class_path = os.path.join(
            dataset_path,
            class_name
        )


        if not os.path.isdir(class_path):

            continue


        for file_name in os.listdir(class_path):

            extension = os.path.splitext(
                file_name
            )[1].lower()


            if extension not in supported_extensions:

                continue


            file_path = os.path.join(
                class_path,
                file_name
            )


            try:

                file_hash = calculate_file_hash(
                    file_path
                )


                record = {

                    "dataset": dataset_name,

                    "class_name": class_name,

                    "file_name": file_name,

                    "file_path": file_path

                }


                hash_records[file_hash].append(
                    record
                )


                total_files += 1


            except Exception as e:

                print(
                    "\nError reading:",
                    file_path
                )

                print(e)


# ==================================================
# PREPARE CLEAN DATASET
# ==================================================

if os.path.exists(clean_dataset_path):

    print("\nClean dataset folder already exists.")

    print(
        "Delete or rename it manually before running again."
    )

    raise SystemExit


os.makedirs(
    clean_dataset_path
)


for class_name in classes:

    class_path = os.path.join(
        clean_dataset_path,
        class_name
    )


    os.makedirs(
        class_path
    )


# ==================================================
# CLEAN HASH GROUPS
# ==================================================

kept_files = 0

removed_conflicting_files = 0

removed_duplicate_files = 0

conflicting_groups = 0

same_label_groups = 0


class_counts = defaultdict(int)


for file_hash, records in hash_records.items():

    class_names = set()


    for record in records:

        class_names.add(
            record["class_name"]
        )


    # ----------------------------------------------
    # Remove conflicting label groups
    # ----------------------------------------------

    if len(class_names) > 1:

        conflicting_groups += 1

        removed_conflicting_files += len(
            records
        )

        continue


    # ----------------------------------------------
    # Same-label duplicate group
    # ----------------------------------------------

    if len(records) > 1:

        same_label_groups += 1


    # ----------------------------------------------
    # Keep one file from this hash group
    # ----------------------------------------------

    selected_record = records[0]


    source_path = selected_record[
        "file_path"
    ]


    class_name = selected_record[
        "class_name"
    ]


    file_name = selected_record[
        "file_name"
    ]


    destination_folder = os.path.join(
        clean_dataset_path,
        class_name
    )


    destination_path = os.path.join(
        destination_folder,
        file_name
    )


    # Prevent filename collision

    if os.path.exists(destination_path):

        base_name = os.path.splitext(
            file_name
        )[0]


        extension = os.path.splitext(
            file_name
        )[1]


        destination_path = os.path.join(

            destination_folder,

            base_name + "_copy" + extension

        )


    shutil.copy2(
        source_path,
        destination_path
    )


    kept_files += 1

    class_counts[class_name] += 1


    removed_duplicate_files += len(
        records
    ) - 1


# ==================================================
# FINAL SUMMARY
# ==================================================

print("\n")

print("=" * 60)

print("CLEAN DATASET SUMMARY")

print("=" * 60)


print(
    "Total files checked:",
    total_files
)


print(
    "Unique audio hashes:",
    len(hash_records)
)


print(
    "Conflicting label groups removed:",
    conflicting_groups
)


print(
    "Files removed due to conflicting labels:",
    removed_conflicting_files
)


print(
    "Same-label duplicate groups:",
    same_label_groups
)


print(
    "Repeated copies removed:",
    removed_duplicate_files
)


print(
    "Files kept:",
    kept_files
)


print("\nClass counts in clean dataset:")


for class_name in classes:

    print(

        class_name,

        ":",

        class_counts[class_name]

    )


print("\n")

print(
    "Clean dataset path:",
    clean_dataset_path
)


print(
    "Dataset cleaning completed!"
)