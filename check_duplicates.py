
import os
import hashlib


# ==================================================
# DATASET PATHS
# ==================================================

dataset_paths = {

    "Original": r"D:\Baby_Cry_Project\Baby Cry Set",

    "External": r"D:\Baby_Cry_Project\External Dataset\Baby Cry Dataset"

}


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
# COLLECT FILE INFORMATION
# ==================================================

hash_records = {}


total_files = 0


for dataset_name, dataset_path in dataset_paths.items():

    print(
        "\nReading:",
        dataset_name
    )


    for class_name in os.listdir(dataset_path):

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

                    "file_path": file_path

                }


                if file_hash not in hash_records:

                    hash_records[file_hash] = []


                hash_records[file_hash].append(
                    record
                )


                total_files += 1


            except Exception as e:

                print(
                    "Error reading:",
                    file_path
                )

                print(e)


# ==================================================
# ANALYZE DUPLICATES
# ==================================================

duplicate_groups = 0

duplicate_files = 0

conflicting_groups = 0

same_label_groups = 0


conflicting_examples = []


for file_hash, records in hash_records.items():

    if len(records) <= 1:

        continue


    duplicate_groups += 1


    duplicate_files += len(records)


    class_names = set()


    for record in records:

        class_names.add(
            record["class_name"]
        )


    if len(class_names) == 1:

        same_label_groups += 1


    else:

        conflicting_groups += 1


        if len(conflicting_examples) < 10:

            conflicting_examples.append(
                records
            )


# ==================================================
# FINAL SUMMARY
# ==================================================

print("\n")

print("=" * 60)

print("DUPLICATE ANALYSIS SUMMARY")

print("=" * 60)


print(
    "Total audio files checked:",
    total_files
)


print(
    "Unique audio hashes:",
    len(hash_records)
)


print(
    "Duplicate groups:",
    duplicate_groups
)


print(
    "Files inside duplicate groups:",
    duplicate_files
)


print(
    "Same-label duplicate groups:",
    same_label_groups
)


print(
    "Conflicting-label duplicate groups:",
    conflicting_groups
)


# ==================================================
# SHOW EXAMPLES OF CONFLICTS
# ==================================================

print("\n")

print("=" * 60)

print("CONFLICTING LABEL EXAMPLES")

print("=" * 60)


if len(conflicting_examples) == 0:

    print(
        "No conflicting duplicate groups found."
    )


else:

    for group_number, records in enumerate(
        conflicting_examples,
        start=1
    ):

        print("\n")

        print(
            "Conflict group:",
            group_number
        )


        for record in records:

            print(
                "Dataset:",
                record["dataset"]
            )


            print(
                "Class:",
                record["class_name"]
            )


            print(
                "File:",
                record["file_path"]
            )


            print()


print("\n")

print(
    "Duplicate analysis completed!"
)