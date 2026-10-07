
import os
import librosa
import numpy as np


# ==================================================
# DATASET PATHS
# ==================================================

original_dataset_path = r"D:\Baby_Cry_Project\Baby Cry Set"

external_dataset_path = r"D:\Baby_Cry_Project\External Dataset\Baby Cry Dataset"


dataset_paths = {
    "Original Dataset": original_dataset_path,
    "External Dataset": external_dataset_path
}


# ==================================================
# AUDIO EXTENSIONS
# ==================================================

supported_extensions = [
    ".wav",
    ".ogg",
    ".mp3"
]


# ==================================================
# ANALYZE HUNGRY AUDIO
# ==================================================

for dataset_name, dataset_path in dataset_paths.items():

    hungry_path = os.path.join(
        dataset_path,
        "hungry"
    )


    print("\n")
    print("=" * 60)
    print(dataset_name)
    print("=" * 60)


    if not os.path.isdir(hungry_path):

        print(
            "Hungry folder not found:",
            hungry_path
        )

        continue


    file_count = 0

    durations = []

    rms_values = []

    sample_rates = []

    error_count = 0


    for file_name in sorted(
        os.listdir(hungry_path)
    ):

        file_extension = os.path.splitext(
            file_name
        )[1].lower()


        if file_extension not in supported_extensions:

            continue


        file_path = os.path.join(
            hungry_path,
            file_name
        )


        try:

            audio, sample_rate = librosa.load(
                file_path,
                sr=None,
                mono=True
            )


            duration = len(audio) / sample_rate


            rms = np.sqrt(
                np.mean(
                    audio ** 2
                )
            )


            file_count += 1

            durations.append(
                duration
            )

            rms_values.append(
                rms
            )

            sample_rates.append(
                sample_rate
            )


        except Exception as e:

            error_count += 1

            print(
                "Error reading:",
                file_path
            )

            print(e)


    if file_count == 0:

        print(
            "No readable audio files found."
        )

        continue


    print(
        "\nAudio files:",
        file_count
    )


    print(
        "Average duration:",
        round(
            np.mean(durations),
            2
        ),
        "seconds"
    )


    print(
        "Minimum duration:",
        round(
            np.min(durations),
            2
        ),
        "seconds"
    )


    print(
        "Maximum duration:",
        round(
            np.max(durations),
            2
        ),
        "seconds"
    )


    print(
        "Average RMS:",
        round(
            np.mean(rms_values),
            6
        )
    )


    print(
        "Minimum RMS:",
        round(
            np.min(rms_values),
            6
        )
    )


    print(
        "Maximum RMS:",
        round(
            np.max(rms_values),
            6
        )
    )


    print(
        "Sample rates:",
        sorted(
            set(sample_rates)
        )
    )


    print(
        "Read errors:",
        error_count
    )


print("\n")
print("Hungry class analysis completed!")