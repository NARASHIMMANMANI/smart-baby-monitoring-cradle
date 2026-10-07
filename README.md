\# Smart Baby Monitoring Cradle



An IoT-enabled smart cradle for real-time baby monitoring with machine learning-based baby cry classification.



\## Project Overview



This project focuses on identifying different types of baby sounds using machine learning techniques. Audio recordings are processed to extract meaningful features, and machine learning models are trained to classify the sounds into different categories.



\## Baby Cry Classes



The current dataset contains the following sound categories:



\* Belly Pain

\* Burping

\* Cold/Hot

\* Discomfort

\* Hungry

\* Laugh

\* Noise

\* Silence

\* Tired



\## Machine Learning



The project extracts audio features from the recordings using `librosa` and uses machine learning algorithms from `scikit-learn` for classification.



Current models include:



\* Support Vector Machine (SVM)

\* Random Forest

\* Extra Trees Classifier



The project also uses PCA for feature visualization.



\## Technologies Used



\* Python

\* Librosa

\* NumPy

\* Scikit-learn

\* Matplotlib



\## Project Structure



```text

smart-baby-monitoring-cradle/

│

├── main.py

├── check\_duplicates.py

├── check\_hungry.py

├── check\_original\_conflicts.py

├── clean\_dataset.py

├── create\_conflict\_csv.py

├── inspect\_conflicts.py

├── original\_clean\_counts.py

├── requirements.txt

├── README.md

└── .gitignore

```



\## Dataset



The audio datasets are not included in this repository because of their size and project/data considerations.



Place the required datasets in the appropriate local project directories before running the program.



\## Installation



Clone the repository and enter the project directory:



```bash

git clone https://github.com/NARASHIMMANMANI/smart-baby-monitoring-cradle.git

cd smart-baby-monitoring-cradle

```



Create and activate a virtual environment:



```bash

python -m venv venv

```



Windows:



```bash

venv\\Scripts\\activate

```



Install the required packages:



```bash

pip install -r requirements.txt

```



\## Running the Project



Run the main program using:



```bash

python main.py

```



\## Project Status



This project is currently under development. The machine learning pipeline is being improved and additional baby-monitoring functionality will be added during development.



\## Author



Narashimman Mani



