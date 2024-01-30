# About

This script was developed as part of the Trifork recruitment process assignment. Its primary goal is to facilitate the conversion of a COCO dataset to the YOLO format. Additionally, the script provides functionalities such as image resizing and dataset splitting into training, validation, and test sets.

# How to Use

Clone the repository:

   ```bash
   git clone https://github.com/jsalisburyv/Trifork-Assignment.git
   cd Trifork-Assignment
   ```

## 1. Pure Python

### Prerequisites

- [Python 3.11](https://www.python.org/downloads/release/python-3117/)
- Dependencies listed in `requirements.txt`
    ```bash
    pip install -r requirements.txt
   ```
### Running
```bash
python src/main_script.py -c path/to/coco.json -i path/to/images -o path/to/output
```
Example:
```bash
python src/main_script.py -c data/coco.json -i data/images -o output
```

## 2. Docker Container
This method eliminates the need for pre-installing Python and requirements, ensuring both isolation and portability.

### Prerequisites
- [Docker Engine](https://docs.docker.com/engine/install/)
- Modify script parameters in the Dockerfile (Optional) 

### Running
First build the docker image using the provided Dockerfile.
```bash
docker build -t assignment:latest .
```
Then run the image with the following command:
```bash
docker run assignment:latest
```
### Obtaining results:
To visualize output files generated by your Python script inside a Docker container, you have a few options:
#### 1. Copy Files from the Container to the Host
Find the container id of using:
```bash
docker ps -a
```
Afterwards copy the output directory to the desired folder
```bash
docker cp container_id:/app/output /local/path
```

#### 2. Bind Mount a Local Directory:
When running mount a local directory to the container:
```bash
docker run -v /local/path:/app/output assignment:latest
```
## 3. (Preferred) Docker-Compose
### Prerequisites
- [Docker Desktop](https://docs.docker.com/desktop/)
- Modify script parameters in the Dockerfile (Optional) 

### Running
Start a docker container using the provided docker-compose.yaml:
```bash
docker-compose run preprocessing
```
This command generates the image if it doesn't exist, and conveniently sets up the mounted directory for the output.

### Notes
**docker-compose up vs run**

As noted in the Known Issues of [`TQDM`](https://pypi.org/project/tqdm/), the progress bar won't show until the next line is printed in the console, so for expected results use `run` instead of `up`.

# Implementation Considerations
When developing this script, the following key considerations were taken into account to ensure robustness and flexibility:

## 1. Modularity and Separation of Concerns
The script is divided into modular functions and classes to enhance code readability and maintainability.
Each module (data_loader, annotation_converter, preprocessing) focuses on specific tasks, promoting a clear separation of concerns.

## 2. Portability
The script is designed to run seamlessly within a Docker container, minimizing dependencies and ensuring consistency across different environments.
Docker provides isolation, making it easier to manage dependencies and execute the script without worrying about the specific setup of the host system.

## 3. Command-Line Interface (CLI) Design:
The script utilizes the argparse library for a user-friendly command-line interface, allowing users to easily configure parameters such as input paths, output paths, and image resizing options.

## 4. Error Handling:
Robust error handling is implemented throughout the script to catch and raise meaningful exceptions, providing clear feedback in case of issues.
Low-level error catching mechanisms were chosen over propagating errors to the main function, ensuring localized handling and precise identification of problems within specific components.

## 5. Documentation:
The script is well-documented with inline comments and clear docstrings to facilitate understanding and usage.
This README file provides comprehensive instructions on how to use the script, including setup, execution, and considerations.

## 6. Inclusion of Original Data in Repository
By including the original COCO dataset (images and annotations) within the repository, users can effortlessly clone it and immediately run the script without the need for separate downloads. This decision was motivated by the desire to simplify the correction process for recruiters evaluating the assignment, ensuring a smoother and more efficient evaluation experience. This aligns with the convenience of providing default parameters inside the Dockerfiles and examples in this README.

The decision to include the dataset directly in the repository was influenced by its manageable size. For larger datasets, alternative strategies would be employed, such as offering explicit instructions on obtaining the dataset from an external source or utilizing data versioning tools like DVC (Data Version Control) to manage large datasets independently.

# Further Work
While the current version of the script achieves its primary objectives, there are opportunities for enhancement and expansion in the future. Some potential areas for further work include:

1. **Optimization for Large Datasets:** Investigate optimization strategies for handling larger datasets efficiently, ensuring the script remains scalable and performant. Some examples may be Parallel or Batch processing, Streaming I/O or Caching.

2. **Graphical User Interface (GUI):** Develop a user-friendly GUI to simplify interaction with the script, making it accessible to individuals without programming experience.

3. **Integration with Continuous Integration (CI):** Implement CI/CD workflows to automate testing and deployment processes, ensuring the reliability and stability of the script across different environments.