# Fake News Detection project
AITP project on fake news detection.

## Environment Setup
### Prerequisites
- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- Git

### Setup Instructions

#### Option 1: Using Conda (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/natalielai1/Fake-News-Group-3.git
cd Fake-News-Group-3
```

2. Create the conda environment:
```bash
conda env create -f environment.yml
```

3. Activate the environment:
```bash
conda activate fake-news-detection
```

#### Option 2: Using pip only
TO-DO

### Updating the Environment
Throughout the project we will probably discover new dependencies we need. When new dependencies are added:
```bash
conda activate ml-project
conda env update -f environment.yml --prune
```

### Deactivating the Environment
```bash
conda deactivate
```

### Removing the Environment
```bash
conda env remove -n fake-news-detection
```