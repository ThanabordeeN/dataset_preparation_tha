# Dataset Translation and Preparations to Thai Language

This repository provides tools and scripts for translating datasets from Hugging Face to Thai using the LLAMA3.1 8b model and transforming them into a format suitable for fine-tuning.

## Overview

The process involves the following steps:

1. **Dataset Loading:** Load a dataset from Hugging Face.
2. **Translation:** Translate the relevant columns of the dataset into Thai using the LLAMA3.1 8b model.
3. **Data Transformation:** Transform the translated dataset into a format suitable for fine-tuning a custom LLM.

## Requirements

* Python 3.10
* `datasets`
* `ollama`

## Installation

```bash
pip install  datasets ollama
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.

