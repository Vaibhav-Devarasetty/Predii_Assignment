# Predii_Assign

## Devarasetty Sri Vaibhav - Roll Number: 2101CS24

This is my submission for the assignment given to me by Predii for the Research Engineer FTE Role (2101CS24).

## Overview

In this project, I have worked on processing and summarizing a dataset provided as part of the assignment. Below is an overview of the steps I followed:

### 1. Data Cleaning
- I started by cleaning the dataset `FLAT_RCL.txt` which I downloaded from the website provided in the assignment.
- I removed irrelevant rows and filtered out the columns of interest, focusing on the **MAKE** column with only **FORD** and **TOYOTA** values, as specified in the assignment.
- The cleaned dataset was then saved as a CSV file named `FLAT_RCL_3.csv`, which is located in the `dataset` directory.

### 2. Code Implementation
- For finding relevant documents, I used several vector embeddings such as:
  - **GloVe**
  - **Word2Vec**
  - **BERT**
  - **OpenAI Embeddings**
  - **Sentence Transformers**
  
- After extracting the relevant rows based on these embeddings, I applied **RAG** (Retriever-Augmented Generation) to retrieve the most relevant documents.

### 3. Summarization
- For summarizing the relevant rows, I used two different models:
  - **Facebook's BART Large**
  - **Llama 3.1-8B**
  
  Both models were utilized for generating concise summaries of the filtered rows.

### 4. Hardware and Computation
- I used my **MacBook Pro (M1 Pro)** for computation, utilizing the **MPS (Metal Performance Shader)** backend to run the models efficiently.

---

## Instructions to Run

1. **Ensure the Dataset is Available**  
   Make sure the dataset file `FLAT_RCL.txt` is available in the `dataset` directory. This file is used for data cleaning and filtering.

2. **Run the Python Script**  
   To clean the dataset, extract relevant rows, and generate summaries, run the Python script by executing the following command in your terminal:
   Replace your_script_name.py with the actual name of the Python script you are using for this task.
   ```bash
   python your_script_name.py
   

This repository showcases the work I have done as part of the assignment for Predii.

Thank you for reviewing my submission!