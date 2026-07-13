# EHR Clinical Chatbot

An AI clinical assistant that converts natural language or voice questions into SQL, queries a MIMIC-IV demo EHR database, and returns a concise clinical summary. The project demonstrates an end-to-end healthcare AI workflow: model fine-tuning, schema-aware text-to-SQL generation, database execution, speech input, document extraction, and a Streamlit user interface.

## Recruiter Snapshot

- Built a domain-specific text-to-SQL assistant for electronic health record data.
- Fine-tuned `Qwen/Qwen2.5-Coder-7B-Instruct` using QLoRA/PEFT for medical database question answering.
- Integrated a DuckDB version of the MIMIC-IV demo database with 31 clinical tables and 342 schema columns.
- Added a Streamlit application for text queries, voice queries, generated SQL inspection, raw result review, and clinical answer generation.
- Used Gemini for speech transcription, final natural-language answer generation, and PDF-to-structured-data extraction.
- Packaged the fine-tuned LoRA adapter and tokenizer artifacts for inference.

## Problem It Solves

Clinical data is often locked inside complex relational databases. Non-technical users may know the clinical question they want to ask, but not the SQL needed to retrieve it. This project bridges that gap by allowing users to ask questions such as:

- "What medications does patient 10004235 take?"
- "How many patients are in the database?"
- "What is the gender of patient 10004235?"
- "When was a patient admitted?"

The assistant generates SQL, executes it against the EHR database, and summarizes the result in readable language.

## Key Features

### Natural Language to SQL

The fine-tuned Qwen model receives the database schema and a clinical question, then generates a SQL query tailored to the EHR database.

### Voice-Based Clinical Queries

The Streamlit app records audio questions with `st-audiorec`, sends the audio to Gemini for transcription, and passes the transcription, through the same text-to-SQL pipeline.

### SQL Execution on EHR Data

Generated queries are executed on `mimic_iv_demo.duckdb`, a local DuckDB database containing representative MIMIC-IV demo clinical tables.

### Clinical Summary Generation

After query execution, Gemini transforms the raw query result into a clear response for the user, while the app still exposes the generated SQL and raw dataframe for transparency.

### PDF Document Ingestion

The app includes a document upload flow that extracts structured medical data from PDFs using Gemini, validates it against expected table schemas, generates insert queries, and updates a writable DuckDB copy.

### Fallback Query Handling

For common queries around patient demographics, medications, admissions, age, and counts, the app includes fallback SQL templates when model-generated SQL fails.

## Architecture

```text
User question
  |
  |-- Text input
  |-- Voice input -> Gemini transcription
  |
Schema-aware prompt
  |
Fine-tuned Qwen2.5-Coder model
  |
Generated SQL
  |
DuckDB MIMIC-IV demo database
  |
Query result dataframe
  |
Gemini clinical response generation
  |
Streamlit UI response + SQL + raw data
```

## Tech Stack

- Python
- Streamlit
- PyTorch
- Transformers
- PEFT / LoRA
- TRL `SFTTrainer`
- BitsAndBytes 4-bit quantization
- DuckDB
- Pandas
- Google Gemini API
- Kaggle notebooks and Kaggle secrets
- ngrok for public Streamlit preview links

## Model Training

The training workflow is in `training/finetune.ipynb`.

Training approach:

- Base model: `Qwen/Qwen2.5-Coder-7B-Instruct`
- Fine-tuning method: QLoRA with PEFT
- Quantization: 4-bit NF4 via BitsAndBytes
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Trainer: TRL `SFTTrainer`
- Train/validation split: 95/5
- Epochs: 2
- Max sequence length: 1024

The saved adapter is included in `finetuned_model/`.

## Database

The project uses a DuckDB version of the MIMIC-IV demo database:

- Database file: `database/mimic_iv_demo.duckdb`
- Size: about 38 MB
- Tables: 31
- Schema columns: 342
- Patients table rows: 100
- Largest table: `chartevents` with 668,862 rows

Important clinical tables include:

- `patients`
- `admissions`
- `icustays`
- `chartevents`
- `labevents`
- `prescriptions`
- `diagnoses_icd`
- `procedures_icd`
- `transfers`

The schema metadata is stored in `table schema/tables.json`.

## Project Structure

```text
ehr-clinical-chatbot/
  app_inference/
    streamlit-text2sql.ipynb      # Streamlit app generation and inference workflow
  database/
    mimic_iv_demo.duckdb          # Demo EHR database
  dataset/                        # Dataset workspace
  finetuned_model/
    adapter_model.safetensors     # Fine-tuned LoRA adapter
    adapter_config.json           # PEFT adapter configuration
    tokenizer.json                # Tokenizer artifacts
    README.md                     # Auto-generated model card stub
  table schema/
    tables.json                   # MIMIC-IV demo schema metadata
  training/
    finetune.ipynb                # QLoRA fine-tuning notebook
```

## How to Run

### 1. Install dependencies

```bash
pip install torch transformers peft trl bitsandbytes datasets duckdb pandas streamlit google-generativeai st-audiorec pyngrok
```

### 2. Configure Gemini

The inference notebook expects a Gemini API key from Kaggle secrets:

```text
GEMINI_API_KEY
```

For local execution, replace the Kaggle secret loading logic with an environment variable such as:

```python
import os
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
```

### 3. Run the inference app

Open `app_inference/streamlit-text2sql.ipynb` and run the Streamlit app generation cell. The notebook writes an `app.py` file and launches it with:

```bash
streamlit run app.py
```

The current notebook paths are Kaggle-specific, so update these constants if running locally:

```python
FINAL_MODEL_DIR = "finetuned_model"
SCHEMA_FILE_PATH = "table schema/tables.json"
ORIGINAL_DUCKDB_FILE = "database/mimic_iv_demo.duckdb"
```

## What This Project Demonstrates

- Practical LLM fine-tuning for a structured healthcare data task.
- Schema-grounded prompting for reliable text-to-SQL generation.
- Integration of LLM inference with an analytical database.
- Human-friendly UX through text input, voice input, quick actions, SQL visibility, and raw data inspection.
- Awareness of production concerns such as fallback logic, writable database copies, error handling, and secrets management.

## Limitations and Next Steps

- The inference notebook is currently optimized for Kaggle paths and should be refactored into a standalone `app.py` for easier local deployment.
- The repository does not include benchmark metrics such as exact-match SQL accuracy or execution accuracy.
- Generated SQL should be sandboxed and restricted before any production use on sensitive data.
- The PDF ingestion flow should add stronger validation before database insertion.
- Real API tokens should never be committed; use environment variables or managed secrets.

## Why It Matters

This project shows the ability to build beyond a simple chatbot. It combines fine-tuning, retrieval from structured clinical data, speech input, document processing, database engineering, and an interactive UI into one applied healthcare AI system. For recruiters, it is a strong signal of hands-on LLM application development with real data infrastructure and practical product thinking.
