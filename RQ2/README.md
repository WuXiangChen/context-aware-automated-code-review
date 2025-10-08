# RQ2 - Context-Aware Code Review Evaluation

## 📁 Folder Structure Overview

This folder contains the implementation and experimental results for Research Question 2 (RQ2), which evaluates the effectiveness of context-aware automated code review using various Large Language Models (LLMs).

```
RQ2/
├── Data/                       # Input datasets
├── Database/                   # Data models and structures
├── Metrics/                    # Evaluation metric implementations
├── Results_Metrics/            # Experimental results organized by rounds
├── config/                     # Configuration files
├── models/                     # Data model definitions
├── utils/                      # Utility functions and helpers
├── _main_3_5.py               # Main experiment orchestration script
├── _3_query_WithContextInfo_DS_LLM.py    # LLM querying with context
├── _4_get_metric_for_dsLLM_result.py     # Metrics calculation
└── _5_analysis_for_reAndContextState.py  # Context state analysis
```

---

## 📂 Directory Details

### **Data/**
Contains input datasets for experiments:
- `rq2_rl_train_NN_5.xlsx` - Training dataset with repository information and selected nodes
- `nodeList_size_3000.pkl` - Pickled context information for code review items
- `case_study.xlsx` - Case study dataset with Q&A pairs

### **Database/**
Database-related data structures:
- `node.py` - Data classes for code review items:
  - `CodeReviewItem` - Main data structure for code review information
  - `GraphStatistics` - Graph-based statistical metrics
  - `GraphSummary` - Graph summary information
  - `GraphData` - Graph data structures (nodes, edges, attributes)

### **Metrics/**
Evaluation metric implementations:
- `bleu.py` - BLEU score calculation
- `smooth_bleu.py` - Smoothed BLEU metric
- `cal_metrics_with_CodeReviewer.py` - Comprehensive metrics computation
- `stopwords.txt` - Stopwords for text processing

### **Results_Metrics/**
Experimental results organized by rounds (round_1, round_2, round_3):
- Each round contains:
  - Results for different LLM models (DeepSeek, GPT-4.1, GPT-5, Gemini, etc.)
  - Both "with context" and "no context" variants
  - `all_results.xlsx` - Aggregated results for each round
- Aggregate files:
  - `avg_all_metrics.xlsx` - Average metrics across all experiments
  - `avg_bleudiff_languageSpecific.xlsx` - Language-specific BLEU differences

### **config/**
Configuration management:
- `settings.py` - Application and database configuration:
  - `DatabaseConfig` - MongoDB connection settings
  - `AppConfig` - Application-level configuration
  - Supports both local and remote environment settings

### **models/**
Data model definitions:
- `node.py` - Node information data structure
- `edge.py` - Edge relationship definitions
- `nodeInfoForContextGen.py` - Context generation models:
  - `NodeCInfo` - Artifact process information with comments
  - `Comment` - Comment data structure
  - Helper functions for context serialization

### **utils/**
Utility functions and helper modules:

#### **Core Utilities**
- `data_util.py` - Data processing utilities:
  - `JSONLReader` - Read JSONL format files
  - `save_results()` - Save experiment results to JSON
  - `pr_quintuple_to_string()` - Convert PR context to string format
  - `save_to_pkl()` / `load_from_pkl()` - Pickle file operations

- `CONSTANT.py` - Global constants and API configurations:
  - LLM API keys and base URLs
  - `BASE_MODEL` dictionary with model configurations (DeepSeek, GPT, Gemini)

- `logger.py` - Logging configuration with file and console handlers

- `multi_processor.py` - Multi-processing support:
  - `process_single_row()` - Process individual data rows
  - `process_dataframe_multiprocess()` - Parallel DataFrame processing
  - `multi_processor_main()` - Main entry point for multi-process execution

- `general_LLMs_hooker.py` - LLM interaction wrapper:
  - `QAProcessor` - Process Q&A pairs with LLMs
  - Concurrent processing support
  - Result saving and queue management

#### **Subdirectories**
- `generalLLMs/` - Remote LLM client implementations:
  - `remote_server.py` - `DeepSeekClient` for code review API calls

---

## 🔄 Workflow and Data Flow

### **1. Main Execution Pipeline (_main_3_5.py)**
Orchestrates the complete experiment workflow for multiple models and contexts:

```
For each experiment round (1-3):
  For each model (ds, dsReasoner, gpt-4.1, gpt-5, gemini-2.5-pro, etc.):
    For each context setting (with/without context):
      1. Query LLM with/without context (_3_main)
      2. Calculate metrics (_4_main)
      3. Analyze results (_5_main)
```

### **2. LLM Query with Context (_3_query_WithContextInfo_DS_LLM.py)**
```
Input: 
  - Context pickle file (nodeList_size_3000.pkl)
  - Q&A dataset (case_study.xlsx)
  - Model name and context flag

Process:
  1. Load context information from pickle
  2. Match Q&A pairs with context
  3. Extract node information for selected nodes
  4. Query LLM with/without context using multi-processing
  
Output:
  - Pickle file with LLM responses (Results/processed_*.pkl)
```

### **3. Metrics Calculation (_4_get_metric_for_dsLLM_result.py)**
```
Input:
  - LLM result pickle file

Process:
  1. Load experiment results
  2. Calculate BLEU, ROUGE-1, ROUGE-2, ROUGE-L metrics
  3. Compute precision, recall, and perfect prediction count
  
Output:
  - Excel file with metrics (Results_Metrics/re_*.xlsx)
```

### **4. Context State Analysis (_5_analysis_for_reAndContextState.py)**
```
Input:
  - Metrics Excel file
  - Original context dataset

Process:
  1. Load experimental results and context info
  2. Calculate graph-based metrics (#N, #E, #CC, AvgCC, ClustC, AvgD, DegCE)
  3. Enrich results with context statistics
  
Output:
  - Enriched Excel file (Analysis/*.xlsx)
```

---

## 🚀 Usage Examples

### **Run Complete Experiment Pipeline**
```python
python _main_3_5.py
```
This will:
- Process all models defined in `CONSTANT.BASE_MODEL`
- Run experiments with and without context
- Execute 3 rounds of experiments
- Save results to `Results_Metrics/round_*/`

### **Query Single Model**
```python
from _3_query_WithContextInfo_DS_LLM import _3_main

result_path = _3_main(
    model_name="ds",
    withContext=True,
    context_pickle_path="Data/nodeList_size_3000.pkl",
    qa_path="Data/case_study.xlsx",
    index=0,
    num_processes=4
)
```

### **Calculate Metrics for Results**
```python
from _4_get_metric_for_dsLLM_result import _4_main

output_path, mean_bleu = _4_main(
    re_path="Results/processed_ds_with_Context_results_0.pkl"
)
print(f"Mean BLEU: {mean_bleu}")
print(f"Metrics saved to: {output_path}")
```

### **Analyze Context State**
```python
from _5_analysis_for_reAndContextState import _5_main

analysis_path = _5_main(
    filename="Results_Metrics/re_ds_with_Context_results.xlsx",
    original_dataset_path="Data/pre_experimental_original_context_dataset.pickle"
)
```

---

## 📊 Evaluation Metrics

### **BLEU (Bilingual Evaluation Understudy)**
Measures n-gram overlap between generated and reference text

### **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest common subsequence

### **Precision & Recall**
Standard information retrieval metrics

### **Perfect Prediction**
Count of exactly matching predictions

### **Graph-based Metrics**
- **#N**: Number of nodes
- **#E**: Number of edges
- **#CC**: Number of connected components
- **AvgCC**: Average clustering coefficient
- **ClustC**: Global clustering coefficient (transitivity)
- **AvgD**: Average degree
- **DegCE**: Degree centrality entropy

---

## 🔧 Configuration

### **Database Configuration (config/settings.py)**
```python
# Local environment
config = AppConfig.default()

# Remote environment
config = AppConfig.from_remote_env()
```

### **Model Configuration (utils/CONSTANT.py)**
Add new models to `BASE_MODEL` dictionary:
```python
BASE_MODEL = {
    "model_name": {
        "API_KEY": "your_api_key",
        "BASE_URL": "https://api.example.com",
        "MODEL_NAME": "model-version"
    }
}
```

---

## 📝 Notes

1. **Multi-processing**: The system supports parallel processing via `num_processes` parameter
2. **Result Persistence**: All intermediate results are saved as pickle files
3. **Experiment Rounds**: Results are organized by rounds (round_1, round_2, round_3)
4. **Context Variants**: Each experiment runs with both "with context" and "no context" settings
5. **Model Support**: Currently supports DeepSeek, GPT (various versions), and Gemini models

---

## 🔍 Key Data Structures

### **CodeReviewItem**
Main data structure for code review information:
- Basic identification (repo, repo_id, ghid)
- Code changes (old_hunk, new, old, hunk)
- Review information (comment, lang)
- Graph data and statistics

### **NodeCInfo**
Artifact process information including:
- Node information (artInfo)
- Comments list
- Status, milestone, labels
- Methods for comment management and string formatting

---

## 📈 Results Organization

Results are organized hierarchically:
```
Results_Metrics/
├── round_1/
│   ├── re_<model>_with_Context_results.xlsx
│   ├── re_<model>_No_Context_results.xlsx
│   └── all_results.xlsx
├── round_2/
│   └── ... (similar structure)
├── round_3/
│   └── ... (similar structure)
├── avg_all_metrics.xlsx
└── avg_bleudiff_languageSpecific.xlsx
```

Each result file contains:
- Repository and PR information
- Ground truth comments
- Model-generated suggestions
- Evaluation metrics (BLEU, ROUGE, precision, recall)
- Graph-based context metrics (when analyzed)
