# Local Abstracts Analyzer with Ollama

This repository provides a Python-based pipeline to analyze scientific abstracts in bulk using a local Ollama Large Language Model (LLM). It replaces OpenAI API calls with Ollama’s local client, enabling faster, cost-effective, and privacy-focused processing of textual data.

---

## Advantages

- **Local Processing**: Executes entirely on your machine, eliminating network latency and dependency on external APIs.
- **Cost Savings**: No per-request charges; use your own hardware for unlimited queries.
- **Privacy & Security**: Data never leaves your environment, ensuring compliance with sensitive data requirements.
- **Scalability**: Harness multiple CPU cores via multiprocessing to process large datasets in parallel.
- **Flexibility**: Easily adjust model parameters (e.g., temperature), batch sizes, and retry logic.

---

## Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.com/) installed and configured locally
- Git
- (Optional) Virtual environment tool (venv, conda)

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/local-abstracts-analyzer.git
   cd local-abstracts-analyzer
   ```

2. **Set up a virtual environment** (recommended)

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\\Scripts\\activate  # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Ollama**

   Ensure Ollama is running and your desired model (e.g., `gemma3:27b`) is available:

   ```bash
   ollama pull gemma3:27b
   ```

---

## Usage

1. **Prepare your dataset**

   Create a CSV file (e.g., `abstracts.csv`) with at least one column named `abstract` containing the text to analyze.

2. **Adjust settings**

   In `analyze_abstracts_local_ollama.py`, modify the configuration constants at the top:

   - `MODEL`: Ollama model identifier (e.g., `gemma3:27b`)
   - `TEMPERATURE`: Float value for generation randomness
   - `BATCH_SIZE`: Number of abstracts per request
   - `WORKERS`: Number of parallel processes
   - `QUERY`: Analysis question for the LLM

3. **Run the script**

   ```bash
   python analyze_abstracts_local_ollama.py
   ```

   - Outputs incremental results to `resultados_parciais.csv`.
   - Logs execution details in `log_execucao.txt`.

4. **Resume processing**

   If interrupted, rerun the script; it will resume from the last saved index.

---

## Customization

- **Prompt Tuning**: Edit the `QUERY` constant to change the analysis question.
- **Model Options**: Explore different Ollama model parameters via the `options` dict in `call_local_llm()`.
- **Error Handling**: Adjust retry limits and backoff strategy in `process_batch_with_retry()`.

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## Contact

For questions or support, contact Jefferson Duarte at <jefferson.duarte@ifrn.edu.br>.

## License

MIT License. See [LICENSE](LICENSE) for details.

