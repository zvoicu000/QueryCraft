# QueryCraft

QueryCraft is a powerful Streamlit-based data exploration and visualization tool that leverages LLMs (Large Language Models) including Azure OpenAI and Google Gemini to help users generate, validate, and analyze SQL queries on SQLite and PostgreSQL databases. It provides interactive charts, advanced statistics, and data quality assessments, all in a modern, user-friendly interface.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Azure OpenAI](https://img.shields.io/badge/Azure%20OpenAI-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)

## Features
- **LLM-powered SQL generation**: Generate and validate SQL queries using AI (Azure OpenAI & Gemini integration).
- **Supports SQLite and PostgreSQL**: Connect, explore, and query your databases.
- **Interactive data visualization**: Create bar charts, line charts, scatter plots, histograms, pie charts, and more.
- **Advanced analysis**: Perform statistical analysis, outlier detection, time series decomposition, and feature relationships.
- **Data quality assessment**: Check for missing values, duplicates, consistency issues, and anomalies.
- **Query history**: Save, search, and re-run previous queries.
- **Export results**: Download query results as CSV, Excel, or JSON.


### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/zvoicu000/QueryCraft.git
   cd QueryCraft
   ```
2. (Recommended) Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration
1. Create a `.env` file in the project root and add your API keys and model deployment names for Azure OpenAI and Gemini. Example:

   ```env
   # Azure OpenAI Configuration
   LLM_PROVIDER=AZURE
   OPENAI_ENDPOINT="https://<your-endpoint>.openai.azure.com"
   OPENAI_API_VERSION="2024-08-01-preview"
   OPENAI_API_KEY="<your-azure-openai-api-key>"
   MODEL_NAME="<your-azure-openai-deployment-name>"

   # Gemini Configuration
   LLM_PROVIDER=GEMINI
   GEMINI_API_KEY="<your-gemini-api-key>"
   ```
2. Launch the app:
   ```bash
   streamlit run app/QueryCraft.py
   ```

## Usage
- Upload a SQLite database or connect to a PostgreSQL database.
- Select tables and enter your query or let the LLM generate one for you.
- Explore results, visualize data, and export findings.

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## Contact
For questions or support, open an issue or contact [zvoicu000](https://github.com/zvoicu000).
