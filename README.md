# QueryCraft

QueryCraft is a powerful Streamlit-based data exploration and visualization tool that leverages LLMs (Large Language Models) to help users generate, validate, and analyze SQL queries on SQLite and PostgreSQL databases. It provides interactive charts, advanced statistics, and data quality assessments, all in a modern, user-friendly interface.

## Features
- **LLM-powered SQL generation**: Generate and validate SQL queries using AI.
- **Supports SQLite and PostgreSQL**: Connect, explore, and query your databases.
- **Interactive data visualization**: Create bar charts, line charts, scatter plots, histograms, pie charts, and more.
- **Advanced analysis**: Perform statistical analysis, outlier detection, time series decomposition, and feature relationships.
- **Data quality assessment**: Check for missing values, duplicates, consistency issues, and anomalies.
- **Query history**: Save, search, and re-run previous queries.
- **Export results**: Download query results as CSV, Excel, or JSON.

## Getting Started

### Prerequisites
- Python 3.8+
- [Streamlit](https://streamlit.io/)
- [Plotly](https://plotly.com/python/)
- [Statsmodels](https://www.statsmodels.org/)
- [dotenv](https://pypi.org/project/python-dotenv/)
- Other dependencies listed in `requirements.txt`

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
1. Create a `.env` file in the project root and add your API keys and model deployment names for Azure OpenAI and Gemini (see example in `.env`).
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
