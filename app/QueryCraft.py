import sys 
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))
import io 
import json 
import re 
import logging
from typing import Dict, List, Optional, Union, TypedDict
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.seasonal import seasonal_decompose
import streamlit as st
from dotenv import load_dotenv
from streamlit_extras.colored_header import colored_header
import streamlit_nested_layout
import numpy as np
from streamlit_extras.dataframe_explorer import dataframe_explorer
import src.database.DB_Config as DB_Config
from src.prompts.Base_Prompt import SYSTEM_MESSAGE
from src.api.LLM_Config import get_completion_from_messages
import hashlib
from datetime import datetime
from time import time
from collections import defaultdict
from jsonschema import validate as json_validate, ValidationError

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

SUPPORTED_CHART_TYPES = {
    "Bar Chart": "A chart that presents categorical data with rectangular bars.",
    "Line Chart": "A chart that displays information as a series of data points called 'markers' connected by straight line segments.",
    "Scatter Plot": "A plot that displays values for typically two variables for a set of data.",
    "Area Chart": "A chart that displays quantitative data visually, using the area below the line.",
    "Histogram": "A graphical representation of the distribution of numerical data.",
    "Pie Chart": "A chart that shows proportions of a whole using slices.",
    "Box Plot": "A chart that shows the distribution of data based on quartiles."
}

st.set_page_config(
    page_icon='📁',
    page_title="QueryCraft",
    layout="wide"
)

def apply_custom_theme():
    custom_css=f"""
    <style>
    /* Global Styles */
    body, .stApp {{
        background-color: #1e1e1e;
        color: #64ffda;
        font-family: sans-serif;
    }}
    /* Sidebar */
    .css-1d391kg, .stSidebar .sidebar-content {{
        background-color: #333333;
    }}
    /* Buttons */
    .stButton>button {{
        background-color: #00ADB5;
        color: #fff;
        border: none;
    }}
    /* Expander */
    .stExpander {{
        background-color: #333333;
        border: none;
        border-radius: 8px;
        padding: 0.5rem;
    }}
    .stExpander .stExanderHeader, .stExpander .stExanderContent {{
        color: #64ffda;
    }}
    /* Tabs */
    .stTabs [data-baseweb="tab"] {{
        background-color: #333333;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        color: #64ffda;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background-color: #00ADB5;
        color: #fff;
    }}
    /* Code Blocks */
    pre {{
        background-color: #333333;
        color: #64ffda;
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

apply_custom_theme()

load_dotenv()

@st.cache_resource
def load_system_message(schemas: dict) -> str:
    """Load and format the system message with JSON-serialized schemas."""
    return SYSTEM_MESSAGE.format(schemas=json.dumps(schemas, indent=2))

def validate_sql_query(query: str) -> bool:
    """
    Ensure the SQL query is valid and safe (select queries only).

    Parameters:
    - query (str): The SQL query to validate.

    Returns:
    - bool: True if the query is valid and safe, False otherwise.
    """
    if not isinstance(query,str):
        return False
    
    disallowed_keywords = r'\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|EXEC)\b'

    if re.search(disallowed_keywords, query, re.IGNORECASE):
        return False
    
    if not query.strip().lower().startswith(('select','with')):
        return False
    
    if query.count('(')!= query.count(')'):
        return False
    
    return True

def validate_query_tables(query: str, schemas: dict) -> bool:
    """
    Very basic check: warn if any known schema table name is missing in the query.
    This is a heuristic check.
    """
    lower_query=query.lower()
    missing=[]
    for table in schemas.keys():
        if table.lower() not in lower_query:
            missing.append(table)
    if missing:
        logging.warning(f"LLM query does not mention these tables from the schema: {', '.join(missing)}")
        return False
    return True

def get_data(query: str, db_name: str, db_type: str, host: Optional[str]=None, user:Optional[str]= None, password: Optional[str]=None) -> pd.DataFrame:
    """Run the specified query and return the complete resulting DataFrame."""
    if not validate_sql_query(query):
        logger.error("Invalid or unsafe SQL query.")
        return pd.DataFrame()
    return DB_Config.query_database(query,db_name,db_type,host,user,password)

def save_temp_file(uploaded_file) -> str:
    """Saves an uploaded file to a temporary location."""
    temp_file_path="temp_database.db"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())
    return temp_file_path

class Path(TypedDict):
    description: str 
    tables: List[str]
    columns: List[List[str]]
    score: int 

class TableColumn(TypedDict):
    table: str 
    columns: List[str]
    reason:str 

class DecisiondLog(TypedDict):
    query_input_details: List[str]
    preprocessing_steps: List[str]
    path_identification: List[Path]
    ambiguity_detection: List[str]
    resolution_criteria: List[str]
    chosen_path_explanation: List[TableColumn]
    generated_sql_query: str
    alternative_paths: List[str]
    execution_feedback: List[str]
    final_summary: str
    visualization_suggestion: Optional[str]

DECISION_LOG_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The generated SQL query"},
        "error": {"type": ["string", "null"], "description": "Error message if query generation failed"},
        "decision_log": {
            "type": "object",
            "properties": {
                "query_input_details": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Details about the input query"
                },
                "preprocessing_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Steps taken to preprocess the query"
                },
                "path_identification": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "tables": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "columns": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            },
                            "score": {"type": "integer"}
                        },
                        "required": ["description", "tables", "columns", "score"]
                    }
                },
                "ambiguity_detection": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "resolution_criteria": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "chosen_path_explanation": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "string"},
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "reason": {"type": "string"}
                        },
                        "required": ["table", "columns", "reason"]
                    }
                },
                "generated_sql_query": {"type": "string"},
                "alternative_paths": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "execution_feedback": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "final_summary": {"type": "string"},
                "visualization_suggestion": {"type": ["string", "null"]}
            },
            "required": [
                "query_input_details",
                "preprocessing_steps",
                "path_identification",
                "ambiguity_detection",
                "resolution_criteria",
                "chosen_path_explanation",
                "generated_sql_query",
                "alternative_paths",
                "execution_feedback",
                "final_summary"
            ]
        }
    },
    "required": ["query", "decision_log"]
}

def generate_sql_query(user_message:str, schemas:dict, max_attempts:int=1) -> dict:
    """Generate a SQL query using LLM responses and validate output structure."""
    formatted_system_message=f"""
    {load_system_message(schemas)}
    Important: Your response must be valid JSON matching this schema:
    {json.dumps(DECISION_LOG_SCHEMA,indent=2)}

    Ensure all reponses strictly follow this format. Include a final_summary and visualization_suggestion in the decision_log.
    """

    for attempt in range(max_attempts):
        try:
            response=get_completion_from_messages(formatted_system_message,user_message)

            response = re.sub(r'^```json\s*', '', response.strip())
            response = re.sub(r'```$', '', response.strip())

            json_response=json.loads(response)

            try:
                json_validate(instance=json_response, schema=DECISION_LOG_SCHEMA)
            except ValidationError as ve:
                logger.warning(f"JSON schema validation error: {ve.message}. Attempt: {attempt + 1}")
                continue

            if not validate_query_tables(json_response.get('query', ''), schemas):
                logger.warning("Generated SQL query contains non-existent tables/columns.")

            return {
                "query": json_response.get('query'),
                "error": json_response.get('error'),
                "decision_log": json_response['decision_log'],
                "visualization_recommendation":json_response['decision_log'].get('visualization_suggestion')   
            }
        
        except json.JSONDecodeError as e:
            logger.exception(f"Invalid JSON response: {response}, Error:{e}")
            continue
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            continue

    return {
        "error":"Failed to generate a valid SQL query after multiple attempts.",
        "decision_log":{
            "execution_feedback":["Failed to generate a valid response after multiple attempts."],
            "final_summary":"Query generation failed."
        }
    }
            
def validate_response_structure(response: dict) -> bool:
    """Check if the LLM response follows the expected JSON schema."""
    try:
        if not all(key in response for key in ["query", "decision_log"]):
            return False
        
        decision_log=response["decision_log"]
        required_sections=[
            "query_input_details",
            "preprocessing_steps",
            "path_identification",
            "ambiguity_detection",
            "resolution_criteria",
            "chosen_path_explanation",
            "generated_sql_query",
            "alternative_paths",
            "execution_feedback",
            "final_summary"
        ]

        if not all(key in decision_log for key in required_sections):
            return False
        
        for path in decision_log["path_identification"]:
            if not all(key in path for key in ["description","tables","columns","score"]):
                return False
        
        for explanation in decision_log["chosen_path_explanation"]:
            if not all(key in explanation for key in ["table","columns","reason"]):
                return False
            
        return True
    
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False
    
def build_markdown_decision_log(decision_log: Dict) -> str:
    """Convert the decision log into a markdown-formatted string."""
    markdown_log=[]

    if query_details := decision_log.get("query_input_details"):
        markdown_log.extend([
            "### Query Input Analysis",
            "\n".join(f"- {detail}" for detail in query_details),
            ""
        ])
    
    if preprocessing := decision_log.get("preprocessing_steps"):
        markdown_log.extend([
            "### Preprocessing Steps",
            "\n".join(f"- {step}" for step in preprocessing),
            ""
        ])

    if paths := decision_log.get("path_identification"):
        markdown_log.extend([
            "### Path Identification",
            "\n".join([
                f"**Path {i+1}** (Score: {path['score']})\n"
                f"- Description: {path['description']}\n"
                f"- Tables: {', '.join(path['tables'])}\n"
                f"- Columns: {', '.join([', '.join(cols) for cols in path['columns']])}"
                for i, path in enumerate(paths)
            ]),
            ""
        ])
    
    if ambiguities := decision_log.get("ambiguity_detection"):
        markdown_log.extend([
            "### Ambiguity Analysis",
            "\n".join(f"- {ambiguity}" for ambiguity in ambiguities),
            ""
        ])
    
    if criteria := decision_log.get("resolution_criteria"):
        markdown_log.extend([
            "### Resolution Criteria",
            "\n".join(f"- {criterion}" for criterion in criteria),
            ""
        ])

    if chosen_path := decision_log.get("chosen_path_explanation"):
        markdown_log.extend([
            "### Selected Tables and Columns",
            "\n".join([
                f"**{table['table']}**\n"
                f"- Columns: {', '.join(table['columns'])}\n"
                f"- Reason: {table['reason']}"
                for table in chosen_path
            ]),
            ""
        ])

    if sql_query := decision_log.get("generated_sql_query"):
        markdown_log.extend([
            "### Generated SQL Query",
            f"```sql\n{sql_query}\n```",
            ""
        ])
    
    if alternatives := decision_log.get("alternative_paths"):
        markdown_log.extend([
            "### Alternative Approaches",
            "\n".join(f"- {alt}" for alt in alternatives),
            ""
        ])
    
    if feedback := decision_log.get("execution_feedback"):
        markdown_log.extend([
            "### Execution Feedback",
            "\n".join(f"- {item}" for item in feedback),
            ""
        ])

    if summary := decision_log.get("final_summary"):
        markdown_log.extend([
            "### Summary",
            summary,
            ""
        ])
    
    if viz_suggestion := decision_log.get("visualization_suggestion"):
        markdown_log.extend([
            "### Visualization Recommendation"
            f"Suggested visualization type:{repr(viz_suggestion)}",
            ""
        ])
    
    return "\n".join(line.rstrip() for line in markdown_log)

def create_chart(df: pd.DataFrame, chart_type: str, x_col: str, y_col: str) -> Optional[any]:
    """Construct a Plotly chart without color column and trendline options."""
    try:
        if chart_type == "Bar Chart":
            fig = px.bar(df, x=x_col, y=y_col, title=f"Bar Chart of {y_col} by {x_col}")
        elif chart_type == "Line Chart":
            fig = px.line(df, x=x_col, y=y_col, title=f"Line Chart of {y_col} by {x_col}", markers=True)
        elif chart_type == "Scatter Plot":
            fig = px.scatter(df, x=x_col, y=y_col, title=f"Scatter Plot of {y_col} vs {x_col}", hover_data=[x_col, y_col])
        elif chart_type == "Area Chart":
            fig = px.area(df, x=x_col, y=y_col, title=f"Area Chart of {y_col} by {x_col}")
        elif chart_type == "Histogram":
            fig = px.histogram(df, x=x_col, nbins=30, title=f"Histogram of {x_col}")
        elif chart_type == "Pie Chart":
            fig = px.pie(df, names=x_col, values=y_col, title=f"Pie Chart of {x_col}")
        elif chart_type == "Box Plot":
            fig = px.box(df, x=x_col, y=y_col, title=f"Box Plot of {y_col} grouped by {x_col}")
        else:
            st.warning("Chart type not recognized.")
            return None

        fig.update_layout(autosize=True, margin=dict(l=20, r=20, t=40, b=20))
        return fig

    except Exception as e:
        st.error(f"Error generating the chart: {e}")
        logger.error(f"Error generating chart: {e}")
        return None
    
def display_summary_statistics(df: pd.DataFrame) -> None:
    """Show optimized summary statistics, filtering out unnecessary metrics."""

    if df.empty:
        st.warning("The DataFrame is empty. Unable to display summary statistics.")
        return
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object','category']).columns
    datetime_cols=df.select_dtypes(include=['datetime']).columns

    tab1, tab2, tab3 = st.tabs(["Numeric Summary","Categorical Analysis","Missing Data & Correlations"])

    with tab1:
        st.markdown("### Numeric Summary Statistics")
        filtered_stats = df[numeric_cols].describe().T

        filtered_stats=filtered_stats.drop(columns=["count"], errors="ignore")

        filtered_stats["median"]=df[numeric_cols].median()
        filtered_stats["iqr"]=filtered_stats["75%"] - filtered_stats["25%"]
        filtered_stats["std"]=df[numeric_cols].std()

        filtered_stats=filtered_stats.loc[filtered_stats["std"]>0]

        st.dataframe(filtered_stats.style.format("{:.2f}").highlight_max(axis=0,color="lightgreen"))
        
        for col in numeric_cols:
            if df[col].nunique() >1:
                st.markdown(f"**Distribution of {col}**")
                st.plotly_chart(px.histogram(df, x=col, nbins=30, title=f"Histogram of {col}"),use_container_width=True)

    with tab2:
        st.markdown("### Categorical Data Insights")
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            unique_count=value_counts.shape[0]

            if unique_count < len(df) *0.8:
                st.markdown(f"**{col}:** {unique_count} unique values")
                freq_table=value_counts.reset_index()
                freq_table.columns=["Category", "Count"]
                freq_table["Percentage"] = (freq_table["Count"]/len(df)*100).round(2)
                st.table(freq_table.style.format({"Percentage": "{:.2f}%"}))
            
            if unique_count <= 10:
                st.plotly_chart(px.pie(freq_table, names="Category", values="Count", title=f"Pie Chart for {col}"), use_container_width=True)
            else:
                st.plotly_chart(px.bar(freq_table, x="Category", y="Count", title=f"Bar Chart for {col}"), use_container_width=True)
    
    with tab3:
        st.markdown("### Missing Data Analysis")

        missing_data=df.isnull().sum()
        missing_data=missing_data[missing_data > 0]
        if not missing_data.empty:
            missing_df=missing_data.reset_index()
            missing_df.columns=["Column","Missing Values"]
            missing_df["Percentage"]=(missing_df["Missing Values"]/len(df)*100).round(2)
            st.table(missing_df.style.format({"Percentage": "{:.2f}%"}))
        else:
            st.success("No missing data detected.")

        st.markdown("### Correlation Matrix")
        if len(numeric_cols) >= 2:
            correlation_matrix = df[numeric_cols].corr()
            heat_fig=px.imshow(correlation_matrix,text_auto=True,aspect="auto",title="Correlation Matrix")
            st.plotly_chart(heat_fig, use_container_width=True)
        else:
            st.info("Not enough numeric solumns for correlation analysis.")

def perform_advanced_analysis(df: pd.DataFrame) -> None:
    """Perform advanced statistical analysis on the dataset."""
    st.markdown("## 📊 Advanced Statistical Analysis")

    tabs= st.tabs(["Distribution Analysis", "Outlier Detection", "Time Series Analysis", "Feature Relationships"])

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns

    with tabs[0]:
        st.markdown("### 📈 Distribution Analysis")
        if len(numeric_cols) > 0:
            col= st.selectbox("Select column for distribution analysis", numeric_cols)

            skewness = stats.skew(df[col].dropna())
            kurtosis = stats.kurtosis(df[col].dropna())

            fig = ff.create_distplot([df[col].dropna()],[col],bin_size=0.2)
            st.plotly_chart(fig, use_container_width=True)

            col1,col2,col3=st.columns(3)
            col1.metric("Skewness", f"{skewness:.2f}")
            col2.metric("Kurtosis", f"{kurtosis:.2f}")
            col3.metric("Normality Test p-value",f"{stats.normaltest(df[col].dropna())[1]:.4f}")
    
    with tabs[1]:
        st.markdown("### 🔍 Outlier Detection")
        if len(numeric_cols) >0:
            col=st.selectbox("Select column for outlier detection", numeric_cols, key="outlier_col")

            Q1 = df[col].quantile(0.25)
            Q3=df[col].quantile(0.75)
            IQR = Q3-Q1
            outliers = df[(df[col]<(Q1 - 1.5 * IQR)) | (df[col]>(Q3 + 1.5 * IQR))][col]

            fig=go.Figure()
            fig.add_trace(go.Box(y=df[col], name=col))
            st.plotly_chart(fig, use_container_width=True)

            if not outliers.empty:
                st.markdown(f"**Found {len(outliers)} outliers:**")
                st.dataframe(outliers)
    
    with tabs[2]:
        st.markdown("### ⏳ Time Series Analysis")
        if len(datetime_cols) >0:
            date_col = st.selectbox("Select date column", datetime_cols)
            value_col = st.selectbox("Select value column", numeric_cols)

            ts_data = df[[date_col, value_col]].sort_values(date_col)
            ts_data=ts_data.set_index(date_col)

            period = st.number_input("Enter the period for seasonal decomposition (default is 12)", min_value=1, value=12)

            try:
                decomposition = seasonal_decompose(ts_data[value_col], period=period)

                fig=go.Figure()
                fig.add_trace(go.Scatter(x=ts_data.index, y=decomposition.trend,name='Trend'))
                fig.add_trace(go.Scatter(x=ts_data.index,y=decomposition.seasonal,name='Seasonal'))
                fig.add_trace(go.Scatter(x=ts_data.index,y=decomposition.resid,name='Residual'))
                fig.update_layout(title='Time Series Decomposition')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning("Could not perform seasonal decomposition. Ensure enough data points and regular intervals.")
    
    with tabs[3]:
        st.markdown("### 🔗 Feature Relationships")
        if len(numeric_cols) >=2:
            correlation = df[numeric_cols].corr()

            fig = px.imshow(correlation,
                            labels=dict(color="Correlation"),
                            title="Feature Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)

            if st.checkbox("Show Variance Inflation Factor (VIF) Analysis"):
                if len(numeric_cols) < 2:
                    st.warning("At least two numeric columns are required to calculate VIF.")
                else:
                    try:
                        X=df[numeric_cols].dropna()
                        vif_data=pd.DataFrame()
                        vif_data["Feature"]=numeric_cols
                        vif_data["VIF"]=[variance_inflation_factor(X.values, i)
                                         for i in range(X.shape[1])]
                        st.dataframe(vif_data.sort_values('VIF', ascending=False))
                    except Exception as e:
                        st.warning("Could not calculate VIF. Check for multicollinearity or missing values.")

def assess_data_quality(df: pd.DataFrame) -> None:
    """Assess the quality of the dataset and provide detailed insights."""
    st.markdown("## 🔍 Data Quality Assessment")

    tabs= st.tabs(["Overview", "Missing Values", "Duplicates", "Consistency", "Anomalies"])

    with tabs[0]:
        st.markdown("### 📊 Data Quality Overview")

        total_rows = len(df)
        total_cols = len(df.columns)
        memory_usage=df.memory_usage(deep=True).sum() /1024**2

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", f"{total_rows:,}")
        col2.metric("Total Columns", total_cols)
        col3.metric("Memory Usage", f"{memory_usage:.2f} MB")
        col4.metric("Data Types", len(df.dtypes.unique()))

        dtype_counts=df.dtypes.value_counts()
        fig = px.pie(values=dtype_counts.values,
                     names=dtype_counts.index.astype(str),
                     title="Column Data Type Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.markdown("### ❌ Missing Values Analysis")

        missing = df.isnull().sum()
        missing__pct=(missing/len(df)*100).round(2)
        missing_df=pd.DataFrame({
            'Column': missing.index,
            'Missing Count':missing.values,
            'Missing Percentage':missing__pct.values
        }).sort_values("Missing Percentage", ascending=False)

        if missing_df['Missing Count'].sum()>0:
            st.dataframe(missing_df)

            fig=px.bar(missing_df,
                       x='Column',
                       y='Missing Percentage',
                       title="Missing Values by Column")
            st.plotly_chart(fig, use_container_width=True)

            if st.checkbox("Show Missing Value Patterns"):
                fig=go.Figure()
                for col in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index[df[col].isnull()],
                        y=[col] * df[col].isnull().sum(),
                        mode='markers',
                        name=col 
                    ))
                fig.update_layout(title="Missing Value Patterns Across Records")
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.success("No missing values found in the dataset!")
    
    with tabs[2]:
        st.markdown("### 🔄 Duplicate Analysis")

        duplicates = df.duplicated()
        duplicate_count = duplicates.sum()

        if duplicate_count>0:
            st.warning(f"Found {duplicate_count} duplicate rows ({(duplicate_count/len(df)*100):.2f}% of data)")

            if st.checkbox("Show Duplicate Rows"):
                st.dataframe(df[duplicates])

            col_duplicates={col:df[col].duplicated().sum() for col in df.columns}
            col_dup_df = pd.DataFrame({
                'Column' : col_duplicates.keys(),
                'Duplicate Count': col_duplicates.values(),
                'Duplicate Percentage': [v/len(df)*100 for v in col_duplicates.values()]
            }).sort_values('Duplicate Count', ascending=False)

            st.markdown("### Column-wise Duplicates")
            st.dataframe(col_dup_df)
        else:
            st.success("No duplicate rows found in the dataset!")
                       