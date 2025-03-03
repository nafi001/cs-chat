import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import google.generativeai as genai
from typing import List, Dict, Any, Tuple

# Page configuration
st.set_page_config(page_title="CSV Data Analyst Chatbot By Nafi", layout="wide")

def initialize_gemini_api():
    """Initialize the Gemini API safely"""
    try:
        # Check if API key is in Streamlit secrets
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
        else:
            st.warning("⚠️ Gemini API Key not found in secrets!")
            
            # Ask for API key input
            api_key = st.text_input(
                "Enter your Gemini API Key (free to create at https://aistudio.google.com/)",
                type="password"
            )
            
            if not api_key:
                st.info("You need a Google AI Studio account to get a free Gemini API key.")
                st.info("1. Visit https://aistudio.google.com/")
                st.info("2. Create a free account")
                st.info("3. Get your API key from the settings")
                return None
        
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Set the default model to Gemini 2.0 Flash
        model_name = "models/gemini-2.0-flash"
        
        # Save key to session state for current session only
        st.session_state.temp_api_key = api_key
        st.session_state.selected_model = model_name
        
        st.success(f"Using model: Gemini 2.0 Flash")
        
        # Create the model instance
        return genai.GenerativeModel(model_name)
            
    except Exception as e:
        st.error(f"Error initializing Gemini API: {str(e)}")
        return None

def load_csv_file(file):
    """Load a CSV file into a pandas DataFrame"""
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None

def get_dataframe_info(df):
    """Extract basic information about the dataframe"""
    buffer = io.StringIO()
    
    # Basic info
    buffer.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n\n")
    
    # Column info
    buffer.write("Column Information:\n")
    column_info = []
    for col in df.columns:
        dtype = df[col].dtype
        n_unique = df[col].nunique()
        n_missing = df[col].isna().sum()
        sample_values = df[col].dropna().sample(min(3, n_unique)).tolist() if n_unique > 0 else []
        
        column_info.append({
            "column": col,
            "dtype": str(dtype),
            "unique_values": n_unique,
            "missing_values": n_missing,
            "sample_values": sample_values
        })
    
    for info in column_info:
        buffer.write(f"- {info['column']} (Type: {info['dtype']})\n")
        buffer.write(f"  * Unique values: {info['unique_values']}\n")
        buffer.write(f"  * Missing values: {info['missing_values']} ({info['missing_values']/df.shape[0]:.1%})\n")
        buffer.write(f"  * Sample values: {info['sample_values']}\n")
    
    # Statistical summary for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        buffer.write("\nNumeric Column Statistics:\n")
        stats_df = df[numeric_cols].describe().transpose()
        buffer.write(stats_df.to_string())
    
    return buffer.getvalue()

def execute_data_analysis(df, analysis_type):
    """Execute a specific type of data analysis on the dataframe"""
    if analysis_type == "summary_stats":
        return df.describe().to_string()
    
    elif analysis_type == "missing_values":
        missing = df.isna().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Missing Values': missing,
            'Percentage': missing_pct
        })
        return missing_df.to_string()
    
    elif analysis_type == "correlation":
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return "Not enough numeric columns for correlation analysis."
        
        correlation = numeric_df.corr().round(2)
        return correlation.to_string()
    
    elif analysis_type == "value_counts":
        results = []
        for col in df.columns:
            if df[col].nunique() <= 20:  # Only for columns with limited unique values
                results.append(f"Value counts for {col}:\n{df[col].value_counts().to_string()}\n")
        
        return "\n".join(results) if results else "No suitable columns for value counts analysis."
    
    return "Unsupported analysis type."

def generate_data_visualization(df, question, chart_type=None):
    """Generate a visualization based on the question and dataframe"""
    # Create a figure and save to buffer
    fig = plt.figure(figsize=(10, 6))
    
    try:
        if "histogram" in question.lower() or chart_type == "histogram":
            # Extract column name from question
            cols = [col for col in df.columns if col.lower() in question.lower()]
            if cols:
                col = cols[0]
                if pd.api.types.is_numeric_dtype(df[col]):
                    sns.histplot(data=df, x=col, kde=True)
                    plt.title(f"Histogram of {col}")
                    plt.tight_layout()
                else:
                    return None, "The selected column is not numeric."
            else:
                return None, "No valid column identified for histogram."
                
        elif "scatter" in question.lower() or chart_type == "scatter":
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) >= 2:
                x_col = num_cols[0]
                y_col = num_cols[1]
                
                # Try to extract columns from question
                for col in df.columns:
                    if col.lower() in question.lower() and col in num_cols:
                        if 'vs' in question.lower():
                            parts = question.lower().split('vs')
                            for part in parts:
                                for c in df.columns:
                                    if c.lower() in part and c in num_cols:
                                        if c.lower() in parts[0]:
                                            x_col = c
                                        else:
                                            y_col = c
                        else:
                            y_col = col
                
                sns.scatterplot(data=df, x=x_col, y=y_col)
                plt.title(f"{y_col} vs {x_col}")
                plt.tight_layout()
            else:
                return None, "Not enough numeric columns for scatter plot."
                
        elif "bar" in question.lower() or chart_type == "bar":
            cols = [col for col in df.columns if col.lower() in question.lower()]
            if cols:
                col = cols[0]
                # Use value_counts() to get frequencies
                if df[col].nunique() <= 20:  # Limit to columns with reasonable number of categories
                    df[col].value_counts().sort_values().plot(kind='barh')
                    plt.title(f"Count of {col}")
                    plt.tight_layout()
                else:
                    return None, f"Too many unique values in {col} for a bar chart."
            else:
                # Default to first categorical column
                cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                if cat_cols and df[cat_cols[0]].nunique() <= 20:
                    df[cat_cols[0]].value_counts().sort_values().plot(kind='barh')
                    plt.title(f"Count of {cat_cols[0]}")
                    plt.tight_layout()
                else:
                    return None, "No suitable categorical column found for bar chart."
                    
        elif "correlation" in question.lower() or chart_type == "heatmap":
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] < 2:
                return None, "Not enough numeric columns for correlation heatmap."
            
            corr = numeric_df.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
            plt.title("Correlation Matrix")
            plt.tight_layout()
            
        elif "boxplot" in question.lower() or chart_type == "boxplot":
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                # Try to extract column from question
                col = next((col for col in num_cols if col.lower() in question.lower()), num_cols[0])
                sns.boxplot(y=col, data=df)
                plt.title(f"Boxplot of {col}")
                plt.tight_layout()
            else:
                return None, "No numeric columns available for boxplot."
                
        elif "time series" in question.lower() or chart_type == "line":
            # Try to identify date column
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if date_cols and num_cols:
                date_col = date_cols[0]
                # Try to parse as datetime
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                    df = df.sort_values(by=date_col)
                    
                    # Find numeric column for Y axis
                    y_col = next((col for col in num_cols if col.lower() in question.lower()), num_cols[0])
                    
                    plt.plot(df[date_col], df[y_col])
                    plt.title(f"{y_col} Over Time")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                except:
                    return None, "Could not parse date column for time series."
            else:
                return None, "No suitable date and numeric columns found for time series."
        else:
            # Default to a summary of numeric columns
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                df[num_cols[:5]].hist(figsize=(10, 6), bins=20, layout=(2, 3))
                plt.suptitle("Histograms of Numeric Columns")
                plt.tight_layout()
            else:
                return None, "No numeric columns available for default visualization."
        
        # Save plot to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close(fig)
        
        # Encode plot for HTML
        img_str = base64.b64encode(buffer.read()).decode()
        img_html = f'<img src="data:image/png;base64,{img_str}" style="max-width: 100%; height: auto;">'
        
        return img_html, None
        
    except Exception as e:
        plt.close(fig)
        return None, f"Error generating visualization: {str(e)}"

def prepare_df_context(df):
    """Prepare a comprehensive context about the dataframe for the LLM"""
    context = ""
    
    # Basic info
    context += f"DataFrame Info: {df.shape[0]} rows and {df.shape[1]} columns\n\n"
    
    # Column info with data types and sample values
    context += "Columns:\n"
    for col in df.columns:
        dtype = df[col].dtype
        n_unique = df[col].nunique()
        n_missing = df[col].isna().sum()
        
        sample_values = ", ".join(map(str, df[col].dropna().sample(min(3, n_unique)).tolist())) if n_unique > 0 else ""
        
        context += f"- {col} (Type: {dtype})\n"
        context += f"  * Unique values: {n_unique}, Missing values: {n_missing} ({n_missing/df.shape[0]:.1%})\n"
        context += f"  * Sample values: {sample_values}\n"
    
    # Statistical summary for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        context += "\nNumeric Column Statistics:\n"
        desc = df[numeric_cols].describe().to_string()
        context += desc + "\n\n"
    
    # Correlation for numeric columns (if available)
    if len(numeric_cols) >= 2:
        context += "Correlation between numeric columns:\n"
        corr = df[numeric_cols].corr().round(2).to_string()
        context += corr + "\n\n"
    
    # Sample data
    context += "Sample data (first 5 rows):\n"
    context += df.head().to_string() + "\n"
    
    return context

def detect_analysis_needs(question):
    """Detect what kind of analysis is needed based on the question"""
    question_lower = question.lower()
    
    analysis_types = {
        "visualization": ["plot", "graph", "chart", "visual", "histogram", "bar chart", "scatter", "boxplot", "show me", "visualize"],
        "summary": ["summary", "describe", "statistics", "stats", "overview", "profile"],
        "missing": ["missing", "null", "na ", "empty"],
        "correlation": ["correlation", "relationship", "related", "correlate"],
        "groupby": ["group by", "grouped", "aggregate", "aggregation", "average by", "mean by"],
        "filter": ["filter", "where", "select", "find rows", "search for"],
        "sort": ["sort", "order", "rank", "top", "bottom", "highest", "lowest"],
        "unique": ["unique", "distinct", "different", "categories"]
    }
    
    detected_types = []
    for analysis_type, keywords in analysis_types.items():
        if any(keyword in question_lower for keyword in keywords):
            detected_types.append(analysis_type)
    
    chart_types = {
        "histogram": ["histogram", "distribution"],
        "bar": ["bar chart", "bar graph", "frequency"],
        "scatter": ["scatter", "correlation plot", " vs ", "versus"],
        "boxplot": ["box plot", "boxplot", "box and whisker"],
        "line": ["line chart", "line graph", "trend", "over time"],
        "heatmap": ["heatmap", "heat map", "correlation matrix"]
    }
    
    chart_type = None
    for ctype, keywords in chart_types.items():
        if any(keyword in question_lower for keyword in keywords):
            chart_type = ctype
            break
    
    return detected_types, chart_type

def execute_query(df, question):
    """Execute a data analysis query based on natural language question"""
    analysis_types, chart_type = detect_analysis_needs(question)
    
    result = ""
    visualization = None
    error = None
    
    # If we need a visualization
    if "visualization" in analysis_types or chart_type:
        visualization, error = generate_data_visualization(df, question, chart_type)
        if error:
            result += f"Visualization Error: {error}\n\n"
    
    # Handle other analysis types
    if "summary" in analysis_types:
        result += "Summary Statistics:\n"
        result += df.describe().to_string() + "\n\n"
    
    if "missing" in analysis_types:
        result += "Missing Value Analysis:\n"
        missing = df.isna().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Missing Values': missing,
            'Percentage': missing_pct
        })
        result += missing_df.to_string() + "\n\n"
    
    if "correlation" in analysis_types:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] >= 2:
            result += "Correlation Analysis:\n"
            result += numeric_df.corr().round(2).to_string() + "\n\n"
        else:
            result += "Not enough numeric columns for correlation analysis.\n\n"
    
    if "unique" in analysis_types:
        # Extract column name from question if present
        cols = [col for col in df.columns if col.lower() in question.lower()]
        
        if cols:
            col = cols[0]
            result += f"Unique values for {col}:\n"
            unique_vals = df[col].unique()
            if len(unique_vals) <= 20:
                result += str(unique_vals) + "\n\n"
            else:
                result += f"There are {len(unique_vals)} unique values. Here's a sample: {unique_vals[:10]}\n\n"
        else:
            for col in df.columns:
                if df[col].nunique() <= 10:
                    result += f"Unique values for {col}: {df[col].unique()}\n"
            result += "\n"
    
    if "filter" in analysis_types:
        # This is a complex operation that would require more sophisticated NLP
        # For now, we'll just provide basic info about potential filters
        result += "For filtering data, you could use these columns and their sample values:\n"
        for col in df.columns:
            unique_vals = df[col].nunique()
            if unique_vals <= 10:
                result += f"- {col}: {df[col].unique()[:5]}\n"
        result += "\n"
    
    # If nothing specific was detected, provide general info
    if not analysis_types:
        result += get_dataframe_info(df)
    
    return result, visualization

def generate_gemini_response(model, df, question):
    """Generate a response using Gemini model based on the dataframe and question"""
    if not model:
        return "API configuration error. Please check your API key and model selection.", None
    
    # First try simple direct analysis
    result, visualization = execute_query(df, question)
    
    # Prepare dataframe context for Gemini
    df_context = prepare_df_context(df)
    
    # Compose prompt for Gemini
    prompt = f"""You are a data analyst assistant helping with CSV data analysis. 
    
Here is information about the dataframe:
{df_context}

The user asked: "{question}"

I've already performed the following analysis:
{result}

Based on this information, provide a clear, concise analysis that answers the user's question. 
Include insights about patterns, trends, or anomalies.
If the question can't be answered with the available data, explain why.
Keep your response focused and professional, like a junior data analyst would.
"""

    try:
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
        
        response = model.generate_content(prompt, generation_config=generation_config)
        
        # Combine direct analysis with Gemini insights
        final_response = response.text
        
        return final_response, visualization
    except Exception as e:
        return f"Error generating response: {str(e)}", visualization

def handle_follow_up_operation(df, operation, previous_question):
    """Handle follow-up operations like sorting, filtering, grouping based on previous query"""
    try:
        if operation == "sort":
            # Extract numeric columns
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                # Try to find column in previous question
                col = next((col for col in num_cols if col.lower() in previous_question.lower()), num_cols[0])
                result = f"Sorting by {col}:\n"
                result += df.sort_values(by=col, ascending=False).head(10).to_string()
                return result
            else:
                return "No numeric columns available for sorting."
        
        elif operation == "filter":
            # This is a more complex operation that would require sophisticated NLP
            # For demonstrative purposes, we'll do a simple operation
            result = "Sample filter operation:\n"
            
            # Try to find column in previous question
            for col in df.columns:
                if col.lower() in previous_question.lower():
                    if df[col].nunique() <= 20:  # Only for columns with limited unique values
                        # Take the most common value
                        most_common = df[col].value_counts().index[0]
                        result += f"Filtering where {col} = {most_common}:\n"
                        result += df[df[col] == most_common].head(10).to_string()
                        return result
            
            # Default behavior
            col = df.columns[0]
            if df[col].nunique() <= 20:
                value = df[col].value_counts().index[0]
                result += f"Filtering where {col} = {value}:\n"
                result += df[df[col] == value].head(10).to_string()
                return result
            else:
                return "No suitable columns found for a simple filter operation."
        
        elif operation == "groupby":
            # Extract categorical columns
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if cat_cols and num_cols:
                # Try to find columns in previous question
                cat_col = next((col for col in cat_cols if col.lower() in previous_question.lower()), cat_cols[0])
                num_col = next((col for col in num_cols if col.lower() in previous_question.lower()), num_cols[0])
                
                result = f"Grouping by {cat_col}, calculating mean of {num_col}:\n"
                if df[cat_col].nunique() <= 20:  # Only for columns with limited unique values
                    result += df.groupby(cat_col)[num_col].mean().to_string()
                    return result
                else:
                    return f"Too many unique values in {cat_col} for meaningful groupby."
            else:
                return "No suitable categorical and numeric columns for groupby operation."
        
        return "Unsupported operation."
    except Exception as e:
        return f"Error performing operation: {str(e)}"

def display_app_limitations():
    """Display information about the app's capabilities and limitations"""
    with st.expander("ℹ️ About this Data Analyst Chatbot"):
        st.markdown("""
        ### CSV Data Analyst Chatbot
        
        **Features:**
        - Upload any CSV file for instant analysis
        - Ask questions about your data in plain English
        - Get visualizations, statistics, and insights
        
        **Capabilities:**
        - Descriptive statistics and data exploration
        - Data visualization (histograms, scatter plots, bar charts, etc.)
        - Correlation analysis and trend identification
        - Basic data filtering and grouping
        - Identifying missing values and data quality issues
        
        **Limitations:**
        - Works best with clean, structured CSV data
        - For large files, analysis may be limited to a sample of data
        - Complex statistical modeling or machine learning is not supported
        - Time series forecasting capabilities are limited
        
        **Tips for Best Results:**
        - Ask specific questions about your data
        - Mention column names in your questions when possible
        - For visualizations, specify the type of chart you want
        - Break complex analyses into multiple simple questions
        """)

def main():
    st.title("📊 Data Analyst Chatbot By Nafi")
    st.write("Upload a CSV file and chat with your data for instant insights!")
    
    # Display app capabilities and limitations
    display_app_limitations()
    
    # Initialize the model
    model = None
    if "temp_api_key" in st.session_state and "selected_model" in st.session_state:
        genai.configure(api_key=st.session_state.temp_api_key)
        model = genai.GenerativeModel(st.session_state.selected_model)
        st.success(f"Using model: Gemini 2.0 Flash")
    else:
        model = initialize_gemini_api()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize dataframe
    if "df" not in st.session_state:
        st.session_state.df = None
    
    # Process uploaded file
    if uploaded_file:
        if st.button("Process CSV File") or st.session_state.df is None:
            with st.spinner("Processing CSV data..."):
                df = load_csv_file(uploaded_file)
                if df is not None:
                    # Cap rows for very large files to avoid performance issues
                    if len(df) > 100000:
                        st.warning(f"File has {len(df)} rows. Using first 100,000 rows for performance.")
                        df = df.head(100000)
                    
                    st.session_state.df = df
                    st.session_state.file_name = uploaded_file.name
                    
                    # Display basic info
                    st.write(f"File processed successfully: **{uploaded_file.name}**")
                    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
                    
                    # Show the first few rows
                    st.write("Preview of data:")
                    st.dataframe(df.head())
                    
                    # Add starter message
                    if not st.session_state.messages:
                        starter_msg = """I'm your Data Analyst Chatbot! I can help analyze this CSV file.
                        
Try asking questions like:
- What's the overall summary of this dataset?
- Show me the distribution of [column]
- What are the correlations between numeric columns?
- Create a scatter plot of [column1] vs [column2]
- What insights can you give about this data?
"""
                        st.session_state.messages.append({"role": "assistant", "content": starter_msg, "visualization": None})
    
    # Display chat interface if data is loaded
    if st.session_state.get("df") is not None:
        st.write(f"Chatting about: **{st.session_state.get('file_name', 'Data')}**")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("visualization"):
                    st.markdown(message["visualization"], unsafe_allow_html=True)
        
        # Quick action buttons
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("📊 Data Summary"):
                question = "Give me a summary of this dataset"
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)
                process_question(model, question)
        
        with col2:
            if st.button("📈 Show Correlations"):
                question = "What are the correlations between variables?"
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)
                process_question(model, question)
        
        with col3:
            if st.button("🔍 Missing Values"):
                question = "Show missing values analysis"
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)
                process_question(model, question)
        
        with col4:
            if st.button("📉 Visualize Data"):
                question = "Create visualizations of key columns"
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)
                process_question(model, question)
        
        # Handle user input
        if question := st.chat_input("Ask about your data..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": question})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(question)
            
            process_question(model, question)
    
    # Add option to clear chat history
    if st.session_state.get("df") is not None and st.session_state.messages:
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.experimental_rerun()

def process_question(model, question):
    """Process a question from the user and generate a response"""
    # Generate response with visualization if applicable
    with st.spinner("Analyzing data..."):
        answer, visualization = generate_gemini_response(model, st.session_state.df, question)
    
    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "visualization": visualization
    })
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)
        if visualization:
            st.markdown(visualization, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
