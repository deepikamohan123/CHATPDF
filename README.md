# ChatPDF: Your AI-Driven PDF Insight Tool

ChatPDF is an intuitive Streamlit application that extracts and analyzes text from PDF files, harnessing the power of Google's Generative AI, particularly the Gemini-PRO model. This tool employs a Retrieval-Augmented Generation (RAG) approach to deliver accurate, context-sensitive responses to user inquiries based on the contents of uploaded documents.

## Key Features

- **Rapid Insights**: Quickly extracts and analyzes text from uploaded PDF files to deliver immediate insights.
- **Retrieval-Augmented Generation**: Leverages Google's Generative AI model Gemini-PRO for producing high-quality and contextually appropriate answers.
- **Secure API Key Entry**: Provides a secure method for entering Google API keys to access generative AI services.

## Getting Started

### Requirements

- Google API Key: Acquire a Google API key to access Google's Generative AI models. Visit Google API Key Setup for your key (https://makersuite.google.com/app/apikey).

- Streamlit: This application is developed using Streamlit. Make sure to have Streamlit installed in your environment.

### Installation

Clone this repository or download the source code to your local machine. Navigate to the application directory and install the necessary Python packages:

```bash
pip install -r requirements.txt
```

### How to Use

1. **Launch the App**: Start the Streamlit application by executing the command:

   ```bash
   streamlit run <path_to_script.py>
   ```

   Ensure you replace <path_to_script.py> with the actual path to your script file.

2. **Input Your Google API Key**: Enter your Google API key securely when prompted. This key is necessary for the application to utilize Google’s Generative AI models.

3. **Upload Your PDFs**: You have the option to upload single or multiple PDF documents. The application will review the contents of these documents to provide responses to your queries.

4. **Pose Your Questions1**: After your documents have been processed, feel free to ask any questions related to the content of your uploaded files.

### Technical Overview

- **PDF Text Extraction**: This application employs PyPDF2 for extracting textual data from PDF files.
- **Chunking Text**: It uses the RecursiveCharacterTextSplitter from LangChain to break down the extracted text into smaller, manageable sections.
- **Creating a Vector Store**: The app utilizes FAISS to generate a searchable vector database from the text chunks.
- **Generating Answers**: It relies on ChatGoogleGenerativeAI from LangChain to formulate responses to user inquiries, utilizing the context from the uploaded documents.
