# Long Text Summarization Using LLMs and Clustering with K-means

Welcome to the GitHub repository for the project on Long Text Summarization using Large Language Models (LLMs) and Clustering with K-means. In this project, we explore the evolution of text summarization and leverage advanced AI models like GPT-3 to efficiently summarize lengthy texts.

## Overview

In today's information age, dealing with an overwhelming volume of textual information is a common challenge. From research papers to legal documents, the complexity of these texts often hinders access to valuable insights. This project addresses the need for summarizing long texts, making knowledge more accessible.

## Approach

### 1. Extract text from PDF and Cleaning using PyMuPDF

We utilize PyMuPDF to extract text from PDF files, ensuring a natural order even with columns in pages. The cleaning process involves various functions like lowercase conversion, URL removal, contraction expansion, space removal, special character removal, tag, and mention removal, and image removal.

### 2. Chunking the Text

Introducing LangChain, an open-source framework, we define chunks effortlessly using a single line of code. This simplifies the handling of AI and machine learning components for developers.

### 3. Embedding

Text embedding is crucial for representing text as vectors containing information about its meaning. We explore two methods: OpenAI Embedding API and Huggingface, each requiring API keys stored in a .env file.

### 4. Clustering with K-means

We adopt K-means clustering with 10 clusters to group similar chunks, providing a comprehensive yet varied perspective on the text. The aim is to identify representative passages that encapsulate the essence of the document.

### 5. Prompting

Prompt engineering in GPT models involves careful design to influence model behavior. Custom prompts are created to guide the generation of meaningful summaries.

### 6. Summarize Chosen Chunks

Using gpt-3.5-turbo, we summarize selected chunks one by one, aiming for efficiency and accuracy in the summarization process.

### 7. Combine Prompting

With individual summaries in hand, we create a prompt to combine them and generate a more comprehensive summary of the entire text.

### 8. Summarize Summaries using gpt-4

Finally, we use gpt-4 to regroup all summaries and retrieve the final comprehensive summary of the document.


## Final Notes

In the project, a frontend layer using Streamlit is added. To deploy the app:
1. Create a GitHub repository for the app.
2. Go to Streamlit Community Cloud, select the repository, branch, and app file.
3. Click on the Deploy! button.

If you find this project helpful, please give it a star and share it!

## References

- [5 Levels Of LLM Summarizing: Novice to Expert](https://www.youtube.com/watch?v=qaPMdcCqtWk) - Kamradt (Data Indy)
- [Chat with Multiple PDFs](https://www.youtube.com/watch?v=dXxQ0LR-3Hg) - Alejandro AO
- [Extract Text from PDF Resumes Using PyMuPDF and Python](https://neurond.com/blog/extract-text-from-pdf-pymupdf-and-python) - Trinh Nguyen
