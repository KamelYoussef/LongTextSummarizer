from dotenv import load_dotenv
from Functions import *
import fitz
import streamlit as st
from htmlTemplates import css
from tempfile import NamedTemporaryFile
import time

# Loaders
from langchain.schema import Document

# Splitters
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Model
from langchain.chat_models import ChatOpenAI

# Embedding Support
from langchain.embeddings import OpenAIEmbeddings

# Summarizer we'll use for Map Reduce
from langchain.chains.summarize import load_summarize_chain

from langchain import PromptTemplate


def main():
    load_dotenv()
    llm = ChatOpenAI()

    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    st.header("Summary of the PDF")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_doc = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", type=["pdf"]
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                start = time.time()
                # get pdf text
                with NamedTemporaryFile(dir=".", suffix=".pdf") as f:
                    f.write(pdf_doc.getbuffer())
                    text = merge_text(fitz.open(f.name))

                print(f"This book has {llm.get_num_tokens(text)} tokens in it")

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=10000, chunk_overlap=3000
                )

                docs = text_splitter.create_documents([text])

                print(f"Now our book is split up into {len(docs)} documents")

                embeddings = OpenAIEmbeddings()

                vectors = embeddings.embed_documents([x.page_content for x in docs])

                embedding_time = time.time()

                selected_indices = clustering(vectors)

                clustering_time = time.time()

                llm3 = ChatOpenAI(temperature=0, max_tokens=1000, model="gpt-3.5-turbo")
                llm4 = ChatOpenAI(
                    temperature=0, max_tokens=3000, model="gpt-4", request_timeout=120
                )

                map_prompt = """
                    You will be given a single passage of a book. This section will be enclosed in triple backticks (```)
                    Your goal is to give a summary of this section so that a reader will have a full understanding of what happened.
                    Your response should be at least three paragraphs and fully encompass what was said in the passage.

                    ```{text}```
                    FULL SUMMARY:
                    """
                map_prompt_template = PromptTemplate(
                    template=map_prompt, input_variables=["text"]
                )
                map_chain = load_summarize_chain(
                    llm=llm3, chain_type="stuff", prompt=map_prompt_template
                )

                selected_docs = [docs[doc] for doc in selected_indices]

                # Make an empty list to hold your summaries
                summary_list = []

                # Loop through a range of the lenght of your selected docs
                for i, doc in enumerate(selected_docs):
                    # Go get a summary of the chunk
                    chunk_summary = map_chain.run([doc])

                    # Append that summary to your list
                    summary_list.append(chunk_summary)

                summaries = "\n".join(summary_list)

                # Convert it back to a document
                summaries = Document(page_content=summaries)

                print(
                    f"Your total summary has {llm.get_num_tokens(summaries.page_content)} tokens"
                )

                combine_prompt = """
                    You will be given a series of summaries from a book. The summaries will be enclosed in triple backticks (```)
                    Your goal is to give a verbose summary in french in three paragraphs of what happened in those summaries.
                    The reader should be able to grasp what happened in the entire book.

                    ```{text}```
                    VERBOSE SUMMARY:
                    """
                combine_prompt_template = PromptTemplate(
                    template=combine_prompt, input_variables=["text"]
                )

                reduce_chain = load_summarize_chain(
                    llm=llm4,
                    chain_type="stuff",
                    prompt=combine_prompt_template,
                )
                output = reduce_chain.run([summaries])

                end = time.time()
                print(end - start)

                save_file("Output/Summary", output)

    output = open_file("Output/Summary")
    st.write(output)
    f = open('Output/Summary.txt', 'r+')
    f.truncate(0)


if __name__ == "__main__":
    main()
