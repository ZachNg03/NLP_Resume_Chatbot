import streamlit as st
import os
from PyPDF2 import PdfReader
from transformers import pipeline
from Models import *
from sentence_transformers import SentenceTransformer
from transformers import BertModel, BertTokenizer, BertForQuestionAnswering
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
from transformers import AlbertTokenizer, AlbertForQuestionAnswering
from transformers import ElectraTokenizer, ElectraForQuestionAnswering
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import textwrap
import re
import torch

from nltk.stem import WordNetLemmatizer
from nltk import pos_tag                                         # for parts of speech
from nltk.corpus import stopwords                                # for stop words
from nltk.tokenize import word_tokenize

from langchain.text_splitter import RecursiveCharacterTextSplitter

import random

responses = {
    "hi": ["Hello!", "Hi there!", "Hey!", "Hey, how can I help you?", "Hi! What can I do for you?"],
    "how are you": ["I'm doing well, thank you!", "I'm good, thanks for asking!", "All good, thanks!", "I'm great, thanks for asking!"],
    "bye": ["Goodbye!", "Bye bye!", "See you later!", "Take care!", "Farewell!"],
    "default": ["I'm sorry, I didn't understand that.", "Could you please rephrase that?", "I'm still learning!", "I'm not sure I understand. Could you provide more context?"]
}

# Ensure the directory for model storage exists
def ensure_model_directory(base_dir="saved_models/qa_models"):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

# Retrieve the full path for a model
def get_model_path(model_name, base_dir="saved_models/qa_models"):
    return os.path.join(base_dir, model_name.replace("/", "_"))

# Save both the model and its tokenizer to the disk
def save_model(model, tokenizer, model_path):
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

# Load or download the model depending on its availability
def load_or_download_model(model_name, base_dir="saved_models/qa_models"):
    model_path = get_model_path(model_name, base_dir)
    if not os.path.exists(model_path):
        print(f"Downloading and saving model: {model_name}")
        model = pipeline("question-answering", model=model_name)
        save_model(model.model, model.tokenizer, model_path)
    else:
        print(f"Loading model from disk: {model_name}")
        model = pipeline("question-answering", model=model_path, tokenizer=model_path)
    return model

# Initialize once at start
ensure_model_directory()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to extract text from PDF using PyPDF2
def extract_text_from_pdf(uploaded_file):
    pdf_text = ""
    with open(os.path.join("Data-Sample", uploaded_file.name), "rb") as file:
        reader = PdfReader(file)
        num_pages = len(reader.pages)
        for page_number in range(num_pages):
            page = reader.pages[page_number]
            pdf_text += page.extract_text()
    return pdf_text


# Function to save uploaded file to local directory
def save_uploaded_file(uploaded_file, save_dir="Data-Sample"):
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save the uploaded file to the specified directory
    with open(os.path.join(save_dir, uploaded_file.name), "wb") as file:
        file.write(uploaded_file.getbuffer())


# Function to upload resume
def upload_resume():
    st.sidebar.header("Upload Your Resume")
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file is None:
        st.markdown("<div style='color: #f63366;'>Please upload a PDF file.</div>", unsafe_allow_html=True)
        return None
    save_uploaded_file(uploaded_file)  # Save the uploaded file to the local directory
    return uploaded_file  # Return the file object


# Clear the chat messages
def clear_chat():
    st.session_state.messages.clear()
    st.experimental_rerun()  # Rerun the Streamlit app to ensure updated session state

# Function to select model
def select_model():
    st.sidebar.header("Select Model")
    model = st.sidebar.selectbox("Select a QA Pipeline Model:", list(QA_MODEL_NAMES.values()))
    return model

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Not meaningful words
stop = stopwords.words('english')

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation and special characters (excluding %, - for year, 's)
    text = re.sub(r"[^a-zA-Z0-9\s%\s@\s,\$\-\'s()]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove spaces between alphabetic characters for abbreviation
    text = re.sub(r"(?<=\w)\s+(?=\w)", " ", text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Initialize WordNet lemmatizer
    lemmatizer = WordNetLemmatizer()

    # parts of speech
    tags_list = pos_tag(tokens, tagset=None)

    lema_words = []  # empty list
    for token, pos_token in tags_list:  # lemmatize according to POS, let the words become original state
        if pos_token.startswith('V'):  # Verb
            pos_val = 'v'
        elif pos_token.startswith('J'):  # Adjective
            pos_val = 'a'
        elif pos_token.startswith('R'):  # Adverb
            pos_val = 'r'
        else:
            pos_val = 'n'  # Noun
        lema_token = lemmatizer.lemmatize(token, pos_val)

        if lema_token not in stop:
            lema_words.append(lema_token)  # appending the lemmatized token into a list

    return " ".join(lema_words)

# Split all the text into chunks
def split_text_into_chunks(text, chunk_size=700, chunk_overlap=10):
    # Initialize the Recursive Character Text Splitter
    textSplitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    # Split the text into chunks
    chunks = textSplitter.split_text(text=text)
    return chunks

#preprocess chunks
def preprocessed_chunk(chunk):
    chunks = preprocess_text(chunk)
    return chunks

# Function to generate a response from the QA model
# Generate a response from the QA model based on the context and prompt
def generate_response(selected_model, prompt, context):
    try:
        if prompt.lower() == "hi":
            # Select a random greeting message
            return random.choice(responses["hi"])
        elif prompt.lower() == "bye":
            # Select a random goodbye message
            return random.choice(responses["bye"])
        elif prompt.lower() == "help":
            return "1)You can ask some question like what is the name, what is the driving license, what is the skill and so on."
        else:
            qa_pipeline = selected_model
            return qa_pipeline({'question': prompt, 'context': context})['answer']
    except Exception as e:
        st.error(f"Failed to generate response: {e}")
        return "An error occurred while generating the response."

# Function to calculate the highest confidence score and answer
def get_highest_confidence_answer(prompt, context_chunks, tokenizer, model):
    highest_confidence_score = -1
    highest_confidence_answer = None

    for chunk in context_chunks:
        # Tokenize input question and context chunk
        inputs = tokenizer.encode_plus(prompt, chunk, return_tensors="pt", max_length=512, truncation=True)

        # Extract input token IDs
        input_ids = inputs["input_ids"]

        # Get model predictions
        outputs = model(input_ids=input_ids)

        # Extract start and end scores from the model output
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        # Get the most likely answer and its confidence scores
        answer_start = torch.argmax(start_scores).item()
        answer_end = torch.argmax(end_scores).item() + 1
        confidence_start = torch.max(start_scores).item()
        confidence_end = torch.max(end_scores).item()

        # Calculate the average confidence score for the answer
        confidence_score = (confidence_start + confidence_end) / 2

        # Decode the tokens into a string
        answer = tokenizer.decode(input_ids[0][answer_start:answer_end], skip_special_tokens=True)

        # Update highest confidence answer
        if confidence_score > highest_confidence_score and answer != "":
            highest_confidence_answer = answer
            highest_confidence_score = confidence_score

    return highest_confidence_answer

import os

# Function to create PDF from chat history
def create_pdf(messages, model_name):
    base_filename = f"Chat_with_{model_name}_results.pdf"
    filename = base_filename
    counter = 2

    # Check if the file already exists
    while os.path.exists(filename):
        filename = f"{base_filename[:-4]}_{counter}.pdf"
        counter += 1

    c = canvas.Canvas(filename, pagesize=A4)
    c.setFont("Helvetica", 12)

    # Define initial y-coordinate for text
    y_coordinate = A4[1] - 50  # Start from top of the page

    for i, message in enumerate(messages, 1):
        role = message["role"]
        content = message["content"]

        # Draw role (user/assistant)
        role_text = f"{role.capitalize()}:"
        y_coordinate = draw_text(c, role_text, y_coordinate, align='left')

        # Split long content into multiple lines
        content_lines = split_text(content, max_width=A4[0]-100)

        # Draw each line of the content
        for line in content_lines:
            y_coordinate = draw_text(c, line, y_coordinate, align='left')

        # Add larger spacing between messages
        y_coordinate -= 20

        # Check if y-coordinate is too low to fit another message
        if y_coordinate < 50:
            c.showPage()  # Create a new page
            y_coordinate = A4[1] - 50  # Reset y-coordinate for the new page

    c.save()
    st.success(f"PDF created: {filename}")

# Function to draw text on PDF canvas
def draw_text(canvas, text, y_coordinate, align='left', spacing=20):
    lines = textwrap.wrap(text, width=70)  # Wrap text to fit within page width
    for line in lines:
        if align == 'left':
            x_coordinate = 50
        elif align == 'right':
            x_coordinate = A4[0] - 50 - canvas.stringWidth(line)
        else:  # Center alignment
            x_coordinate = (A4[0] - canvas.stringWidth(line)) / 2
        canvas.drawString(x_coordinate, y_coordinate, line)
        y_coordinate -= spacing
    return y_coordinate

# Function to split text into multiple lines
def split_text(text, max_width):
    lines = textwrap.wrap(text, width=max_width)
    return lines


#Main function
def main():

    st.title("CVGenie: Your Resume's 24/7 Wingman")
    # Check if the chat history is empty
    if not st.session_state.messages:
        # Display welcome message sent by the assistant
        st.session_state.messages.append(
            {"role": "assistant", "content": "Hi I am CVGenie, what do you wish to inquire about the Resume?"})

    # Sidebar for upload and model selection
    uploaded_resume = upload_resume()

    if uploaded_resume:  # Check if resume was uploaded successfully
        selected_QA_model_name = select_model()

        with st.spinner(text='Loading QA model...'):

            QA_model_key = [key for key, value in QA_MODEL_NAMES.items() if value == selected_QA_model_name][0]
            QA_model = load_or_download_model(QA_model_key)

        # Extract text from the uploaded resume
        resume_text = extract_text_from_pdf(uploaded_resume)

        if selected_QA_model_name == "BERT large model (uncased) whole word masking finetuned on SQuAD":
            tokenizer = BertTokenizer.from_pretrained(QA_model_key)
            model = BertForQuestionAnswering.from_pretrained(QA_model_key)

        elif selected_QA_model_name == "BERT base model (uncased)":
            tokenizer = BertTokenizer.from_pretrained(QA_model_key)
            model = BertForQuestionAnswering.from_pretrained(QA_model_key)

        elif selected_QA_model_name == "DistilBERT base model (uncased)":
            tokenizer = DistilBertTokenizer.from_pretrained(QA_model_key)
            model = DistilBertForQuestionAnswering.from_pretrained(QA_model_key)

        elif selected_QA_model_name == "ALBERT Base v2":
            tokenizer = AlbertTokenizer.from_pretrained(QA_model_key)
            model = AlbertForQuestionAnswering.from_pretrained(QA_model_key)

        elif selected_QA_model_name == "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators":
            tokenizer = ElectraTokenizer.from_pretrained(QA_model_key)
            model = ElectraForQuestionAnswering.from_pretrained(QA_model_key)

        elif selected_QA_model_name == "roberta-base for QA":
            tokenizer = AutoTokenizer.from_pretrained(QA_model_key)
            model = AutoModelForQuestionAnswering.from_pretrained(QA_model_key)

        # Display chatbot interface
        if prompt := st.chat_input("Ask a question about the resume..."):
            prompt = prompt.lower()  # Convert user input to lowercase
            with st.spinner(text="Generating response..."):
                # Load the tokenizer based on the selected QA model
                context_chunks = split_text_into_chunks(resume_text)
                highest_confidence_answer = get_highest_confidence_answer(prompt, context_chunks, tokenizer, model)
                if highest_confidence_answer:
                    context = "".join(
                        [str(doc) if isinstance(doc, str) else doc.page_content for doc in highest_confidence_answer])
                else:
                    # Handle the case when highest_confidence_answer is None
                    # You can set a default value or raise an error
                    context = "No answer found"
                response = generate_response(QA_model, prompt, context)

            # Append the user's query and the chatbot's response to the conversation history
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response})

        # Display the conversation history
        for message in st.session_state.messages:
            if message["role"] == "user":
                with open('avatar/user_role.png', 'rb') as f:
                    avatar_image = f.read()
                with st.chat_message("You", avatar=avatar_image):
                    st.markdown(message["content"])
            elif message["role"] == "assistant":
                with open('avatar/assistant_role.png', 'rb') as f:
                    avatar_image = f.read()
                with st.chat_message("CVGenie", avatar=avatar_image):
                    st.markdown(message["content"])

    # Button to save chat history as PDF
    if st.sidebar.button("Save Chat as PDF"):
        create_pdf(st.session_state.messages, "Chatbot")

    # New chat button to clear chat history
    if st.sidebar.button("Start New Chat"):
        clear_chat()

if __name__ == "__main__":
    main()

