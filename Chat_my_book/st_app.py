from llama_index import ServiceContext, StorageContext, load_index_from_storage, SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index.prompts import PromptTemplate
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index import LLMPredictor
from langchain.chat_models.openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
import tiktoken
import os
import time

load_dotenv()
openai_api_key = os.environ['OPENAI_API_KEY']

model = "local:BAAI/bge-small-en"

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# app = Flask(__name__)
# @app.route('/chat-api', methods=['POST'])
def chat_function(title, chunk_size, chunk_overlap, nodes, llm_model, prompt, query):

    node_parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name=llm_model))

    service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
    node_parser=node_parser,
    embed_model=model,
    )
    folder_name = title.split()
    
    # fetch relevant embeddings of the book
    directory = f"embeddings_{folder_name[0]}_{folder_name[1]}_{folder_name[2]}"

    storage_context = StorageContext.from_defaults(persist_dir=directory)
    index = load_index_from_storage(
                                storage_context,
                                service_context=service_context
                                )
    
    # prompt template to quide the chatbot
    prompt = prompt.format(title=title, query=query)
    qa_template = PromptTemplate(prompt)

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=nodes,
        text_qa_template=qa_template
    )

    query_engine = RetrieverQueryEngine(retriever=retriever).from_args(retriever=retriever, text_qa_template=qa_template)
    # query_engine = index.as_query_engine(text_qa_template = qa_template)
    response = query_engine.query(query)
    return response

def main():

    with open('params.txt','r',encoding="utf-8") as pa:
        param_data = pa.read()
    params = param_data.split(',')

    with open('prompt.txt', 'r',encoding="utf-8") as pr:
        prompt_saved = pr.read()
    st.title("Chat My Book")

    _, col2 = st.columns([3,1])
    with col2:
        model = st.button('Save Final Model')
    with st.form("search_form"):
        query = st.text_input("Enter your query", value="")
        submit_button = st.form_submit_button("Send")
    with st.sidebar:
        book = st.selectbox('Select Book', ['The 7 Habits of Highly Effective People', 'The 8th Habit','First Things First'])
        openai_models = ['text-davinci-003', 'gpt-3.5-turbo','gpt-3.5-turbo-16k','gpt-4','gpt-4-32k','gpt-4-0613','gpt-4-32k-0613','gpt-4-1106-preview']
        index_ = openai_models.index(params[0])
        llm_model = st.selectbox("Choose OpenAI's Model", openai_models, index=index_)
        size = st.number_input('Chunk Size', value = int(params[1]), min_value = 512)
        overlap = st.number_input('Chunk Overlap', value = int(params[2]), min_value = 50)
        nodes = st.number_input('Chunks Retrieved: ', value = int(params[3]), min_value = 3)
        default_prompt = prompt_saved
        prompt = st.text_area("Editable Prompt Template", value=default_prompt, height=200)

        # Display the edited text
        # st.write("Edited Text:", edited_text)
        # save = st.button("Save Model")
    if submit_button and query:
        start = time.time()
        response = chat_function(book, size, overlap, nodes, llm_model, prompt, query)
        total_time = time.time() - start
        tokens = num_tokens_from_string(prompt, llm_model)
        
        with st.expander('Response', expanded=True):
            st.write(response.response)
            st.write('\n*Response Time:* ', round(total_time,3), ' seconds')
            st.write('*Number of Tokens:* ', tokens)

        st.markdown(f"<h6 style='font-size:20px;'>Source Nodes/Chunks</h5>", unsafe_allow_html=True)
        i = 1
        for node in response.source_nodes:
            with st.expander(f'Node {i}', expanded=False):
                st.markdown(node.text, unsafe_allow_html=True)
            i += 1
        #     st.text_area(label ="",value=node.text, height =100)


        # print('submitted')
        # print(book, llm_model, size, overlap)
    
    if model:
        params = f'{llm_model},{size},{overlap},{nodes}'
        prompt_template = f'{prompt}'
        with open("params.txt", "w") as f:
            f.write(params) 
        with open("prompt.txt", "w") as f:
            f.write(prompt_template) 

if __name__ == '__main__':
    # app.run(debug=True, host="0.0.0.0",port=8000)
    main()
