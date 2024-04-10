from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.chains import create_retrieval_chain
import constants


# Fetch data -> process data -> ready final cleaned data 

#Loading data --> split data into chunks --> embed data to form vector rep --> store it in vector db --> retrieve data 

loader = TextLoader('test.txt')
data = loader.load()


splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = splitter.split_documents(data)
embeddings = CohereEmbeddings(cohere_api_key=constants.COHERE_API_KEY)
vector = FAISS.from_documents(documents, embeddings)

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

#prompt | llm 

#retriever | prompt | llm 

#input | retriever | prompt | llm | response

llm = ChatAnthropic(anthropic_api_key=constants.ANTHROPIC_API_KEY,model="claude-3-sonnet-20240229", temperature=0.2, max_tokens=1024)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
input = input("Give me instructions - ")

response = retrieval_chain.invoke({"input": input})
print(response["answer"])