import os
import gradio as gr
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Load environment variables
load_dotenv()

class QdrantRAGPipeline:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        # Initialize OpenAI components
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            openai_api_key=self.openai_api_key
        )
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key
        )
        
        # Store collections and vector stores
        self.collections = {}
        self.vector_stores = {}
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200)),
            length_function=len,
        )
    

    
    def load_document(self, file_path: str, collection_name: str, chunk_size: int = None) -> Dict:
        """Load and process a document into the specified collection"""
        try:
            # Update chunk size if provided
            if chunk_size:
                self.text_splitter.chunk_size = chunk_size
            
            # Load document
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            if not documents:
                return {"status": "error", "message": "В документе не найдено содержимое"}
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Create collection if it doesn't exist
            try:
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=1536,  # OpenAI embeddings dimension
                        distance=Distance.COSINE
                    )
                )
            except Exception as e:
                # Collection might already exist
                print(f"Collection {collection_name} might already exist: {e}")
            
            # Create vector store for this collection using from_documents
            vector_store = QdrantVectorStore.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                collection_name=collection_name,
            )
            
            # Store the vector store for later use
            self.vector_stores[collection_name] = vector_store
            self.collections[collection_name] = {
                "document_count": len(chunks),
                "original_file": Path(file_path).name
            }
            
            return {
                "status": "success",
                "message": f"Успешно загружено {len(chunks)} фрагментов в коллекцию '{collection_name}'",
                "chunks": len(chunks)
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Ошибка загрузки документа: {str(e)}"}
    
    def get_collections(self) -> List[str]:
        """Get list of available collections"""
        try:
            # Get collections from Qdrant
            collections_info = self.qdrant_client.get_collections()
            qdrant_collections = [col.name for col in collections_info.collections]
            
            # Merge with locally tracked collections
            all_collections = list(set(list(self.collections.keys()) + qdrant_collections))
            return all_collections
        except Exception as e:
            print(f"Error getting collections: {e}")
            return list(self.collections.keys())
    
    def chat_with_document(self, query: str, collection_name: str) -> str:
        """Chat with a specific document collection"""
        if collection_name not in self.vector_stores:
            # Try to load existing collection
            try:
                # First check if collection exists
                collections_info = self.qdrant_client.get_collections()
                collection_names = [col.name for col in collections_info.collections]
                
                if collection_name not in collection_names:
                    return f"Ошибка: Коллекция '{collection_name}' не найдена. Пожалуйста, сначала загрузите документ."
                
                vector_store = QdrantVectorStore.from_existing_collection(
                    collection_name=collection_name,
                    embedding=self.embeddings,
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key,
                )
                self.vector_stores[collection_name] = vector_store
            except Exception as e:
                return f"Ошибка подключения к коллекции '{collection_name}': {str(e)}"
        
        try:
            # Create retrieval chain
            vector_store = self.vector_stores[collection_name]
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            
            # Get answer using invoke instead of __call__
            result = qa_chain.invoke({"query": query})
            
            # Format response with sources
            answer = result["result"]
            sources = result.get("source_documents", [])
            
            if sources:
                answer += "\n\n**Источники:**\n"
                for i, doc in enumerate(sources[:2], 1):
                    page = doc.metadata.get("page", "Неизвестно")
                    answer += f"- Страница {page}: {doc.page_content[:100]}...\n"
            
            return answer
            
        except Exception as e:
            return f"Ошибка обработки запроса: {str(e)}"

# Initialize the RAG pipeline
rag_pipeline = QdrantRAGPipeline()

def load_document_interface(file, collection_name, chunk_size):
    """Interface function for loading documents"""
    if not file:
        return "Пожалуйста, выберите файл для загрузки.", "", []
    
    if not collection_name:
        return "Пожалуйста, укажите название коллекции.", "", []
    
    tmp_file_path = None
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            # Handle both bytes and file-like objects
            if isinstance(file, bytes):
                tmp_file.write(file)
            elif hasattr(file, 'read'):
                tmp_file.write(file.read())
            else:
                # If it's a file path or other format, try to read it
                with open(file, 'rb') as f:
                    tmp_file.write(f.read())
            tmp_file_path = tmp_file.name

        # Load document
        result = rag_pipeline.load_document(tmp_file_path, collection_name, chunk_size)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        # Update collections dropdown
        collections = get_collections()
        
        if result["status"] == "success":
            return result["message"], "", gr.update(choices=collections, value=collection_name)
        else:
            return result["message"], "", gr.update(choices=collections)
            
    except Exception as e:
        # Clean up temporary file
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        return f"Ошибка: {str(e)}", "", gr.update()

def chat_interface(message, history, collection_name):
    """Interface function for chatting"""
    if not collection_name:
        history.append({"role": "assistant", "content": "Пожалуйста, сначала выберите коллекцию документов."})
        return history, ""
    
    if not message:
        return history, ""
    
    # Get response from RAG pipeline
    response = rag_pipeline.chat_with_document(message, collection_name)
    
    # Update chat history with the new message format
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    
    return history, ""



def get_collections():
    """Get available collections for dropdown"""
    try:
        return rag_pipeline.get_collections()
    except Exception as e:
        print(f"Error getting collections: {e}")
        return []

# Create Gradio interface
def create_gradio_app():
    with gr.Blocks(title="Система RAG для медицинских документов", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🏥 Система RAG для медицинских документов")
        gr.Markdown("Загружайте медицинские документы и общайтесь с ними, используя поиск и извлечение информации на основе ИИ.")
        
        with gr.Tab("📄 Управление документами"):
            gr.Markdown("## Загрузка и обработка документов")
            
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(
                        label="Выберите PDF документ",
                        file_types=[".pdf"],
                        type="binary"
                    )
                    collection_input = gr.Textbox(
                        label="Название коллекции",
                        placeholder="Введите уникальное название для этой коллекции документов",
                        value=""
                    )
                    chunk_size_input = gr.Slider(
                        minimum=500,
                        maximum=2000,
                        value=int(os.getenv("CHUNK_SIZE", 1000)),
                        step=100,
                        label="Размер фрагмента"
                    )
                    load_btn = gr.Button("📤 Загрузить документ", variant="primary")
                
                with gr.Column(scale=1):
                    load_status = gr.Textbox(
                        label="Статус",
                        interactive=False,
                        lines=3
                    )
        
        with gr.Tab("💬 Чат с документами"):
            gr.Markdown("## Общение с вашими документами")
            
            with gr.Row():
                with gr.Column(scale=1):
                    collection_dropdown = gr.Dropdown(
                        label="Выберите коллекцию документов",
                        choices=[],
                        value=None,
                        allow_custom_value=False
                    )
                    refresh_btn = gr.Button("🔄 Обновить коллекции")
                
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="История чата",
                        height=400,
                        type="messages"
                    )
                    with gr.Row():
                        msg_input = gr.Textbox(
                            label="Ваш вопрос",
                            placeholder="Задайте вопрос о выбранном документе...",
                            scale=4
                        )
                        send_btn = gr.Button("📤 Отправить", scale=1, variant="primary")
        
        with gr.Tab("ℹ️ Информация о системе"):
            gr.Markdown("## Системная информация")
            
            def get_system_info():
                collections = rag_pipeline.get_collections()
                info = f"**Доступные коллекции:** {len(collections)}\n\n"
                for col in collections:
                    col_info = rag_pipeline.collections.get(col, {})
                    info += f"- **{col}**: {col_info.get('document_count', 0)} фрагментов"
                    if 'original_file' in col_info:
                        info += f" (из {col_info['original_file']})"
                    info += "\n"
                return info
            
            system_info = gr.Markdown(get_system_info())
            info_refresh_btn = gr.Button("🔄 Обновить информацию")
        
        # Event handlers
        load_btn.click(
            fn=load_document_interface,
            inputs=[file_input, collection_input, chunk_size_input],
            outputs=[load_status, collection_input, collection_dropdown]
        )
        
        refresh_btn.click(
            fn=lambda: gr.update(choices=get_collections()),
            outputs=[collection_dropdown]
        )
        
        send_btn.click(
            fn=chat_interface,
            inputs=[msg_input, chatbot, collection_dropdown],
            outputs=[chatbot, msg_input]
        )
        
        msg_input.submit(
            fn=chat_interface,
            inputs=[msg_input, chatbot, collection_dropdown],
            outputs=[chatbot, msg_input]
        )
        
        info_refresh_btn.click(
            fn=lambda: get_system_info(),
            outputs=[system_info]
        )
    
    return app

if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", 7878)),
        share=False
    )
