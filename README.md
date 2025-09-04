# Medical Document RAG System

A sophisticated Retrieval-Augmented Generation (RAG) pipeline designed specifically for medical documents. This system uses Qdrant vector database, LangChain, and OpenAI embeddings to enable intelligent document search and chat functionality.

## 🌟 Features

- **Document Upload & Processing**: Upload PDF medical documents and process them into searchable chunks
- **Multiple Collections**: Separate documents into different collections for organized retrieval
- **Intelligent Chat**: Chat with your documents using natural language queries
- **Configurable Chunking**: Adjust chunk size for optimal document processing
- **Docker Deployment**: Complete containerized solution with Docker Compose
- **Vector Search**: Powered by Qdrant vector database for fast and accurate retrieval
- **Modern UI**: Clean and intuitive Gradio interface

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Gradio UI     │────│   RAG Pipeline  │────│   Qdrant DB     │
│                 │    │   (LangChain)   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                       ┌─────────────────┐
                       │   OpenAI API    │
                       │   (Embeddings   │
                       │   + Chat)       │
                       └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API key

### Setup

1. **Clone and setup environment**:
   ```bash
   cd med-doc-rag
   cp .env.example .env
   ```

2. **Configure environment variables**:
   Edit `.env` file and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Start the application**:
   ```bash
   docker-compose up -d
   ```

4. **Access the application**:
   - RAG Interface: http://localhost:7860
   - Qdrant Dashboard: http://localhost:6333/dashboard

## 📖 Usage

### 1. Document Management
- Navigate to the "Document Management" tab
- Upload a PDF document
- Provide a unique collection name
- Adjust chunk size if needed (default: 1000 characters)
- Click "Load Document"

### 2. Chat with Documents
- Go to the "Chat with Documents" tab
- Select a document collection from the dropdown
- Ask questions about your documents
- Get AI-powered answers with source references

### 3. System Information
- Check the "System Info" tab to see:
  - Available collections
  - Document chunk counts
  - Original file names

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required |
| `QDRANT_URL` | Qdrant database URL | `http://qdrant:6333` |
| `QDRANT_API_KEY` | Qdrant API key (optional) | - |
| `CHUNK_SIZE` | Document chunk size | `1000` |
| `CHUNK_OVERLAP` | Chunk overlap size | `200` |
| `GRADIO_SERVER_NAME` | Gradio server host | `0.0.0.0` |
| `GRADIO_SERVER_PORT` | Gradio server port | `7860` |

### Customization

You can modify the following in `app.py`:

- **LLM Model**: Change `model_name` in `ChatOpenAI` initialization
- **Retrieval Parameters**: Adjust `k` value in `search_kwargs`
- **Text Splitting**: Modify `RecursiveCharacterTextSplitter` parameters
- **UI Theme**: Change Gradio theme in `gr.themes`

## 🔧 Development

### Local Development

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Qdrant locally**:
   ```bash
   docker run -p 6333:6333 qdrant/qdrant:v1.7.0
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

### Adding New Features

The codebase is modular and extensible:

- **Document Loaders**: Add support for new file types in `QdrantRAGPipeline`
- **Embeddings**: Switch to different embedding models
- **UI Components**: Extend Gradio interface with new tabs/features
- **Database**: Add metadata filtering and advanced search

## 📁 Project Structure

```
med-doc-rag/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Multi-container setup
├── .env                  # Environment variables
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Errors**:
   - Verify your API key is correct
   - Check your OpenAI account has sufficient credits

2. **Qdrant Connection Issues**:
   - Ensure Qdrant container is running: `docker-compose ps`
   - Check logs: `docker-compose logs qdrant`

3. **Memory Issues**:
   - Reduce chunk size for large documents
   - Increase Docker memory allocation

4. **Upload Failures**:
   - Ensure PDF files are not corrupted
   - Check file size limits

### Logs

View application logs:
```bash
docker-compose logs rag-app
```

View Qdrant logs:
```bash
docker-compose logs qdrant
```

## 🔐 Security Considerations

- Store API keys securely using `.env` file
- Don't commit sensitive information to version control
- Consider implementing authentication for production use
- Regularly update dependencies for security patches

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review Qdrant and LangChain documentation

---

**Note**: This system is designed for medical document processing. Ensure compliance with relevant healthcare data regulations (HIPAA, GDPR, etc.) when handling sensitive medical information.
