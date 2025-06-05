from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
from langchain.schema import Document
from typing import List, Dict, Any
import logging

class RAGChain:
    def __init__(self, vector_store_manager, model_name: str = "gpt-3.5-turbo", temperature: float = 0.2):
        """
        Initialize RAG Chain with vector store manager

        Args:
            vector_store_manager: VectorStoreManager instance
            model_name: OpenAI model to use
            temperature: Temperature for response generation
        """
        self.vector_store_manager = vector_store_manager
        self.model_name = model_name
        self.temperature = temperature

        # Initialize the LLM
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature
        )

        self.qa_chain = None
        self._setup_chain()

        logging.info(f"RAG Chain initialized with model: {model_name}, temperature: {temperature}")

    def _setup_chain(self):
        """Setup the QA chain with custom prompt"""
        try:
            # Enhanced prompt template for better responses
            prompt_template = """You are an expert assistant helping with questions about documents.
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer based on the provided context, just say that you don't know, don't try to make up an answer.
            Be specific and provide detailed answers when possible, citing relevant information from the context.

            Context:
            {context}

            Question: {question}

            Helpful Answer:"""

            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            # Get retriever from vector store manager
            retriever = self.vector_store_manager.get_retriever(
                search_kwargs={"k": 5}
            )

            # Create the QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True,
                verbose=False
            )

            logging.info("QA chain setup completed successfully")

        except Exception as e:
            logging.error(f"Error setting up QA chain: {e}")
            raise

    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system and return comprehensive results

        Args:
            question: User question

        Returns:
            Dictionary with answer, sources, tokens, and cost information
        """
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Call _setup_chain() first.")

        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        try:
            # Use OpenAI callback to track usage
            with get_openai_callback() as cb:
                result = self.qa_chain.invoke({"query": question})

                # Prepare comprehensive response
                response = {
                    "answer": result.get("result", "No answer generated"),
                    "source_documents": result.get("source_documents", []),
                    "question": question,
                    "tokens_used": cb.total_tokens,
                    "cost": cb.total_cost,
                    "model_used": self.model_name,
                    "temperature": self.temperature
                }

                # Add source metadata
                if response["source_documents"]:
                    response["sources_info"] = []
                    for i, doc in enumerate(response["source_documents"]):
                        source_info = {
                            "source_id": i + 1,
                            "page": doc.metadata.get("page", "Unknown"),
                            "chunk_id": doc.metadata.get("chunk_id", i),
                            "content_length": len(doc.page_content),
                            "preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                        }
                        response["sources_info"].append(source_info)

                logging.info(f"Query processed successfully. Tokens: {cb.total_tokens}, Cost: ${cb.total_cost:.4f}")
                return response

        except Exception as e:
            logging.error(f"Error processing query '{question[:50]}...': {e}")
            raise

    def get_relevant_chunks(self, question: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Get relevant document chunks for inspection

        Args:
            question: User question
            k: Number of chunks to retrieve

        Returns:
            List of dictionaries with chunk information
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        try:
            # Get relevant documents from vector store
            relevant_docs = self.vector_store_manager.similarity_search(question, k=k)

            chunks_info = []
            for i, doc in enumerate(relevant_docs):
                chunk_info = {
                    "rank": i + 1,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "content_length": len(doc.page_content),
                    "word_count": len(doc.page_content.split()),
                    "page": doc.metadata.get("page", "Unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", i),
                    "preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                chunks_info.append(chunk_info)

            logging.info(f"Retrieved {len(chunks_info)} relevant chunks for question")
            return chunks_info

        except Exception as e:
            logging.error(f"Error retrieving relevant chunks: {e}")
            raise

    def update_model_settings(self, model_name: str = None, temperature: float = None):
        """
        Update model settings and reinitialize the chain

        Args:
            model_name: New model name (optional)
            temperature: New temperature (optional)
        """
        try:
            # Update settings if provided
            if model_name is not None:
                self.model_name = model_name
            if temperature is not None:
                self.temperature = temperature

            # Reinitialize LLM with new settings
            self.llm = ChatOpenAI(
                model_name=self.model_name,
                temperature=self.temperature
            )

            # Recreate the chain
            self._setup_chain()

            logging.info(f"Model settings updated: {self.model_name}, temperature: {self.temperature}")

        except Exception as e:
            logging.error(f"Error updating model settings: {e}")
            raise

    def get_chain_info(self) -> Dict[str, Any]:
        """
        Get information about the current chain configuration

        Returns:
            Dictionary with chain configuration details
        """
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "chain_type": "RetrievalQA",
            "retriever_k": 5,
            "vector_store_type": "Pinecone" if self.vector_store_manager.use_pinecone else "FAISS",
            "is_initialized": self.qa_chain is not None
        }