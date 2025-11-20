"""
RAG Service - Retrieval-Augmented Generation
Core RAG pipeline implementation
"""

import re
from openai import OpenAI
from typing import List, Dict, Any, Optional
import logging
import numpy as np
from services.vector_store import VectorStore
from services.query_processor import QueryProcessor
from config import settings

logger = logging.getLogger(__name__)


class RAGService:
    """RAG service for query processing and response generation"""
    
    STOPWORDS = {
        "the", "and", "for", "with", "from", "that", "this", "about", "your",
        "what", "when", "where", "have", "will", "you", "are", "how", "much",
        "does", "can", "who", "whose", "into", "than", "then", "they", "them",
        "into", "onto", "ours", "ourselves", "themselves"
    }
    
    def __init__(
        self,
        vector_store: VectorStore,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize RAG service
        
        Args:
            vector_store: Vector store instance
            openai_api_key: OpenAI API key (optional)
        """
        self.vector_store = vector_store
        self.openai_api_key = openai_api_key or settings.OPENAI_API_KEY
        self.query_processor = QueryProcessor()
        
        # Initialize OpenAI client
        if self.openai_api_key and self.openai_api_key != "your_openai_api_key_here":
            self.openai_client = OpenAI(api_key=self.openai_api_key)
            logger.info("OpenAI client initialized")
        else:
            self.openai_client = None
            logger.warning("OpenAI API key not provided, using simulated embeddings")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        if not self.openai_client:
            # Generate random embedding for testing (3072 dimensions for text-embedding-3-large)
            embedding = np.random.rand(3072).tolist()
            # Normalize
            norm = np.linalg.norm(embedding)
            return (np.array(embedding) / norm).tolist()
        
        try:
            response = self.openai_client.embeddings.create(
                model=settings.OPENAI_EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Fallback to random embedding
            embedding = np.random.rand(3072).tolist()
            norm = np.linalg.norm(embedding)
            return (np.array(embedding) / norm).tolist()
    
    def retrieve_context(
        self,
        query: str,
        top_k: int = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        use_query_processing: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for query
        
        Args:
            query: User query
            top_k: Number of results to retrieve
            filter_metadata: Optional metadata filters
            use_query_processing: Whether to use query processing enhancements
            
        Returns:
            List of relevant documents
        """
        top_k = top_k or settings.TOP_K_RESULTS
        
        # Process query if enabled
        processed_query = None
        if use_query_processing:
            processed_query = self.query_processor.process_query(query)
            query = processed_query['processed_query']
            
            # Use metadata filters from query processing if not provided
            if filter_metadata is None:
                filter_metadata = self.query_processor.get_metadata_filters(processed_query)
            
            logger.info(f"Query intent: {processed_query['intent']['intent']} "
                      f"(confidence: {processed_query['intent']['confidence']:.2f})")
        
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2 if use_query_processing else top_k,  # Retrieve more for re-ranking
            filter_metadata=filter_metadata
        )
        
        # Re-rank results if query processing is enabled
        if use_query_processing and processed_query:
            results = self._rerank_results(results, processed_query)
        
        # Filter by similarity threshold (lower for simulated embeddings)
        threshold = settings.SIMILARITY_THRESHOLD
        if not self.openai_client:
            # Lower threshold for simulated embeddings (random won't match well)
            threshold = 0.0  # Accept all results when using simulated embeddings
        
        filtered_results = [
            r for r in results
            if r.get('similarity', 0) >= threshold
        ]
        
        # If no results meet threshold, lower it and try again (be more lenient)
        if not filtered_results and results:
            logger.info(f"No results above threshold {threshold}, using top results with lower threshold")
            threshold = max(0.3, threshold * 0.5)  # Lower threshold to 50% of original, min 0.3
            filtered_results = [
                r for r in results
                if r.get('similarity', 0) >= threshold
            ]
        
        # Deduplicate results (same source URL)
        filtered_results = self._deduplicate_results(filtered_results)
        
        # Limit to top_k, but ensure we return at least some results if available
        filtered_results = filtered_results[:top_k]
        
        # If still no results but we have some from search, return top 3 anyway
        if not filtered_results and results:
            logger.info("No results after filtering, returning top 3 results anyway")
            filtered_results = self._deduplicate_results(results[:3])
        
        logger.info(f"Retrieved {len(filtered_results)} relevant documents (threshold: {threshold})")
        return filtered_results
    
    def _rerank_results(
        self,
        results: List[Dict[str, Any]],
        processed_query: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Re-rank results based on query intent and entities
        
        Args:
            results: Retrieved results
            processed_query: Processed query information
            
        Returns:
            Re-ranked results
        """
        intent = processed_query['intent']['intent']
        entities = processed_query['entities']
        
        # Score each result
        scored_results = []
        for result in results:
            score = result.get('similarity', 0)
            metadata = result.get('metadata', {})
            content = result.get('content', '').lower()
            
            # Boost score based on intent matching
            if intent == 'course_inquiry':
                if 'course' in content or 'program' in content:
                    score *= 1.2
            elif intent == 'admission':
                if any(word in content for word in ['requirement', 'entry', 'admission', 'apply']):
                    score *= 1.2
            elif intent == 'support':
                if any(word in content for word in ['support', 'service', 'help', 'library']):
                    score *= 1.2
            
            # Boost score based on entity matching
            course_levels = entities.get('course_level', [])
            if course_levels:
                for level in course_levels:
                    if level in content:
                        score *= 1.15
            
            subjects = entities.get('subject', [])
            if subjects:
                for subject in subjects:
                    if subject in content:
                        score *= 1.15
            
            scored_results.append({
                **result,
                'similarity': score
            })
        
        # Sort by new score
        scored_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        return scored_results
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate results (same source URL)
        
        Args:
            results: List of results
            
        Returns:
            Deduplicated results
        """
        seen_urls = set()
        unique_results = []
        
        for result in results:
            url = result.get('metadata', {}).get('source_url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
            elif not url:  # Keep results without URLs
                unique_results.append(result)
        
        return unique_results
    
    def build_prompt(
        self,
        query: str,
        context: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        query_info: Optional[Dict[str, Any]] = None,
        selected_sources: Optional[List[str]] = None
    ) -> str:
        """
        Build prompt for LLM
        
        Args:
            query: User query
            context: Retrieved context documents
            conversation_history: Previous conversation messages
            query_info: Optional query processing information
            
        Returns:
            Formatted prompt
        """
        # Enhanced system prompt based on query intent
        base_system_prompt = """You are a helpful assistant for Dublin Business School (DBS). 
Your role is to answer questions about DBS courses, admissions, campus life, and student support 
using the provided context from the DBS website.

Response format (always follow this):
1. **Context Summary** – 2–3 sentences that recap the most relevant facts found in the context.
2. **Key Details** – Bullet list with specific facts (programme names, requirements, deadlines, services, etc.). ONLY cite sources that are explicitly provided below with [Source X] format. If no sources are provided in the context, do NOT use [Source X] citations.
3. **Recommended Actions** – Bullet list describing what the user should do next (apply, contact admissions, review requirements, etc.). Mention if more information is required and point them to DBS contact options.

Note: Do NOT include a "Sources" section in your response. Source links are handled separately by the system.

Guidelines:
- Answer based on the provided context, even if it's not a perfect match
- Use the context to provide the best possible answer about DBS
- If the context doesn't directly answer the question, provide related information from the context that might be helpful
- Only say "I don't have information about that" if the context is completely irrelevant or empty
- Be friendly, professional, and accurate
- Provide specific details when available
- Cite sources when providing specific information
- Never fabricate programme names, requirements, or URLs
"""
        
        # Add intent-specific guidance
        if query_info and query_info.get('intent'):
            intent = query_info['intent']
            if intent == 'course_inquiry':
                base_system_prompt += "\n- For course inquiries, include: course name, duration, level, key modules, and entry requirements if available"
            elif intent == 'admission':
                base_system_prompt += "\n- For admission questions, provide specific requirements, deadlines, and application process details"
            elif intent == 'support':
                base_system_prompt += "\n- For support services, explain what services are available and how to access them"
        
        system_prompt = base_system_prompt + "\n\nContext Information:\n"
        
        # Filter context to only include documents from selected_sources
        valid_sources = set(selected_sources or [])
        filtered_context = []
        source_index_map = {}  # Map URLs to source numbers
        
        if selected_sources:
            # Only include context documents whose URLs are in selected_sources
            for doc in context:
                metadata = doc.get('metadata', {})
                source_url = metadata.get('source_url') or metadata.get('parent_url') or ''
                if source_url in valid_sources:
                    filtered_context.append(doc)
                    # Assign source number based on order in selected_sources
                    if source_url not in source_index_map:
                        source_index_map[source_url] = len(source_index_map) + 1
        else:
            # If no selected sources, don't include any context citations
            filtered_context = context
        
        # Add context with relevance indicators
        context_text = ""
        if selected_sources and len(selected_sources) > 0:
            # Only number sources that are actually selected
            for doc in filtered_context:
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})
                source_url = metadata.get('source_url') or metadata.get('parent_url') or ''
                title = metadata.get('title', 'Untitled')
                similarity = doc.get('similarity', 0)
                
                # Get source number from map
                source_num = source_index_map.get(source_url, 0)
                if source_num == 0:
                    continue
                
                # Truncate content if too long (keep within token limits)
                max_content_length = settings.MAX_CONTEXT_LENGTH // max(len(filtered_context), 1)
                if len(content) > max_content_length:
                    content = content[:max_content_length] + "..."
                
                context_text += f"\n[Source {source_num} - {title}]\n"
                context_text += f"URL: {source_url}\n"
                if similarity > 0:
                    context_text += f"Relevance: {similarity:.2f}\n"
                context_text += f"Content: {content}\n"
        else:
            # No selected sources - just provide context without citations
            for doc in filtered_context:
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})
                similarity = doc.get('similarity', 0)
                
                # Truncate content if too long
                max_content_length = settings.MAX_CONTEXT_LENGTH // max(len(filtered_context), 1)
                if len(content) > max_content_length:
                    content = content[:max_content_length] + "..."
                
                context_text += f"\nContent:\n"
                if similarity > 0:
                    context_text += f"Relevance: {similarity:.2f}\n"
                context_text += f"{content}\n"
        
        # Add conversation history if available
        history_text = ""
        if conversation_history:
            history_text = "\n\nPrevious Conversation:\n"
            for msg in conversation_history[-3:]:  # Last 3 messages
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                history_text += f"{role.capitalize()}: {content}\n"
        
        # Note: Sources are handled separately by the system, no need to include in prompt
        
        # Build final prompt
        prompt = f"""{system_prompt}{context_text}{history_text}

User Question: {query}

Please provide a helpful and accurate answer based on the context above:"""
        
        return prompt
    
    def generate_response(
        self,
        query: str,
        context: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
        query_info: Optional[Dict[str, Any]] = None,
        selected_sources: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate response using LLM
        
        Args:
            query: User query
            context: Retrieved context
            conversation_history: Previous conversation
            stream: Whether to stream response
            query_info: Optional query processing information
            
        Returns:
            Generated response with metadata
        """
        # Validate context
        if not context:
            return {
                "response": "I couldn't find relevant information to answer your question. Please try rephrasing or ask about DBS courses, admissions, or student support.",
                "sources": [],
                "model": "no_context",
                "tokens_used": 0
            }
        
        # Build prompt
        prompt = self.build_prompt(query, context, conversation_history, query_info, selected_sources)
        
        if not self.openai_client:
            # Simulated response for testing
            return {
                "response": "I'm a simulated response. Please provide your OpenAI API key to enable full functionality.",
                "sources": selected_sources or [
                    doc.get('metadata', {}).get('source_url', '')
                    for doc in context
                    if doc.get('metadata', {}).get('source_url', '')
                ],
                "model": "simulated",
                "tokens_used": 0
            }
        
        try:
            # Generate response
            response = self.openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for Dublin Business School."},
                    {"role": "user", "content": prompt}
                ],
                temperature=settings.OPENAI_TEMPERATURE,
                max_tokens=settings.OPENAI_MAX_TOKENS,
                stream=stream
            )
            
            if stream:
                # Handle streaming response
                return {"stream": response}
            else:
                # Handle regular response
                return {
                    "response": response.choices[0].message.content,
                    "sources": selected_sources or [
                        doc.get('metadata', {}).get('source_url', '')
                        for doc in context
                        if doc.get('metadata', {}).get('source_url', '')
                    ],
                    "model": settings.OPENAI_MODEL,
                    "tokens_used": response.usage.total_tokens if response.usage else 0
                }
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "response": f"I apologize, but I encountered an error: {str(e)}",
                "sources": [],
                "model": "error",
                "tokens_used": 0
            }
    
    def process_query(
        self,
        query: str,
        top_k: int = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
        use_query_processing: bool = True
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve + generate
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            filter_metadata: Optional metadata filters
            conversation_history: Previous conversation
            stream: Whether to stream response
            use_query_processing: Whether to use query processing enhancements
            
        Returns:
            Complete response with context and sources
        """
        # Process query for intent and entities
        processed_query = None
        if use_query_processing:
            processed_query = self.query_processor.process_query(query)
            logger.info(f"Processed query - Intent: {processed_query['intent']['intent']}, "
                       f"Entities: {processed_query['entities']}")
        
        # Step 1: Retrieve relevant context
        context = self.retrieve_context(
            query=query,
            top_k=top_k,
            filter_metadata=filter_metadata,
            use_query_processing=use_query_processing
        )
        
        if not context:
            result = {
                "response": "I couldn't find relevant information to answer your question. Please try rephrasing or ask about DBS courses, admissions, or student support.",
                "sources": [],
                "context": [],
                "model": "no_context"
            }
            # Add query processing info even when no context
            if processed_query:
                result["query_info"] = {
                    "intent": processed_query['intent']['intent'],
                    "confidence": processed_query['intent']['confidence'],
                    "entities": processed_query['entities']
                }
            return result
        
        # Step 2: Prepare dynamic sources
        selected_sources = self._select_sources(
            context=context,
            original_query=query,
            processed_query=processed_query
        )
        
        # Step 3: Generate response
        query_info_for_response = None
        if processed_query:
            query_info_for_response = {
                "intent": processed_query['intent']['intent'],
                "confidence": processed_query['intent']['confidence']
            }
        
        result = self.generate_response(
            query=query,
            context=context,
            conversation_history=conversation_history,
            stream=stream,
            query_info=query_info_for_response,
            selected_sources=selected_sources
        )
        
        # Add context to result
        result["context"] = []
        for doc in context:
            metadata = doc.get('metadata', {}) or {}
            source_url = metadata.get('source_url') or metadata.get('parent_url') or ''
            snippet = doc.get('content', '')[:200]
            if snippet:
                snippet = snippet.strip() + ("..." if len(doc.get('content', '')) > 200 else "")
            result["context"].append({
                "content": snippet or "",
                "source": source_url,
                "similarity": doc.get('similarity', 0)
            })
        
        # Add query processing info
        if processed_query:
            result["query_info"] = {
                "intent": processed_query['intent']['intent'],
                "confidence": processed_query['intent']['confidence'],
                "entities": processed_query['entities']
            }
        
        return result

    def _select_sources(
        self,
        context: List[Dict[str, Any]],
        original_query: str,
        processed_query: Optional[Dict[str, Any]] = None,
        max_sources: int = 3
    ) -> List[str]:
        """
        Select the most relevant sources based on similarity and query coverage.
        """
        if not context:
            return []
        
        query_terms = self._extract_query_terms(original_query)
        grouped_sources: Dict[str, Dict[str, float]] = {}
        
        for doc in context:
            metadata = doc.get("metadata", {})
            url = metadata.get("source_url") or metadata.get("parent_url")
            if not url:
                continue
            
            content = (doc.get("content") or "").lower()
            similarity = doc.get("similarity") or 0.0
            
            coverage = 0.0
            if query_terms:
                hits = sum(1 for term in query_terms if term in content)
                coverage = hits / len(query_terms)
            
            # Intent-aware boost (e.g., prefer course pages for course intent)
            intent = processed_query['intent']['intent'] if processed_query else None
            category = (metadata.get("category") or "").lower()
            if intent == "course_inquiry" and "course" in category:
                similarity *= 1.1
            elif intent == "admission" and "admission" in category:
                similarity *= 1.1
            elif intent == "support" and any(word in category for word in ["support", "service"]):
                similarity *= 1.1
            
            relevance = (similarity * 0.7) + (coverage * 0.3)
            
            if url not in grouped_sources or relevance > grouped_sources[url]["relevance"]:
                grouped_sources[url] = {
                    "relevance": relevance,
                    "similarity": similarity,
                    "coverage": coverage
                }
        
        if not grouped_sources:
            return []
        
        # Determine adaptive thresholds
        best_relevance = max(entry["relevance"] for entry in grouped_sources.values())
        min_relevance = max(0.35, best_relevance * 0.55)
        min_similarity = max(0.5, settings.SIMILARITY_THRESHOLD * 0.9)
        
        sorted_sources = sorted(
            grouped_sources.items(),
            key=lambda item: item[1]["relevance"],
            reverse=True
        )
        
        filtered_urls = [
            url for url, data in sorted_sources
            if data["relevance"] >= min_relevance and data["similarity"] >= min_similarity
        ]
        
        if not filtered_urls and sorted_sources:
            # Fallback to the single best source to avoid empty citations
            filtered_urls = [sorted_sources[0][0]]
        
        # Preserve order, limit to max_sources
        unique_urls = []
        for url in filtered_urls:
            if url not in unique_urls:
                unique_urls.append(url)
            if len(unique_urls) >= max_sources:
                break
        
        return unique_urls

    def _extract_query_terms(self, query: str) -> set:
        """
        Extract meaningful terms from the original query for coverage checks.
        """
        tokens = re.findall(r"\b\w+\b", query.lower())
        return {token for token in tokens if len(token) > 3 and token not in self.STOPWORDS}

    def _has_sufficient_context(
        self,
        query: str,
        context: List[Dict[str, Any]],
        min_coverage: float = 0.4
    ) -> bool:
        """
        Check whether retrieved context adequately covers the query terms.
        """
        terms = self._extract_query_terms(query)
        if not terms:
            return True
        
        matched_terms = set()
        for term in terms:
            for doc in context:
                content = (doc.get("content") or "").lower()
                if term in content:
                    matched_terms.add(term)
                    break
        
        coverage = len(matched_terms) / len(terms)
        logger.info(f"Context coverage for query '{query}': {coverage:.2f}")
        return coverage >= min_coverage

