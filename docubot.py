"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import glob
import re

# Common function words that appear everywhere and carry no signal for matching
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "on", "at", "by", "for", "with", "about",
    "from", "into", "through", "during", "before", "after", "above",
    "below", "between", "and", "or", "but", "if", "so", "yet", "both",
    "not", "no", "nor", "as", "up", "out", "any", "these", "those",
    "this", "that", "it", "its", "i", "you", "he", "she", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "our", "their",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "there", "here", "all", "each", "every", "some", "such", "than",
    "then", "also", "more", "most", "doc", "docs", "document", "documents",
    "mention", "mentions", "mentioned",
}

class DocuBot:
    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Build a retrieval index (implemented in Phase 1)
        self.index = self.build_index(self.documents)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        return docs

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def build_index(self, documents):
        """
        Build an inverted index mapping lowercase words to the document
        filenames that contain them.

        Structure:
        {
            "token": ["AUTH.md", "API_REFERENCE.md"],
            "database": ["DATABASE.md"]
        }
        """
        index = {}
        for filename, text in documents:
            words = set(re.findall(r'[a-z]+', text.lower())) - STOPWORDS
            for word in words:
                if word not in index:
                    index[word] = []
                index[word].append(filename)
        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """
        Return a relevance score: total count of query word occurrences in text.
        More occurrences = higher score, giving longer matching passages a boost.
        """
        query_words = [w for w in re.findall(r'[a-z]+', query.lower()) if w not in STOPWORDS]
        text_lower = text.lower()
        return sum(len(re.findall(r'\b' + re.escape(word) + r'\b', text_lower)) for word in query_words)

    def retrieve(self, query, top_k=3):
        """
        Split each document into paragraphs, score each paragraph against
        the query, and return the top_k highest-scoring (filename, paragraph)
        pairs sorted by score descending.

        Returns an empty list when no paragraph scores above zero — this
        triggers the "I do not know" guardrail in both answer modes.
        """
        # Use the index to find candidate filenames containing any query word
        query_words = set(re.findall(r'[a-z]+', query.lower())) - STOPWORDS
        candidate_files = set()
        for word in query_words:
            for filename in self.index.get(word, []):
                candidate_files.add(filename)

        # Score at paragraph level so snippets are focused, not whole files
        scored = []
        for filename, text in self.documents:
            if filename not in candidate_files:
                continue
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            for para in paragraphs:
                score = self.score_document(query, para)
                if score > 0:
                    scored.append((score, filename, para))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [(filename, para) for _, filename, para in scored[:top_k]]

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode.
        Returns raw snippets and filenames with no LLM involved.
        """
        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        formatted = []
        for filename, text in snippets:
            formatted.append(f"[{filename}]\n{text}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
