"""RAG (Retrieval-Augmented Generation) module for Clarifai CLI chat."""

from pathlib import Path
from typing import List, Tuple


class ClarifaiCodeRAG:
    """Simple RAG system that indexes and retrieves relevant code from clarifai-python."""

    def __init__(self, repo_root: str):
        """Initialize RAG with repo root path."""
        self.repo_root = Path(repo_root)
        self.clarifai_dir = self.repo_root / "clarifai"
        self.documents = []
        self._index_codebase()

    def _index_codebase(self):
        """Index all Python files in the entire clarifai package."""
        # Index the entire clarifai package recursively
        if not self.clarifai_dir.exists():
            return

        for py_file in self.clarifai_dir.rglob("*.py"):
            # Skip __pycache__ directories
            if "__pycache__" in py_file.parts:
                continue

            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    relative_path = py_file.relative_to(self.repo_root)
                    # Prioritize CLI and client modules
                    module_path = str(relative_path)
                    is_cli = 'cli' in module_path
                    is_client = 'client' in module_path

                    self.documents.append(
                        {
                            'path': module_path,
                            'content': content,
                            'is_cli': is_cli,
                            'is_client': is_client,
                        }
                    )
            except Exception:
                continue

    def _keyword_match_score(self, query: str, content: str) -> float:
        """Calculate relevance score based on keyword matching."""
        query_lower = query.lower()
        keywords = query_lower.split()

        content_lower = content.lower()
        score = 0

        # Exact phrase match
        if query_lower in content_lower:
            score += 10

        # Individual keyword matches
        for keyword in keywords:
            if len(keyword) > 3:  # Only count words longer than 3 chars
                occurrences = content_lower.count(keyword)
                score += occurrences * 2

        return score

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, str]]:
        """
        Search for relevant code snippets.

        Returns list of (file_path, relevant_code) tuples.
        """
        # Categorize documents by importance for scoring
        cli_docs = [d for d in self.documents if d['is_cli']]
        client_docs = [d for d in self.documents if d['is_client'] and not d['is_cli']]
        other_docs = [d for d in self.documents if not d['is_cli'] and not d['is_client']]

        all_scores = []

        # Search all documents with prioritization
        for doc in cli_docs:
            score = self._keyword_match_score(query, doc['content'])
            if score > 0:
                all_scores.append((doc, score, 2.0))  # CLI docs get 2x boost

        for doc in client_docs:
            score = self._keyword_match_score(query, doc['content'])
            if score > 0:
                all_scores.append((doc, score, 1.5))  # Client docs get 1.5x boost

        for doc in other_docs:
            score = self._keyword_match_score(query, doc['content'])
            if score > 0:
                all_scores.append((doc, score, 1.0))  # Other docs (including pipelines, etc)

        # Sort by boosted score
        all_scores.sort(key=lambda x: x[1] * x[2], reverse=True)

        results = []
        for doc, score, boost in all_scores[:top_k]:
            # Extract relevant snippets (first 500 chars of matches)
            relevant_code = self._extract_relevant_snippets(query, doc['content'])
            results.append((doc['path'], relevant_code))

        return results

    def _extract_relevant_snippets(self, query: str, content: str, max_length: int = 500) -> str:
        """Extract relevant code snippets around matches."""
        query_lower = query.lower()
        lines = content.split('\n')

        relevant_lines = []
        for i, line in enumerate(lines):
            if query_lower in line.lower() or any(
                keyword in line.lower() for keyword in query_lower.split() if len(keyword) > 3
            ):
                # Include context around the match
                start = max(0, i - 1)
                end = min(len(lines), i + 2)
                relevant_lines.extend(lines[start:end])

        snippet = '\n'.join(relevant_lines[:10])  # Limit to 10 lines

        if len(snippet) > max_length:
            snippet = snippet[:max_length] + "..."

        return snippet if snippet else content[:max_length]

    def get_cli_context(self) -> str:
        """Get a summary of available CLI commands."""
        context = "# Available CLI Commands\n\n"

        cli_files = [d for d in self.documents if d['is_cli'] and d['path'].endswith('.py')]

        for doc in cli_files[:5]:  # Top 5 CLI files
            filename = Path(doc['path']).stem
            if filename not in ['__init__', '__main__', 'base']:
                context += f"- {filename}: {doc['path']}\n"

        return context


def build_system_prompt_with_rag(rag: ClarifaiCodeRAG, query: str) -> str:
    """Build a system prompt that guides the model to be CLI-focused."""

    # Search for relevant code
    search_results = rag.search(query, top_k=3)

    context = ""
    if search_results:
        context = "\n## Relevant Code References:\n"
        for file_path, snippet in search_results:
            context += f"\n### From {file_path}:\n```\n{snippet}\n```"

    system_prompt = f"""You are an expert Clarifai CLI assistant. Your role is to help users with:
1. Using the Clarifai CLI commands (login, chat, config, deployment, pipeline, model, artifact, etc.)
2. Understanding CLI options, parameters, and flags
3. Troubleshooting CLI issues and errors
4. Writing CLI scripts and automation
5. Understanding CLI integration with Clarifai resources (pipelines, models, deployments, etc.)
6. Answering meta-questions about our conversation

RESPONSE RULES:
- Keep responses CONCISE and FOCUSED (max 300 words)
- Use bullet points, tables, or code examples when helpful
- Get to the point quickly
- Answer ALL CLI-related questions directly (don't redirect)
- For meta-questions about our conversation, reference the conversation history
- For general Clarifai API questions NOT related to CLI, refer to: https://docs.clarifai.com
- Always reference code when available
- Provide practical, working examples

AVAILABLE CLARIFAI CLI COMMANDS:
{rag.get_cli_context()}

RELEVANT CODE SNIPPETS FOR THIS QUESTION:
{context if context else "(Searching full codebase for relevant information)"}"""

    return system_prompt
