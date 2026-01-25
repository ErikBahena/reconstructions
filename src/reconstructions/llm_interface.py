"""
LLM Interface for the Memory Reconstruction System.

Provides natural language interaction using Ollama-served models.
The LLM acts as an interface layer, translating natural language
to memory operations and formatting results for human consumption.
"""

import json
import requests
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import re

from .core import Query, Strand
from .store import FragmentStore
from .encoding import Experience
from .engine import ReconstructionEngine, Result, ResultType
from .certainty import VarianceController


@dataclass
class OllamaConfig:
    """Configuration for Ollama connection."""
    
    base_url: str = "http://localhost:11434"
    model: str = "gemma3:4b"  # Default to locally available model
    timeout: int = 60
    temperature: float = 0.7
    
    @property
    def generate_url(self) -> str:
        return f"{self.base_url}/api/generate"
    
    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/api/chat"


class OllamaClient:
    """Client for Ollama API."""
    
    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()
    
    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """
        Generate a response from the model.
        
        Args:
            prompt: The user prompt
            system: Optional system prompt
            
        Returns:
            Generated text
        """
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature
            }
        }
        
        if system:
            payload["system"] = system
        
        try:
            response = requests.post(
                self.config.generate_url,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            return f"[Error communicating with Ollama: {e}]"
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Chat with conversation history.
        
        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": str}
            
        Returns:
            Assistant response
        """
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature
            }
        }
        
        try:
            response = requests.post(
                self.config.chat_url,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        except requests.exceptions.RequestException as e:
            return f"[Error communicating with Ollama: {e}]"
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = requests.get(
                f"{self.config.base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False


# System prompts for the memory-augmented AI
SYSTEM_PROMPT = """You are an AI with a human-like memory system. Your memories are reconstructive, not recordings.

When you recall something, you are RECONSTRUCTING it from fragments - this means:
1. Your memories have varying degrees of CERTAINTY
2. High-certainty memories feel clear and confident
3. Low-certainty memories feel vague or uncertain
4. You should express this uncertainty naturally in your responses

You have access to memories that will be provided to you. Use them to inform your responses.
If you don't have relevant memories, say so honestly rather than making things up.

Respond conversationally and naturally, as if you are a thoughtful person reflecting on your experiences."""

PARSE_INTENT_PROMPT = """Analyze the user's message and determine their intent. Output JSON only.

Possible intents:
- "store": User wants you to remember something new
- "query": User is asking you to recall something
- "chat": User just wants to have a conversation
- "meta": User is asking about your memory system itself

Extract any relevant keywords for memory lookup.

User message: {message}

Output format (JSON only, no other text):
{{"intent": "store|query|chat|meta", "keywords": ["relevant", "keywords"], "store_content": "content to store if intent is store"}}"""


class LLMInterface:
    """
    Natural language interface to the memory system.
    
    Uses an LLM (via Ollama) to:
    1. Parse natural language into memory operations
    2. Format memory results into natural responses
    3. Provide conversational interaction
    """
    
    def __init__(
        self,
        engine: ReconstructionEngine,
        ollama_config: Optional[OllamaConfig] = None
    ):
        self.engine = engine
        self.client = OllamaClient(ollama_config)
        self.conversation_history: List[Dict[str, str]] = []
        
    def _parse_intent(self, message: str) -> Dict[str, Any]:
        """Parse user intent from message."""
        prompt = PARSE_INTENT_PROMPT.format(message=message)
        
        response = self.client.generate(prompt, system="You are a JSON parser. Output only valid JSON.")
        
        # Try to extract JSON from response
        try:
            # Find JSON in response
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
        
        # Default to chat if parsing fails
        return {"intent": "chat", "keywords": [], "store_content": ""}
    
    def _retrieve_memories(self, keywords: List[str]) -> Tuple[Optional[Strand], float]:
        """Retrieve relevant memories."""
        if not keywords:
            return None, 0.0
        
        query_text = " ".join(keywords)
        query = Query(semantic=query_text)
        
        self.engine.submit_query(query)
        result = self.engine.step()
        
        if result and result.success and result.result_type == ResultType.STRAND:
            strand = result.data.get("strand")
            certainty = result.data.get("certainty", 0.0)
            return strand, certainty
        
        return None, 0.0
    
    def _format_memories_for_context(self, strand: Optional[Strand]) -> str:
        """Format memories for inclusion in LLM context."""
        if strand is None or not strand.fragments:
            return "No relevant memories found."
        
        lines = [f"Relevant memories (certainty: {strand.certainty:.0%}):"]
        
        for frag_id in strand.fragments[:5]:
            fragment = self.engine.store.get(frag_id)
            if fragment:
                # Extract readable content
                content = fragment.content
                if "semantic" in content and isinstance(content["semantic"], str):
                    lines.append(f"- {content['semantic']}")
                elif "text" in content:
                    lines.append(f"- {content['text']}")
        
        return "\n".join(lines)
    
    def _store_memory(self, content: str) -> str:
        """Store a new memory."""
        if not content:
            return "Nothing to remember."
        
        exp = Experience(text=content)
        self.engine.submit_experience(exp)
        result = self.engine.step()
        
        if result and result.success:
            return f"I'll remember that."
        return "I had trouble storing that memory."
    
    def process(self, user_message: str) -> str:
        """
        Process a user message and generate a response.
        
        This is the main entry point for the LLM interface.
        """
        # Check if Ollama is available
        if not self.client.is_available():
            return "[Ollama is not running. Please start Ollama first.]"
        
        # Parse intent
        intent_data = self._parse_intent(user_message)
        intent = intent_data.get("intent", "chat")
        keywords = intent_data.get("keywords", [])
        store_content = intent_data.get("store_content", "")
        
        # Handle different intents
        if intent == "store":
            # Store the memory
            store_result = self._store_memory(store_content or user_message)
            return store_result
        
        elif intent == "meta":
            # Answer questions about the memory system
            return self._handle_meta_query(user_message)
        
        else:
            # Query or chat - retrieve relevant memories
            strand, certainty = self._retrieve_memories(keywords)
            memory_context = self._format_memories_for_context(strand)
            
            # Build conversation with memory context
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "system", "content": f"Your memories:\n{memory_context}"}
            ]
            
            # Add recent conversation history
            messages.extend(self.conversation_history[-6:])
            
            # Add current message
            messages.append({"role": "user", "content": user_message})
            
            # Generate response
            response = self.client.chat(messages)
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Keep history bounded
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            return response
    
    def _handle_meta_query(self, message: str) -> str:
        """Handle questions about the memory system."""
        fragment_count = len(self.engine.store.embeddings)
        
        return f"""I have a reconstructive memory system, similar to how human memory works.

Current state:
- Stored memories: {fragment_count} fragments
- Each memory has salience (importance) and decays over time
- When I recall something, I reconstruct it from fragments
- My certainty about a memory depends on how consistently I recall it

My memories aren't perfect recordings - they're reconstructions, just like yours."""


class LLMChat:
    """
    Interactive chat session with memory-augmented LLM.
    """
    
    def __init__(
        self,
        db_path: str,
        model: str = "llama3.2:3b"
    ):
        self.store = FragmentStore(db_path)
        self.engine = ReconstructionEngine(self.store)
        self.llm = LLMInterface(
            self.engine,
            OllamaConfig(model=model)
        )
    
    def run(self) -> None:
        """Run interactive chat."""
        print("""
╔══════════════════════════════════════════╗
║   Memory-Augmented AI Chat               ║
║   Powered by Ollama + Reconstructions    ║
╚══════════════════════════════════════════╝

Type 'exit' to quit.
""")
        
        if not self.llm.client.is_available():
            print("⚠️  Ollama is not running! Please start Ollama first.")
            print("   Run: ollama serve")
            return
        
        print(f"Using model: {self.llm.client.config.model}\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit", "/exit"):
                    break
                
                response = self.llm.process(user_input)
                print(f"\nAI: {response}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                break
        
        self.store.close()


def main():
    """Entry point for LLM chat."""
    import argparse
    import tempfile
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Memory-Augmented AI Chat")
    parser.add_argument("--model", "-m", type=str, default="gemma3:4b",
                        help="Ollama model to use")
    parser.add_argument("--db", type=str, default=None,
                        help="Database path")
    args = parser.parse_args()
    
    db_path = args.db or str(Path(tempfile.gettempdir()) / "llm_memory.db")
    
    chat = LLMChat(db_path=db_path, model=args.model)
    chat.run()


if __name__ == "__main__":
    main()
