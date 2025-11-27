"""llm interaction using google gemini for plagiarism analysis"""

import google.generativeai as genai
from typing import List, Dict
import json
import time
from .config import GEMINI_API_KEY, GEMINI_MODEL
from .chunking import CodeChunk


class GeminiLLM:
    """wrapper for gemini api calls"""

    def __init__(self, model_name: str = GEMINI_MODEL, temperature: float = 0.0):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(model_name)
        self.temperature = temperature
        self.generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=2048,
        )

    def generate(self, prompt: str, retry_count: int = 3) -> str:
        """sends prompt to gemini and returns response text"""
        for attempt in range(retry_count):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config
                )
                return response.text
            except Exception as e:
                if attempt < retry_count - 1:
                    print(f"generation failed (attempt {attempt + 1}): {e}")
                    time.sleep(2 ** attempt)
                else:
                    raise

    def analyze_plagiarism_with_context(
        self,
        query_code: str,
        context_functions: List[CodeChunk],
        return_json: bool = True
    ) -> Dict:
        """checks if query code is plagiarized from retrieved context functions"""
        context_str = self._build_context(context_functions)

        prompt = f"""You are a code plagiarism detection expert. Analyze if the QUERY CODE is plagiarized from any code in the REFERENCE FUNCTIONS below.

REFERENCE FUNCTIONS:
{context_str}

QUERY CODE:
```python
{query_code}
```

Analyze the code and determine:
1. Is the query code plagiarized from any reference function?
2. If yes, which function(s) is it most similar to?
3. What transformations were applied (e.g., variable renaming, refactoring)?
4. Confidence score (0-100)

Respond in the following JSON format:
{{
    "is_plagiarism": true/false,
    "confidence": 0-100,
    "matched_function": "function_name or null",
    "matched_file": "file_path or null",
    "reasoning": "brief explanation",
    "transformations": ["list of transformations detected"]
}}
"""

        response = self.generate(prompt)

        if return_json:
            return self._parse_json_response(response)
        return {"raw_response": response}

    def analyze_plagiarism_direct(self, query_code: str) -> Dict:
        """zero-shot analysis without any reference corpus - just looks for suspicious patterns"""
        prompt = f"""You are a code plagiarism detection expert. Analyze if the following code shows signs of plagiarism WITHOUT having access to any reference corpus.

Look for indicators like:
- Generic variable names (e.g., x, y, temp)
- Common algorithmic patterns
- Standard library usage
- Typical code structure

If the code appears original and well-crafted, mark as NOT plagiarism.
If it shows suspicious patterns (overly generic, seems copied from tutorials), mark as possible plagiarism.

CODE TO ANALYZE:
```python
{query_code}
```

Respond in JSON format:
{{
    "is_plagiarism": true/false,
    "confidence": 0-100,
    "reasoning": "brief explanation",
    "suspicious_patterns": ["list of suspicious patterns if any"]
}}
"""

        response = self.generate(prompt)
        result = self._parse_json_response(response)

        # fill in missing fields for consistency
        result.setdefault('matched_function', None)
        result.setdefault('matched_file', None)
        result.setdefault('transformations', result.get('suspicious_patterns', []))

        return result

    def analyze_plagiarism_with_full_corpus(
        self,
        query_code: str,
        full_corpus: List[CodeChunk]
    ) -> Dict:
        """tries to fit entire corpus in context - might hit token limits"""
        return self.analyze_plagiarism_with_context(query_code, full_corpus)

    def _build_context(self, functions: List[CodeChunk], max_functions: int = 20) -> str:
        """formats code chunks into a readable context string"""
        parts = []

        for i, func in enumerate(functions[:max_functions]):
            parts.append(
                f"--- function {i+1}: {func.function_name} (from {func.file_path}) ---\n"
                f"```python\n{func.content}\n```\n"
            )

        if len(functions) > max_functions:
            parts.append(f"\n... and {len(functions) - max_functions} more functions ...")

        return "\n".join(parts)

    def _parse_json_response(self, response: str) -> Dict:
        """extracts json from llm response, handles markdown code blocks"""
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)

            return {
                "is_plagiarism": False,
                "confidence": 0,
                "matched_function": None,
                "matched_file": None,
                "reasoning": "failed to parse json response",
                "transformations": [],
                "raw_response": response
            }
        except json.JSONDecodeError:
            return {
                "is_plagiarism": False,
                "confidence": 0,
                "matched_function": None,
                "matched_file": None,
                "reasoning": "json parsing failed",
                "transformations": [],
                "raw_response": response
            }
