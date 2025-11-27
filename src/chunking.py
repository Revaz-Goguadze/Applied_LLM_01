"""extracts functions from python files using ast parsing"""

import ast
from typing import List, Dict, Optional
from dataclasses import dataclass
import os


@dataclass
class CodeChunk:
    """represents a single function extracted from source code"""
    content: str
    function_name: str
    file_path: str
    start_line: int
    end_line: int
    docstring: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'content': self.content,
            'function_name': self.function_name,
            'file_path': self.file_path,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'docstring': self.docstring
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'CodeChunk':
        return cls(**data)

    def __hash__(self):
        return hash((self.file_path, self.function_name, self.start_line))

    def __eq__(self, other):
        if not isinstance(other, CodeChunk):
            return False
        return (self.file_path == other.file_path and
                self.function_name == other.function_name and
                self.start_line == other.start_line)


class PythonFunctionExtractor:
    """walks through python files and pulls out function definitions"""

    def __init__(self, min_lines: int = 3, max_lines: int = 500):
        self.min_lines = min_lines
        self.max_lines = max_lines

    def extract_functions_from_file(self, file_path: str) -> List[CodeChunk]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
        except Exception as e:
            print(f"couldn't read {file_path}: {e}")
            return []

        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            print(f"syntax error in {file_path}: {e}")
            return []

        functions = []
        source_lines = source_code.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                chunk = self._extract_function_node(node, source_lines, file_path)
                if chunk:
                    functions.append(chunk)

        return functions

    def _extract_function_node(
        self,
        node: ast.FunctionDef,
        source_lines: List[str],
        file_path: str
    ) -> Optional[CodeChunk]:
        start_line = node.lineno - 1
        end_line = node.end_lineno

        docstring = ast.get_docstring(node)

        function_lines = source_lines[start_line:end_line]
        num_lines = len(function_lines)

        # skip tiny functions
        if num_lines < self.min_lines:
            return None

        # chop off huge ones
        if num_lines > self.max_lines:
            function_lines = function_lines[:self.max_lines]
            end_line = start_line + self.max_lines

        content = '\n'.join(function_lines)

        return CodeChunk(
            content=content,
            function_name=node.name,
            file_path=file_path,
            start_line=start_line + 1,
            end_line=end_line,
            docstring=docstring
        )

    def extract_from_directory(self, directory: str) -> List[CodeChunk]:
        """recursively grabs all functions from .py files in a folder"""
        all_chunks = []

        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    chunks = self.extract_functions_from_file(file_path)
                    all_chunks.extend(chunks)

        return all_chunks


def normalize_code(code: str) -> str:
    """strips comments and extra whitespace for cleaner comparison"""
    lines = []
    for line in code.split('\n'):
        if '#' in line:
            line = line[:line.index('#')]
        line = line.rstrip()
        if line:
            lines.append(line)

    return '\n'.join(lines)
