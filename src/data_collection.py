"""Data collection utilities for GitHub repositories."""

import requests
import os
import json
from typing import List, Dict, Optional
import time
from pathlib import Path


class GitHubCollector:
    """Collect Python repositories from GitHub."""

    def __init__(self, output_dir: str = "data/reference_corpus"):
        """
        Initialize GitHub collector.

        Args:
            output_dir: Directory to save collected repositories
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def clone_repository(self, repo_url: str, local_name: Optional[str] = None) -> str:
        """
        Clone a GitHub repository.

        Args:
            repo_url: GitHub repository URL
            local_name: Optional local directory name

        Returns:
            Path to cloned repository
        """
        if local_name is None:
            local_name = repo_url.split('/')[-1].replace('.git', '')

        local_path = os.path.join(self.output_dir, local_name)

        if os.path.exists(local_path):
            print(f"Repository already exists: {local_path}")
            return local_path

        print(f"Cloning {repo_url} to {local_path}...")
        os.system(f"git clone {repo_url} {local_path}")

        return local_path

    def download_file_from_url(self, url: str, save_path: str):
        """
        Download a single file from a URL.

        Args:
            url: URL to download from
            save_path: Local path to save the file
        """
        response = requests.get(url)
        response.raise_for_status()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

        print(f"Downloaded: {save_path}")


class DatasetConstructor:
    """Construct labeled test dataset for plagiarism detection."""

    def __init__(self, output_file: str = "data/test_dataset.json"):
        """
        Initialize dataset constructor.

        Args:
            output_file: Path to save the dataset
        """
        self.output_file = output_file
        self.test_cases = []

    def add_test_case(
        self,
        code: str,
        is_plagiarism: bool,
        source_file: Optional[str] = None,
        source_function: Optional[str] = None,
        transformation_type: Optional[str] = None,
        description: Optional[str] = None
    ):
        """
        Add a test case to the dataset.

        Args:
            code: The code snippet to test
            is_plagiarism: True if this is plagiarized code
            source_file: Source file if plagiarized
            source_function: Source function name if plagiarized
            transformation_type: Type of transformation applied (for positive cases)
            description: Human-readable description
        """
        test_case = {
            'id': len(self.test_cases),
            'code': code,
            'is_plagiarism': is_plagiarism,
            'source_file': source_file,
            'source_function': source_function,
            'transformation_type': transformation_type,
            'description': description
        }

        self.test_cases.append(test_case)

    def save(self):
        """Save dataset to JSON file."""
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_cases, f, indent=2, ensure_ascii=False)

        print(f"Dataset saved to: {self.output_file}")
        print(f"Total test cases: {len(self.test_cases)}")
        print(f"Positive examples: {sum(1 for tc in self.test_cases if tc['is_plagiarism'])}")
        print(f"Negative examples: {sum(1 for tc in self.test_cases if not tc['is_plagiarism'])}")

    def load(self):
        """Load dataset from JSON file."""
        with open(self.output_file, 'r', encoding='utf-8') as f:
            self.test_cases = json.load(f)

        print(f"Dataset loaded: {len(self.test_cases)} test cases")


class CodeTransformer:
    """Apply transformations to code for creating positive examples."""

    @staticmethod
    def rename_variables(code: str, mapping: Optional[Dict[str, str]] = None) -> str:
        """
        Rename variables in code.

        Args:
            code: Original code
            mapping: Dictionary of old_name -> new_name. If None, use generic mapping.

        Returns:
            Transformed code
        """
        if mapping is None:
            # Default generic mapping
            mapping = {
                'data': 'values',
                'result': 'output',
                'temp': 'tmp',
                'i': 'index',
                'j': 'idx',
                'n': 'count',
                'x': 'item',
                'y': 'element'
            }

        transformed = code
        for old, new in mapping.items():
            # Simple string replacement (not perfect but works for simple cases)
            import re
            # Replace whole words only
            transformed = re.sub(r'\b' + old + r'\b', new, transformed)

        return transformed

    @staticmethod
    def remove_comments(code: str) -> str:
        """
        Remove all comments from code.

        Args:
            code: Original code

        Returns:
            Code without comments
        """
        lines = []
        in_multiline_string = False

        for line in code.split('\n'):
            stripped = line.lstrip()

            # Handle docstrings
            if '"""' in line or "'''" in line:
                in_multiline_string = not in_multiline_string

            # Remove single-line comments if not in docstring
            if not in_multiline_string and '#' in line:
                line = line[:line.index('#')].rstrip()

            if line.strip():  # Keep non-empty lines
                lines.append(line)

        return '\n'.join(lines)

    @staticmethod
    def reorder_statements(code: str) -> str:
        """
        Reorder independent statements (simple version).

        Args:
            code: Original code

        Returns:
            Code with reordered statements
        """
        # This is a simplified version - just reverses line order within function body
        lines = code.split('\n')

        # Find function definition line
        def_line_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                def_line_idx = i
                break

        # Reverse lines after function definition (excluding docstring)
        body_start = def_line_idx + 1

        # Skip docstring if present
        if body_start < len(lines) and ('"""' in lines[body_start] or "'''" in lines[body_start]):
            body_start += 1
            while body_start < len(lines) and '"""' not in lines[body_start] and "'''" not in lines[body_start]:
                body_start += 1
            body_start += 1

        if body_start < len(lines):
            header = lines[:body_start]
            body = lines[body_start:]
            body.reverse()
            lines = header + body

        return '\n'.join(lines)

    @staticmethod
    def change_formatting(code: str) -> str:
        """
        Change code formatting (whitespace, line breaks).

        Args:
            code: Original code

        Returns:
            Reformatted code
        """
        # Remove extra whitespace and compact code
        lines = [line.rstrip() for line in code.split('\n')]
        # Remove blank lines
        lines = [line for line in lines if line.strip()]
        return '\n'.join(lines)
