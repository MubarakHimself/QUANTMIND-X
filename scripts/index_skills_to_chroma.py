#!/usr/bin/env python3
"""
Index Skills to ChromaDB

Reads skill definitions from /docs/skills/ and indexes them into ChromaDB
for semantic search and retrieval via the Assets Hub.

Usage:
    python scripts/index_skills_to_chroma.py
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Check dependencies
try:
    import chromadb
except ImportError:
    print("ChromaDB not found. Install with: pip install chromadb")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("sentence-transformers not found. Install with: pip install sentence-transformers")
    sys.exit(1)

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_SKILLS_DIR = PROJECT_ROOT / "docs" / "skills"
CHROMA_PATH = PROJECT_ROOT / "data" / "chromadb"
COLLECTION_NAME = "agentic_skills"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# HNSW index configuration (optimized for performance)
HNSW_CONFIG = {
    "hnsw:space": "cosine",
    "hnsw:M": 16,
    "hnsw:construction_ef": 100,
    "hnsw:search_ef": 50
}


def extract_frontmatter(content: str) -> tuple[Dict[str, Any], str]:
    """
    Extract YAML frontmatter from markdown content.

    Args:
        content: Markdown file content

    Returns:
        Tuple of (frontmatter_dict, content_without_frontmatter)
    """
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            frontmatter_text = parts[1].strip()
            meta = {}
            current_list_key = None
            current_list = []

            for line in frontmatter_text.split('\n'):
                stripped = line.strip()

                # Handle list items (lines starting with "-")
                if stripped.startswith('- '):
                    item = stripped[2:].strip()
                    # Remove quotes if present
                    if item.startswith('"') and item.endswith('"'):
                        item = item[1:-1]
                    elif item.startswith("'") and item.endswith("'"):
                        item = item[1:-1]

                    if current_list_key is not None:
                        current_list.append(item)
                    continue

                # Save any pending list
                if current_list_key is not None:
                    meta[current_list_key] = current_list
                    current_list_key = None
                    current_list = []

                # Handle key-value pairs
                if ':' in line:
                    key, val = line.split(':', 1)
                    key = key.strip()
                    val = val.strip()

                    # Check if this is a list header (key ending with colon, value empty)
                    if not val:
                        current_list_key = key
                        continue

                    # Remove quotes if present
                    if val.startswith('"') and val.endswith('"'):
                        val = val[1:-1]
                    elif val.startswith("'") and val.endswith("'"):
                        val = val[1:-1]
                    elif val.startswith('[') and val.endswith(']'):
                        # Parse inline list
                        val = [item.strip().strip('"\'') for item in val[1:-1].split(',') if item.strip()]
                    elif val.isdigit():
                        val = int(val)
                    # Don't try to parse semantic versions as floats
                    # Check if it's a simple float (single dot, only digits)
                    elif val.count('.') == 1 and val.replace('.', '').isdigit():
                        try:
                            val = float(val)
                        except ValueError:
                            pass  # Keep as string

                    meta[key] = val

            # Save any pending list at end
            if current_list_key is not None and current_list:
                meta[current_list_key] = current_list

            return meta, parts[2].strip()
    return {}, content


def extract_code_blocks(content: str) -> str:
    """
    Extract code from markdown code blocks.

    Args:
        content: Markdown content

    Returns:
        Extracted code (first code block found)
    """
    # First try to extract from the "## Code" section specifically
    if "## Code" in content:
        # Get the Code section
        code_section = content.split("## Code")[1].split("##")[0] if "## Code" in content else ""
        # Extract the python code block
        code_match = re.search(r'```python\n(.*?)```', code_section, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
    # Fallback: Match any code block with language identifier
    pattern = r'```(?:python|mq5|yaml)?\n(.*?)```'
    matches = re.findall(pattern, content, re.DOTALL)
    if matches:
        return matches[0].strip()
    return ""


def extract_input_schema(content: str) -> Dict[str, Any]:
    """
    Extract input schema from skill markdown.

    Args:
        content: Markdown content

    Returns:
        JSON Schema for input
    """
    # Look for JSON schema in Input Schema section
    if "## Input Schema" in content:
        section = content.split("## Input Schema")[1].split("##")[0]
        # Extract JSON from code block
        json_match = re.search(r'```json\n(.*?)```', section, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
    return {"type": "object"}


def extract_output_schema(content: str) -> Dict[str, Any]:
    """
    Extract output schema from skill markdown.

    Args:
        content: Markdown content

    Returns:
        JSON Schema for output
    """
    # Look for JSON schema in Output Schema section
    if "## Output Schema" in content:
        section = content.split("## Output Schema")[1].split("##")[0]
        # Extract JSON from code block
        json_match = re.search(r'```json\n(.*?)```', section, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
    return {"type": "object"}


def extract_example_usage(content: str) -> str:
    """
    Extract example usage from skill markdown.

    Args:
        content: Markdown content

    Returns:
        Example usage code
    """
    if "## Example Usage" in content:
        section = content.split("## Example Usage")[1].split("##")[0]
        # Extract first code block
        code_match = re.search(r'```(?:python|yaml)?\n(.*?)```', section, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
    return ""


def process_skill_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Process a skill markdown file and extract all metadata.

    Args:
        file_path: Path to skill .md file

    Returns:
        Dictionary with skill data or None if processing fails
    """
    try:
        content = file_path.read_text(encoding='utf-8')
        frontmatter, body = extract_frontmatter(content)

        # Extract required fields
        name = frontmatter.get('name')
        if not name:
            # Try to extract from filename
            name = file_path.stem

        category = frontmatter.get('category')
        if category not in ['trading_skills', 'system_skills', 'data_skills']:
            # Infer category from directory
            parent_dir = file_path.parent.name
            category = parent_dir if parent_dir in ['trading_skills', 'system_skills', 'data_skills'] else 'system_skills'

        description = frontmatter.get('description', '')
        version = frontmatter.get('version', '1.0.0')
        dependencies = frontmatter.get('dependencies', [])
        if isinstance(dependencies, str):
            dependencies = [dependencies]
        # Filter out empty strings from dependencies
        dependencies = [d for d in dependencies if d and isinstance(d, str) and d.strip()]

        # Extract schemas and code
        input_schema = extract_input_schema(content)
        output_schema = extract_output_schema(content)
        code = extract_code_blocks(content)
        example_usage = extract_example_usage(content)

        # Combine description with tags for better search
        tags = frontmatter.get('tags', [])
        if isinstance(tags, str):
            tags = [tags]
        search_text = f"{description} {' '.join(tags)}"

        # Determine relative path for file_path metadata
        relative_path = file_path.relative_to(PROJECT_ROOT)

        return {
            'id': name,
            'name': name,
            'category': category,
            'description': description,
            'version': version,
            'dependencies': dependencies,
            'tags': tags,
            'file_path': str(relative_path),
            'input_schema': json.dumps(input_schema),
            'output_schema': json.dumps(output_schema),
            'code': code,
            'example_usage': example_usage,
            'search_text': search_text
        }

    except Exception as e:
        print(f"   Error processing {file_path.name}: {e}")
        return None


def main():
    """Main function to index skills to ChromaDB."""
    print("=" * 70)
    print("Index Skills to ChromaDB")
    print("=" * 70)

    # Check if skills directory exists
    if not DOCS_SKILLS_DIR.exists():
        print(f"\nError: Skills directory not found at: {DOCS_SKILLS_DIR}")
        print("Create skill definitions in /docs/skills/ first.")
        sys.exit(1)

    # Initialize ChromaDB
    print(f"\n1. Initializing ChromaDB at: {CHROMA_PATH}")
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # Initialize embedding model
    print("2. Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"   Model: {EMBEDDING_MODEL} (dim={embedding_dim})")

    # Get or create collection
    print(f"3. Getting/creating collection: {COLLECTION_NAME}")
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata=HNSW_CONFIG
    )

    # Find all skill files
    print("4. Scanning for skill files...")
    skill_files = list(DOCS_SKILLS_DIR.rglob("*.md"))
    print(f"   Found {len(skill_files)} skill files")

    if not skill_files:
        print("\nNo skill files found. Create .md files in:")
        print(f"   - {DOCS_SKILLS_DIR}/trading_skills/")
        print(f"   - {DOCS_SKILLS_DIR}/system_skills/")
        print(f"   - {DOCS_SKILLS_DIR}/data_skills/")
        sys.exit(1)

    # Process skills
    print("5. Processing skill files...")
    skills_data = []
    for skill_file in skill_files:
        skill_data = process_skill_file(skill_file)
        if skill_data:
            skills_data.append(skill_data)
            print(f"   Processed: {skill_data['name']} ({skill_data['category']})")

    if not skills_data:
        print("\nNo valid skills found.")
        sys.exit(1)

    # Generate embeddings
    print(f"\n6. Generating embeddings for {len(skills_data)} skills...")
    search_texts = [skill['search_text'] for skill in skills_data]
    embeddings = model.encode(search_texts, show_progress_bar=True)

    # Prepare data for ChromaDB
    ids = [skill['id'] for skill in skills_data]
    documents = [skill['description'] for skill in skills_data]
    metadatas = []
    for skill in skills_data:
        metadata = {
            'name': skill['name'],
            'category': skill['category'],
            'version': skill['version'],
            'dependencies': json.dumps(skill['dependencies']),
            'file_path': skill['file_path'],
            'input_schema': skill['input_schema'],
            'output_schema': skill['output_schema'],
            'example_usage': skill['example_usage'],
            'code': skill['code']  # Include code for load_skill tool
        }
        metadatas.append(metadata)

    # Clear existing data (optional - comment out to keep existing)
    print("7. Clearing existing skill data...")
    try:
        existing_ids = collection.get()['ids']
        if existing_ids:
            collection.delete(ids=existing_ids)
            print(f"   Deleted {len(existing_ids)} existing entries")
    except Exception as e:
        print(f"   Note: {e}")

    # Add skills to ChromaDB
    print("8. Adding skills to ChromaDB...")
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    # Verify
    count = collection.count()
    print(f"\n{'=' * 70}")
    print(f"Successfully indexed {count} skills to ChromaDB!")
    print(f"{'=' * 70}")
    print(f"\nCollection: {COLLECTION_NAME}")
    print(f"Storage: {CHROMA_PATH}")
    print(f"HNSW Config: M={HNSW_CONFIG['hnsw:M']}, ef_construction={HNSW_CONFIG['hnsw:construction_ef']}")

    # Show breakdown by category
    print("\nSkills by category:")
    categories = {}
    for skill in skills_data:
        cat = skill['category']
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items()):
        print(f"   {cat}: {count}")

    print("\nYou can now use the MCP tools:")
    print("   - load_skill: Load a skill by name")
    print("   - search_knowledge_base: Search skills semantically")


if __name__ == "__main__":
    main()
