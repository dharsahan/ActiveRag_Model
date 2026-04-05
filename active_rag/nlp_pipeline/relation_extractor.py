"""LLM-based Relation Extraction for Knowledge Graph enrichment."""

import json
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
from active_rag.config import Config
from active_rag.schemas.relationships import RELATIONSHIP_SCHEMAS

logger = logging.getLogger(__name__)

_RELATION_SYSTEM_PROMPT = """
You are a Knowledge Graph specialist. Your task is to extract relationships between entities from the provided text.

GUIDELINES:
1. Identify clear relationships between the provided entities.
2. If a "Chunk" ID is provided, determine the most descriptive relationship between that Chunk and the entities it contains.
   - Instead of just 'MENTIONS', use verbs like: 'DEFINES', 'DISCUSSES', 'REPORTS', 'CRITICIZES', 'LAUDS', 'EXPLAINS'.
3. The predicate (relationship type) should be a concise verb, in UPPERCASE_WITH_UNDERSCORES.
4. Be as specific as possible. 

Return ONLY a JSON object with a "relationships" key containing a list of objects with this structure:
{
  "relationships": [
    {
      "subject_id": "id_of_subject", 
      "subject_label": "Chunk",
      "predicate": "EXPLAINS",
      "object_id": "id_of_object",
      "object_label": "Person",
      "properties": {"certainty": 0.9}
    }
  ]
}

Note: subject_id and object_id MUST match exactly the IDs provided in the list below.
If no clear relationships are found, return [].
"""

class RelationExtractor:
    """Extracts relationships from text using LLM."""

    def __init__(self, config: Config):
        self._config = config
        self._client = OpenAI(
            base_url=config.ollama_base_url,
            api_key=config.api_key,
        )

    def extract_relations(self, text: str, entities: List[Dict[str, Any]], chunk_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract relationships between the provided entities (and optionally a Chunk) from text."""
        if not text or not entities:
            return []

        # Prepare context for the LLM
        entity_list = entities.copy()
        if chunk_id:
            # Add the Chunk as a virtual entity for the LLM to relate to others
            entity_list.append({
                "label": "Chunk",
                "properties": {"id": chunk_id, "name": "The provided text chunk"}
            })

        entity_context = "\n".join([
            f"- {e['properties']['id']} ({e['label']}): {e['properties'].get('name', '')}"
            for e in entity_list
        ])

        prompt = f"Text:\n{text}\n\nEntities to link:\n{entity_context}\n\nExtract relationships in JSON format:"

        try:
            response = self._client.chat.completions.create(
                model=self._config.model_name,
                messages=[
                    {"role": "system", "content": _RELATION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content or "{}"
            data = json.loads(content)
            
            # Robust extraction of the list
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                # Check for common keys
                for key in ["relationships", "relations", "links"]:
                    if key in data and isinstance(data[key], list):
                        return data[key]
                # If it's a dict but no list key, maybe it's a single relation?
                if "subject_id" in data:
                    return [data]
            
            return []
        except Exception as e:
            logger.warning(f"Relation extraction failed: {e}")
            return []
