"""
LangChain extraction chain for trading concept extraction.

This module implements the extraction chain that uses the EXTRACTION_PROMPT
to extract structured trading concepts from unstructured content.
"""

from typing import Dict, Any, List, Optional
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from ..prompts.templates import EXTRACTION_PROMPT
from pydantic import BaseModel, Field
import json
import logging

# Configure logging
logger = logging.getLogger(__name__)


class ExtractionResult(BaseModel):
    """Pydantic model for validated extraction results."""
    strategy_name: Optional[str] = Field(None, description="Name of the trading strategy")
    overview: Optional[str] = Field(None, description="Brief overview of the strategy")
    entry_conditions: List[str] = Field(default_factory=list, description="List of entry conditions")
    exit_conditions: Dict[str, Any] = Field(default_factory=dict, description="Exit conditions including take profit, stop loss, etc.")
    filters: List[str] = Field(default_factory=list, description="List of trading filters")
    indicators: List[Dict[str, Any]] = Field(default_factory=list, description="List of technical indicators with settings")
    position_sizing: Dict[str, Any] = Field(default_factory=dict, description="Position sizing and risk management details")
    mentioned_concepts: List[str] = Field(default_factory=list, description="Key concepts mentioned in the content")


class ExtractionChain:
    """LangChain extraction chain for trading concept extraction."""

    def __init__(self, llm: BaseLanguageModel):
        """
        Initialize the extraction chain.

        Args:
            llm: BaseLanguageModel instance to use for extraction
        """
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=ExtractionResult)

        # Create the chain
        self.chain = (
            RunnablePassthrough()
            | EXTRACTION_PROMPT
            | self.llm
            | self.parser
        )

    async def extract(self, content: str, content_type: str = "video_transcript", keywords: List[str] = None) -> Dict[str, Any]:
        """
        Extract trading concepts from content using the extraction chain.

        Args:
            content: The content to analyze (transcript + metadata)
            content_type: Type of content being analyzed
            keywords: Optional list of keywords to guide extraction

        Returns:
            Dictionary containing extracted trading concepts

        Raises:
            ValueError: If JSON parsing fails or structure is invalid
        """
        try:
            # Prepare input for the chain
            input_data = {
                "content": content,
                "content_type": content_type,
                "keywords": keywords or []
            }

            # Run the extraction chain
            result = await self.chain.ainvoke(input_data)

            # Validate the result structure
            validated_result = ExtractionResult(**result)

            # Convert to plain dict for return
            return validated_result.dict()

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in extraction: {str(e)}")
            raise ValueError(f"Failed to parse JSON response from LLM: {str(e)}")
        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            raise ValueError(f"Extraction failed: {str(e)}")


async def extraction_chain(
    content: str,
    content_type: str = "video_transcript",
    keywords: List[str] = None,
    llm: BaseLanguageModel = None
) -> Dict[str, Any]:
    """
    Extract trading concepts from content using LangChain extraction chain.

    This function creates a temporary extraction chain and processes the content.
    It handles JSON parsing errors gracefully and returns structured data.

    Args:
        content: The content to analyze (transcript + metadata)
        content_type: Type of content being analyzed
        keywords: Optional list of keywords to guide extraction
        llm: Optional BaseLanguageModel instance. If None, a default will be used.

    Returns:
        Dictionary containing extracted trading concepts with the following structure:
        {
            "strategy_name": str,
            "overview": str,
            "entry_conditions": List[str],
            "exit_conditions": Dict[str, Any],
            "filters": List[str],
            "indicators": List[Dict[str, Any]],
            "position_sizing": Dict[str, Any],
            "mentioned_concepts": List[str]
        }

    Raises:
        ValueError: If JSON parsing fails or structure is invalid
    """
    if llm is None:
        # In a real implementation, you would import and use a default LLM
        # For this example, we'll assume llm is provided or will be handled elsewhere
        raise ValueError("LLM must be provided")

    # Create extraction chain instance
    chain = ExtractionChain(llm)

    # Perform extraction
    return await chain.extract(content, content_type, keywords)