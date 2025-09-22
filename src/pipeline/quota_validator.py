"""
Section quota validator to ensure newsletter meets production requirements.

This module provides quota validation to catch quota violations before email compilation.
"""

import logging
from typing import Dict, List, Optional

# Use string constants to avoid circular dependency with content_aggregator
class Section:
    """Section constants - matches content_aggregator.Section"""
    BREAKING_NEWS = "breaking_news"
    BUSINESS = "business"
    TECH_SCIENCE = "tech_science"
    RESEARCH_PAPERS = "research_papers"
    POLITICS = "politics"
    MISCELLANEOUS = "miscellaneous"


class QuotaValidationError(Exception):
    """Raised when section quotas are not met"""
    pass


class QuotaValidator:
    """
    Validates that newsletter sections meet required quotas.
    
    Production requirements:
    - Breaking News: exactly 3
    - Business: exactly 3
    - Tech & Science: exactly 3
    - Research Papers: exactly 5
    - Politics: exactly 2
    - Miscellaneous: exactly 5
    """
    
    # Required quotas per section
    REQUIRED_QUOTAS = {
        Section.BREAKING_NEWS: 3,
        Section.BUSINESS: 3,
        Section.TECH_SCIENCE: 3,
        Section.RESEARCH_PAPERS: 5,
        Section.POLITICS: 2,
        Section.MISCELLANEOUS: 5,
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_ranked_items(self, items_by_section: Dict[str, List]) -> Dict[str, int]:
        """
        Validate that ranked items meet quotas before selection.
        
        Args:
            items_by_section: Dictionary mapping section names to lists of RankedItem objects
            
        Returns:
            Dictionary mapping section names to actual counts
            
        Raises:
            QuotaValidationError: If any section has fewer items than required
        """
        violations = []
        actual_counts = {}
        
        for section, required_count in self.REQUIRED_QUOTAS.items():
            section_items = items_by_section.get(section, [])
            actual_count = len(section_items)
            actual_counts[section] = actual_count
            
            if actual_count < required_count:
                violations.append({
                    'section': section,
                    'required': required_count,
                    'actual': actual_count,
                    'missing': required_count - actual_count
                })
        
        if violations:
            error_msg = "Section quota violations detected:\n"
            for v in violations:
                error_msg += f"  - {v['section']}: {v['actual']}/{v['required']} items (missing {v['missing']})\n"
            
            self.logger.error(error_msg)
            raise QuotaValidationError(error_msg)
        
        self.logger.info(f"✅ All section quotas validated successfully")
        return actual_counts
    
    def validate_selected_items(self, selected_by_section: Dict[str, List]) -> Dict[str, int]:
        """
        Validate that selected items meet quotas after selection.
        
        This should be called before email compilation to ensure we have
        exactly the right number of items for each section.
        
        Args:
            selected_by_section: Dictionary mapping section names to lists of selected items
            
        Returns:
            Dictionary mapping section names to actual counts
            
        Raises:
            QuotaValidationError: If any section has incorrect number of items
        """
        violations = []
        actual_counts = {}
        
        for section, required_count in self.REQUIRED_QUOTAS.items():
            section_items = selected_by_section.get(section, [])
            actual_count = len(section_items)
            actual_counts[section] = actual_count
            
            if actual_count != required_count:
                violations.append({
                    'section': section,
                    'required': required_count,
                    'actual': actual_count,
                    'difference': actual_count - required_count
                })
        
        if violations:
            error_msg = "Section quota violations after selection:\n"
            for v in violations:
                diff_str = f"+{v['difference']}" if v['difference'] > 0 else str(v['difference'])
                error_msg += f"  - {v['section']}: {v['actual']}/{v['required']} items (difference: {diff_str})\n"
            
            self.logger.error(error_msg)
            raise QuotaValidationError(error_msg)
        
        self.logger.info(f"✅ All selected items meet quotas: {actual_counts}")
        return actual_counts
    
    def validate_section_summaries(self, summaries_by_section: Dict) -> Dict[str, int]:
        """
        Validate that section summaries meet quotas.
        
        This should be called before email compilation.
        
        Args:
            summaries_by_section: Dictionary mapping section names to SectionSummaries objects
            
        Returns:
            Dictionary mapping section names to actual counts
            
        Raises:
            QuotaValidationError: If any section has incorrect number of items
        """
        violations = []
        actual_counts = {}
        
        for section, required_count in self.REQUIRED_QUOTAS.items():
            section_data = summaries_by_section.get(section)
            
            if section_data is None:
                actual_count = 0
            elif hasattr(section_data, 'summaries'):
                # SectionSummaries object
                actual_count = len(section_data.summaries)
            elif hasattr(section_data, 'items'):
                # Fallback to items
                actual_count = len(section_data.items)
            elif isinstance(section_data, list):
                # Direct list
                actual_count = len(section_data)
            else:
                actual_count = 0
            
            actual_counts[section] = actual_count
            
            if actual_count != required_count:
                violations.append({
                    'section': section,
                    'required': required_count,
                    'actual': actual_count,
                    'difference': actual_count - required_count
                })
        
        if violations:
            error_msg = "Section quota violations in summaries:\n"
            for v in violations:
                diff_str = f"+{v['difference']}" if v['difference'] > 0 else str(v['difference'])
                error_msg += f"  - {v['section']}: {v['actual']}/{v['required']} items (difference: {diff_str})\n"
            
            self.logger.error(error_msg)
            raise QuotaValidationError(error_msg)
        
        self.logger.info(f"✅ All section summaries meet quotas: {actual_counts}")
        return actual_counts

