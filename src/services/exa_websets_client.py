"""
Exa Websets API Client

Wrapper around exa-py SDK providing:
- Robust error handling and retry logic
- Comprehensive logging for debugging
- Async search with timeout management
- CSV data retrieval and cleanup

References:
- Exa Websets API Guide: InternalDocs/exa-websets-guide.md
- Exa Python SDK: https://github.com/exa-labs/exa-py
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from exa_py import Exa
from exa_py.websets.types import CreateWebsetParameters


@dataclass
class SearchResult:
    """Result from an Exa Websets search operation."""
    webset_id: str
    search_id: str
    status: str
    items_found: int
    search_time: float
    success: bool
    error_message: Optional[str] = None


class ExaWebsetsError(Exception):
    """Base exception for Exa Websets client errors."""
    pass


class ExaWebsetsClient:
    """
    Wrapper around Exa Websets API with production-grade error handling.
    
    Features:
    - Async search with configurable timeouts
    - Exponential backoff retry logic
    - Comprehensive logging for debugging
    - Automatic cleanup of temporary Websets
    """
    
    def __init__(
        self,
        api_key: str,
        timeout_seconds: int = 600,
        max_retries: int = 3,
        retry_backoff_base: int = 2,
    ):
        """
        Initialize Exa Websets client.

        Args:
            api_key: Exa API key
            timeout_seconds: Maximum time to wait for search completion (default: 600s = 10 min)
            max_retries: Maximum number of retry attempts
            retry_backoff_base: Base for exponential backoff (seconds)
        """
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base
        
        # Initialize Exa client
        self.client = Exa(api_key=api_key)
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Track created Websets for cleanup
        self._created_websets: List[str] = []
    
    async def create_search(
        self,
        query: str,
        count: int,
        entity_type: str = "article",
        criteria: Optional[List[Dict[str, str]]] = None,
        enrichments: Optional[List[Dict[str, Any]]] = None,
    ) -> SearchResult:
        """
        Create a Webset search and wait for completion.

        Args:
            query: Search query text
            count: Number of results to retrieve
            entity_type: Type of entity to search for (article, research_paper, etc.)
            criteria: Optional list of criteria dicts with "description" keys
            enrichments: Optional list of enrichment dicts with "description" and "format" keys

        Returns:
            SearchResult with webset_id, status, and items found

        Raises:
            ExaWebsetsError: If search fails after retries
        """
        start_time = time.time()
        webset = None

        criteria_count = len(criteria) if criteria else 0
        self.logger.info(
            f"🔍 Creating Exa webset: entity={entity_type}, count={count}, "
            f"criteria={criteria_count}"
        )
        self.logger.debug(f"Query: {query[:100]}...")

        try:
            # Prepare search parameters
            search_params = {
                "query": query,
                "count": count,
            }

            # Add entity type if specified
            if entity_type:
                search_params["entity"] = {"type": entity_type}

            # Add criteria if specified (already in correct format)
            if criteria:
                search_params["criteria"] = criteria

            # Create Webset with retry logic
            self.logger.info("📤 Sending webset creation request to Exa API...")
            webset = await self._create_webset_with_retry(search_params, enrichments)

            if not webset:
                self.logger.error("❌ Failed to create webset after all retries")
                return SearchResult(
                    webset_id="",
                    search_id="",
                    status="failed",
                    items_found=0,
                    search_time=time.time() - start_time,
                    success=False,
                    error_message="Failed to create Webset after retries"
                )

            # Track for cleanup
            self._created_websets.append(webset.id)
            self.logger.info(
                f"✅ Webset created successfully: {webset.id} (status: {webset.status})"
            )

            # Wait for search completion
            completed = await self._wait_for_completion(webset.id)

            if not completed:
                # Cleanup on timeout to prevent orphaned webset
                self.logger.warning(
                    f"Search timed out after {self.timeout_seconds}s, cleaning up webset {webset.id}"
                )
                await self.cleanup_webset(webset.id)

                return SearchResult(
                    webset_id=webset.id,
                    search_id="",
                    status="timeout",
                    items_found=0,
                    search_time=time.time() - start_time,
                    success=False,
                    error_message=f"Search timed out after {self.timeout_seconds}s"
                )

            # Get final webset state
            final_webset = self.client.websets.get(webset.id)

            # Count items
            items_response = self.client.websets.items.list(webset.id, limit=1)
            # Note: We'll get actual count when retrieving all items

            search_time = time.time() - start_time

            self.logger.info(
                f"Search completed: webset_id={webset.id}, "
                f"status={final_webset.status}, time={search_time:.2f}s"
            )

            return SearchResult(
                webset_id=webset.id,
                search_id=webset.searches[0].id if webset.searches else "",
                status=final_webset.status,
                items_found=0,  # Will be counted during retrieval
                search_time=search_time,
                success=True,
            )

        except Exception as e:
            # Cleanup on exception to prevent orphaned webset
            if webset:
                self.logger.warning(
                    f"Exception during search, cleaning up webset {webset.id}: {e}"
                )
                await self.cleanup_webset(webset.id)
            raise
    
    async def get_items(
        self,
        webset_id: str,
        batch_size: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all items from a Webset.

        Args:
            webset_id: Webset ID to retrieve items from
            batch_size: Number of items to fetch per page

        Returns:
            List of item dictionaries with properties and metadata

        Raises:
            ExaWebsetsError: If retrieval fails
        """
        self.logger.info(f"Retrieving items from webset: {webset_id}")

        # Check webset status before retrieving items
        try:
            webset = self.client.websets.get(webset_id)
            self.logger.info(f"📊 Webset status: {webset.status}")

            # Note: Enrichment status is checked via item.enrichments, not webset.enrichments
            # The expand parameter only supports 'items', not 'enrichments'
        except Exception as e:
            self.logger.warning(f"Could not check webset status: {e}")

        all_items = []
        cursor = None
        page = 0

        try:
            while True:
                page += 1
                self.logger.debug(f"Fetching page {page} (batch_size={batch_size})")

                response = self.client.websets.items.list(
                    webset_id,
                    limit=batch_size,
                    cursor=cursor
                )

                # Convert items to dictionaries
                for item in response.data:
                    item_dict = self._item_to_dict(item)
                    all_items.append(item_dict)

                # Check if more pages exist
                if not response.has_more:
                    break

                cursor = response.next_cursor

            self.logger.info(f"✅ Retrieved {len(all_items)} items from webset {webset_id}")

            # Log sample of items for debugging
            if all_items:
                self.logger.info(f"📊 Sample item structure: {list(all_items[0].keys())}")
                self.logger.debug(f"First item details: {all_items[0]}")

                # Count items with enrichment_summary
                enriched_count = sum(1 for item in all_items if item.get("enrichment_summary"))
                self.logger.info(
                    f"📈 Enrichment stats: {enriched_count}/{len(all_items)} items have enrichment_summary"
                )

            return all_items

        except Exception as e:
            self.logger.error(f"Failed to retrieve items from webset {webset_id}: {e}")
            raise ExaWebsetsError(f"Item retrieval failed: {e}") from e
    
    async def cleanup_webset(self, webset_id: str) -> bool:
        """
        Delete a Webset to free resources.
        
        Args:
            webset_id: Webset ID to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            self.logger.debug(f"Deleting webset: {webset_id}")
            self.client.websets.delete(webset_id)
            
            # Remove from tracking
            if webset_id in self._created_websets:
                self._created_websets.remove(webset_id)
            
            self.logger.info(f"Deleted webset: {webset_id}")
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to delete webset {webset_id}: {e}")
            return False
    
    async def cleanup_all(self) -> int:
        """
        Clean up all Websets created by this client.

        Returns:
            Number of Websets successfully deleted
        """
        deleted_count = 0

        for webset_id in list(self._created_websets):
            if await self.cleanup_webset(webset_id):
                deleted_count += 1

        self.logger.info(f"Cleaned up {deleted_count} websets")
        return deleted_count

    async def create_enrichment(
        self,
        webset_id: str,
        description: str,
        format: str = "text",
        title: Optional[str] = None,
        options: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create an enrichment for a Webset.

        Args:
            webset_id: ID of the Webset
            description: Description of the enrichment task
            format: Format of the enrichment response (text, date, number, options, email, phone, url)
            title: Optional title for the enrichment
            options: Optional list of option dicts (for format="options")
            metadata: Optional metadata dict

        Returns:
            Enrichment object as dictionary with id, status, etc.

        Raises:
            ExaWebsetsError: If enrichment creation fails
        """
        try:
            self.logger.info(f"🔧 Creating enrichment for Webset {webset_id}: {description[:100]}...")

            # Build enrichment parameters - SDK uses body dict, not kwargs
            enrichment_body = {
                "description": description,
                "format": format,
            }
            if title:
                enrichment_body["title"] = title
            if options:
                enrichment_body["options"] = options
            if metadata:
                enrichment_body["metadata"] = metadata

            # Create enrichment using SDK - pass body as single argument
            enrichment = self.client.websets.enrichments.create(
                webset_id,
                enrichment_body
            )

            # Convert to dict
            enrichment_dict = {
                "id": enrichment.id,
                "object": getattr(enrichment, "object", "webset_enrichment"),
                "status": enrichment.status,
                "webset_id": webset_id,
                "description": description,
                "format": format,
                "created_at": getattr(enrichment, "created_at", None),
                "updated_at": getattr(enrichment, "updated_at", None),
            }

            self.logger.info(f"✅ Created enrichment {enrichment.id} with status: {enrichment.status}")
            return enrichment_dict

        except Exception as e:
            error_msg = f"Failed to create enrichment for Webset {webset_id}: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            raise ExaWebsetsError(error_msg) from e

    async def get_enrichment(
        self,
        webset_id: str,
        enrichment_id: str,
    ) -> Dict[str, Any]:
        """
        Get enrichment status.

        Args:
            webset_id: ID of the Webset
            enrichment_id: ID of the enrichment

        Returns:
            Enrichment object as dictionary

        Raises:
            ExaWebsetsError: If retrieval fails
        """
        try:
            # Get webset with enrichments expanded
            webset = self.client.websets.get(
                id=webset_id,
                expand=["enrichments"]
            )

            # Find the enrichment
            if hasattr(webset, "enrichments") and webset.enrichments:
                for enrichment in webset.enrichments:
                    if enrichment.id == enrichment_id:
                        return {
                            "id": enrichment.id,
                            "status": enrichment.status,
                            "webset_id": webset_id,
                            "description": getattr(enrichment, "description", ""),
                            "format": getattr(enrichment, "format", "text"),
                        }

            raise ExaWebsetsError(f"Enrichment {enrichment_id} not found in Webset {webset_id}")

        except ExaWebsetsError:
            raise
        except Exception as e:
            error_msg = f"Failed to get enrichment {enrichment_id} from Webset {webset_id}: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            raise ExaWebsetsError(error_msg) from e

    async def poll_enrichment_completion(
        self,
        webset_id: str,
        enrichment_id: str,
        max_wait_seconds: int = 300,
        poll_interval: int = 5,
    ) -> Dict[str, Any]:
        """
        Poll enrichment until it completes or times out.

        Args:
            webset_id: ID of the Webset
            enrichment_id: ID of the enrichment
            max_wait_seconds: Maximum time to wait (default: 300s = 5 min)
            poll_interval: Seconds between polls (default: 5s)

        Returns:
            Final enrichment status dict

        Raises:
            ExaWebsetsError: If enrichment fails or times out
        """
        start_time = time.time()
        poll_count = 0

        self.logger.info(f"⏳ Polling enrichment {enrichment_id} (max {max_wait_seconds}s, interval {poll_interval}s)")

        while True:
            poll_count += 1
            elapsed = time.time() - start_time

            # Check timeout
            if elapsed > max_wait_seconds:
                error_msg = f"Enrichment {enrichment_id} timed out after {max_wait_seconds}s"
                self.logger.error(f"⏰ {error_msg}")
                raise ExaWebsetsError(error_msg)

            # Get enrichment status
            enrichment = await self.get_enrichment(webset_id, enrichment_id)
            status = enrichment.get("status", "unknown")

            self.logger.debug(f"📊 Poll #{poll_count} ({elapsed:.1f}s): Enrichment {enrichment_id} status = {status}")

            # Check if completed
            if status == "completed":
                self.logger.info(f"✅ Enrichment {enrichment_id} completed after {elapsed:.1f}s ({poll_count} polls)")
                return enrichment

            # Check if failed
            if status in ["failed", "canceled"]:
                error_msg = f"Enrichment {enrichment_id} {status}"
                self.logger.error(f"❌ {error_msg}")
                raise ExaWebsetsError(error_msg)

            # Wait before next poll
            await asyncio.sleep(poll_interval)

    async def list_all_websets(self) -> List[Dict[str, Any]]:
        """
        List all websets in the account (not just those created by this client).

        Returns:
            List of webset dictionaries with id, status, title, created_at
        """
        try:
            response = self.client.websets.list(limit=100)
            websets = []

            for webset in response.data:
                websets.append({
                    "id": webset.id,
                    "status": webset.status,
                    "title": getattr(webset, "title", "Untitled"),
                    "created_at": webset.created_at,
                })

            self.logger.info(f"Found {len(websets)} websets in account")
            return websets

        except Exception as e:
            self.logger.error(f"Failed to list websets: {e}")
            return []

    async def cleanup_all_websets(self) -> int:
        """
        Delete ALL websets in the account (use with caution!).

        This is useful for cleaning up orphaned websets from failed tests.

        Returns:
            Number of websets deleted
        """
        websets = await self.list_all_websets()
        deleted_count = 0

        self.logger.info(f"Attempting to delete {len(websets)} websets from account")

        for webset in websets:
            if await self.cleanup_webset(webset["id"]):
                deleted_count += 1

        self.logger.info(f"Deleted {deleted_count}/{len(websets)} websets from account")
        return deleted_count
    
    # Private helper methods
    
    async def _create_webset_with_retry(
        self,
        search_params: Dict[str, Any],
        enrichments: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Any]:
        """Create Webset with exponential backoff retry."""
        for attempt in range(self.max_retries):
            try:
                # Build webset parameters
                webset_params = {"search": search_params}

                # Add enrichments if provided
                if enrichments:
                    from exa_py.websets.types import CreateEnrichmentParameters
                    webset_params["enrichments"] = [
                        CreateEnrichmentParameters(**enrichment)
                        for enrichment in enrichments
                    ]

                webset = self.client.websets.create(
                    params=CreateWebsetParameters(**webset_params)
                )
                self.logger.debug(f"Webset created: {webset.id}")
                return webset
                
            except Exception as e:
                self.logger.warning(
                    f"Webset creation attempt {attempt + 1}/{self.max_retries} failed: {e}"
                )
                
                if attempt < self.max_retries - 1:
                    delay = self.retry_backoff_base ** attempt
                    self.logger.debug(f"Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"Webset creation failed after {self.max_retries} attempts")
                    return None
        
        return None
    
    async def _wait_for_completion(
        self,
        webset_id: str,
        poll_interval: int = 5
    ) -> bool:
        """Wait for Webset search to complete with timeout."""
        start_time = time.time()
        last_status = None
        poll_count = 0

        # Log every 30 seconds (6 polls at 5s interval) to track progress
        log_interval_polls = 6

        while True:
            elapsed = time.time() - start_time
            poll_count += 1

            # Warn at 50% timeout
            if elapsed > self.timeout_seconds * 0.5 and poll_count == log_interval_polls * 6:
                self.logger.warning(
                    f"⚠️  Webset {webset_id} still running after {elapsed:.0f}s "
                    f"({elapsed/self.timeout_seconds*100:.0f}% of timeout)"
                )

            if elapsed > self.timeout_seconds:
                self.logger.error(
                    f"❌ Webset {webset_id} TIMED OUT after {self.timeout_seconds}s "
                    f"(last status: {last_status})"
                )
                return False

            try:
                webset = self.client.websets.get(webset_id)
                current_status = webset.status

                # Log status changes or every N polls
                if current_status != last_status or poll_count % log_interval_polls == 0:
                    self.logger.info(
                        f"📊 Webset {webset_id} status: {current_status} "
                        f"(elapsed: {elapsed:.1f}s, poll #{poll_count})"
                    )
                    last_status = current_status

                if webset.status == "idle":
                    self.logger.info(
                        f"✅ Webset {webset_id} completed successfully "
                        f"(total time: {elapsed:.1f}s, {poll_count} polls)"
                    )
                    return True
                elif webset.status in ["paused", "failed"]:
                    self.logger.error(
                        f"❌ Webset {webset_id} ended with status: {webset.status} "
                        f"after {elapsed:.1f}s"
                    )
                    return False

                # Still running, wait and poll again
                await asyncio.sleep(poll_interval)

            except Exception as e:
                self.logger.error(
                    f"❌ Error checking webset {webset_id} status: {e} "
                    f"(elapsed: {elapsed:.1f}s)",
                    exc_info=True
                )
                return False
    
    def _item_to_dict(self, item: Any) -> Dict[str, Any]:
        """Convert Exa item object to dictionary."""
        # Extract URL and convert to string
        url = ""
        if hasattr(item.properties, "url"):
            url_obj = item.properties.url
            url = str(url_obj) if url_obj else ""

        item_dict = {
            "id": item.id,
            "type": item.properties.type if hasattr(item.properties, "type") else "unknown",
            "url": url,
            "description": item.properties.description if hasattr(item.properties, "description") else "",
        }

        # Add type-specific fields
        if hasattr(item.properties, "article"):
            article = item.properties.article
            item_dict["title"] = getattr(article, "title", "")
            item_dict["published_date"] = getattr(article, "published_date", None)
            item_dict["author"] = getattr(article, "author", "")

        if hasattr(item.properties, "research_paper"):
            paper = item.properties.research_paper
            item_dict["title"] = getattr(paper, "title", "")
            item_dict["authors"] = getattr(paper, "authors", [])
            item_dict["published_date"] = getattr(paper, "published_date", None)
            item_dict["abstract"] = getattr(paper, "abstract", "")

        # Add evaluations (verification results)
        if hasattr(item, "evaluations") and item.evaluations:
            item_dict["evaluations"] = [
                {
                    "criterion": e.criterion,
                    "satisfied": e.satisfied,
                    "reasoning": e.reasoning,
                }
                for e in item.evaluations
            ]

        # Add enrichment data (extract summary from enrichments array)
        enrichment_found = False
        if hasattr(item, "enrichments") and item.enrichments:
            self.logger.debug(f"🔍 Item {item.id[:8]}... has {len(item.enrichments)} enrichment(s)")

            # Find completed text enrichments and extract the result
            for idx, enrichment in enumerate(item.enrichments):
                # Log enrichment structure for debugging
                enr_status = getattr(enrichment, "status", "no_status")
                enr_format = getattr(enrichment, "format", "no_format")
                enr_has_result = hasattr(enrichment, "result")

                self.logger.debug(
                    f"  Enrichment {idx}: status={enr_status}, format={enr_format}, "
                    f"has_result={enr_has_result}"
                )

                if enr_has_result:
                    result_value = enrichment.result
                    result_type = type(result_value).__name__
                    result_len = len(result_value) if hasattr(result_value, "__len__") else "N/A"
                    self.logger.debug(f"    Result type: {result_type}, length: {result_len}")

                    if result_value:
                        self.logger.debug(f"    Result preview: {str(result_value)[:100]}...")

                if (hasattr(enrichment, "status") and enrichment.status == "completed" and
                    hasattr(enrichment, "format") and enrichment.format == "text" and
                    hasattr(enrichment, "result") and enrichment.result):
                    # result is an array of strings, join them
                    enrichment_text = " ".join(enrichment.result) if isinstance(enrichment.result, list) else str(enrichment.result)
                    item_dict["enrichment_summary"] = enrichment_text
                    enrichment_found = True
                    self.logger.info(
                        f"✅ Extracted enrichment for {item.id[:8]}...: {len(enrichment_text)} chars"
                    )
                    break  # Use the first completed text enrichment
        else:
            self.logger.debug(f"⚠️  Item {item.id[:8]}... has NO enrichments attribute or empty enrichments")

        if not enrichment_found and hasattr(item, "enrichments"):
            self.logger.warning(
                f"❌ No usable enrichment found for item {item.id[:8]}... "
                f"(has enrichments: {hasattr(item, 'enrichments')}, "
                f"count: {len(item.enrichments) if hasattr(item, 'enrichments') and item.enrichments else 0})"
            )

        return item_dict

