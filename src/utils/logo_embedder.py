"""Logo URL provider for email templates."""
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def get_embedded_logo_url() -> str:
    """
    Get the logo URL for embedding in emails.
    
    Best practice for email images:
    - Use HTTPS hosted images for maximum compatibility
    - Avoid base64 (blocked by Gmail)
    - Avoid CID attachments (inconsistent support)
    
    Returns:
        HTTPS URL string for embedding in HTML img src
    """
    # Check for environment variable override first
    custom_logo_url = os.getenv("NEWSLETTER_LOGO_URL")
    if custom_logo_url:
        logger.info(f"Using custom logo URL from environment: {custom_logo_url}")
        return custom_logo_url
    
    # Default to a reliable hosted URL
    # Options:
    # 1. Upload to a CDN like Cloudinary, Imgur, or AWS S3
    # 2. Use GitHub Pages to host the image
    # 3. Use a dedicated image hosting service
    
    # For now, using a placeholder that should be replaced with actual hosted URL
    default_url = "https://i.imgur.com/YOUR_IMAGE_ID.png"  # Replace with actual hosted image
    
    logger.warning(
        "Using placeholder logo URL. Please set NEWSLETTER_LOGO_URL environment variable "
        "or update default_url in logo_embedder.py with your hosted image URL"
    )
    
    return default_url


def get_logo_html(width: int = 100, height: int = 100, alt_text: str = "Fourier Forecast Logo") -> str:
    """
    Get complete HTML img tag with embedded logo.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        alt_text: Alt text for accessibility
        
    Returns:
        Complete HTML img tag
    """
    logo_url = get_embedded_logo_url()
    return f'<img src="{logo_url}" alt="{alt_text}" width="{width}" height="{height}" style="display:block; margin:0 auto;">'