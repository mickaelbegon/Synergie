import socket
import requests
import logging

logger = logging.getLogger(__name__)

def has_internet_connection() -> bool:
    """
    Check if the computer is connected to the internet using both socket and HTTP methods.
    Returns:
        bool: True if internet connection is available, False otherwise.
    """
    # First, attempt a socket connection
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(3)
            sock.connect(("8.8.8.8", 53))
        logger.info("Internet connection verified via socket.")
        return True
    except socket.timeout:
        logger.warning("Socket connection timed out.")
    except socket.error as e:
        logger.error(f"Socket error: {e}")
    except Exception as e:
        logger.error(f"Unexpected socket error: {e}")

    # If socket connection fails, attempt an HTTP request
    try:
        response = requests.get("https://www.google.com", timeout=3)
        if response.status_code == 200:
            logger.info("Internet connection verified via HTTP request.")
            return True
    except requests.ConnectionError:
        logger.warning("HTTP connection failed.")
    except requests.Timeout:
        logger.warning("HTTP connection timed out.")
    except Exception as e:
        logger.error(f"Unexpected HTTP error: {e}")

    # If both methods fail, assume no internet connection
    logger.error("No internet connection detected.")
    return False


def check_internet_connection() -> bool:
    """
    Check if the computer is connected to the internet.
    This function is a simple wrapper around has_internet_connection.
    It returns True if internet is connected, False otherwise.

    Returns:
        bool: True if internet connection is available, False otherwise.
    """
    return has_internet_connection()
