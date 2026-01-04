import os
import sys
from dotenv import load_dotenv, dotenv_values
from loguru import logger
from agents.chain_agent_reviewer import ChainAgentReviewer

# Load environment variables
load_dotenv()

config = dotenv_values(".env")

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")

# Check for API Key
if not config['OPENAI_API_KEY']:
    logger.warning("OPENAI_API_KEY not found in environment variables. Please set it in .env file.")

def review_architure(decision: str, context: str, image_path: str):
     logger.info("Initializing Chain Agent Reviewer...")
    try:
        reviewer = ChainAgentReviewer(api_key=config['OPENAI_API_KEY'])
    
        print("\n--- AI Agent Architecture Reviewer ---")
        print("This tool reviews an architectural decision against a diagram.")
        
        decision = input("\nEnter the proposed decision (e.g., 'Migrate to Microservices'): ")
        if not decision: 
            logger.info("No decision provided. Exiting.")
            return

        result = reviewer.run(decision=decision)

        if img_input:
            image_path = img_input
            
        if os.path.exists(image_path):
            logger.info("Processing... This may take a moment.")
            result = reviewer.run(decision=decision, context=context, image_path=image_path)
            print("\nFinal Analysis:\n", result)
        else:
            logger.error(f"Image file '{image_path}' does not exist. Cannot proceed with Diagram Review.")

    except Exception as e:
        logger.exception(f"An error occurred: {e}")


def review_architure_with_diagram():
    logger.info("Initializing Chain Agent Reviewer...")
    try:
        reviewer = ChainAgentReviewer(api_key=config['OPENAI_API_KEY'])
        
        # Example Usage inputs
        # Ensure you have a valid image path. 
        # For demonstration, we'll check if a dummy path exists or ask the user to provide one.
        image_path = "architecture.png" 
        
        if not os.path.exists(image_path):
             logger.info(f"Note: '{image_path}' not found. Please ensure you have an image file if you want to test the full flow.")
             # We can't really run the vision part without a real image, so we'll just print a message.
             # But if the user provides one, it will work.
        
        # Using print for UI interaction is still cleaner than logging, or we can use input directly.
        # But we will use log info for status updates.
        print("\n--- AI Agent Architecture Reviewer ---")
        print("This tool reviews an architectural decision against a diagram.")
        
        decision = input("\nEnter the proposed decision (e.g., 'Migrate to Microservices'): ")
        if not decision: 
            logger.info("No decision provided. Exiting.")
            return

        context = input("Enter context (optional): ")
        
        img_input = input(f"Enter path to architecture diagram image (default: {image_path}): ")

        result = reviewer.run(decision=decision, context=context, image_path=image_path)

        if img_input:
            image_path = img_input
            
        if os.path.exists(image_path):
            logger.info("Processing... This may take a moment.")
            result = reviewer.run(decision=decision, context=context, image_path=image_path)
            print("\nFinal Analysis:\n", result)
        else:
            logger.error(f"Image file '{image_path}' does not exist. Cannot proceed with Diagram Review.")

    except Exception as e:
        logger.exception(f"An error occurred: {e}")

def main():
    review_architure()

if __name__ == "__main__":
    main()
