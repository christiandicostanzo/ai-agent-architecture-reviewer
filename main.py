from typing import Dict
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from agents.decision_reviewer_agent import DecisionReviewerAgent
from agents.diagram_reviewer_agent import DiagramReviewerAgent
from dotenv import load_dotenv
import os

load_dotenv()

def create_review_chain():
    # Initialize Models
    # Note: Diagram agent needs a vision model like gpt-4o
    model_text = ChatOpenAI(model="gpt-3.5-turbo")
    model_vision = ChatOpenAI(model="gpt-4o")

    # Initialize Agents
    decision_agent = DecisionReviewerAgent(model_text)
    diagram_agent = DiagramReviewerAgent(model_vision)

    # 1. Define Step for Decision Review
    # Input: { "decision": ..., "context": ..., "image_path": ... }
    def run_decision_review(inputs: Dict) -> str:
        print(f"Running Decision Review for: {inputs['decision']}")
        return decision_agent.review(inputs["decision"], inputs.get("context", ""))

    # 2. Define Step for Diagram Review
    # It needs the original decision AND the output from step 1 (feedback)
    def run_diagram_review(inputs: Dict) -> str:
        print("Running Diagram Review...")
        image_path = inputs["image_path"]
        decision = inputs["decision"]
        feedback = inputs["decision_feedback"]
        
        return diagram_agent.review_diagram(image_path, decision, feedback)

    # 3. Construct the Chain using LCEL
    # We use RunnablePassthrough.assign to keep the original inputs and add the new 'decision_feedback'
    
    chain = (
        RunnablePassthrough.assign(
            decision_feedback=RunnableLambda(run_decision_review)
        )
        | RunnableLambda(run_diagram_review)
    )

    return chain

if __name__ == "__main__":
    # Test the chain
    chain = create_review_chain()
    
    # Example Usage inputs
    # Ensure you have a valid image path in 'architecture.png' or update the path
    inputs = {
        "decision": "We will replace the SQL Database with a NoSQL solution to improve scale.",
        "context": "The current system makes heavy use of complex joins for reporting.",
        "image_path": "architecture.png" 
    }
    
    # Only run if image exists to avoid errors in this dry run
    if os.path.exists(inputs["image_path"]):
        result = chain.invoke(inputs)
        print("\nFinal Analysis:\n", result)
    else:
        print(f"Please provide a valid image at {inputs['image_path']} to test.")
