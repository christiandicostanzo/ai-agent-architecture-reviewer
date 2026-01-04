from typing import Dict
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from agents.decision_reviewer_agent import DecisionReviewerAgent
from agents.diagram_reviewer_agent import DiagramReviewerAgent
from loguru import logger


class ChainAgentDecisionReviewer:
    
    def __init__(self, api_key: str):

        self.model_text = ChatOpenAI(model="gpt-4o-mini-2024-07-18", api_key=api_key)

        self.decision_agent = DecisionReviewerAgent(self.model_text)
        self.chain = self._create_review_chain()

    def _create_review_chain(self):
        # 1. Define Step for Decision Review
        def run_decision_review(inputs: Dict) -> str:
            logger.info(f"Running Decision Review for: {inputs['decision']}")
            return self.decision_agent.review(inputs["decision"], inputs.get("context", ""))

        # 3. Construct the Chain using LCEL
        chain = (
            RunnablePassthrough.assign(
                decision_feedback=RunnableLambda(run_decision_review)
            )
        )
        return chain

    def run(self, decision: str) -> str:
        """
        Executes the review chain.
        """
        inputs = {
            "decision": decision,
        }
        return self.chain.invoke(inputs)

class ChainAgentReviewer:

    def __init__(self, api_key: str):
        # Initialize Models
        # Note: Diagram agent needs a vision model like gpt-4o

        self.model_text = ChatOpenAI(model="gpt-4o-mini-2024-07-18", api_key=api_key)
        self.model_vision = ChatOpenAI(model="gpt-4o-mini-2024-07-18", api_key=api_key)

        # Initialize Agents
        self.decision_agent = DecisionReviewerAgent(self.model_text)
        self.diagram_agent = DiagramReviewerAgent(self.model_vision)
        
        # Build the chain
        self.chain = self._create_review_chain()

    def _create_review_chain(self):
        # 1. Define Step for Decision Review
        def run_decision_review(inputs: Dict) -> str:
            logger.info(f"Running Decision Review for: {inputs['decision']}")
            return self.decision_agent.review(inputs["decision"], inputs.get("context", ""))

        # 2. Define Step for Diagram Review
        def run_diagram_review(inputs: Dict) -> str:
            logger.info("Running Diagram Review...")
            image_path = inputs["image_path"]
            decision = inputs["decision"]
            feedback = inputs["decision_feedback"]
            
            return self.diagram_agent.review_diagram(image_path, decision, feedback)

        # 3. Construct the Chain using LCEL
        chain = (
            RunnablePassthrough.assign(
                decision_feedback=RunnableLambda(run_decision_review)
            )
            | RunnableLambda(run_diagram_review)
        )
        return chain

    def run(self, decision: str, image_path: str) -> str:
        """
        Executes the review chain.
        """
        inputs = {
            "decision": decision,
            "image_path": image_path
        }
        return self.chain.invoke(inputs)
