from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class DecisionReviewerAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert Decision Reviewer. Your goal is to analyze proposed decisions, identify potential biases, logical fallacies, or overlooked consequences, and provide a critical review."),
            ("human", "Context: {context}\n\nProposed Decision: {decision}\n\nPlease review this decision.")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def review(self, decision: str, context: str = "") -> str:
        """
        Reviews a proposed decision given some context.
        """
        return self.chain.invoke({
            "decision": decision,
            "context": context
        })
