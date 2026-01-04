from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class DecisionReviewerAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

        system_prompt = """
        You are an expert in software architecture and cloud architecture. 
        You are an expert in AWS, Azure, GCP, etc.
        
        Your main task is to review a proposed decision and provide a critical review or feedback.
        """

        user_prompt = """
        A user provided a proposed decision. You must provide a critical review or feedback.
        
        Proposed Decision: {decision}

        Follow this instructions:
        - Provide a critical review or feedback.
        - Provide an output in json format with the following structure:
            {
                "feedback": "Your feedback here",
                "recommendations": "Your recommendations here",
                "advantages": "The advantages of this decision here",
                "disadvantages": "The disadvantages of this decision here"
            }
        - Your only output should be the json format above.            
        - Avoid any other text or explanation.
        """

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_prompt)
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def review(self, decision: str) -> str:
        """
        Reviews a proposed decision given some context.
        """
        return self.chain.invoke({
            "decision": decision,
        })
