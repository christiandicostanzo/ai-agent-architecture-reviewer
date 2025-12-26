import base64
import mimetypes
from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseChatModel

class DiagramReviewerAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def _get_image_data_url(self, image_path: str) -> str:
        """
        Reads an image file and converts it to a base64 data URL.
        """
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or mime_type not in ["image/jpeg", "image/png"]:
            raise ValueError("Unsupported image format. Please use JPEG or PNG.")

        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        
        return f"data:{mime_type};base64,{encoded_string}"

    def review_diagram(self, image_path: str, decision: str, feedback: str = None) -> str:
        """
        Analyzes if a decision is in accordance with the provided diagram image,
        optionally considering previous feedback.
        """
        image_data_url = self._get_image_data_url(image_path)
        
        prompt_text = f"Please analyze if the following decision is consistent with the architecture diagram provided.\n\nProposed Decision: {decision}"
        if feedback:
            prompt_text += f"\n\nContext/Previous Review: {feedback}\n\nPlease verify if the diagram supports this review and the decision."

        message = HumanMessage(
            content=[
                {
                    "type": "text", 
                    "text": prompt_text
                },
                {
                    "type": "image_url", 
                    "image_url": {"url": image_data_url}
                }
            ]
        )

        response = self.llm.invoke([message])
        return response.content
