import json
import requests
from openai import OpenAI

from utils.data_util import pr_quintuple_to_string
from ..CONSTANT import *
# 25.4.22 本节的主要目标，利用本地服务（DS系列）实现S-ACR的基本任务
class DeepSeekClient:
  def __init__(self, base_model="ds_671B"):
    if base_model in list(BASE_MODEL.keys()):
       api_key = BASE_MODEL[base_model]["API_KEY"]
       base_url = BASE_MODEL[base_model]["BASE_URL"]
       self.model_name = BASE_MODEL[base_model]["MODEL_NAME"]
    else:
        raise ValueError(f"Invalid base_model '{base_model}' provided. Please choose from: {', '.join(BASE_MODEL.keys())}")

    self.client = OpenAI(
        api_key=f"{api_key}",
        base_url=base_url,
    )

  def truncate_content(self, content: str, max_bytes: int = 64000) -> str:
    """Truncate content to fit within specified byte size"""
    encoded = content.encode('utf-8')
    if len(encoded) <= max_bytes:
        return content
    return encoded[:max_bytes].decode('utf-8', errors='ignore') + "...[TRUNCATED]"

  def perform_code_review(self, code_diff, pr_id, additional_context=None):
    
    if additional_context is not None:
        node_attr = additional_context
        system_prompt = """
          # Role & Goal
            You are an expert code reviewer for OpenSource Software project. Your task is to analyze the provided code diff, and deliver concise, actionable feedback in a strict JSON format.
          # Instructions:
            0.  **Context-Driven Focus**: Analyze the provided context and identify the primary goal or purpose of given code change (e.g., "Add HTTP feature", "Fix memory leak", "Refactor module X"). If the context is relevant and available, let it be the primary focus when reviewing the code hunk.
            1.  **Context Handling**: The context may be constructed from the conversation history and PR description. Use it to grasp the purpose of the change and ensure that the feedback is consistent with the context style.
            2. **Primary Focus**: Identify the single most important issue that could undermine this purpose (e.g., bug, security, performance, or deviation from best practices).
            3. **Scope**: Suggest a concrete improvement for this issue, focusing on clarity, maintainability, or efficiency.
            4. **Tone & Format**: Keep the feedback short, conversational, and limited to this key point.
          
          # Code Diff for Review: {code_diff} \n
          # Contextual Information: {node_attr} \n
          
          # Output Format:
            You MUST output a valid JSON object with the following exact keys:
            {{
              "code_review_suggestion": "A concise, conversational sentence or two.",
            }}
          """.format(node_attr=node_attr,code_diff=code_diff)
    else:
        system_prompt = """
              # Role & Goal
                You are an expert code reviewer for OpenSource Software project. Your task is to analyze the provided code diff, and deliver concise, actionable feedback in a strict JSON format.
              # Instructions:
                1. **Primary Focus**: Identify the single most important issue that could undermine this purpose (e.g., bug, security, performance, or deviation from best practices).
                2. **Scope**: Suggest a concrete improvement for this issue, focusing on clarity, maintainability, or efficiency.
                3. **Tone & Format**: Keep the feedback short, conversational, and limited to this key point.
              
              # Code Diff for Review: {code_diff}
              # Output Format:
                You MUST output a valid JSON object with the following exact keys:
                {{
                  "code_review_suggestion": "A concise, conversational sentence or two.",
                }}
              """.format(code_diff=code_diff)
    messages = [{"role": "user", "content": system_prompt}]
    response = self.client.chat.completions.create(model=self.model_name, messages=messages, temperature=0)
    data = response.choices[0].message.content
    data = data.strip("```").strip("json").strip()
    return data