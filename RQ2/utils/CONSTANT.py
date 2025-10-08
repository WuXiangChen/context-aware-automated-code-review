
server_address = "XXXX"
port = 111

username = "XXX"
password = "XXX"

DEEPSEEK_API_KEY = "XXX"
MONICA_API_KEY = "XXX"
OPENAI_API_KEY = "XXX"


DS_BASE_URL = "https://api.deepseek.com"
MONICA_BASE_URL = "https://openapi.monica.im/v1"
OPENAI_HUB_URL = ""
## 这里需要实现一个字典，根据base_model来自动选择对应的URL和必要的参数
  
BASE_MODEL = {
  "ds":{
    "API_KEY":DEEPSEEK_API_KEY,
    "BASE_URL":DS_BASE_URL,
    "MODEL_NAME":"deepseek-chat"
  },
  "dsReasoner":{
    "API_KEY":DEEPSEEK_API_KEY,
    "BASE_URL":DS_BASE_URL,
    "MODEL_NAME":"deepseek-reasoner"
  },
  "gemini-2.5-pro":{
    "API_KEY":MONICA_API_KEY,
    "BASE_URL":MONICA_BASE_URL,
    "MODEL_NAME":"gemini-2.5-pro"
  },
  "gpt-4.1-mini":{
    "API_KEY":MONICA_API_KEY,
    "BASE_URL":MONICA_BASE_URL,
    "MODEL_NAME":"gpt-4.1-mini"
  },
  "gpt-5":{
    "API_KEY":MONICA_API_KEY,
    "BASE_URL":MONICA_BASE_URL,
    "MODEL_NAME":"gpt-5"
  }
  # "gpt-4.1-nano":{
  #   "API_KEY":MONICA_API_KEY,
  #   "BASE_URL":MONICA_BASE_URL,
  #   "MODEL_NAME":"gpt-4.1-nano"
}