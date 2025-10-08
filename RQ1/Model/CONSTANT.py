
server_address = "210.28.133.13"
port = 22207

username = "wxc@nju.edu.cn"
password = "wxc123"

##BBBB
DEEPSEEK_API_KEY = "sk-182d8ed5c3ec4c46ad8e1846a29031a7"
MONICA_API_KEY = ""

DS_BASE_URL = "https://api.deepseek.com"
MONICA_BASE_URL = "https://openapi.monica.im/v1"

## 这里需要实现一个字典，根据base_model来自动选择对应的URL和必要的参数
BASE_MODEL = {
  "ds_671B":{
    "API_KEY":DEEPSEEK_API_KEY,
    "BASE_URL":DS_BASE_URL,
    "MODEL_NAME":"deepseek-chat"
  },
  "gpt_4o":{
    "API_KEY":MONICA_API_KEY,
    "BASE_URL":MONICA_BASE_URL,
    "MODEL_NAME":"gpt-4o"
  },
  "ds_coder_v2":{
    "API_KEY":DEEPSEEK_API_KEY,
    "BASE_URL":DS_BASE_URL,
    "MODEL_NAME":"deepseek-coder-v2"
  },
  "ds_reasoner":{
    "API_KEY":DEEPSEEK_API_KEY,
    "BASE_URL":DS_BASE_URL,
    "MODEL_NAME":"deepseek-reasoner"
  },
  "qwen_2.5":{
    "API_KEY":MONICA_API_KEY,
    "BASE_URL":MONICA_BASE_URL,
    "MODEL_NAME":"qwen-2.5-72b-instruct"
  },
  "llama_3.1":{
    "API_KEY":MONICA_API_KEY,
    "BASE_URL":MONICA_BASE_URL,
    "MODEL_NAME":"llama-3.1-405b-instruct"
  }
}