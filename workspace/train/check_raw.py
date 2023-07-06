from workspace.chat.ChatGLM6BWrapper import ChatGLM6BWrapper
from workspace.env import base_model_path

if __name__ == '__main__':
    c = ChatGLM6BWrapper.load(base_model_path, ChatGLM6BWrapper.load_config(base_model_path, 128))
    c.start_in_cli(False)
