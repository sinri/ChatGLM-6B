from workspace.chat.ChatGLM6BWrapper import ChatGLM6BWrapper

if __name__ == '__main__':
    model_path = 'E:\\OneDrive\\Leqee\\ai\\repo\\THUDM\\chatglm-6b'
    c = ChatGLM6BWrapper.load(model_path, ChatGLM6BWrapper.load_config(model_path, 128))
    c.start_in_cli(False)
