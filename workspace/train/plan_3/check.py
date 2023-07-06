import time

from workspace.chat.ChatGLM6BWrapper import ChatGLM6BWrapper

if __name__ == '__main__':
    model_path = 'E:\\OneDrive\\Leqee\\ai\\repo\\THUDM\\chatglm-6b'
    c = ChatGLM6BWrapper.load(model_path, ChatGLM6BWrapper.load_config(model_path, 128))
    c.load_extra_ckpt('E:\\sinri\\ChatGLM-6B\\workspace\\train\\plan_3\\output\\checkpoint-2200')

    # c.start_in_cli(False)

    questions = [
        '胶原蛋白对人有用吗？',
        '用这个过敏了',
        '最新活动价是什么？',
        '面膜要怎么用',
        '可以帮我介绍下 AHC 么？',
        '你们的小神仙水成分怎么样呀',
        'AHC的产地是哪里啊',
        '你们的水里有没有核辐射啊',
    ]

    for question in questions:
        print(f'[{time.time()}] > {question}')
        answer = c.chat(query=question)
        print(f'[{time.time()}] < {answer}')
