import gradio as gr

from workspace.chat.ChatGLM6BWrapper import ChatGLM6BWrapper
from workspace.env import base_model_path, workspace_in_win


class Service:
    def __init__(self):
        self.__wrapper = ChatGLM6BWrapper.load(base_model_path, ChatGLM6BWrapper.load_config(base_model_path, 128))
        self.__wrapper.load_extra_ckpt(f'{workspace_in_win}\\train\\plan_3\\output\\checkpoint-4000')

    def start_gradio(self):
        def question_answer(question: str):
            r = self.__wrapper.chat(question)
            self.__wrapper.clean_history()
            return r

        gr.Interface(fn=question_answer, inputs=["text"], outputs=["textbox"]).launch()


if __name__ == '__main__':
    s = Service()
    s.start_gradio()
