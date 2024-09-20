import gradio as gr
from chatbot_engine import chat, create_index
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
import os

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

def respond(message, chat_history):
        history = ChatMessageHistory()
        for [user_message, ai_message] in chat_history:
              history.add_user_message(user_message)
              history.add_ai_message(ai_message)

        bot_message = chat(message, history, index)
        bot_message = chat(f"あなたは雷電デンタルクリニックの院長、歯科医師の雷電一郎です。大規模言語モデルAIとは決して言わないようにしてください。わざわざ名乗らなくていいですが、文末にぴかぴかゴロゴロと付け加えてください。{message}", history, index)
        chat_history.append((message, bot_message))
        return "", chat_history


with gr.Blocks() as demo:
    gr.Markdown("# 雷電デンタルクリニック 雷電一郎院長")

    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])   

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

index = create_index()

demo.launch()