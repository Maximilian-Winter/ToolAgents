import json
from dotenv import load_dotenv

from ToolAgents.agents import ChatToolAgent
from ToolAgents.data_models.messages import ChatMessage
from ToolAgents.provider import OpenAIChatAPI

load_dotenv()

# System prompt for Daozang OCR
SYSTEM_PROMPT_DAOZANG_OCR = """You are a specialized OCR system for Classical Chinese Daoist scriptures (道藏).

## Task
Extract Chinese text from images of traditional woodblock-printed Daoist texts with maximum accuracy.

## Reading Direction
- Read vertically from TOP to BOTTOM
- Read columns from RIGHT to LEFT
- This is traditional Chinese text layout

## Text Characteristics
- Traditional Chinese characters (繁體字), NOT simplified
- Classical Chinese (文言文) grammar
- Daoist technical terminology

## Common Terms to Recognize Correctly
These terms appear frequently - ensure correct recognition:

- 靈寶 (Língbǎo) - NOT 靈寳
- 無量 (wúliàng) - NOT 有量
- 元始天尊 (Yuánshǐ Tiānzūn) - primordial deity name
- 碧落 (bìluò) - azure void - NOT 睹落
- 眾真 (zhòngzhēn) - assembled perfected ones - NOT 泉真
- 痼疾 (gùjí) - chronic illness - NOT 癡疾
- 懷姙 (huáirèn) - pregnant - NOT 懷雉, 懷姪
- 反黑 (fǎnhēi) - return to black - NOT 及黑
- 反壯 (fǎnzhuàng) - return to vigor - NOT 及壯
- 發泄 (fāxiè) - reveal - NOT 發世
- 金玉 (jīnyù) - gold and jade - NOT 金毛
- 丹霄 (dānxiāo) - red clouds - NOT 舟霄
- 說經一偏 - first recitation (偏 not 編)

## Output Format
- Output extracted text only
- Use <div> and <p> tags for structure
- Do NOT translate or explain
- Do NOT repeat sections
- Do NOT hallucinate content not in image"""

# User prompt
USER_PROMPT_DAOZANG = """Extract the Chinese text from this Daoist scripture image.

Source: 道藏 (Daoist Canon)
Read: Right-to-left, top-to-bottom
Output: Text only, traditional characters"""

api = OpenAIChatAPI(
    api_key="token-abc123",
    base_url="http://127.0.0.1:8080/v1",
    model="chandra"
)

agent = ChatToolAgent(chat_api=api)

settings = api.get_default_settings()
settings.temperature = 0.1
settings.top_p = 1.0
settings.extra_body = {"min_p": 0.00}


def ocr_daozang_page(image_path: str) -> str:
    """OCR a single Daozang page."""
    messages = [
        ChatMessage.create_system_message(SYSTEM_PROMPT_DAOZANG_OCR),
    ]

    user_msg = ChatMessage.create_empty_user_message()
    user_msg.add_image_file_data(image_path, "png")
    user_msg.add_text(USER_PROMPT_DAOZANG)
    messages.append(user_msg)

    result = agent.get_response(messages=messages, settings=settings)
    return result.content


def ocr_daozang_page_streaming(image_path: str):
    """OCR a single Daozang page with streaming output."""
    messages = [
        ChatMessage.create_system_message(SYSTEM_PROMPT_DAOZANG_OCR),
    ]

    user_msg = ChatMessage.create_empty_user_message()
    user_msg.add_image_file_data(image_path, "png")
    user_msg.add_text(USER_PROMPT_DAOZANG)
    messages.append(user_msg)

    full_text = ""
    result = agent.get_streaming_response(messages=messages, settings=settings)
    for res in result:
        print(res.chunk, end="", flush=True)
        full_text += res.chunk

    return full_text


# Example usage
if __name__ == "__main__":
    image_path = "daozang_01_book_page_7.png"
    text = ocr_daozang_page_streaming(image_path)
    print("\n\n--- Extracted Text ---")
    print(text)