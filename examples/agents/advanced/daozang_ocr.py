import json
from dotenv import load_dotenv

from ToolAgents.agents import ChatToolAgent
from ToolAgents.data_models.messages import ChatMessage
from ToolAgents.provider import OpenAIChatAPI

load_dotenv()

# System prompt for Daozang OCR
SYSTEM_PROMPT_DAOZANG_OCR = """You are a specialized OCR system for Classical Chinese Daoist scriptures (é“è—).

## Task
Extract Chinese text from images of traditional woodblock-printed Daoist texts with maximum accuracy.

## Reading Direction
- Read vertically from TOP to BOTTOM
- Read columns from RIGHT to LEFT
- This is traditional Chinese text layout

## Text Characteristics
- Traditional Chinese characters (ç¹é«”å­—), NOT simplified
- Classical Chinese (æ–‡è¨€æ–‡) grammar
- Daoist technical terminology

## Common Terms to Recognize Correctly
These terms appear frequently - ensure correct recognition:

- éˆå¯¶ (LÃ­ngbÇŽo) - NOT éˆå¯³
- ç„¡é‡ (wÃºliÃ ng) - NOT æœ‰é‡
- å…ƒå§‹å¤©å°Š (YuÃ¡nshÇ TiÄnzÅ«n) - primordial deity name
- ç¢§è½ (bÃ¬luÃ²) - azure void - NOT ç¹è½
- çœ¾çœŸ (zhÃ²ngzhÄ“n) - assembled perfected ones - NOT æ³‰çœŸ
- ç—¼ç–¾ (gÃ¹jÃ­) - chronic illness - NOT ç™¡ç–¾
- æ‡·å§™ (huÃ¡irÃ¨n) - pregnant - NOT æ‡·é›‰, æ‡·å§ª
- åé»‘ (fÇŽnhÄ“i) - return to black - NOT åŠé»‘
- åå£¯ (fÇŽnzhuÃ ng) - return to vigor - NOT åŠå£¯
- ç™¼æ³„ (fÄxiÃ¨) - reveal - NOT ç™¼ä¸–
- é‡‘çŽ‰ (jÄ«nyÃ¹) - gold and jade - NOT é‡‘æ¯›
- ä¸¹éœ„ (dÄnxiÄo) - red clouds - NOT èˆŸéœ„
- èªªç¶“ä¸€å - first recitation (å not ç·¨)

## Output Format
- Output extracted text only
- Use <div> and <p> tags for structure
- Do NOT translate or explain
- Do NOT repeat sections
- Do NOT hallucinate content not in image"""

# User prompt
USER_PROMPT_DAOZANG = """Extract the Chinese text from this Daoist scripture image.

Source: é“è— (Daoist Canon)
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
