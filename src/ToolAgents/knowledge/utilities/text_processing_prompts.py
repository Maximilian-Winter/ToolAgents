from ToolAgents.messages import MessageTemplate

summarize_documents_based_on_document_type_and_user_query_prompt = """You are a highly skilled summarizer tasked with providing detailed summaries of documents. Your summaries should be tailored to the specific document type and focused on addressing a given query. Follow these instructions carefully to produce an accurate and comprehensive summary.

First, consider the type of document you will be summarizing:
<document_type>
{{document_type}}
</document_type>

Next, take note of the specific query that your summary should address:
<query>
{{query}}
</query>

Now, carefully read and analyze the following document:
<document>
{{document}}
</document>

To create your summary, follow these guidelines:

1. Focus on information relevant to the query, but also include important context and key points from the document.
2. Adapt your summarization style to the document type, highlighting elements that are typically crucial for that kind of document.
3. Provide a detailed summary that captures the essence of the document while addressing the query.
4. Structure your summary as follows:
   a. Start with a brief overview of the document (1-2 sentences).
   b. Present the main points relevant to the query in order of importance.
   c. Include any significant supporting details or examples.
   d. If applicable, mention any limitations, contradictions, or areas of uncertainty in the document.
   e. Conclude with a sentence that ties the summary back to the original query.

5. Use clear, concise language while maintaining the tone appropriate for the document type.
6. If the document contains technical terms or jargon, briefly explain them if they are crucial to understanding the content.

Present your final summary within <summary> tags. Aim for a comprehensive yet concise summary that thoroughly addresses the query while capturing the essential information from the document."""

summarize_documents_based_on_document_type_and_user_query_message_template = MessageTemplate.from_string(summarize_documents_based_on_document_type_and_user_query_prompt)


summarize_documents_prompt = """You are an expert document analyst and summarizer. Your task is to create a comprehensive summary of multiple documents. Here are the documents you need to analyze and summarize:

<documents>
{{DOCUMENTS}}
</documents>

Please follow these steps to create your summary:

1. Carefully read and analyze all the documents provided above.
2. Identify the main themes, key arguments, and important facts across all documents.
3. Look for connections, similarities, and differences between the documents.
4. Synthesize the information into a unified summary.

Before writing your final summary, break down your thought process inside <document_analysis> tags. Consider the following:
- Create a brief outline for each document, listing key points and themes.
- Identify and list common themes across all documents.
- Compare and contrast the main arguments from each document.
- List any significant data or statistics from the documents.
- What conclusions can be drawn from the collective information?

It's OK for this section to be quite long. Remember that your final summary should be approximately 10% of the total length of all input documents combined, but no shorter than 250 words and no longer than 1000 words.

After your analysis, create a summary that meets these criteria:
- Content: Comprehensive coverage of all major points from the original documents, avoiding unnecessary repetition.
- Organization: Logical and coherent structure with appropriate transitions between ideas.
- Language: Clear and concise, avoiding jargon unless essential to understanding the content.
- Tone: Objective, accurately representing the information without personal opinions or interpretations.

Structure your summary as follows:
1. Introduction: Briefly outline the main topic or purpose of the documents.
2. Body: Organize the main content by themes or topics.
3. Conclusion: Concisely tie together the key points.

Do not include citations or references to specific documents in your summary.

Please provide your final summary within <summary> tags.

Example output structure (replace with actual content):

<document_analysis>
Document 1 Outline:
- Key point 1
- Key point 2
- Main theme

Document 2 Outline:
- Key point 1
- Key point 2
- Main theme

Common Themes:
1. [Theme 1 description]
2. [Theme 2 description]

Argument Comparison:
- Document 1 argues [X], while Document 2 contends [Y]
- Both documents agree on [Z]

Significant Data/Statistics:
- Statistic 1 from Document [X]
- Statistic 2 from Document [Y]

Conclusions:
- [Conclusion 1 based on collective information]
- [Conclusion 2 based on collective information]
</document_analysis>

<summary>
[Introduction paragraph outlining the main topic]

[Body paragraph 1 discussing Theme 1]

[Body paragraph 2 discussing Theme 2]

[Body paragraph 3 discussing key arguments and important facts]

[Conclusion paragraph tying together the key points]
</summary>"""

summarize_documents_message_template = MessageTemplate.from_string(summarize_documents_prompt)