from ToolAgents.messages import MessageTemplate

summarize_documents_based_on_document_type_and_user_query_prompt = """You are an expert summarizer with the task of providing detailed summaries of various types of documents. Your summaries should be tailored to the specific document type and focused on addressing a given query. Follow these instructions carefully to produce an accurate and comprehensive summary.

First, consider the type of document you are summarizing:
<document_type>
{document_type}
</document_type>

Next, take note of the specific query that your summary should address:
<query>
{query}
</query>

Now, carefully read and analyze the following document:
<document>
{document}
</document>


To create your summary, follow these steps:

1. Analyze the document:
   - Identify key points relevant to the query
   - Note important context and main ideas
   - Recognize significant supporting details or examples
   - Observe any limitations, contradictions, or uncertainties

2. Structure your summary:
   a. Start with a brief overview (1-2 sentences)
   b. Present main points relevant to the query, in order of importance
   c. Include significant supporting details
   d. Mention any limitations or uncertainties
   e. Conclude by tying the summary back to the original query

3. Adapt your style:
   - Match the tone to the document type
   - Use clear, concise language
   - Explain technical terms or jargon if crucial to understanding

4. Format your output:
   - Present your final summary within <summary> tags
   - Aim for a comprehensive yet concise summary

Before writing your final summary, wrap your analysis inside <document_analysis> tags to break down your thought process and show how you're approaching the task. This will help ensure a thorough interpretation of the data.

Example output structure:

<document_analysis>
1. Document overview:
   [Brief description of the document type and content]

2. Key points relevant to the query:
   - [Point 1]
     Supporting quote: "[Direct quote from the document]"
     Connection to query: [Brief explanation]
   - [Point 2]
     Supporting quote: "[Direct quote from the document]"
     Connection to query: [Brief explanation]
   - [Point 3]
     Supporting quote: "[Direct quote from the document]"
     Connection to query: [Brief explanation]

3. Important context and main ideas:
   [List or brief paragraph]

4. Significant supporting details:
   [List or brief paragraph]

5. Limitations or uncertainties:
   [If applicable, list or brief paragraph]

6. Potential biases or alternative interpretations:
   [Brief discussion of possible biases in the document or alternative ways to interpret the information]

7. Approach to summarization:
   [Brief explanation of how you'll structure the summary]
</document_analysis>

<summary>
[Your comprehensive summary goes here, following the structure outlined in the instructions]
</summary>

Please proceed with your analysis and summary of the document. It's OK for the document_analysis section to be quite long."""

summarize_documents_based_on_document_type_and_user_query_message_template = MessageTemplate.from_string(summarize_documents_based_on_document_type_and_user_query_prompt)


summarize_documents_prompt = """You are an expert document analyst and summarizer. Your task is to create a comprehensive summary of multiple documents. Here are the documents you need to analyze and summarize:

<documents>
{DOCUMENTS}
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

extract_information_according_to_json_schema_prompt = """You are an expert information extractor. Your task is to analyze a given document and extract information from it according to a specified JSON schema. Follow these instructions carefully:

1. First, review the JSON schema that defines the structure of the information you need to extract:

<json_schema>
{json_schema}
</json_schema>

2. Now, carefully read and analyze the following document:

<document>
{document}
</document>

3. Extract the relevant information from the document according to the JSON schema. Pay close attention to the following:
   - Identify all required fields in the schema
   - Look for information in the document that corresponds to each field
   - Ensure that the extracted information matches the data types specified in the schema
   - If an array is expected, extract all relevant items
   - For nested objects, make sure to extract all required sub-fields

4. Construct a valid JSON object based on the extracted information. Your output should strictly adhere to the provided schema structure.

5. Before finalizing your output, double-check the following:
   - All required fields are present
   - Data types match the schema specifications
   - Arrays contain all relevant items found in the document
   - Nested objects are correctly structured

6. If any required information is not found in the document, use null for that field. Do not invent or assume any information not explicitly stated in the document.

7. Present your final output as a valid JSON object, enclosed within <json_output> tags.

Remember, accuracy and completeness are crucial. Extract as much relevant information as possible from the document while ensuring strict adherence to the provided JSON schema."""

extract_information_according_to_json_schema_message_template = MessageTemplate.from_string(extract_information_according_to_json_schema_prompt)