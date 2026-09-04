"""LangChain prompt template demo using ChatAnthropic."""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that explains topics simply."),
        ("human", "Explain {topic} in {sentence_count} sentences."),
    ]
)
def generate_text_summary_prompt(text, num_words, tone):
    return (
        f"You are an experienced copywriter. Write a {num_words} word "
        f"summary of the following text, using a {tone} tone: {text}"
    )

model = ChatAnthropic(model="claude-haiku-4-5", max_tokens=1024)

chain = prompt | model | StrOutputParser()

if __name__ == "__main__":
    result = chain.invoke({"topic": "prompt engineering", "sentence_count": 3})
    print(result)
