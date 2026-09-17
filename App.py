import os
import requests
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain.agents import create_agent


# -----------------------------
# Load API Keys
# -----------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Single AI Agent",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Single AI Agent")
st.write("Ask me anything about news, weather, or general questions.")


# -----------------------------
# Check API Keys
# -----------------------------
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing in your .env file.")
    st.stop()

if not TAVILY_API_KEY:
    st.error("TAVILY_API_KEY is missing in your .env file.")
    st.stop()


# -----------------------------
# Search Tool
# -----------------------------
search_tool = TavilySearch(
    max_results=3
)


# -----------------------------
# Weather Tool
# -----------------------------
@tool
def get_weather(city: str) -> str:
    """Fetch current weather information for a city."""

    api_key = os.getenv("WEATHERSTACK_API_KEY")

    if not api_key:
        return "WEATHERSTACK_API_KEY is not set."

    url = (
        "http://api.weatherstack.com/current"
        f"?access_key={api_key}"
        f"&query={city}"
    )

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if "current" not in data:
            return f"Could not fetch weather data for {city}."

        return (
            f"City: {city}\n"
            f"Temperature: {data['current']['temperature']}°C\n"
            f"Weather: {data['current']['weather_descriptions'][0]}\n"
            f"Humidity: {data['current']['humidity']}%"
        )

    except Exception as e:
        return f"Weather API error: {e}"


# -----------------------------
# Groq LLM
# -----------------------------
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY
)


# -----------------------------
# Create AI Agent
# -----------------------------
tools = [
    search_tool,
    get_weather
]

agent = create_agent(
    model=llm,
    tools=tools
)


# -----------------------------
# Chat History
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------
# User Input
# -----------------------------
user_input = st.chat_input(
    "Ask your question..."
)


if user_input:

    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)


    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = agent.invoke({
                    "messages": [
                        {
                            "role": "user",
                            "content": user_input
                        }
                    ]
                })

                answer = response["messages"][-1].content

            except Exception as e:

                answer = f"Error: {e}"

            st.markdown(answer)


    # Save response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })