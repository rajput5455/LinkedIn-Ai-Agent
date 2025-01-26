import os 
from dotenv import load_dotenv

from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo

# load enviroment variables
load_dotenv()

GROQ_API_KEY = os.environ['GROQ_API_KEY']

## web search agent
web_search_agent= Agent(
    name="News Relevance Validator",
    role="Critically evaluate AI news for social media posting",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Forcast news as a compelling Linkdin post.",
                  "Include 3-5 key AI news developments.",
                  "Write in a professional, engaging tone.",
                  "Use bulled points for readability.",
                  "Inculde relevent hashtage",
                  "Provide source links for credibility.",
                  "Highlight the borader impact of AI development.",
                  "End with an enagement prompt.",
                  "Ensure content is suitable for professional networking audience.",
                  "Keep the total post length under 3000 character.",
],

    show_tools_calls = True,
    markdown = True,
)

# News Revelance Agent
news_revelevence_agent = Agent(
    name = "News Relevence Validator",
    role = "Critically evalute AI news for social media posting",
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions=["Carefully assess the generated AI news content",
                  "Determine if the content is suitable for linkedin posting",
                   """Check for :
                     - Professinalise
                     - Current revelence
                     - Potential Impact 
                     - Absence of controvertial content""",
                   """Provide a structure evalution with:
                    - Suitability score (0-10)
                    - Posting recommendation (yes/no)
                    - Specific reasons for evalution""",
                   "If not suitable , explain specific reasons",
                   "Suggest potential modifications if needed ",
                   "Respond with 'No' in the posting recommendation if content is not suitable "
],
    show_tools_calls = True,
    markdown = True,       
)


# Execute web search for latest AI news

def main():
    news_response = web_search_agent.run("5 latest significant AI news developments wih sources", stream=False)
    
    # Validate the generated news content 
    validation_response = news_revelevence_agent.run(
        f"Evaluate the following AI content for linkedin posting suitability:\n\n{news_response.content}",
        stream = False 
    )

      # Check if validation recommends not posting
    news_content = news_response.content
    if "<function=duckduckgo_news" in validation_response.content:
        news_content = ""
    else:
        news_content = news_response.content
    # news_content = "" if "No" in validation_response.content else news_respone.content

    return {
        "news_content": news_content,
        "validation": validation_response.content
    }  
  
if __name__ == "__main__":
    result = main()
    print("generated_news:")
    print(result["news_content"])
    print("\n validation Result:")
    print(result["validation"])