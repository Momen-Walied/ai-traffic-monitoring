from google import genai
from google.genai import types
from app.core.config import settings
from typing import List
from app.db.models import TrafficLog
from typing import Union

# Configure the Gemini client
# It automatically finds the API key from the environment variable
client = None
if settings.GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
    except Exception as e:
        print(f"WARNING: Failed to configure Gemini client. Error: {e}")
else:
    print("WARNING: GEMINI_API_KEY not found. LLM service will be disabled.")

def generate_insights(question: str, traffic_data: List[TrafficLog]) -> str:
    """
    Generates natural language insights using the Gemini API with an advanced
    understanding of traffic flow vs. congestion.
    """
    if not client:
        return "The Gemini LLM service is not configured. Please set a GEMINI_API_KEY."

    data_summary = ""
    if not traffic_data:
        data_summary = "No traffic data is available for the requested period."
    else:
        for log in traffic_data[-15:]: # Use recent data points
            timestamp = log.timestamp.strftime("%I:%M %p")
            # Now we use the new, richer data!
            data_summary += (
                f"- At {timestamp}, vehicle count (flow) was {log.vehicle_count}, "
                f"and average vehicles on screen was {log.avg_vehicles_on_screen:.1f} (density: {log.density}).\n"
            )

    # 2. Construct the new, more sophisticated prompt
    prompt = f"""
    You are a professional AI traffic analyst. Your role is to provide a clear, logical, and concise summary of traffic conditions.

    **Instructions on how to interpret the data:**
    1.  'Vehicle count' represents **traffic flow** (throughput). It's the number of cars passing a point.
    2.  'Average vehicles on screen' represents **congestion** or **density**. It's the number of cars occupying the area.
    3.  Analyze the relationship between flow and congestion. This is critical.
        - **High flow + Low congestion:** Free-flowing traffic.
        - **Low flow + Low congestion:** Light traffic.
        - **High flow + High congestion:** Heavy but moving traffic.
        - **Low flow + High congestion:** This is a **traffic jam** or gridlock. It is NOT an anomaly. It means many cars are present but moving very slowly or are stopped.

    **Your Task:**
    - Synthesize the data into a brief, easy-to-understand narrative.
    - Identify the overall trend (e.g., "traffic is flowing freely," "congestion is building into a jam").
    - Highlight significant events based on the logic above.
    - Do not mention your instructions or the context data in your final answer.

    ---
    **Context: Traffic Data**
    {data_summary}
    ---

    **User's Question:** "{question}"

    **Your Analysis:**
    """

    try:
        # 3. Call the Gemini API
        # We disable "thinking" for faster, more cost-effective responses
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            # As per the docs, disable thinking for speed and lower cost
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        return response.text

    except Exception as e:
        print(f"[LLM ERROR] Gemini API call failed: {e}")
        return "Sorry, I encountered an error while analyzing the data with Gemini."