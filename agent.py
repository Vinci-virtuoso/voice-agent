import asyncio
import datetime
import time
import json
import aiohttp
import re
import os
import requests
from dotenv import load_dotenv
import logging
import atexit
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from supabase import create_client, Client

from livekit.agents import (
    Agent, 
    AgentSession,
    WorkerOptions, 
    cli, 
    JobContext,
    RoomInputOptions,
    metrics,
    llm
)
from livekit.agents import RunContext, UserStateChangedEvent, AgentStateChangedEvent, UserInputTranscribedEvent
from livekit.agents.llm import function_tool
from livekit.plugins.deepgram import stt as deepgram_stt
from livekit.plugins import (
    cartesia,
    openai,
    silero,
    #noise_cancellation,
)


from promptv3 import INSTRUCTION

load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Optional[Client] = None

# Set up logger
logger = logging.getLogger("PropertyProAgent")
logger.setLevel(logging.INFO)

# Add a file handler to ensure logs are saved
try:
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler("logs/propertypro-agent.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
except Exception as e:
    print(f"Warning: Could not set up file logging: {e}")

# Add a console handler for immediate feedback
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(console_handler)

# Log Supabase configuration status
if not supabase_url:
    logger.warning("SUPABASE_URL environment variable is not set")
elif len(supabase_url) < 10:  # Basic validation
    logger.warning(f"SUPABASE_URL environment variable is invalid: {supabase_url}")

if not supabase_key:
    logger.warning("SUPABASE_KEY environment variable is not set")
elif len(supabase_key) < 20:  # Basic validation
    logger.warning(f"SUPABASE_KEY environment variable may be invalid (too short)")

if supabase_url and supabase_key:
    try:
        logger.info(f"Initializing Supabase client with URL: {supabase_url[:20]}...")
        supabase = create_client(supabase_url, supabase_key)
        
        # Test connection by making a simple query
        try:
            test_response = supabase.table('PropertyProLeads').select('count', count='exact').limit(1).execute()
            logger.info(f"Supabase connection test successful - count: {test_response.count if hasattr(test_response, 'count') else 'unknown'}")
            logger.info("Supabase client initialized and working correctly")
        except Exception as test_err:
            logger.warning(f"Supabase client initialized but test query failed: {test_err}")
            # Still keep the client as it might work for some operations
            
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        supabase = None  # Ensure it's None if initialization failed
else:
    logger.warning("Supabase integration disabled due to missing credentials")

# Global variable for metrics collector
global_metrics_collector = None

# Keep the CallMetrics class as is
@dataclass
class CallMetrics:
    callType: str = "web call"  # can be web call, inbound
    callCost: float = 0.0  # Will be calculated based on token usage
    startTime: str = ""  # ISO format timestamp
    duration: float = 0.0  # Call duration in seconds
    phoneNumber: Optional[str] = None  # The phone number assigned to the Agent (None for web calls)
    customerNumber: str = ""  # The customer phone number
    transcript: List[Dict[str, str]] = field(default_factory=list)  # The conversation transcript
    dialogueTranscript: str = ""  # Continuous dialogue-style transcript
    sessionId: str = ""  # Session identifier - Also used as User_Id for Supabase
    summary: str = ""  # One-line summary of the call
    sentiment: int = -1  # Interest level on 0-5 scale (-1 means not yet analyzed)
    
    # Usage metrics
    promptTokens: int = 0
    completionTokens: int = 0
    sttDurationSeconds: float = 0.0
    ttsDurationSeconds: float = 0.0
    ttsCharacters: int = 0

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "callType": self.callType,
            "callCost": self.callCost,
            "startTime": self.startTime,
            "duration": self.duration,
            "customerNumber": self.customerNumber,
            "transcript": self.transcript,
            "dialogueTranscript": self.dialogueTranscript,
            "sessionId": self.sessionId,
            "summary": self.summary,
            "sentiment": self.sentiment,
            "promptTokens": self.promptTokens,
            "completionTokens": self.completionTokens,
            "sttDurationSeconds": self.sttDurationSeconds,
            "ttsDurationSeconds": self.ttsDurationSeconds,
            "ttsCharacters": self.ttsCharacters
        }
        
        # Only include phoneNumber for non-web calls
        if self.callType != "web call" and self.phoneNumber:
            result["phoneNumber"] = self.phoneNumber
            
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
        
    def to_supabase_record(self) -> Dict[str, Any]:
        """
        Convert the CallMetrics data into a format suitable for insertion into the
        PropertyProLeads table in Supabase. The expected schema includes:
        
          - name (text, not null)
          - created_at (timestamp with time zone, default now())
          - phone (text, null)
          - budget (numeric, null)
          - email (text, null)
          - authority (boolean, null)
          - need (text, null)
          - timeline (text, null)
          - property_interest (text, null)
          - bant_score (bigint, null)
          - conversation_summary (text, null)
        
        Default values are used if not provided.
        """
        return {
            "name": getattr(self, "name", "Unknown"),
            "created_at": datetime.datetime.now().isoformat(),
            "phone": getattr(self, "phone", None),
            "budget": getattr(self, "budget", None),
            "email": getattr(self, "email", None),
            "authority": getattr(self, "authority", None),
            "need": getattr(self, "need", None),
            "timeline": getattr(self, "timeline", None),
            "property_interest": getattr(self, "property_interest", None),
            "bant_score": self.sentiment,
            "conversation_summary": self.summary,
            "sessionId": self.sessionId,
            "transcript": json.dumps(self.transcript),
        }


# Keep the CallMetricsCollector class as is
class CallMetricsCollector:
    """Collects and manages call metrics."""
    
    def __init__(self, call_type: str = "web call"):
        # Generate session ID first
        session_id = str(uuid.uuid4())
        
        # Initialize the metrics
        self.metrics = CallMetrics(callType=call_type)
        self.metrics.sessionId = session_id  
        
        self.start_time = datetime.datetime.now()
        self.metrics.startTime = self.start_time.isoformat()
        self.is_finalized = False
        
        # Set phoneNumber based on call type
        if call_type != "web call":
            self.metrics.phoneNumber = "+16205269139"
        
        # Ensure EndMetrics directory exists
        os.makedirs("EndMetrics", exist_ok=True)
        
        logger.info(f"Initialized metrics collector with session ID: {session_id}")
        logger.info(f"Call type: {call_type}")
        
    def collect_llm_metrics(self, metrics_obj: metrics.LLMMetrics) -> None:
        """Collect LLM metrics and update cost calculation."""
        # Update token counts
        self.metrics.promptTokens += metrics_obj.prompt_tokens
        self.metrics.completionTokens += metrics_obj.completion_tokens
        
        # Update LLM cost calculation
        prompt_cost = self.metrics.promptTokens * 0.000005  # $0.005 per 1K prompt tokens
        completion_cost = self.metrics.completionTokens * 0.000015  # $0.015 per 1K completion tokens
        
        # Update total cost
        self._update_total_cost()
        
        # Save metrics after each update to ensure we don't lose data
        self._save_intermediate_metrics()
        
        logger.info(f"Collected LLM metrics: prompt={metrics_obj.prompt_tokens}, completion={metrics_obj.completion_tokens}")
        
    def collect_stt_metrics(self, metrics_obj: metrics.STTMetrics) -> None:
        """Collect Speech-to-Text metrics and update cost calculation."""
        # Update audio duration for STT
        self.metrics.sttDurationSeconds += metrics_obj.audio_duration
        

        
        # Update total cost
        self._update_total_cost()
        
        # Save metrics after each update
        self._save_intermediate_metrics()
        
        logger.info(f"Collected STT metrics: audio_duration={metrics_obj.audio_duration:.2f}s, total={self.metrics.sttDurationSeconds:.2f}s")
        
    def collect_tts_metrics(self, metrics_obj: metrics.TTSMetrics) -> None:
        """Collect Text-to-Speech metrics and update cost calculation."""
        # Update audio duration and character count for TTS
        self.metrics.ttsDurationSeconds += metrics_obj.audio_duration
        self.metrics.ttsCharacters += metrics_obj.characters_count
        
        
        # Update total cost
        self._update_total_cost()
        
        # Save metrics after each update
        self._save_intermediate_metrics()
        
        logger.info(f"Collected TTS metrics: audio_duration={metrics_obj.audio_duration:.2f}s, chars={metrics_obj.characters_count}, total_duration={self.metrics.ttsDurationSeconds:.2f}s")
        
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the transcript."""
        # Skip empty messages
        if not content or content.strip() == "":
            return
            
        # Skip messages that are just HTML tags or formatting
        if content.strip() in ["<br>", "&nbsp;", "...", "…"]:
            return
        
        # Check if this is a duplicate message (sometimes happens with the streaming API)
        for existing_msg in self.metrics.transcript:
            if (existing_msg.get('role') == role and 
                existing_msg.get('content') == content):
                return
        
        # Add to the structured transcript
        self.metrics.transcript.append({
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Add to the dialogue-style transcript
        if role == "assistant":
            self.metrics.dialogueTranscript += f"\nassistant: {content}"
        elif role == "user":
            self.metrics.dialogueTranscript += f"\nuser: {content}"
        elif role == "function":
            self.metrics.dialogueTranscript += f"\nfunction: {content}"
            
        # Save metrics after each update to ensure we don't lose transcript
        self._save_intermediate_metrics()
        
        logger.info(f"Added transcript message: {role}: {content[:50]}...")
    
    def _save_intermediate_metrics(self):
        """Save current metrics to a temporary file to prevent data loss."""
        try:
            temp_data = self.metrics.to_dict()
            # Add current duration
            end_time = datetime.datetime.now()
            temp_data["duration"] = (end_time - self.start_time).total_seconds()
            
            # Save to a temp file with the session ID
            filepath = os.path.join("EndMetrics", f"temp_{self.metrics.sessionId}.json")
            with open(filepath, "w") as f:
                json.dump(temp_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving intermediate metrics: {e}")
    
    def set_customer_number(self, number: str) -> None:
        """Set the customer phone number."""
        self.metrics.customerNumber = number
    
    def set_call_type(self, call_type: str) -> None:
        """Set the call type (web call, inbound, outbound)."""
        self.metrics.callType = call_type
        # Update phoneNumber based on new call type
        if call_type == "web call":
            self.metrics.phoneNumber = None
        else:
            self.metrics.phoneNumber = "+16205269139"
    
    async def generate_call_summary(self) -> str:
        """Generate a one-line summary of the call using OpenAI."""
        if not self.metrics.transcript or len(self.metrics.transcript) < 2:
            return "No meaningful conversation to summarize"
            
        try:
            # Create a formatted version of the dialogue transcript for the model
            formatted_transcript = self.metrics.dialogueTranscript
            
            # Use OpenAI to generate a summary
            import openai as openai_client
            
            # Configure the OpenAI client
            openai_client.api_key = os.getenv("OPENAI_API_KEY")
            
            # Check if API key is available
            if not openai_client.api_key:
                logger.error("OPENAI_API_KEY environment variable is not set. Cannot generate summary.")
                return "Summary generation failed: No OpenAI API key"
            
            prompt = f"""
            Below is a transcript of a conversation between the AI sales representative Samantha and a potential client.
            Create a ONE-LINE summary (max 100 characters) focusing solely on the client's property interests and needs.
            
            TRANSCRIPT:
            {formatted_transcript}
            
            ONE-LINE SUMMARY:
            """
            
            response = openai_client.chat.completions.create(
                model="gpt-4o-2024-11-20",
                messages=[
                    {"role": "system", "content": "You create brief, informative one-line summaries of real estate sales calls."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.3
            )
            
            # Extract the summary from the response
            summary = response.choices[0].message.content.strip()
            
            # Ensure summary is truly one line and not too long
            summary = summary.replace("\n", " ")
            if len(summary) > 100:
                summary = summary[:97] + "..."
                
            logger.info(f"Generated call summary: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating call summary: {e}")
            import traceback
            logger.error(f"Summary generation traceback: {traceback.format_exc()}")
            return f"Summary generation failed: {str(e)[:50]}"
    
    async def analyze_sentiment(self) -> int:
        """
        Analyze the conversation transcript using the BANT framework to qualify the potential client.
        Returns:
            int: BANT qualification score (0-5) or -1 if analysis fails.
        """
        if not self.metrics.transcript or len(self.metrics.transcript) < 2:
            logger.warning("Not enough conversation to analyze sentiment")
            return -1
            
        try:
            # Format transcript for analysis
            formatted_transcript = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.metrics.transcript])
            
            import openai as openai_client
            openai_client.api_key = os.getenv("OPENAI_API_KEY")
            if not openai_client.api_key:
                logger.error("OPENAI_API_KEY environment variable is not set. Cannot analyze sentiment.")
                return -1
            
            prompt = f"""
            Below is a transcript of a conversation between the AI sales representative Samantha and a potential client.
            Using the BANT framework, analyze the client's level of interest by evaluating:
            - Budget (client's financial capacity)
            - Authority (decision-making power)
            - Need (specific property requirements)
            - Timeline (urgency of purchase)
            
            Provide a single number (0-5) that represents the client's qualification level.
            
            TRANSCRIPT:
            {formatted_transcript}
            
            BANT Score (0-5):
            """
            
            response = openai_client.chat.completions.create(
                model="gpt-4o-2024-11-20",
                messages=[
                    {"role": "system", "content": "You analyze real estate sales conversations using the BANT framework and provide a score from 0-5."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.3
            )
            
            bant_score_str = response.choices[0].message.content.strip()
            try:
                bant_score = int(bant_score_str[0])
            except Exception:
                bant_score = -1
            
            self.metrics.sentiment = bant_score
            logger.info(f"Determined BANT score: {bant_score}")
            
            # Insert the lead record into the "PropertyProLeads" table
            if supabase:
                lead_record = self.metrics.to_supabase_record()
                try:
                    res = supabase.table("PropertyProLeads").insert(lead_record).execute()
                    logger.info(f"Lead record sent to Supabase: {res.data}")
                except Exception as sup_err:
                    logger.error(f"Error sending lead record to Supabase: {sup_err}")
            else:
                logger.warning("Supabase client not initialized; cannot send lead record.")
            
            return bant_score
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return -1


    
    async def finalize(self) -> Dict[str, Any]:
        """Finalize metrics calculation, generate summary and analyze sentiment, then return the data."""
        if self.is_finalized:
            return self.metrics.to_dict()
            
        end_time = datetime.datetime.now()
        self.metrics.duration = (end_time - self.start_time).total_seconds()
        
        # Generate a one-line summary of the call
        self.metrics.summary = await self.generate_call_summary()
        logger.info(f"Call summary: {self.metrics.summary}")
        
        # Analyze sentiment to determine lead interest level
        self.metrics.sentiment = await self.analyze_sentiment()
        interest_level = {
            -1: "Analysis failed",
            0: "Not interested",
            1: "Low interest",
            2: "Slight interest", 
            3: "Moderate interest",
            4: "Very interested",
            5: "Highly interested"
        }.get(self.metrics.sentiment, "Unknown")
        logger.info(f"Lead interest level: {self.metrics.sentiment}/5 - {interest_level}")    
        self.is_finalized = True
        return self.metrics.to_dict()
    
    def send_to_supabase(self) -> bool:
        """Send metrics data to Supabase table."""
        if supabase is None:
            logger.warning("Supabase client not initialized, can't send metrics to database")
            return False
            
        try:
            # Prepare the record for Supabase
            record = self.metrics.to_supabase_record()
            logger.info(f"Preparing to send call metrics to Supabase: {self.metrics.sessionId}")
            
            # Log the record structure (without sensitive data)
            safe_record = {k: (v if k not in ['transcript', 'dialogueTranscript'] else '[CONTENT OMITTED]') for k, v in record.items()}
            logger.info(f"Record structure to insert: {json.dumps(safe_record)}")
            
            # Validate all data is properly formatted
            for key, value in record.items():
                if isinstance(value, (dict, list)):
                    logger.warning(f"Field {key} contains a complex type that should be JSON string. Fixing...")
                    record[key] = json.dumps(value)
            
            # Insert into the 'PropertyProLeads' table
            table_name = 'PropertyProLeads'  # Ensure this matches your Supabase table name
            logger.info(f"Inserting into Supabase table: {table_name}")
            
            # Try to insert the record
            response = supabase.table(table_name).insert(record).execute()
            
            # Check the response
            if hasattr(response, 'data') and response.data:
                logger.info(f"Successfully sent metrics to Supabase: {self.metrics.sessionId}")
                logger.info(f"Supabase response data: {response.data}")
                return True
            else:
                # Log more details about the response
                logger.error(f"Failed to send metrics to Supabase - empty response data")
                
                # Check for error property and log it
                if hasattr(response, 'error'):
                    logger.error(f"Error from Supabase: {response.error}")
                
                # Try logging the raw response
                try:
                    if hasattr(response, '_raw'):
                        logger.error(f"Raw response: {response._raw}")
                    else:
                        logger.error(f"Response dir: {dir(response)}")
                except:
                    pass
                    
                return False
                
        except Exception as e:
            logger.error(f"Error sending metrics to Supabase: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def save_to_file(self, filename: Optional[str] = None) -> str:
        """Save metrics to a JSON file in EndMetrics folder and return the filename."""
        if not self.is_finalized:
            # We can't await here, so just use the metrics as they are
            if not self.metrics.summary:
                self.metrics.summary = "No summary generated (synchronous save)"
            if self.metrics.sentiment == -1:
                self.metrics.sentiment = -1  # Sentiment not analyzed
            self.is_finalized = True
            
        if filename is None:
            filename = f"PropertyProLeads_{self.metrics.sessionId}.json"
        
        filepath = os.path.join("EndMetrics", filename)
        
        with open(filepath, "w") as f:
            f.write(self.metrics.to_json())
        
        # Clean up temp file if it exists
        temp_path = os.path.join("EndMetrics", f"temp_{self.metrics.sessionId}.json")
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
        
        # Send data to Supabase if client is available
        logger.info("Attempting to send call metrics to Supabase...")
        success = self.send_to_supabase()
        if success:
            logger.info("Successfully sent call metrics to Supabase!")
        else:
            logger.warning("Failed to send call metrics to Supabase. Check logs for details.")
            
        return filepath

# Keep the exit handler
def save_metrics_on_exit():
    """Exit handler to save metrics before shutting down"""
    if global_metrics_collector and not global_metrics_collector.is_finalized:
        try:
            logger.info("Application exiting - saving final metrics...")
            # Since we can't await in a synchronous function, we'll have to finalize without the summary and sentiment
            global_metrics_collector.metrics.summary = "Call ended abruptly - no summary generated"
            global_metrics_collector.metrics.sentiment = -1  # Unable to analyze in sync context
            global_metrics_collector.is_finalized = True
            metrics_data = global_metrics_collector.metrics.to_dict()
            metrics_file = global_metrics_collector.save_to_file()
            logger.info(f"Final call metrics saved to {metrics_file}")
            logger.info(f"Call details: type={metrics_data['callType']}, duration={metrics_data['duration']:.2f}s, cost=${metrics_data['callCost']:.6f}")
        except Exception as e:
            logger.error(f"Error saving metrics on exit: {e}")

# Register the exit handler
atexit.register(save_metrics_on_exit)

# NEW: Create a custom Agent class that extends Agent instead of using FunctionContext
class PropertyProAssistant(Agent):
    def __init__(self) -> None:
        # Move system instructions into the class initialization
        super().__init__(instructions=INSTRUCTION)
    
    @function_tool()
    async def search_properties(
        self,
        context: RunContext,  # New parameter required by v1.0
        property_type: str,
        location: str,
        bedrooms: int,
        min_price: int,
        max_price: int,
        features: str,
        property_number: int,
    ) -> str:
        
        """Search for properties based on the criteria provided by the caller.
        
        Args:
            property_type: The type of property the caller is interested in
            location: The desired location
            bedrooms: The number of bedrooms requested
            min_price: The minimum price in Naira
            max_price: The maximum price in Naira
            features: Additional features or amenities mentioned
            property_number: The specific property number to get details for
        """

        await context.session.generate_reply(
        instructions="Let me check our listings real quick…"
        )
        # Import HTML parser at the function level to ensure it's available throughout
        from html.parser import HTMLParser
        
        # Define HTML stripper class for text extraction
        class MLStripper(HTMLParser):
            def __init__(self):
                super().__init__()
                self.reset()
                self.strict = False
                self.convert_charrefs = True
                self.text = []
            
            def handle_data(self, d):
                self.text.append(d)
            
            def get_data(self):
                return ''.join(self.text)
        
        # Helper function to clean output text
        def clean_output_text(text):
            # Remove any remaining markdown characters
            text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Replace **text** with text
            text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Replace *text* with text
            
            # Remove any leftover asterisks
            text = re.sub(r'\*+', '', text)
            
            # Remove other markdown formatting
            text = re.sub(r'\_\_([^_]+)\_\_', r'\1', text)  # Replace __text__ with text
            text = re.sub(r'\_([^_]+)\_', r'\1', text)      # Replace _text_ with text
            text = re.sub(r'\_+', '', text)                 # Remove any remaining underscores
            
            text = re.sub(r'\~\~([^~]+)\~\~', r'\1', text)  # Replace ~~text~~ with text
            text = re.sub(r'\~+', '', text)                 # Remove any remaining tildes
            
            text = re.sub(r'\`\`\`([^`]+)\`\`\`', r'\1', text)  # Replace ```text``` with text
            text = re.sub(r'\`([^`]+)\`', r'\1', text)          # Replace `text` with text
            text = re.sub(r'\`+', '', text)                     # Remove any remaining backticks
            
            # Replace multiple spaces with single space
            text = re.sub(r'\s+', ' ', text)
            
            # Clean up quotation marks
            text = text.replace('\"\"', '')
            text = text.replace('""', '')
            
            # Replace multiple newlines with double newline
            text = re.sub(r'\n{3,}', '\n\n', text)
            
            return text.strip()
        
        # Build the search query for Trieve
        search_query = ""
        if property_type and property_type != "":
            search_query += f"{property_type} "
        if location and location != "":
            search_query += f"in {location} "
        if bedrooms and bedrooms > 0:
            search_query += f"{bedrooms} bedroom "
        if features and features != "":
            search_query += f"with {features} "
        if min_price and max_price and min_price > 0 and max_price > 0:
            search_query += f"price between {min_price} and {max_price} naira "
        elif max_price and max_price > 0:
            search_query += f"price below {max_price} naira "
        elif min_price and min_price > 0:
            search_query += f"price above {min_price} naira "
        
        if not search_query.strip():
            search_query = "luxury property"  # Default search if no criteria provided
        
        try:
            # Call Trieve API to get property data
            api_key = os.environ.get('TRIEVE_API_KEY')
            dataset_id = os.environ.get('TRIEVE_DATASET_ID')
            
            if not api_key or not dataset_id:
                return "I'm unable to search for properties at the moment due to missing API configuration. Please contact our support team."
            
            headers = {
                "TR-Dataset": dataset_id,
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "query": search_query,
                "search_type": "hybrid",
                "page": 1,
                "page_size": 3,
                "highlight_results": True,
                "highlight_delimiters": ["<mark>", "</mark>"]
            }
            
            async with aiohttp.ClientSession() as session:
                # Log the request details
                print(f"Trieve API Request: URL=https://api.trieve.ai/api/chunk/search")
                print(f"Trieve API Request: Headers={headers}")
                print(f"Trieve API Request: Payload={payload}")
                
                try:
                    async with session.post(
                        "https://api.trieve.ai/api/chunk/search", 
                        headers=headers, 
                        json=payload
                    ) as response:
                        print(f"Trieve API Response Status: {response.status}")
                        
                        # Log the raw response text
                        response_text = await response.text()
                        print(f"Trieve API Response Text: {response_text}")
                        
                        if response.status == 200:
                            try:
                                result = json.loads(response_text)
                                properties = result.get("chunks", [])
                                print(f"Trieve API Chunks Count: {len(properties)}")
                                
                                # Print the structure of the first property if available for debugging
                                if properties and len(properties) > 0:
                                    print(f"First property structure: {json.dumps(properties[0], indent=2)}")
                                
                                if not properties:
                                    return "I couldn't find any properties matching your criteria. Would you like to try a different search?"
                                
                                # Extract property information from the API response
                                extracted_properties = []
                                
                                for i, prop in enumerate(properties, 1):
                                    # Get the HTML content
                                    chunk_data = prop.get("chunk", {})
                                    content = chunk_data.get("chunk_html", "")
                                    
                                    # Strip HTML tags to get clean text
                                    stripper = MLStripper()
                                    stripper.feed(content)
                                    clean_text = stripper.get_data().strip()
                                    
                                    # Remove markdown formatting characters or escape them
                                    clean_text = re.sub(r'\*\*(.+?)\*\*', r'\1', clean_text)  # Replace **text** with text
                                    clean_text = re.sub(r'\*(.+?)\*', r'\1', clean_text)      # Replace *text* with text
                                    clean_text = re.sub(r'\_\_(.+?)\_\_', r'\1', clean_text)  # Replace __text__ with text
                                    clean_text = re.sub(r'\_(.+?)\_', r'\1', clean_text)      # Replace _text_ with text
                                    clean_text = re.sub(r'\~\~(.+?)\~\~', r'\1', clean_text)  # Replace ~~text~~ with text
                                    
                                    # Extract property information
                                    property_info = {
                                        "number": i,
                                        "content": clean_text
                                    }
                                    
                                    # Try to extract title/name
                                    if "Title:" in clean_text:
                                        try:
                                            title_line = [line for line in clean_text.split('\n') if line.startswith("Title:")][0]
                                            property_info["name"] = clean_output_text(title_line.replace("Title:", "").strip())
                                        except:
                                            property_info["name"] = f"Property {i}"
                                    elif "<h1>" in content and "</h1>" in content:
                                        try:
                                            property_info["name"] = clean_output_text(content.split("<h1>")[1].split("</h1>")[0].strip())
                                        except:
                                            property_info["name"] = f"Property {i}"
                                    else:
                                        property_info["name"] = f"Property {i}"
                                    
                                    # Extract location
                                    locations = ["Lekki", "Victoria Island", "Ikoyi", "Banana Island", "Lagos", "Ikeja", "Ajah", "Abuja"]
                                    for loc in locations:
                                        if loc in clean_text:
                                            property_info["location"] = loc
                                            break
                                    if "location" not in property_info:
                                        property_info["location"] = location if location else "Nigeria"
                                    
                                    # Extract price if present
                                    price_patterns = [
                                        r"N(\d+(?:\.\d+)?)\s*million",
                                        r"(\d+(?:\.\d+)?)\s*million\s*naira",
                                        r"rent\s*is\s*N(\d+(?:\.\d+)?)\s*million",
                                        r"priced\s*at\s*N(\d+(?:\.\d+)?)\s*million"
                                    ]
                                    
                                    for pattern in price_patterns:
                                        price_match = re.search(pattern, clean_text, re.IGNORECASE)
                                        if price_match:
                                            try:
                                                price_value = float(price_match.group(1))
                                                property_info["price"] = f"{price_value} million naira"
                                                break
                                            except:
                                                pass
                                    
                                    # Extract bedrooms
                                    bedroom_match = re.search(r'(\d+)[\s-]bedroom', clean_text, re.IGNORECASE)
                                    if bedroom_match:
                                        property_info["bedrooms"] = bedroom_match.group(1)
                                    
                                    # Extract key features
                                    features_list = []
                                    feature_keywords = ["gym", "pool", "swimming", "security", "parking", "garden", "balcony"]
                                    for feature in feature_keywords:
                                        if feature in clean_text.lower():
                                            features_list.append(feature)
                                    
                                    if features_list:
                                        property_info["features"] = ", ".join(features_list)
                                    
                                    # Extract a description
                                    if "Description/Answer:" in clean_text:
                                        try:
                                            description_parts = clean_text.split("Description/Answer:")
                                            if len(description_parts) > 1:
                                                desc_text = description_parts[1].split("Tags:")[0].strip()
                                                # Clean description text of quote marks and other artifacts
                                                desc_text = desc_text.strip('"').strip("'")
                                                desc_text = re.sub(r'^\s*["\']|["\']\s*$', '', desc_text)
                                                property_info["description"] = clean_output_text(desc_text)
                                        except:
                                            pass
                                    
                                    if "description" not in property_info:
                                        # If no specific description section, use the content
                                        clean_content = clean_text[:300] + "..." if len(clean_text) > 300 else clean_text
                                        # Clean description text
                                        clean_content = re.sub(r'^\s*["\']|["\']\s*$', '', clean_content)
                                        property_info["description"] = clean_output_text(clean_content)
                                    
                                    # Ensure all text fields are clean of markdown
                                    for key in property_info:
                                        if isinstance(property_info[key], str) and key != "number":
                                            property_info[key] = clean_output_text(property_info[key])
                                    
                                    extracted_properties.append(property_info)
                                
                                # If a specific property number is requested (1, 2, or 3)
                                if property_number is not None and 1 <= property_number <= len(extracted_properties):
                                    selected_property = extracted_properties[property_number - 1]
                                    
                                    # Format detailed property information
                                    details = []
                                    details.append(f"Property {property_number}: {selected_property.get('name', '')}")
                                    details.append(f"Location: {selected_property.get('location', 'Not specified')}")
                                    
                                    if "bedrooms" in selected_property:
                                        details.append(f"Bedrooms: {selected_property['bedrooms']}")
                                    
                                    if "price" in selected_property:
                                        details.append(f"Price: {selected_property['price']}")
                                    
                                    if "features" in selected_property:
                                        details.append(f"Key Features: {selected_property['features']}")
                                    
                                    details.append(f"Description: {selected_property.get('description', 'Not available')}")
                                    
                                    details.append("\nWould you like to schedule a viewing of this property, see details about another property, or discuss financing options?")
                                    
                                    return clean_output_text("\n\n".join(details))
                                
                                # Otherwise, return a summary of all properties
                                intro = f"These are the {len(extracted_properties)} properties that match your criteria:"
                                
                                property_summaries = []
                                for prop in extracted_properties:
                                    # Create a condensed, single-line summary for each property
                                    summary_parts = []
                                    
                                    # Always include the property number and name
                                    summary_parts.append(f"Property {prop['number']}: {prop['name']}")
                                    
                                    # Include essential details in a single line
                                    details = []
                                    if "location" in prop:
                                        details.append(f"in {prop['location']}")
                                    
                                    if "bedrooms" in prop:
                                        details.append(f"{prop['bedrooms']} bedroom")
                                    
                                    if "price" in prop:
                                        details.append(f"for {prop['price']}")
                                    
                                    if details:
                                        summary_parts.append(" - " + ", ".join(details))
                                    
                                    # Add a very brief features mention if available
                                    if "features" in prop:
                                        summary_parts.append(f"(Includes: {prop['features']})")
                                    
                                    # Clean the entire summary before adding it
                                    property_summaries.append(clean_output_text(" ".join(summary_parts)))
                                
                                summaries = "\n\n".join(property_summaries)
                                conclusion = "\nWhich property would you like to know more about?"
                                
                                return clean_output_text(f"{intro}\n\n{summaries}\n\n{conclusion}")
                            except Exception as e:
                                print(f"Error processing Trieve API response: {e}")
                                return "There was an error processing the search results. Please try again later."
                        else:
                            return f"I'm having trouble searching for properties right now. The search service returned status {response.status}. Could you please try again later?"
                except Exception as e:
                    print(f"Error making Trieve API request: {e}")
                    return f"I'm sorry, but I encountered an error while searching for properties: {str(e)}. Please try again with different criteria or contact our support team for assistance."
        except Exception as e:
            print(f"Error searching properties: {e}")
            error_details = f"""
I apologize, but I'm having technical difficulties with our property search system.

Error details: {str(e)}

Here's what we can do:
1. Try searching with different criteria
2. Contact our support team for direct assistance
3. I can take your contact information and have our team follow up with suitable property options

Would you prefer one of these options, or would you like to tell me more about what you're looking for?
"""
            return error_details

async def entrypoint(ctx: JobContext):
    global global_metrics_collector
     
    # Create initial chat context with system message (updated for v1.0)
    initial_ctx = llm.ChatContext()
    initial_ctx.add_message(role="system", content=INSTRUCTION)

    # Connect to the room (simplified in v1.0)
    await ctx.connect()
    
    # Helper function to attempt reconnection if audio fails
    async def attempt_reconnect():
        logger.warning("Attempting to reconnect audio connections")
        try:
            # Ensure audio is enabled
            session.input.set_audio_enabled(True)
            # Send a simple message to test audio recovery
            await session.generate_reply(instructions="Let me reconnect to better assist you.")
            logger.info("Audio reconnection appears successful")
            return True
        except Exception as e:
            logger.error(f"Failed to reconnect audio: {e}")
            return False
    
    # Determine call type based on room name prefix
    call_type = "web call"
    if ctx.room.name.startswith("propertypro-"):
        call_type = "inbound"
    
    # Initialize call metrics collector
    metrics_collector = CallMetricsCollector(call_type=call_type)
    global_metrics_collector = metrics_collector
    
    # Get customer phone number from participant metadata
    remote_participant = next(iter(ctx.room.remote_participants.values()), None)
    if remote_participant and hasattr(remote_participant, "metadata"):
        try:
            metadata = json.loads(remote_participant.metadata)
            if "phone" in metadata:
                metrics_collector.set_customer_number(metadata["phone"])
        except (json.JSONDecodeError, AttributeError):
            pass

    # Configure STT
    improved_stt = deepgram_stt.STT(
        model="nova-3-general",
        interim_results=True,
        smart_format=True,
        punctuate=True,
        filler_words=True,
        profanity_filter=False,
        language="en-US",
        # Keyterms as in original implementation
        keyterms=[
            # Companies and organizations
            "PropertyPro",
            # Locations
            "Lekki",
            "Ikoyi",
            "Victoria Island",
            "Lagos",
            "Ikeja",
            "Ajah",
            "Abuja",
            "Banana Island",
            "Surulere",
            # Property types
            "apartment",
            "duplex",
            "penthouse",
            "townhouse",
            "mansion",
            "terrace",
            "bungalow",
            "condominium",
            # Features and amenities
            "waterfront",
            "pool",
            "gym",
            "garden",
            "security",
            "garage",
            "balcony",
            "rooftop",
            "furnished",
            "parking",
            "ensuite",
            # Room types
            "bedroom",
            "bathroom",
            "kitchen",
            "living room",
            # General real estate terms
            "property",
            "real estate",
            "mortgage",
            "financing",
            "viewing",
            "tour",
            "inspection",
            "million naira",
            "price range",
            # Property numbers and queries
            "property 1",
            "property 2", 
            "property 3",
            "property one",
            "property two",
            "property three",
            "first property",
            "second property",
            "third property",
            "details about",
            "tell me more",
            "more information",
            "more details",
            "I want property 1",
            "I want property 2",
            "I want property 3",
            "show me property 1",
            "show me property 2",
            "show me property 3",
            "tell me about property 1",
            "tell me about property 2",
            "tell me about property 3",
            "which property",
            "select property",
            "choose property",
            "number 1",
            "number 2",
            "number 3",
            "number one",
            "number two", 
            "number three",
            "property number 1",
            "property number 2",
            "property number 3",
            "the first one",
            "the second one",
            "the third one",
        ]
    )
    # Configure LLM
    improved_llm = openai.LLM(
        model="gpt-4.1",
        temperature=0.4,
    )

    # Create the agent session with VAD-only turn detection
    session = AgentSession(
        stt=improved_stt,
        llm=improved_llm,
        tts=cartesia.TTS(model="sonic-2",voice="bf0a246a-8642-498a-9950-80c35e9276b5"),
        vad=silero.VAD.load(),
        turn_detection="vad",  
        allow_interruptions=True,
    )

    # Create the assistant instance
    assistant = PropertyProAssistant()

    # Set up event handlers before starting the session
    
    # Metrics collection
    @session.on("metrics_collected")
    def on_metrics(metrics_obj):
        logger.info(f"Conversation metrics: {metrics_obj}")
        
        if isinstance(metrics_obj, metrics.LLMMetrics):
            metrics_collector.collect_llm_metrics(metrics_obj)
        elif isinstance(metrics_obj, metrics.STTMetrics):
            metrics_collector.collect_stt_metrics(metrics_obj)
        elif isinstance(metrics_obj, metrics.TTSMetrics):
            metrics_collector.collect_tts_metrics(metrics_obj)
    
    # New event: transcription - correctly typed
    @session.on("transcription")
    def on_transcription(ev):
        # Extract content from the transcription event
        content = ev.text if hasattr(ev, 'text') else str(ev)
        
        # Only process final transcriptions
        if hasattr(ev, 'is_final') and not ev.is_final:
            return
            
        # Update activity time whenever we get user speech
        nonlocal last_activity_time
        last_activity_time = time.time()
            
        logger.info(f"User said: {content}")
        metrics_collector.add_message("user", content)
    
    # New event: agent_state_changed
    @session.on("agent_state_changed")
    def on_agent_state_changed(ev: AgentStateChangedEvent):
        logger.info(f"Agent state changed from {ev.old_state} to {ev.new_state}")
        
        # When agent transitions to speaking, capture what's being said
        if ev.new_state == "speaking" and hasattr(ev, 'content'):
            content = ev.content if isinstance(ev.content, str) else str(ev.content)
            logger.info(f"Agent said: {content}")
            metrics_collector.add_message("assistant", content)
        
        # If agent transitions from speaking to idle without a user state change,
        # it might indicate an issue with audio delivery
        if ev.old_state == "speaking" and ev.new_state == "idle":
            logger.debug("Agent finished speaking")
        
        # Reset activity time whenever agent is active
        nonlocal last_activity_time
        if ev.new_state in ["speaking", "thinking"]:
            last_activity_time = time.time()
    
    # New event: agent_speech_completed to ensure TTS completes properly
    @session.on("agent_speech_completed")
    def on_agent_speech_completed(ev):
        logger.info("Agent finished speaking")
    
    # New event: agent_speech_failed to detect TTS errors
    @session.on("agent_speech_failed")
    def on_agent_speech_failed(ev):
        logger.error(f"TTS speech failed: {ev}")
        # Try to recover by reconnecting and sending a message
        try:
            asyncio.create_task(attempt_reconnect())
        except Exception as e:
            logger.error(f"Error recovering from speech failure: {e}")
    
    # New event: user_state_changed
    @session.on("user_state_changed")
    def on_user_state_changed(ev: UserStateChangedEvent):
        logger.info(f"User state changed from {ev.old_state} to {ev.new_state}")
        
        # If user disconnects, log it
        if ev.new_state == "away" and ev.old_state != "away":
            logger.warning("User appears to be disconnected")
            
        # If user returns after being away, greet them
        if ev.new_state != "away" and ev.old_state == "away":
            logger.info("User has returned after being away")
            try:
                asyncio.create_task(session.generate_reply(
                    instructions="Welcome back! I'm still here to help with your property search."
                ))
            except Exception as e:
                logger.error(f"Error sending welcome back message: {e}")
                
    # Function call tracking - similar to original
    @session.on("function_calls_finished")
    def on_function_calls_finished(called_functions):
        for called_func in called_functions:
            if called_func.result is not None:
                logger.info(f"Function call result: {called_func.name} -> {called_func.result}")
                metrics_collector.add_message("function", str(called_func.result))
    
    # This handler manages activity timing for the inactivity prompt
    last_activity_time = time.time()
    
    @session.on("transcription")
    @session.on("agent_state_changed") 
    def update_activity(*args):
        nonlocal last_activity_time
        last_activity_time = time.time()
    
    # Add event handler for audio-related issues
    @session.on("error")
    def on_error(error):
        logger.error(f"Session error: {error}")
        # Try to recover from the error
        try:
            asyncio.create_task(session.generate_reply(
                instructions="I apologize for the technical difficulty. Let me continue helping you."
            ))
        except Exception as e:
            logger.error(f"Error recovering from session error: {e}")
    
    # Event handler for connection state changes
    @session.on("connection_state_changed")
    def on_connection_state_changed(ev):
        logger.info(f"Connection state changed: {ev.old_state} -> {ev.new_state}")
        if ev.new_state == "disconnected":
            # Try to reconnect if disconnected
            try:
                asyncio.create_task(attempt_reconnect())
            except Exception as e:
                logger.error(f"Error attempting to reconnect: {e}")
    
    # Event handler for audio processing
    @session.on("audio")
    def on_audio(ev):
        # This will let us know about audio processing
        nonlocal last_activity_time
        last_activity_time = time.time()
    
    # Setup heartbeat to keep connection alive
    async def send_keepalive():
        try:
            while True:
                await asyncio.sleep(15)  # Send keepalive every 15 seconds
                # Check if we should send a health check
                if time.time() - last_activity_time > 30:  # If no activity for 30 seconds
                    logger.info("Sending keepalive health check")
                    # Try to interact with the session to verify connection
                    try:
                        room_state = ctx.room.state
                        logger.info(f"Room state: {room_state}")
                    except Exception as e:
                        logger.error(f"Failed to check room state: {e}")
                        # If we can't check room state, try to reconnect
                        asyncio.create_task(attempt_reconnect())
        except asyncio.CancelledError:
            logger.info("Keepalive task cancelled")
        except Exception as e:
            logger.error(f"Error in keepalive task: {e}")
    
    # Start keepalive task
    keepalive_task = asyncio.create_task(send_keepalive())
    
    # Start the session with the agent and room
    logger.info("Starting agent session...")
    await session.start(
        room=ctx.room,
        agent=assistant,
        room_input_options=RoomInputOptions(
            audio_enabled=True,
            video_enabled=False,
            text_enabled=True,
        ),
    )
    
    try:
        await asyncio.sleep(1)
        
        # Dynamic greeting based on time of day
        current_hour = datetime.datetime.now().hour
        greeting = "Good morning" if 5 <= current_hour < 12 else "Good afternoon" if 12 <= current_hour < 18 else "Good evening"
        greeting_message = f"{greeting}! This is Samantha from PropertyPro Real Estate. How can I assist you today?"
        
        # Use the correct parameter for generate_reply
        logger.info(f"Sending initial greeting: {greeting_message}")
        
        # In v1.0, generate_reply uses 'instructions' parameter
        await session.generate_reply(instructions=greeting_message)
        
        # Add the greeting to metrics
        metrics_collector.add_message("assistant", greeting_message)
        
        # Keep the session alive while there are remote participants
        heartbeat_counter = 0
        while len(ctx.room.remote_participants) > 0:
            await asyncio.sleep(5)  # Check more frequently
            
            # Increment heartbeat counter
            heartbeat_counter += 1
            
            # Every 30 seconds (6 * 5s), log a heartbeat 
            if heartbeat_counter % 6 == 0:
                logger.info(f"Agent heartbeat - remote participants: {len(ctx.room.remote_participants)}")
                
                # If we've gone more than 3 minutes without activity, try to ensure the connection is alive
                if time.time() - last_activity_time > 180:
                    try:
                        logger.info("Long period of inactivity - checking agent health")
                        # Try to ping the agent with a simple message that doesn't speak out loud
                        # This keeps the connection alive without annoying the user
                        await asyncio.sleep(0.1)
                    except Exception as e:
                        logger.error(f"Error sending health check: {e}")
            
            # If no activity for 30 seconds (reduced from 60), prompt the user
            if time.time() - last_activity_time > 30:
                try:
                    prompt_message = "Are you still there? I can help you find the perfect property."
                    logger.info("Sending inactivity prompt")
                    await session.generate_reply(instructions=prompt_message)
                    last_activity_time = time.time()  # Reset the timer
                    metrics_collector.add_message("assistant", prompt_message)
                except Exception as e:
                    logger.error(f"Error sending inactivity prompt: {e}")
                    # Try to restart the session if there's an error
                    try:
                        logger.info("Attempting to restart audio connection...")
                        # Send a simple message to check if connection is working
                        await session.generate_reply(instructions="Let me know if you need any help with property listings.")
                        last_activity_time = time.time()
                    except Exception as restart_error:
                        logger.error(f"Failed to restart connection: {restart_error}")
            
            # Periodically send a ping message to keep connection alive
            try:
                # Don't use update_metadata which is not available
                # Instead, just log a keepalive message
                logger.debug("Sending keepalive ping")
                # If needed, we could publish data to the room as a keepalive
                # but a simple log is sufficient to maintain the event loop
            except Exception as e:
                logger.error(f"Error sending keepalive: {e}")
                
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        # Finalize and save metrics
        try:
            logger.info("Finalizing call metrics")
            metrics_data = await metrics_collector.finalize()
            metrics_file = metrics_collector.save_to_file(f"PropertyProLeads_{metrics_collector.metrics.sessionId}.json")
            logger.info(f"Call metrics saved to {metrics_file}")
            logger.info(f"Call summary: {metrics_data['summary']}")
            
            # Log detailed metrics
        except Exception as e:
            logger.error(f"Error finalizing metrics: {e}")
            import traceback
            logger.error(f"Metrics finalization traceback: {traceback.format_exc()}")
        
        # Cleanup
        logger.info("Conversation ended, cleaning up resources")
        await session.aclose()  # Close the session instead of assistant.aclose()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint)) 