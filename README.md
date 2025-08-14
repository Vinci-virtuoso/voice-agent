# PropertyPro AI Sales Representative Samantha

## Overview

**PropertyPro AI Sales Representative Samantha** is a real-time voice lead qualification system designed for PropertyPro. Samantha is an AI-powered sales representative that engages inbound leads—whether through web inquiries, phone calls, or marketing follow-ups—with a friendly, calm, and warm demeanor. The system leverages modern voice processing, natural language understanding, and real-time database updates to ensure that no potential lead is missed.

## Purpose

The project aims to automate the lead qualification process by:
- **Answering property inquiries in real time** using voice-driven interactions.
- **Qualifying leads using the BANT framework** (Budget, Authority, Need, Timeline) along with sentiment analysis.
- **Storing structured lead information in Supabase** as soon as each BANT answer is provided.

This results in a streamlined process that minimizes manual errors and ensures rapid follow-up on promising leads.

## Architecture

The system integrates several key components:

- **LiveKit Voice Agent**: Facilitates real-time conversation between the prospect and the AI representative.
- **Speech-to-Text (STT)**: Converts voice input to text (via AssemblyAI or similar service).
- **Text-to-Speech (TTS)**: (if enabled) Converts text responses to a natural-sounding voice.
- **Search-Properties**: Fetches the latest property listings of  PropertyPro.ng from the indexed TRIEVE DATASET.
- **BANT Framework Evaluation**: Analyzes conversation transcripts using the BANT framework to determine the prospect’s qualification level.
- **Supabase Database**: Stores lead information in real time in the `PropertyProLeads` table.

The following diagram summarizes the overall system:

```
 ┌──────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
 │  Prospect    │◄────►│ LiveKit Voice AI    │◄────►│  AssemblyAI (STT)   │
 └──────────────┘      │      Agent          │      └─────────────────────┘
                     │ (Direct Integration)  │
                     │   with Firecrawl and  │      ┌─────────────────────┐
                     │   Supabase API        │─────►│ PropertyPro Listings│
                     │                       │      └─────────────────────┘
                     │                       │      ┌─────────────────────┐
                     │                       │─────►│ Lead Storage DB      │
                     │                       │      │ (Live BANT Updates)  │
                     └───────────────────────┘      └─────────────────────┘
```

## Supabase Schema

The leads are stored in Supabase, following the schema of the `PropertyProLeads` table:

```sql
CREATE TABLE public."PropertyProLeads" (
  name text not null,
  created_at timestamp with time zone not null default now(),
  phone text null,
  budget numeric null,
  email text null,
  authority boolean null,
  need text null,
  timeline text null,
  property_interest text null,
  bant_score bigint null,
  conversation_summary text null,
  id uuid null default gen_random_uuid (),
  constraint PropertyProLeads_pkey primary key (name)
) TABLESPACE pg_default;
```

## Features

- **Real-Time Voice Interaction**: Utilizes LiveKit to handle live voice conversations.
- **BANT Framework Analysis**: Processes conversation transcripts with OpenAI models to derive a qualification score based on budget, authority, need, and timeline.
- **Instant Database Updates**: The processed lead data is immediately stored in Supabase, ensuring real-time tracking and follow-up.
- **Scalability and Speed**: Designed to be built and deployed in less than 4 hours using modern cloud and voice technologies.

## Setup and Installation

1. **Clone the Repository**

   ```bash
   git clone <repository_url>
   cd voice-agent
   ```

2. **Environment Variables**

   Create a `.env` file in the project root and configure the following:

   ```env
   SUPABASE_URL=https://your-supabase-project.supabase.co
   SUPABASE_KEY=your_supabase_key
   OPENAI_API_KEY=your_openai_api_key
   # Other relevant environment variables (e.g., for Firecrawl, if applicable)
   ```

3. **Install Dependencies**

   Ensure you are using a virtual environment and install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Agent**

   Use your preferred method (e.g., via a script or command-line) to start the agent. For example:

   ```bash
   python agent.py
   ```

## Code Structure

- **agent.py**: Contains the core logic of the LiveKit voice agent, including metrics collection, BANT analysis, and Supabase integration.
- **promptv3.py**: Contains prompt templates and related functions.
- **PRD.md**: The product requirements document outlining project goals, desired features, and system architecture.
- **README.md**: This file, providing an overview and guide to the project.
- **setup.sh**: (If applicable) a shell script for setting up the development environment.

## Development Phases

The project was developed in multiple phases, with a focus on rapid deployment:

1. **Environment Setup**: Integrate LiveKit, Supabase, and other services.
2. **BANT Logic Implementation**: Develop functionality for real-time sentiment analysis using OpenAI and the BANT framework.
3. **Real-Time Database Writes**: Ensure every BANT answer is immediately pushed to Supabase.
4. **Testing and Validation**: Verify stability and functionality through simulated calls and real data.
5. **Deployment**: Finalize the system for production use with continuous monitoring via Supabase dashboards.

## Troubleshooting

- **Supabase Connection Issues**: Ensure that the `SUPABASE_URL` and `SUPABASE_KEY` environment variables are correctly set. A common error like `[Errno 11001] getaddrinfo failed` indicates that the host could not be resolved.
- **OpenAI API Key**: Verify that your `OPENAI_API_KEY` is valid and has sufficient quota for generating summaries and sentiment analysis.
- **LiveKit Integration**: Check network configurations if issues arise with real-time voice sessions.
