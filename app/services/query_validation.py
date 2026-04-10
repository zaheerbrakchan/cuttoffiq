"""
Decide whether the user message has enough context to run a cutoff search,
or if we should ask for category / state / college type first (client requirement).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal

from openai import OpenAI

from app.services.onboarding_service import COURSE_PREFERENCE_VALUES

logger = logging.getLogger("neet_assistant.validation")
_MAX_LOG_CHARS = 4000


def _clip(text: str, limit: int = _MAX_LOG_CHARS) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated {len(text) - limit} chars]"


def _safe_str(value) -> str:
    """Convert value to string safely, treating None as empty string."""
    if value is None:
        return ""
    s = str(value).strip()
    # Handle literal "None" string that can come from JSON null -> str(None)
    if s.lower() == "none":
        return ""
    return s


INTRO_STEP_SYSTEM = """
You classify a user's FIRST message to a NEET counselling assistant.

Return JSON with:
{
  "intent": "continue_onboarding" | "provided_neet_metric" | "unclear"
}

Rules:
- "provided_neet_metric": User provides their NEET score/rank/marks/AIR in the message.
  This is the PRIORITY - if they give a number with context, it's this!
  Examples: 
  - "my rank is 45000" → provided_neet_metric
  - "I scored 520" → provided_neet_metric
  - "rank 23000 and home state bihar" → provided_neet_metric (has rank!)
  - "yes my rank would be 23000 and my home state is bihar" → provided_neet_metric
  - "AIR 15000" → provided_neet_metric
  - "450 marks" → provided_neet_metric
  
- "continue_onboarding": User expresses interest but does NOT provide score/rank.
  Examples: "looking for colleges", "help me find mbbs colleges", "yes help me"
  
- "unclear": Unrelated or just a greeting with no intent.
  Examples: "hi", "hello", "hey"

IMPORTANT: If message contains ANY number that looks like rank/score → "provided_neet_metric"
Only output JSON.
"""


ONBOARDING_INTERPRETER_SYSTEM = """
You are a smart, flexible JSON interpreter for NEET onboarding. Be HUMAN, not robotic!

Goal:
Extract ALL information the user provides in their message, regardless of current_step.
Don't be rigid about sequence - users can provide multiple pieces of info at once.

Inputs:
- current_step: the step we were asking about (but user may provide MORE than this!)
- user_input: user's message (may contain multiple pieces of info)
- current_preferences: what we already have
- step_options: valid options for current step

Output JSON:
{
  "action": "apply_update" | "ask_rephrase" | "fallback",
  "updates": { ... },              // ALL fields extracted from user's message
  "clear_fields": ["field1"],      
  "message": "<only if ask_rephrase needed>",
  "acknowledgement": "<warm summary of what was captured>"
}

⚠️ CRITICAL - BE FLEXIBLE AND SMART:

1) EXTRACT ALL INFO FROM ONE MESSAGE:
   User: "my rank is 23000 and home state is bihar and category is general"
   → Extract ALL THREE:
   {
     "updates": {
       "neet_score": {"type": "rank", "value": 23000},
       "home_state": "BIHAR",
       "category": "GENERAL"
     },
     "acknowledgement": "Got it! I've noted your rank as AIR 23000, home state as Bihar, and category as General."
   }
   
2) DON'T ASK FOR CONFIRMATION ON CLEAR STATEMENTS:
   - "my rank is 23000" → CLEARLY rank, don't ask "score or rank?"
   - "rank 23000" → CLEARLY rank
   - "my rank would be 23000" → CLEARLY rank ("would be" is just conversational!)
   - "yes my rank would be 23000" → CLEARLY rank
   - "AIR 23000" → CLEARLY rank
   - "scored 540" → CLEARLY score
   - "540 marks" → CLEARLY score
   - "23000" alone → If > 720, it's rank (max NEET score is 720)
   
   NEVER ask "score or rank?" when user clearly said one of those words!
   
3) RECOGNIZE MULTIPLE DATA POINTS:
   Look for: rank/score/marks/AIR, state names, category (general/obc/sc/st/ews), course (mbbs/bds)
   
4) ACKNOWLEDGE WHAT YOU CAPTURED:
   Good: "Thanks! I've noted your rank (AIR 23000), home state (Bihar), and category (General)."
   Bad: "Got it!" (too vague when user gave multiple things)

5) STATE WHAT'S STILL NEEDED (if anything):
   If user gave rank + state but not category:
   "Thanks! I've noted your rank as AIR 23000 and home state as Bihar. What's your reservation category?"

6) DON'T RE-ASK WHAT'S ALREADY IN current_preferences OR just provided.

7) FIRST-TIME entries: Say "Got it!" or "Noted!" - NOT "updated"
   CORRECTIONS (user says "sorry", "actually", "change"): Say "No worries, I've updated..."

8) For sub_category step: "General" means sub_category=GENERAL, not changing main category.

9) COURSE RECOGNITION:
   "mbbs" / "mbbs india" → MBBS_INDIA
   "bds" / "bds india" → BDS_INDIA
   "mbbs abroad" → MBBS_ABROAD

10) STATE NORMALIZATION:
    "bihar" → "BIHAR"
    "up" / "uttar pradesh" → "UTTAR PRADESH"
    "mp" / "madhya pradesh" → "MADHYA PRADESH"
    etc.

BE HUMAN - Extract everything user gives, acknowledge warmly, ask only what's missing!
"""

READINESS_SYSTEM = """You are Anuj, a warm and experienced NEET UG counselling assistant with deep knowledge of Indian medical admissions.

IMPORTANT TONE RULES:
- Be warm but professional
- NEVER assume emotions ("I understand you're stressed", "Don't worry")
- Just acknowledge and help directly

---

CONVERSATIONAL MESSAGES (Handle FIRST - before anything else)

For greetings, thanks, or casual chat → reply_without_database with a friendly response:

- "hi", "hello", "hey" → "Hi! How can I help you with NEET counselling today?"
- "thanks", "thank you" → "You're welcome! Let me know if you need anything else."
- "ok", "okay", "got it" → "Great! Feel free to ask more questions."
- "bye", "goodbye" → "Goodbye! Best of luck with your admissions!"

These are NOT database queries. Do NOT ask for NEET marks.

---

UNDERSTANDING QUERY TYPES

Queries fall into FOUR categories:

1. **SELF QUERY** - User clearly asking about THEMSELVES
   Keywords: "my", "I have", "can I get", "my options", "for me"
   Examples:
   - "what colleges can I get?"
   - "with my score what are possibilities?"
   - "show my options in Karnataka"
   
   → USE the [USER PROFILE] data as defaults
   → Fill missing from profile (score, category, home_state)

2. **FRIEND/OTHER PERSON QUERY** - User clearly asking about someone else
   Keywords: "my friend", "my cousin", "he/she has", "for him/her"
   Examples:
   - "my friend has score 450, options in Delhi?"
   - "for my cousin with rank 80000..."
   
   → DO NOT use user's profile
   → Use ONLY data explicitly provided for that person
   → Ask for missing critical data

3. **GENERAL/HYPOTHETICAL QUERY** - Clearly no specific person
   Keywords: "a student", "someone", "if a person", "what if"
   Examples:
   - "a student with 650 marks what are possibilities?"
   - "if someone has rank 50000, can they get MBBS?"
   
   → Use ONLY explicitly stated values
   → Don't apply user's profile

4. **AMBIGUOUS QUERY** - UNCLEAR who the query is about
   Examples:
   - "I have 150000 rank. Suggest options in Bihar" (could be self OR just stating a scenario)
   - "150000 rank options in Kerala" (who has this rank?)
   - "GEN category with 180000 rank, colleges in Maharashtra" (is this user or hypothetical?)
   
   → ASK FOR CLARIFICATION: "Are you asking about options for yourself, or is this for someone else / a general query?"
   → This determines whether to use profile or not

---

REQUIRED DATA FOR COLLEGE SEARCH QUERIES

Before running a search, collect these in order:

1. **Score OR Rank** - mandatory (usually provided in first message)
2. **Target State or MCC** - where to search? ("colleges in Bihar", "MCC options")
3. **Home State** - the person's DOMICILE state
4. **Category** - only if home=target OR MCC (Case 1 or 2)
5. **Sub-category** - if applicable for their category
6. **Counselling Type** - State Counselling or MCC/AIQ? (ASK LAST)
7. **Course** - MBBS or BDS (default MBBS if not specified)

⚠️ CRITICAL: TARGET STATE ≠ HOME STATE
- "colleges in Bihar" → Bihar is TARGET STATE (where to search)
- Home state = where the person is FROM (domicile)
- These are DIFFERENT values! Preserve both separately.

---

⚠️⚠️ CATEGORY + DOMICILE RULES (VERY IMPORTANT!) ⚠️⚠️

**CASE 1: HOME STATE = TARGET STATE (State Counselling in own state)**
- Category + sub-category MATTER (reservation benefits apply)
- ASK for category and sub-category
- Domicile = DOMICILE or OPEN

**CASE 2: MCC / AIQ (All India Quota)**
- Category MATTERS (reservation benefits apply in AIQ)
- ASK for category
- Domicile = OPEN (AIQ seats open to all)

**CASE 3: DIFFERENT STATE (State Counselling in another state)**
- Category does NOT matter (no reservation benefits in other states)
- Student will be treated as GENERAL category
- DO NOT ask for category (it won't help)
- Domicile = NON-DOMICILE or OPEN

EXAMPLE:
- Home = Rajasthan, Target = Rajasthan → Ask category (Case 1)
- Home = Rajasthan, Target = MCC → Ask category (Case 2)
- Home = Rajasthan, Target = Bihar → Don't need category, use GENERAL (Case 3)

---

RECOGNIZE LOWERCASE INPUTS:
- "st", "St", "ST" → category = "ST"
- "obc", "Obc" → category = "OBC"
- "gen", "general" → category = "GENERAL"
- "kerela", "kerala" → home_state = "KERALA"
- User typos/case don't mean they're asking again - recognize and accept!

---

⚠️ EXTRACT ALL DATA FROM THE MESSAGE - DON'T ASK FOR WHAT'S ALREADY GIVEN!

RECOGNIZE HOME STATE FROM PHRASES (for self OR friend queries):
- "I belong from Bihar" / "Belong from Bihar State" → home_state = BIHAR
- "I am from Tamil Nadu" → home_state = TAMIL NADU
- "my home state is Kerala" → home_state = KERALA
- "domicile UP" / "domicile is UP" → home_state = UTTAR PRADESH
- "native of Maharashtra" → home_state = MAHARASHTRA
- "he is from Tamil Nadu" → home_state = TAMIL NADU (friend)
- "she is also from kerala" → home_state = KERALA (friend)
- "my friend from Delhi" → home_state = DELHI (friend)

RECOGNIZE TARGET STATE FROM PHRASES:
- "options in Tamil Nadu" → target_state = TAMIL NADU
- "colleges in Bihar" → target_state = BIHAR
- "suggest options in Delhi" → target_state = DELHI
- "what are the options there" (same state context) → target_state = home_state

⚠️ IMPORTANT: When BOTH home_state and target_state are the SAME:
- This is CASE 1 (same state) - category APPLIES!
- Questions needed: category, sub-category (if SC/ST/OBC), counselling_type

Example: "I have 150000 NEET rank. Belong from Bihar State. Suggest college options in Bihar"
→ home_state = BIHAR (from "Belong from Bihar State")
→ target_state = BIHAR (from "options in Bihar")
→ Same state = Case 1!
→ Questions to ask: category, sub-category (skip counselling_type if clearly state counselling)

Example: "I have 150000 NEET Rank. Suggest college options in Bihar"
→ home_state = UNKNOWN (not mentioned!)
→ target_state = BIHAR (from "options in Bihar")
→ Questions to ask: home_state FIRST, then category, sub-category, counselling_type

---

DATA COLLECTION RULES:
- SELF queries: Fill missing from [USER PROFILE], don't ask what's in profile
- FRIEND/GENERAL queries: MUST ASK for each missing field

QUESTIONS TO ASK (in order, skip if already known):
1. "What is the home/domicile state?"
2. If Case 1 or 2: "What is the reservation category?"
3. If Case 1 or 2 and category is SC/ST/OBC: "Is there a specific sub-category?"
4. "Are you looking at State Counselling or MCC/All India Quota?" (ASK LAST)

---

MULTI-TURN CONVERSATION HANDLING

Data builds up across turns for the SAME subject:

**Self query multi-turn:**
Turn 1: "show colleges in my state" → Use profile's home_state
Turn 2: "what about private colleges?" → Same but college_type=PRIVATE

**Friend query multi-turn:**
Turn 1: "friend scored 380, looking in Delhi"
Turn 2: "his home state is Bihar"
Turn 3: "he's OBC category"
Turn 4: "yes please" → Now have: friend, 380, Bihar→Delhi, OBC

**Multiple different subjects:**
Turns 1-4: First friend data
Turn 5: "now another friend with rank 65400" → NEW subject, start fresh

---

DATA FROM COUNSELLOR RESPONSES

Counsellor messages contain:
- **Clarification questions** → Help understand query (useful context)
- **Search results** (college names, cutoffs) → OUTPUT only, NOT input parameters

NEVER extract score/rank from result text like "for your score of 450..."
Result numbers are OUTPUT, not parameters for new queries.

---

USER PROFILE USAGE

The [USER PROFILE] shows stored data about THIS logged-in user.
ONLY use profile for SELF queries. NEVER use profile for FRIEND/GENERAL queries.

USE PROFILE WHEN:
✅ Query type = SELF
✅ User says "my options", "can I get", "for me"
✅ Clear self-reference language

DO NOT USE PROFILE WHEN:
❌ Query type = FRIEND (friend has different profile)
❌ Query type = GENERAL (hypothetical, not about them)
❌ Query type = AMBIGUOUS (don't assume - ASK first)
❌ User explicitly states different values
❌ NEVER assume home_state = target_state for friend/general queries

---

DECISION LOGIC

0. CHECK FOR CONVERSATION/GREETING:
   → reply_without_database with friendly response

1. IDENTIFY QUERY TYPE: self / friend / general / ambiguous
   - If AMBIGUOUS → ask_clarification: "Is this for yourself or someone else?"

2. FOR FRIEND/GENERAL QUERIES - Check each required field:
   ☐ home_state - If not provided, ASK: "What is the home/domicile state?"
   ☐ category - If not provided, ASK: "What is the reservation category?"
   ☐ sub_category - If applicable
   ☐ counselling_type - If unclear, ASK: "State Counselling or MCC?"
   
   ASK FOR MISSING FIELDS ONE AT A TIME, in this order.
   Do NOT run query until all fields collected.

3. FOR SELF QUERIES:
   - Fill missing from profile
   - Only ask if something essential is missing from profile too

4. DECIDE ACTION:
   - AMBIGUOUS subject → ask_clarification
   - Missing home_state (friend/general) → ask_clarification
   - Missing category → ask_clarification
   - All required data available → run_database_query
   - Process/info question → reply_without_database

---

OUTPUT: Valid JSON only.

{
  "action": "run_database_query" | "ask_clarification" | "reply_without_database",
  "message": "<clarification question OR friendly response if no DB needed, else empty>",
  "extracted": {
    "query_mode": "eligibility" | "informational" | "unknown",
    "query_type": "self" | "friend" | "general" | "ambiguous",
    "metric_type": "score" | "rank" | "none",
    "metric_value": <number or null>,
    "home_state_for_query": "<subject's home state or null if NOT YET PROVIDED>",
    "target_state": "<target state/MCC or null>",
    "category": "<category or null>",
    "sub_category": "<sub-category or null>",
    "college_type": "<GOVERNMENT/PRIVATE/DEEMED/ALL or null>",
    "course": "<MBBS/BDS>",
    "counselling_type": "state" | "mcc" | "unknown",
    "use_profile_defaults": true | false,
    "missing_slots": ["counselling_type", "home_state", "category", "sub_category", ...],
    "data_source": "profile" | "conversation" | "both" | "explicit"
  }
}

CLARIFICATION EXAMPLES (warm, direct):

For counselling type:
"Are you looking at State Counselling or MCC/All India Quota?"

For ambiguous subject:
"Are you asking about options for yourself, or is this for someone else?"

For missing home state:
"What is this person's home/domicile state?"

For missing category (only ask if home = target OR MCC):
"What is the reservation category - General, OBC, SC, ST, or EWS?"

For missing sub-category:
"Is there a specific sub-category for [category]?"

---

⚠️ CRITICAL SCENARIO EXAMPLES - FOLLOW THESE EXACTLY:

**SCENARIO A: Home state NOT mentioned**
Query: "I have 150000 NEET Rank. Suggest college options in Bihar"
Analysis:
- home_state: NOT PROVIDED → MUST ASK
- target_state: BIHAR
- category: NOT PROVIDED
Questions to ask (in order):
1. "Which is your home/domicile state?"
2. "What is your reservation category?" (after knowing home state)
3. "Is there a specific sub-category?" (after knowing category)
4. "Are you looking at State Counselling or MCC?" (ask LAST)

**SCENARIO B: Home state IS mentioned**
Query: "I have 150000 NEET rank. Belong from Bihar State. Suggest college options in Bihar"
Analysis:
- home_state: BIHAR (extracted from "Belong from Bihar State")
- target_state: BIHAR (from "options in Bihar")
- home = target → CASE 1 (same state, category applies!)
Questions to ask (in order):
1. "What is your reservation category?" (home state already known!)
2. "Is there a specific sub-category?" (after knowing category)
(Skip counselling_type if clearly state counselling from context)

⚠️ KEY DIFFERENCE: In Scenario B, DON'T ask for home state - it's already given!

---

EXAMPLE FLOW 1: DIFFERENT STATE (Category NOT needed)
Turn 1: "rank 150000 suggest college in bihar"
→ Ambiguous subject. ASK: "Are you asking for yourself or someone else?"

Turn 2: "in general for someone"
→ query_type=general. ASK: "What is this person's home/domicile state?"

Turn 3: "rajasthan"
→ home=RAJASTHAN, target=BIHAR. Different states = Case 3!
→ Category doesn't matter. ASK: "State Counselling or MCC/All India Quota?"

Turn 4: "state counselling"
→ All data collected for Case 3.
→ RUN QUERY: state=BIHAR, category=GENERAL, domicile=NON-DOMICILE/OPEN

---

EXAMPLE FLOW 2: SAME STATE (Category needed)
Turn 1: "rank 150000 suggest college in rajasthan"
→ ASK: "Are you asking for yourself or someone else?"

Turn 2: "for myself"
→ query_type=self. home=RAJASTHAN from profile, target=RAJASTHAN. Same state = Case 1!
→ Category matters. If profile has category=ST, use it.
→ ASK: "State Counselling or MCC/All India Quota?"

Turn 3: "state counselling"
→ All data collected. Category applies!
→ RUN QUERY: state=RAJASTHAN, category=ST, domicile=DOMICILE/OPEN

---

EXAMPLE FLOW 3: FRIEND QUERY (Home state given in first message)
Turn 1: "my friend scored 540 and he is from Tamil Nadu, what options?"
→ query_type=friend, score=540, home_state=TAMIL NADU, target_state=TAMIL NADU (same state implied)
→ Same state = Case 1! Category matters.
→ ASK: "What is your friend's reservation category - General, OBC, SC, ST, or EWS?"

Turn 2: "ST"
→ category=ST. ASK: "Is there a specific sub-category?"

Turn 3: "no"
→ ASK: "Is your friend looking at State Counselling or MCC/All India Quota?"

Turn 4: "state"
→ All data collected. Category applies!
→ RUN QUERY: state=TAMIL NADU, category=ST, domicile=DOMICILE/OPEN

---

EXAMPLE FLOW 4: MCC (Category needed)
Turn 1: "rank 150000 options"
→ No target specified. ASK: "Are you asking for yourself or someone else?"

Turn 2: "for a friend"
→ query_type=friend. ASK: "Which state or MCC are you looking at?"

Turn 3: "MCC"
→ target=MCC. MCC = Case 2, category will matter.
→ ASK: "What is your friend's home state?"

Turn 4: "UP"
→ home=UTTAR PRADESH. ASK: "What is your friend's reservation category?"

Turn 5: "OBC"
→ category=OBC. MCC doesn't need counselling_type question.
→ RUN QUERY: state=MCC, category=OBC, domicile=OPEN
"""


@dataclass(frozen=True)
class QueryGateResult:
    action: Literal["run_database_query", "ask_clarification", "reply_without_database"]
    message: str
    extracted: dict


def _parse_json_object(raw: str) -> dict:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def gate_user_query(
    client: OpenAI,
    user_question: str,
    *,
    request_id: str | None = None,
) -> QueryGateResult:
    """Single LLM call to route: search vs clarify vs small talk."""
    rid = request_id or "-"
    logger.info("[%s] Gate LLM system prompt:\n%s", rid, _clip(READINESS_SYSTEM))
    logger.info("[%s] Gate LLM user prompt:\n%s", rid, _clip(user_question.strip()))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": READINESS_SYSTEM},
            {"role": "user", "content": user_question.strip()},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = (response.choices[0].message.content or "").strip()
    logger.info("[%s] Gate LLM raw output:\n%s", rid, _clip(raw))
    try:
        data = _parse_json_object(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("[%s] Query gate JSON parse failed: %s — raw=%r", rid, exc, raw[:200])
        return QueryGateResult(
            action="ask_clarification",
            message=(
                "Thanks for sharing that. If you want me to suggest possible colleges, please tell me your "
                "category (e.g. GENERAL, OBC, SC, ST, EWS), whether you want your state or MCC/all India, "
                "and if you prefer government, private, or deemed colleges."
            ),
            extracted={
                "query_mode": "unknown",
                "metric_type": "none",
                "metric_value": None,
                "subject": "unknown",
                "home_state_for_query": "",
                "target_state": "",
                "category": "",
                "college_type": "",
                "use_profile_defaults": False,
                "missing_slots": [],
                "needs_confirmation": False,
                "confirmation_question": "",
            },
        )

    action = data.get("action", "ask_clarification")
    message = (data.get("message") or "").strip()

    if action not in ("run_database_query", "ask_clarification", "reply_without_database"):
        action = "ask_clarification"
    if action == "run_database_query":
        message = ""
    elif action == "reply_without_database" and not message:
        message = (
            "Hi! When you want cutoff-based college options, share your category, "
            "state or MCC/all India, and government vs private vs deemed preference."
        )
    elif not message:
        message = (
            "Thanks for sharing. If you're looking for college options, please share your category, "
            "state (or MCC/all India), and preferred college type (government / private / deemed)."
        )

    extracted = data.get("extracted")
    if not isinstance(extracted, dict):
        extracted = {}

    query_mode = str(extracted.get("query_mode", "unknown")).lower()
    if query_mode not in ("eligibility", "informational", "unknown"):
        query_mode = "unknown"
    metric_type = str(extracted.get("metric_type", "none")).lower()
    if metric_type not in ("score", "rank", "none"):
        metric_type = "none"
    metric_value = extracted.get("metric_value")
    if isinstance(metric_value, str) and metric_value.strip().isdigit():
        metric_value = int(metric_value.strip())
    if not isinstance(metric_value, (int, float)):
        metric_value = None
    subject = str(extracted.get("subject", "unknown")).lower()
    if subject not in ("self", "friend", "general", "ambiguous", "unknown"):
        subject = "unknown"
    use_profile_defaults = bool(extracted.get("use_profile_defaults", False))
    college_type = str(extracted.get("college_type", "")).strip().upper()
    if college_type not in ("", "GOVERNMENT", "PRIVATE", "DEEMED", "ALL", "AIIMS", "JIPMER", "BHU", "AMU"):
        college_type = ""
    
    # Counselling type
    counselling_type = str(extracted.get("counselling_type", "unknown")).lower()
    if counselling_type not in ("state", "mcc", "unknown"):
        counselling_type = "unknown"
    
    raw_missing = extracted.get("missing_slots", [])
    if not isinstance(raw_missing, list):
        raw_missing = []
    valid_slots = (
        "state_or_mcc", "category", "sub_category", "college_type", 
        "neet_metric", "home_state", "target_state", "subject_clarification",
        "counselling_type"
    )
    missing_slots = [
        str(x).strip().lower()
        for x in raw_missing
        if str(x).strip().lower() in valid_slots
    ]

    # Handle both query_type (new) and subject (old) for compatibility
    query_type = str(extracted.get("query_type", "")).lower()
    if query_type not in ("self", "friend", "general", "ambiguous"):
        # Fallback to subject field
        query_type = str(extracted.get("subject", "unknown")).lower()
        if query_type not in ("self", "friend", "general", "ambiguous", "unknown"):
            query_type = "unknown"

    extracted_clean = {
        "query_mode": query_mode,
        "query_type": query_type,  # self / friend / general / ambiguous
        "subject": query_type,  # Keep for backward compatibility
        "metric_type": metric_type,
        "metric_value": int(metric_value) if isinstance(metric_value, (int, float)) else None,
        "home_state_for_query": _safe_str(extracted.get("home_state_for_query")),
        "target_state": _safe_str(extracted.get("target_state")),
        "category": _safe_str(extracted.get("category")),
        "sub_category": _safe_str(extracted.get("sub_category")),
        "college_type": college_type,
        "course": str(extracted.get("course", "")).strip().upper(),
        "counselling_type": counselling_type,
        "use_profile_defaults": use_profile_defaults or (query_type == "self"),
        "missing_slots": missing_slots,
        "data_source": str(extracted.get("data_source", "")).lower(),
    }

    # Trust the LLM decision - no hardcoded overrides
    # The prompt has clear instructions for greetings vs clarification vs database queries

    logger.info(
        "[%s] Query gate parsed: action=%s, query_type=%s, mode=%s, metric=%s:%s, use_profile=%s",
        rid,
        action,
        extracted_clean["query_type"],
        extracted_clean["query_mode"],
        extracted_clean["metric_type"],
        extracted_clean["metric_value"],
        extracted_clean["use_profile_defaults"],
    )
    return QueryGateResult(action=action, message=message, extracted=extracted_clean)


def classify_intro_step_intent(
    client: OpenAI,
    user_text: str,
    *,
    request_id: str | None = None,
) -> Literal["continue_onboarding", "provided_neet_metric", "unclear"]:
    rid = request_id or "-"
    prompt = (user_text or "").strip()
    logger.info("[%s] Intro-intent LLM prompt:\n%s", rid, _clip(prompt))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": INTRO_STEP_SYSTEM.strip()},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = (response.choices[0].message.content or "").strip()
    logger.info("[%s] Intro-intent LLM raw output:\n%s", rid, _clip(raw))
    try:
        data = _parse_json_object(raw)
        intent = str(data.get("intent", "unclear")).strip().lower()
        if intent in ("continue_onboarding", "provided_neet_metric", "unclear"):
            return intent  # type: ignore[return-value]
    except Exception:
        pass

    # Fallback only if LLM output malformed.
    low = prompt.lower()
    if re.search(r"\b(?:air|rank|score|marks?)\b", low) or re.search(r"\b\d{2,7}\b", low):
        return "provided_neet_metric"
    if re.search(r"\b(?:yes|yeah|yep|ok|okay|start|proceed|help|find)\b", low):
        return "continue_onboarding"
    return "unclear"


def interpret_onboarding_response(
    client: OpenAI,
    *,
    current_step: str,
    user_input: str,
    current_preferences: dict,
    step_options: list[dict] | None,
    request_id: str | None = None,
) -> dict:
    """
    LLM-first onboarding parser. Returns normalized JSON contract:
    {
      action: apply_update|ask_rephrase|fallback,
      updates: dict,
      clear_fields: list[str],
      message: str,
      acknowledgement: str
    }
    """
    rid = request_id or "-"
    payload = {
        "current_step": current_step,
        "user_input": user_input,
        "current_preferences": current_preferences or {},
        "step_options": step_options or [],
    }
    logger.info("[%s] Onboarding interpreter input: %s", rid, _clip(json.dumps(payload, ensure_ascii=True)))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": ONBOARDING_INTERPRETER_SYSTEM.strip()},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = (response.choices[0].message.content or "").strip()
    logger.info("[%s] Onboarding interpreter raw output:\n%s", rid, _clip(raw))
    try:
        data = _parse_json_object(raw)
    except Exception:
        return {
            "action": "fallback",
            "updates": {},
            "clear_fields": [],
            "message": "",
            "acknowledgement": "",
        }

    action = str(data.get("action", "fallback")).strip().lower()
    if action not in ("apply_update", "ask_rephrase", "fallback"):
        action = "fallback"
    updates = data.get("updates")
    if not isinstance(updates, dict):
        updates = {}
    # Fix common LLM mistake: course preference written under category.
    cat_mis = updates.get("category")
    if isinstance(cat_mis, str):
        for v in COURSE_PREFERENCE_VALUES:
            if cat_mis.strip().upper() == v.upper():
                updates = dict(updates)
                updates.pop("category", None)
                if not updates.get("course"):
                    updates["course"] = v
                break
    clear_fields = data.get("clear_fields")
    if not isinstance(clear_fields, list):
        clear_fields = []
    clear_fields = [str(x).strip() for x in clear_fields if str(x).strip()]
    message = str(data.get("message", "")).strip()
    acknowledgement = str(data.get("acknowledgement", "")).strip()
    return {
        "action": action,
        "updates": updates,
        "clear_fields": clear_fields,
        "message": message,
        "acknowledgement": acknowledgement,
    }
