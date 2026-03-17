## Prompt

You are a qualitative research interview designer.

Generate a complete interview protocol JSON file for a **[RESEARCH TOPIC]** study targeting **[TARGET PARTICIPANT GROUP]**.

The protocol must follow the schema below exactly. All fields are required unless marked optional.

### Schema

```json
{
  "protocol_name": "string — short descriptive title for the interview",
  "version": "string — e.g. '1.0'",
  "total_minutes": "number — total interview duration in minutes (typically 30–60)",
  "wrapup_minutes": "number — minutes reserved for wrap-up at the end (typically 3–5)",
  "max_followups_per_thread": "number — max follow-up probes per topic before auto-transitioning (typically 5–10)",
  "description": "string — 2–3 sentence plain-language explanation of the study's purpose and what the participant can expect. Written directly to the participant. End with a reassurance that there are no right or wrong answers.",
  "topics": [
    {
      "topic_id": "string — snake_case unique identifier",
      "topic_title": "string — short display name for the topic",
      "budget_minutes": "number — time allocated for this topic in minutes",
      "guiding_question": "string — the opening question the interviewer asks to introduce this topic. Should be open-ended, conversational, and non-leading.",
      "objectives": [
        "string — what the interviewer is trying to learn from this topic (3 items)",
        "string",
        "string"
      ]
    }
  ]
}
```

### Design Guidelines

**Topics:**
- Include 5–7 topics. The first topic should establish context and rapport. The last topic should be a closing/reflection topic.
- `budget_minutes` across all topics should sum to approximately `total_minutes - wrapup_minutes`.
- Allocate more time to substantive topics and less to opening/closing topics.

**Guiding questions:**
- Use open-ended phrasing (e.g. "Tell me about…", "Walk me through…", "How has…").
- Avoid yes/no questions and avoid leading the participant toward a particular answer.
- Each guiding question should be specific enough to focus the topic but broad enough to allow varied responses.

**Objectives:**
- Write exactly 3 objectives per topic.
- Each objective should start with an action verb (e.g. "Understand", "Explore", "Identify", "Surface", "Capture").
- Objectives should describe what the *researcher* wants to learn, not what the participant should do.
- Make them specific to the topic — avoid generic objectives that could apply to any topic.

**Description:**
- Written in second person ("Your responses will…", "We are interested in…").
- Should convey the purpose of the research and reassure the participant that there are no right or wrong answers.

### Example Output (for reference)

```json
{
  "protocol_name": "Family Caregiver Experience Interview",
  "version": "1.0",
  "total_minutes": 45,
  "wrapup_minutes": 5,
  "max_followups_per_thread": 10,
  "description": "This study explores the lived experiences of family caregivers — what the role involves day-to-day, how it affects your wellbeing and relationships, and what support would make a real difference. Your perspective will help us better understand and address the needs of caregivers. There are no right or wrong answers — please share as much or as little as you feel comfortable with.",
  "topics": [
    {
      "topic_id": "caregiving_background",
      "topic_title": "Caregiving Background",
      "budget_minutes": 6,
      "guiding_question": "Can you tell me a little about the person you care for and how you came to take on that role?",
      "objectives": [
        "Understand who the participant cares for and the nature of the care relationship",
        "Explore how the participant came into the caregiving role (planned vs. unexpected)",
        "Establish baseline context about the duration, intensity, and type of caregiving"
      ]
    },
    {
      "topic_id": "closing",
      "topic_title": "Closing Reflections",
      "budget_minutes": 4,
      "guiding_question": "Is there anything about your experience as a caregiver that you feel is important for others to understand, that we haven't touched on yet?",
      "objectives": [
        "Give the participant space to share any perspectives not yet captured",
        "Invite reflection on what the participant most wants others to understand",
        "Close the interview respectfully and affirm the value of their contribution"
      ]
    }
  ]
}
```

### Instructions

1. Replace **[RESEARCH TOPIC]** and **[TARGET PARTICIPANT GROUP]** with the actual study context.
2. Paste this prompt into your LLM of choice.
3. Review the output — particularly check that `budget_minutes` sums correctly and that guiding questions are genuinely open-ended.
4. Save the generated file to `sample_protocols/` and point `AI_PROTOCOL_PATH` at it in your `.env`.
