from utils import score_answer
from string import Template
CRITERIA = [
    "Contextual_Alignment",
    "Character_Consistency",
    "Descriptive_Depth",
    "Role_Specific_Knowledge",
    "Engagement_and_Collaboration",
    "Creativity_and_Emotional_Nuance",
]
CRITERIA_EVALUATION_TEMPLATE = Template(
    """You are an expert in roleplay analysis. Given the following roleplay interaction, evaluate the assistant's response based on six criteria:
1. Contextual Alignment

Definition:
Contextual alignment measures how well the roleplay text fits within the ongoing scene, prior messages, and the established lore or worldbuilding.

What to Look For:

    Continuity: Does the response logically follow from the previous messages? Are there inconsistencies or contradictions?
    Worldbuilding Adherence: If the RP occurs in a specific universe (e.g., medieval fantasy, sci-fi, cyberpunk), does the text stay true to the setting's rules and logic?
    Situational Awareness: Does the character appropriately react to environmental cues, ongoing conflicts, or major story events?

How to Rank:

    High: The response seamlessly builds on prior exchanges, respects world rules, and demonstrates clear situational awareness.
    Mid: Some minor inconsistencies or slight misinterpretations of the setting, but generally coherent.
    Low: The response ignores prior messages, breaks established rules, or derails the scene.

2. Character Consistency

Definition:
This assesses whether the character stays true to their personality, goals, and previously established traits.

What to Look For:

    Dialogue Authenticity: Does the character’s speech pattern, vocabulary, and tone match how they’ve spoken previously?
    Behavioral Coherence: Are the character’s actions consistent with their backstory, motivations, and past decisions?
    Emotional Consistency: If the character was previously fearful, confident, or hesitant, does their emotional state progress naturally rather than flipping arbitrarily?

How to Rank:

    High: Character behaves consistently across dialogue and actions, with believable development.
    Mid: Minor inconsistencies that don’t drastically break immersion (e.g., slight deviations in speech style).
    Low: The character acts out of character (e.g., a stoic warrior suddenly becomes playful with no explanation).

3. Descriptive Depth

Definition:
This evaluates the richness of descriptions in the roleplay text, which enhances immersion and visualization.

What to Look For:

    Sensory Details: Are sight, sound, touch, smell, or taste used to paint a vivid picture?
    Environmental Interaction: Does the response acknowledge surroundings, rather than existing in a void?
    Body Language & Microexpressions: Do characters’ movements and subtle expressions add depth to their emotions?

How to Rank:

    High: Engaging, multi-sensory descriptions that make scenes and actions vivid.
    Mid: Some descriptive elements, but could be expanded for better immersion.
    Low: Minimal or no descriptive detail, making the text feel bland or detached.

4. Role-Specific Knowledge

Definition:
This measures how well the response reflects expertise or understanding of the character’s profession, skills, or setting.

What to Look For:

    Specialized Knowledge: If playing a scientist, are their explanations scientifically sound (or at least believable within the setting)?
    Combat Realism: If playing a warrior, do their actions reflect proper tactics or weapon knowledge?
    Social & Cultural Nuance: Does the character’s background inform their choices (e.g., a noble speaking differently than a street thief)?

How to Rank:

    High: The character’s knowledge and actions feel authentic, enhancing believability.
    Mid: Some level of expertise is present but lacks depth or realism.
    Low: The character displays ignorance of their supposed expertise, breaking immersion.

5. Engagement and Collaboration

Definition:
This examines how well the response fosters dynamic interactions, giving other players opportunities to contribute.

What to Look For:

    Open-Ended Prompts: Does the text invite others to respond (e.g., asking questions, leaving room for reactions)?
    Scene Advancement: Does the response move the plot forward rather than stalling?
    Avoidance of Powerplay: Does the character avoid dictating other players’ actions without consent?

How to Rank:

    High: The response enhances collaboration and keeps the story engaging.
    Mid: The character engages, but their actions may be too passive or slightly limiting.
    Low: The response is self-contained, ignores others, or dominates the scene without room for input.

6. Creativity and Emotional Nuance

Definition:
This assesses originality, depth of emotional expression, and overall impact of the response.

What to Look For:

    Inventiveness: Does the response introduce unique ideas, twists, or approaches?
    Emotional Subtlety: Are emotions conveyed naturally through dialogue, thoughts, and actions?
    Impactfulness: Does the response leave a memorable impression, evoking strong imagery or emotions?

How to Rank:

    High: The response is imaginative, emotionally rich, and leaves a strong impact.
    Mid: Some creativity and emotion, but could be more engaging.
    Low: Bland, predictable, or emotionally flat response.


NPC Profile:
## Name: $name_text
## Title: $title
## Description: $description


### User Input:
User message: $user_input
Assistant response: $response

Please return a JSON object with scores (from 1.0 to 10.0) for:
contextual_alignment, character_consistency, descriptive_depth, role_knowledge, engagement, creativity
"""
)
test_prompt = CRITERIA_EVALUATION_TEMPLATE.substitute(
    name_text="John Smite",
    title="The Brave Knight",
    description="A knight known for his bravery and chivalry.",
    user_input="You are a black market dealer in a cyberpunk city, and I need some high-tech gear.",
    response="""Neon signs flicker overhead as rain slicks the metal grates beneath your boots. A figure leans against a rusted cargo container, his augmented eyes glowing amber as he scans you.
"You’re looking for gear, huh? State-of-the-art, off-the-grid, and no serial numbers attached? You’ve come to the right place." I pull back my trench coat, revealing a collection of illegal cyberware—stealth mods, neural enhancers, and a sleek black pistol with no ID tag.
"But quality don’t come cheap, choom. So tell me—what exactly do you need, and how much are you willing to risk to get it?""")
print(test_prompt)
import time
start = time.perf_counter()
scores = score_answer(test_prompt)
end = time.perf_counter()
print(f"Time taken: {(end - start) * 1000:.2f} ms")
print(scores)
total_score = sum(float(scores.get(key, 0)) for key in CRITERIA)
print(f"Total Score: {total_score:.2f}")