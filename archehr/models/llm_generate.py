import os
import openai
from rich import print
import re
from typing import List, Tuple


PROMPT = """
You are an expert medical assistant tasked with generating a template for the answer to a specific clinical question about a patient.

CONTEXT:

==PATIENT NARRATIVE==
{patient_narrative}

==CLINICIAN'S QUESTION==
{clinician_question}

==NOTE EXCERPT==
{note_excerpt}

==RELEVANT SENTENCES==
{relevant_sentences}

TASK:
- Generate a template for the answer to the clinician's question based on the patient's narrative and the note excerpt.
- The template must include <<RELEVANT SENTENCE>> in the template for the answer to be filled in.

===EXAMPLE===

==PATIENT NARRATIVE==
Took my 59 yo father to ER ultrasound discovered he had an aortic aneurysm. He had a salvage repair (tube graft). Long surgery / recovery for couple hours then removed packs. why did they do this surgery????? After this time he spent 1 month in hospital now sent home.

==CLINICIAN'S QUESTION==
Why did they perform the emergency salvage repair on him?

==NOTE EXCERPT==
He was transferred to the hospital on 2025-1-20 for emergent repair of his ruptured thoracoabdominal aortic aneurysm. He was immediately taken to the operating room where he underwent an emergent salvage repair of ruptured thoracoabdominal aortic aneurysm with a 34-mm Dacron tube graft using deep hypothermic circulatory arrest. Please see operative note for details which included cardiac arrest x2. Postoperatively he was taken to the intensive care unit for monitoring with an open chest. He remained intubated and sedated on pressors and inotropes. On 2025-1-22, he returned to the operating room where he underwent exploration and chest closure. On 1-25 he returned to the OR for abd closure JP/ drain placement/ feeding jejunostomy placed at that time for nutritional support.
Thoracoabdominal wound healing well with exception of very small open area mid wound that is @1cm around and 1/2cm deep, no surrounding erythema. Packed with dry gauze and covered w/DSD.

==RELEVANT SENTENCES==
He was transferred to the hospital on 2025-1-20 for emergent repair of his ruptured thoracoabdominal aortic aneurysm.
He was immediately taken to the operating room where he underwent an emergent salvage repair of ruptured thoracoabdominal aortic aneurysm with a 34-mm Dacron tube graft using deep hypothermic circulatory arrest.

==TEMPLATE==
The procedure was performed because:
- <<RELEVANT SENTENCE>>
- <<RELEVANT SENTENCE>>

==END OF EXAMPLE==

Respond only with the template (no other text) for the answer to the clinician's question (always include <<RELEVANT SENTENCE>> in the template).

Output:
"""

SUMMARIZATION_PROMPT = """
You are an expert medical assistant tasked with creating a concise summary of medical information.

Below are relevant sentences from a medical note, each with an ID number.
Your task is to create a concise summary (under 50 words) that preserves the key information.

==CLINICIAN'S QUESTION==
{clinician_question}

==RELEVANT SENTENCES==
{sentences_with_ids}

INSTRUCTIONS:
1. Create a concise summary that captures the essential information relevant to the clinician's question
2. Group related information from multiple sentences together
3. Include the sentence IDs for each piece of information in your summary using brackets |ID|   
4. When a statement combines information from multiple sentences, include all relevant IDs |ID1,ID2|
5. Keep medical terminology intact where possible
6. Your summary must be under 65 words
7. Your summary must be verbatim from the relevant sentences
8. Your summary must be one sentence per line

EXAMPLE OUTPUT FORMAT (one sentence per line):
The patient had severe abdominal pain |3| and was diagnosed with appendicitis based on elevated WBC and CT findings |1,4|. 
Laparoscopic appendectomy was performed without complications |5,7|.

Output only the summary, nothing else.
"""


class LLMModelGenerate:
    def __init__(self, model_name: str):
        self.model_name = model_name

        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.temperature = float(os.getenv("LLM_TEMP", "0.7"))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "300"))
        self.threads = int(os.getenv("LLM_THREADS", "30"))
        self.api_key = os.getenv("OPENAI_API_KEY", "EMPTY")

        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def generate_template(
        self,
        patient_narrative: str,
        clinician_question: str,
        note_excerpt: str,
        relevant_sentences: list[str],
    ) -> str:
        """Generate a template for the answer to the clinician's question based on the patient's narrative and the note excerpt.
        This method is generating a template for the answer to the clinician's question based on the patient's narrative and the note excerpt. And then it will use the relevant sentences to fill in the template.

        :param patient_narrative: The patient's narrative.
        :param clinician_question: The clinician's question.
        :param note_excerpt: The note excerpt.
        :param relevant_sentences: The relevant sentences.
        :return: The template for the answer to the clinician's question.
        """

        formatted_prompt = PROMPT.format(
            patient_narrative=patient_narrative,
            clinician_question=clinician_question,
            note_excerpt=note_excerpt,
            relevant_sentences="\n".join(
                [f"- {sentence}" for sentence in relevant_sentences]
            ),
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": formatted_prompt}],
        )

        return response.choices[0].message.content.strip()

    def generate_grouped_summary(
        self, clinician_question: str, sentences: List[Tuple[int, str]]
    ) -> str:
        """Generate a summary of the relevant sentences with grouped citations.

        :param clinician_question: The clinician's question
        :param sentences: List of tuples with (sentence_id, sentence_text)
        :return: A summary with grouped citations throughout the text
        """
        # Format sentences with their IDs
        sentences_with_ids = "\n".join(
            [f"|{id}| {sentence}" for id, sentence in sentences]
        )

        formatted_prompt = SUMMARIZATION_PROMPT.format(
            clinician_question=clinician_question, sentences_with_ids=sentences_with_ids
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": formatted_prompt}],
            max_tokens=150,
            temperature=0.3,  # Lower temperature for more focused summary
        )

        # Change accidental [ID] to |ID|
        summary = (
            response.choices[0]
            .message.content.strip()
            .replace("[", "|")
            .replace("]", "|")
        )

        return summary

    def generate(
        self,
        patient_narrative: str,
        clinician_question: str,
        note_excerpt: str,
        sentences: list[str],
        relevant: list[bool],
    ) -> str:
        """Generate an answer to the clinician's question based on the patient's narrative and the note excerpt.
        This method is generating a template for the answer to the clinician's question based on the patient's narrative and the note excerpt. And then it will use the relevant sentences to fill in the template.

        :param patient_narrative: The patient's narrative.
        :param clinician_question: The clinician's question.
        :param note_excerpt: The note excerpt.
        :param sentences: The sentences to predict.
        :param relevant: The relevance of the sentences.
        :return: The answer to the clinician's question.
        """

        relevant_sentences = [
            (i + 1, sentence)
            for i, (sentence, is_relevant) in enumerate(zip(sentences, relevant))
            if is_relevant
        ]

        template = self.generate_template(
            patient_narrative,
            clinician_question,
            note_excerpt,
            [sentence for _, sentence in relevant_sentences],
        )

        if "<<RELEVANT SENTENCE>>" not in template:
            print("Warning: <<RELEVANT SENTENCE>> not found in template")

        # Fill in the template with the relevant sentences
        for sentence_id, sentence in relevant_sentences:
            # Replace the first <<RELEVANT SENTENCE>> with the sentence, if not found, just add it to the end
            if "<<RELEVANT SENTENCE>>" not in template:
                template += f"\n{sentence} |{sentence_id}|"
            else:
                template = template.replace(
                    "<<RELEVANT SENTENCE>>", f"\n{sentence} |{sentence_id}|", 1
                )

        # If relevant sentences are empty, remove <<RELEVANT SENTENCE>> from the template
        if not relevant_sentences:
            template = template.replace("<<RELEVANT SENTENCE>>", "")

        # Check if the template is longer than 75 words and create a summarized version if necessary
        template_words = [w for w in template.split(" ") if w.strip()]
        if len(template_words) > 75:
            print(
                f"Warning: Template is longer than 75 words ({len(template_words)} words). Creating grouped summary..."
            )
            return self.create_summarized_response(
                template, clinician_question, relevant_sentences
            )

        return template

    def create_summarized_response(
        self,
        original_text: str,
        clinician_question: str,
        relevant_sentences: List[Tuple[int, str]],
    ) -> str:
        """Create a summarized response with grouped citations.

        :param original_text: The original template with filled sentences
        :param clinician_question: The clinician's question
        :param relevant_sentences: List of tuples with (sentence_id, sentence_text)
        :return: A summarized response with grouped citations
        """
        # Generate the grouped summary
        grouped_summary = self.generate_grouped_summary(
            clinician_question, relevant_sentences
        )

        # Validate that all sentence IDs are included in the summary
        expected_ids = set(str(id) for id, _ in relevant_sentences)
        found_ids = set()

        # Find all IDs in the summary (both single and grouped IDs)
        id_matches = re.findall(r"\|(\d+(?:,\d+)*)\|", grouped_summary)
        for match in id_matches:
            # Handle both single IDs and grouped IDs (e.g., "1,2,3")
            ids = match.split(",")
            found_ids.update(ids)

        # Check for missing IDs
        missing_ids = expected_ids - found_ids

        missing_ids = list(missing_ids)
        missing_ids.sort()

        if missing_ids:
            print(
                f"Warning: The following sentence IDs are missing from the summary: {', '.join(missing_ids)}"
            )

            # If one, just add it to the summary
            if len(missing_ids) == 1:
                grouped_summary += f"\nSee also: |{missing_ids[0]}|"
            else:
                # Add the missing IDs to the summary
                grouped_summary += f"\nSee also: |{','.join(missing_ids)}|"

        return grouped_summary
