from archehr.models.base import ArchehrModel
import os
import openai
from concurrent.futures import ThreadPoolExecutor
from rich import print


PROMPT_ZERO_SHOT = """
You are an expert medical assistant tasked with determining whether a sentence from a medical note is relevant to answering a specific clinical question about a patient.

CONTEXT:

==PATIENT NARRATIVE==
{patient_narrative}

==CLINICIAN'S QUESTION==
{clinician_question}

==NOTE EXCERPT==
{note_excerpt}

==SENTENCE TO EVALUATE==
{sentence}

TASK:
Determine if this specific sentence contains information that is relevant to answering the clinician's question about this patient.

A sentence is considered RELEVANT if it:
1. Directly answers the clinician's question
2. Provides essential context needed to answer the question
3. Contains supplementary information that helps understand the patient's condition related to the question

A sentence is considered NOT RELEVANT if it:
1. Contains general medical information unrelated to the question
2. Mentions details about the patient that don't pertain to the question
3. Is administrative or procedural content not specific to the patient's condition
4. Is boilerplate text that doesn't provide specific information about this patient

INSTRUCTIONS:
- Focus only on the relevance of the given sentence, not the entire note
- Consider the patient's specific condition as described in the narrative
- The sentence must have a clear connection to answering the clinician's question

Respond with "RELEVANT" or "NOT RELEVANT", then give a short explanation for your answer.
"""


PROMPT = """
You are an expert medical assistant tasked with determining whether a sentence from a medical note is relevant to answering a specific clinical question about a patient.

CONTEXT:

==PATIENT NARRATIVE==
{patient_narrative}

==CLINICIAN'S QUESTION==
{clinician_question}

==NOTE EXCERPT==
{note_excerpt}

==SENTENCE TO EVALUATE==
{sentence}

TASK:
Determine if this specific sentence contains information that is relevant to answering the clinician's question about this patient.

A sentence is considered RELEVANT if it:
1. Directly answers the clinician's question
2. Provides essential context needed to answer the question
3. Contains supplementary information that helps understand the patient's condition related to the question

A sentence is considered NOT RELEVANT if it:
1. Contains general medical information unrelated to the question
2. Mentions details about the patient that don't pertain to the question
3. Is administrative or procedural content not specific to the patient's condition
4. Is boilerplate text that doesn't provide specific information about this patient

INSTRUCTIONS:
- Focus only on the relevance of the given sentence, not the entire note
- Consider the patient's specific condition as described in the narrative
- The sentence must have a clear connection to answering the clinician's question

==EXAMPLE==

==PATIENT NARRATIVE==
Took my 59 yo father to ER ultrasound discovered he had an aortic aneurysm. He had a salvage repair (tube graft). Long surgery / recovery for couple hours then removed packs. why did they do this surgery????? After this time he spent 1 month in hospital now sent home.

==CLINICIAN'S QUESTION==
Why did they perform the emergency salvage repair on him?

==NOTE EXCERPT==
He was transferred to the hospital on 2025-1-20 for emergent repair of his ruptured thoracoabdominal aortic aneurysm. He was immediately taken to the operating room where he underwent an emergent salvage repair of ruptured thoracoabdominal aortic aneurysm with a 34-mm Dacron tube graft using deep hypothermic circulatory arrest. Please see operative note for details which included cardiac arrest x2. Postoperatively he was taken to the intensive care unit for monitoring with an open chest. He remained intubated and sedated on pressors and inotropes. On 2025-1-22, he returned to the operating room where he underwent exploration and chest closure. On 1-25 he returned to the OR for abd closure JP/ drain placement/ feeding jejunostomy placed at that time for nutritional support.
Thoracoabdominal wound healing well with exception of very small open area mid wound that is @1cm around and 1/2cm deep, no surrounding erythema. Packed with dry gauze and covered w/DSD.

==SENTENCE TO EVALUATE==
He was immediately taken to the operating room where he underwent an emergent salvage repair of ruptured thoracoabdominal aortic aneurysm with a 34-mm Dacron tube graft using deep hypothermic circulatory arrest.

==ANSWER==
RELEVANT, because it directly answers the question about why they performed the emergency salvage repair.

==END OF EXAMPLE==

Respond with "RELEVANT" or "NOT RELEVANT", then give a short explanation for your answer.
"""


class LLMModel(ArchehrModel):
    def __init__(self, model_name: str, zero_shot: bool = False):
        self.model_name = model_name
        self.zero_shot = zero_shot

        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.temperature = float(os.getenv("LLM_TEMP", "0.0"))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "200"))
        self.threads = int(os.getenv("LLM_THREADS", "30"))
        self.api_key = os.getenv("OPENAI_API_KEY", "EMPTY")

        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def _process_sentence(
        self,
        patient_narrative: str,
        clinician_question: str,
        note_excerpt: str,
        sentence: str,
    ) -> tuple[bool, str]:
        if self.zero_shot:
            formatted_prompt = PROMPT_ZERO_SHOT.format(
                patient_narrative=patient_narrative,
                clinician_question=clinician_question,
                note_excerpt=note_excerpt,
                sentence=sentence,
            )
        else:
            formatted_prompt = PROMPT.format(
                patient_narrative=patient_narrative,
                clinician_question=clinician_question,
                note_excerpt=note_excerpt,
                sentence=sentence,
            )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical assistant that determines sentence relevance.",
                },
                {"role": "user", "content": formatted_prompt},
            ],
        )

        result = response.choices[0].message.content.strip().upper()

        relevant = "NOT RELEVANT" not in result
        explanation = result.split("RELEVANT")[1].strip()

        return relevant, explanation

    def predict(
        self,
        patient_narrative: str,
        clinician_question: str,
        note_excerpt: str,
        sentences: list[str],
    ) -> list[bool]:
        """
        Predict the relevance of a sentence in a case.

        :param patient_narrative: The patient's narrative.
        :param clinician_question: The clinician's question.
        :param note_excerpt: The note excerpt.
        :param sentences: The sentences to predict.
        :return: The relevance of the sentences.
        """
        results = []
        explanations = []

        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            futures = []
            for i, sentence in enumerate(sentences):
                future = executor.submit(
                    self._process_sentence,
                    patient_narrative,
                    clinician_question,
                    note_excerpt,
                    sentence,
                )
                futures.append(future)

            for future in futures:
                try:
                    is_relevant, explanation = future.result()
                    results.append(is_relevant)
                    explanations.append(explanation)
                except Exception as e:
                    print(f"Error in sentence processing: {e}")

        return results
