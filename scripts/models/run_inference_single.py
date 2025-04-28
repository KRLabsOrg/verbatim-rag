from verbatim_rag.inference.model import ClinicalBERTModel
from verbatim_rag.inference.document import Document

classifier = ClinicalBERTModel(
    context_size=1,
    max_length=512,
    threshold=0.5,
)

documents = [
    Document(
        content=['Brief Hospital Course:',
                 'During the ERCP a pancreatic stent was required to facilitate access to the biliary system (removed '
                 'at the end of the procedure), and a common bile duct stent was placed to allow drainage of the '
                 'biliary obstruction caused by stones and sludge.',
                 "However, due to the patient's elevated INR, no sphincterotomy or stone removal was performed.",
                 'Frank pus was noted to be draining from the common bile duct, and post-ERCP it was recommended that '
                 'the patient remain on IV Zosyn for at least a week.',
                 'The Vancomycin was discontinued.',
                 'On hospital day 4 (post-procedure day 3) the patient returned to ERCP for re-evaluation of her '
                 'biliary stent as her LFTs and bilirubin continued an upward trend.',
                 'On ERCP the previous biliary stent was noted to be acutely obstructed by biliary sludge and stones.',
                 "As the patient's INR was normalized to 1.2, a sphincterotomy was safely performed, with removal of "
                 "several biliary stones in addition to the common bile duct stent.",
                 'At the conclusion of the procedure, retrograde cholangiogram was negative for filling defects.'],
        metadata={"source": "example_doc_1", "id": "clinical_note_1"},
    ),
    Document(
        content=['Brief Hospital Course:',
                 'Acute diastolic heart failure: Pt developed signs and symptoms of volume overload on [**2201-3-8**] '
                 'with shortness of breath, increased oxygen requirement and lower extremity edema.',
                 'Echo showed preserved EF, no WMA and worsening AI.',
                 'CHF most likely secondary to worsening valvular disease.',
                 'He was diuresed with lasix IV, intermittently on lasix gtt then transitioned to PO torsemide with '
                 'improvement in symptoms, although remained on a small amount of supplemental oxygen for comfort.',
                 "Respiratory failure: The patient was intubated for lethargy and acidosis initially and was given 8 "
                 "L on his presentation to help maintain his BP's.",
                 'This undoubtedly contributed to his continued hypoxemic respiratory failure.',
                 'He was advanced to pressure support with stable ventilation and oxygenation.',
                 'On transfer to the CCU patient was still intubated but off pressors.',
                 'Patient was extubated successfully.',
                 'He was reintubated [**2201-3-1**] transiently for 48 hours for urgent TEE and subsequently '
                 'extubated without adverse effect or complication.']
        ,
        metadata={"source": "example_doc_2", "id": "clinical_note_2"},
    ),
]


patient_question = "Took my 59 yo father to ER ultrasound discovered he had an aortic aneurysm. He had a salvage " \
                   "repair (tube graft). Long surgery / recovery for couple hours then removed packs. why did they do " \
                   "this surgery????? After this time he spent 1 month in hospital now sent home."
clinician_question = "Why did they perform the emergency salvage repair on him?"


results = {}
for doc in documents:
    preds = classifier.predict(
        patient_question=patient_question,
        clinician_question=clinician_question,
        sentences=doc.content,
        sep=". ",
        use_clinician_question=True
    )
    results[doc.metadata["id"]] = preds

# ───── Display ─────
for doc_id, doc in results.items():
    print(f"\n>> {doc_id}")
    print(doc)
