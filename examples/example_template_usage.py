"""
Clean template usage examples - showing all three modes.
"""

from verbatim_rag import VerbatimRAG, VerbatimIndex, TemplateManager
from verbatim_rag.vector_stores import LocalMilvusStore
from verbatim_rag.embedding_providers import SpladeProvider

# Create RAG system
sparse_provider = SpladeProvider(model_name="naver/splade-v3", device="cpu")
vector_store = LocalMilvusStore(
    db_path="./example_index.db",
    enable_dense=False,
    enable_sparse=True,
)
index = VerbatimIndex(vector_store=vector_store, sparse_provider=sparse_provider)
rag = VerbatimRAG(index)

print("=== TEMPLATE MODES ===\n")

# Mode 1: Single template for everything
print("1. SINGLE MODE - Same template for all questions")
rag.template_manager.use_single_mode()
# or with custom template:
# rag.template_manager.use_single_mode("Custom template: [RELEVANT_SENTENCES]")

response = rag.query("What are the findings?")
print(f"Template used: {rag.template_manager.get_template('any question')[:50]}...")
print()

# Mode 2: Random templates for variety (your main use case!)
print("2. RANDOM MODE - Generate 50 diverse templates")
rag.template_manager.generate_random_templates(50)

print(f"Generated {len(rag.template_manager._random_templates)} templates")
print("Sample templates:")
for i in range(3):
    template = rag.template_manager.get_template()
    print(f"  {i + 1}. {template[:60]}...")
print()

# Mode 3: Question-specific templates
print("3. QUESTION-SPECIFIC MODE - Different templates for different questions")
rag.template_manager.use_question_specific_mode()

# Add specific templates
rag.template_manager.add_question_template(
    "What are the findings?",
    "Research findings from the documents:\n\n[RELEVANT_SENTENCES]",
)
rag.template_manager.add_question_template(
    "Who are the authors?", "Authors mentioned:\n\n[RELEVANT_SENTENCES]"
)

print("Added question-specific templates:")
print(
    f"  Findings: {rag.template_manager.get_template('What are the findings?')[:50]}..."
)
print(f"  Authors: {rag.template_manager.get_template('Who are the authors?')[:50]}...")
print(f"  Unknown: {rag.template_manager.get_template('Random question?')[:50]}...")
print()

print("=== CONVENIENCE FEATURES ===\n")

# Quick template creation
simple_template = TemplateManager.create_simple(
    intro="Here's what I found:", outro="Hope this helps!"
)
print(f"Simple template: {simple_template}")
print()

academic_template = TemplateManager.create_academic()
print(f"Academic template: {academic_template}")
print()

# Template persistence
print("4. SAVING AND LOADING")
rag.template_manager.save("my_templates.json")
print("Templates saved to my_templates.json")

# Load in new RAG instance
new_rag = VerbatimRAG(index)
new_rag.template_manager.load("my_templates.json")
print(f"Loaded templates - mode: {new_rag.template_manager.info()}")
print()

print("=== INTEGRATION EXAMPLES ===\n")

# Your main use case: Generate 50 templates and use randomly
rag = VerbatimRAG(index)
rag.template_manager.generate_random_templates(50)

questions = [
    "What are the main findings?",
    "What is the methodology?",
    "Who are the authors?",
    "What are the conclusions?",
]

print("Random template variety for same questions:")
for question in questions:
    template = rag.template_manager.get_template(question)
    print(f"Q: {question}")
    print(f"T: {template[:70]}...")
    print()

# Show current state
print(f"Template manager info: {rag.template_manager.info()}")
