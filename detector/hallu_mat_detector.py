import re
import torch
import faiss
import networkx as nx
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rank_bm25 import BM25Okapi
from mpl_toolkits.mplot3d import Axes3D
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import wikipediaapi
import random
import scipy.stats
import community as community_louvain  # pip install python-louvain
from tabulate import tabulate  # pip install tabulate

# ------------------------------------------------------------------------------
# Step 0: Load models and pipelines
# ------------------------------------------------------------------------------

# Load the causal language model (LLM) for generating answers.
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)

# Load the NLI model for intrinsic confidence and entailment checks.
nli_model_name = "microsoft/deberta-v3-large"
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)

# Load the embedding model for text embeddings (used in FAISS and knowledge graph).
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ------------------------------------------------------------------------------
# Step 1: Basic Retrieval Functions (Only used if Intrinsic Evaluation fails)
# ------------------------------------------------------------------------------
def generate_chunks(query):
    """
    Split the query into all possible contiguous word chunks.
    """
    words = query.split()
    chunks = set()
    for n in range(1, len(words) + 1):
        for i in range(len(words) - n + 1):
            chunk = " ".join(words[i:i + n])
            chunks.add(chunk)
    return list(chunks)

def rerank_chunks_bm25(query, chunks):
    """
    Rerank the chunks based on BM25 scores.
    """
    tokenized_chunks = [chunk.split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    scores = bm25.get_scores(query.split())
    return sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

def retrieve_data(query, reranked_chunks):
    """
    Retrieve supporting documents from Wikipedia.
    """
    retrieved_docs = []
    # Wikipedia retrieval
    wiki = wikipediaapi.Wikipedia(user_agent="MyResearchBot/1.0", language="en")
    for chunk, _ in reranked_chunks:
        try:
            page = wiki.page(chunk)
            if page.exists():
                retrieved_docs.append(Document(page_content=page.summary,
                                               metadata={"source": "Wikipedia", "link": page.fullurl}))
        except Exception as e:
            print(f"⚠️ Error retrieving Wikipedia data for '{chunk}': {e}")
    return retrieved_docs

def display_retrieved_docs(retrieved_docs):
    """
    Print a summary of the retrieved documents.
    """
    if not retrieved_docs:
        print("\n❌ No valid data retrieved.")
        return
    print("\n📖 **Retrieved Knowledge Base:**")
    for i, doc in enumerate(retrieved_docs):
        print(f"\n🔹 Source: {doc.metadata['source']}")
        print(f"🔗 Link: {doc.metadata.get('link', 'No link available')}")
        print(f"📜 Content: {doc.page_content[:500]}...")
        print("-" * 80)

def retrieve_with_faiss(query, retrieved_docs):
    """
    Retrieve supporting documents using FAISS-based dense retrieval.
    """
    if not retrieved_docs:
        return []
    texts = [doc.page_content for doc in retrieved_docs]
    embeddings = embedding_model.embed_documents(texts)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    query_embedding = embedding_model.embed_query(query)
    distances, indices = index.search(np.array([query_embedding]), k=min(5, len(texts)))
    return [retrieved_docs[i] for i in indices[0]]

# ------------------------------------------------------------------------------
# Step 2: Generate LLM Output WITHOUT Retrieval (Intrinsic-Only Mode)
# ------------------------------------------------------------------------------
def generate_llm_output_no_retrieval(query):
    """
    Generate an answer using only the query without external retrieval.
    """
    prompt = f"Query: {query}"
    llm_response = text_generator(prompt, max_new_tokens=200, truncation=True)
    return llm_response[0]['generated_text']

# ------------------------------------------------------------------------------
# Step 3: Intrinsic Evaluation Functions (Upgraded)
# ------------------------------------------------------------------------------
def chain_of_verification(llm_response, num_iterations=2):
    """
    Repeatedly ask the LLM to verify its own response.
    Returns 1 if consistent across iterations, else 0.
    """
    consistent = True
    current_response = llm_response
    for i in range(num_iterations):
        prompt = f"Is the following statement logically sound and factually consistent? '{current_response}'"
        revised_response = text_generator(prompt, max_new_tokens=200, truncation=True)[0]['generated_text']
        if revised_response.strip() != current_response.strip():
            consistent = False
        current_response = revised_response
    return int(consistent)

def calculate_confidence_variance(llm_response):
    """
    Calculate the maximum confidence and its variance from token-level probabilities.
    """
    inputs = nli_tokenizer(llm_response, return_tensors="pt", truncation=True)
    outputs = nli_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().numpy()[0]
    max_confidence = np.max(probs)
    confidence_std = np.std(probs)
    return max_confidence, confidence_std

def entropy_analysis_multiple(llm_response, num_samples=5):
    """
    Compute the average entropy and standard deviation of the LLM response over several samples.
    """
    entropy_scores = []
    for _ in range(num_samples):
        inputs = nli_tokenizer(llm_response, return_tensors="pt", truncation=True)
        outputs = nli_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).tolist()[0]
        entropy_scores.append(scipy.stats.entropy(probs))
    return np.mean(entropy_scores), np.std(entropy_scores)

def iterative_self_refinement(llm_response, max_iterations=3):
    """
    Ask the LLM to iteratively refine its response to improve consistency.
    """
    refined_response = llm_response
    for i in range(max_iterations):
        prompt = f"Review the following statement and correct any inconsistencies or errors: '{refined_response}'"
        refined_response = text_generator(prompt, max_new_tokens=200, truncation=True)[0]['generated_text']
    return refined_response

def detect_internal_contradictions(llm_response):
    """
    Ask the LLM to check for internal contradictions in its response.
    Returns 1 if no contradictions are found.
    """
    prompt = f"Does the following statement contain any internal contradictions? '{llm_response}' Answer 'Yes' or 'No'."
    response = text_generator(prompt, max_new_tokens=50, truncation=True)[0]['generated_text']
    return int("no" in response.lower())

def perturb_query(original_query):
    """
    Generate a slightly modified version of the query.
    """
    templates = [
        "Is it accurate to say that ’[statement]’?",
        "Would you consider the statement ’[statement]’ to be correct?",
        "Can we confirm that ’[statement]’ is true?",
        "Does the statement ’[statement]’ hold true?",
        "Is ’[statement]’ a valid statement?",
        "Is there accuracy in the claim ’[statement]’?",
        "Could ’[statement]’ be considered a factual statement?",
        "Is it correct to assume that ’[statement]’ is true?",
        "Would it be right to say ’[statement]’ is accurate?",
        "Does the statement ’[statement]’ accurately reflect the truth?"
    ]
    template = random.choice(templates)
    return template.replace("[statement]", original_query)

def self_consistency_check(query, llm_response):
    """
    Compare the original LLM response with the response to a perturbed query.
    """
    perturbed_query = perturb_query(query)
    print(f"\n **Perturbed Query:** {perturbed_query}")
    perturbed_response = text_generator(perturbed_query, max_new_tokens=200, truncation=True)[0]['generated_text']
    print(f" **Perturbed LLM Output:** {perturbed_response}")
    return int(perturbed_response.strip() == llm_response.strip())

def in_context_self_verification(llm_response):
    """
    Ask the LLM in a different context to verify its response.
    """
    context_query = f"Do you know if this statement is true? {llm_response}"
    verification_response = text_generator(context_query, max_new_tokens=100, truncation=True)[0]['generated_text']
    return int("not sure" not in verification_response.lower())

# ------------------------------------------------------------------------------
# Step 4: Extrinsic Verification Functions (Unchanged)
# ------------------------------------------------------------------------------
def check_entailment(claim, evidence):
    """
    Use the NLI model to check if the evidence supports the claim.
    """
    inputs = nli_tokenizer(claim, evidence, return_tensors="pt", truncation=True)
    outputs = nli_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}
    return label_map[torch.argmax(probs).item()]

def retrieve_facts(query, retrieved_docs):
    """
    Retrieve supporting facts using FAISS.
    """
    if not retrieved_docs:
        return []
    texts = [doc.page_content for doc in retrieved_docs]
    embeddings = embedding_model.embed_documents(texts)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    query_embedding = embedding_model.embed_query(query)
    distances, indices = index.search(np.array([query_embedding]), k=min(5, len(texts)))
    return [retrieved_docs[i] for i in indices[0]]

# ------------------------------------------------------------------------------
# Step 5: Compute Final Reliability Score (Combine Intrinsic and Extrinsic)
# ------------------------------------------------------------------------------
def compute_reliability_score(self_consistency, confidence_std, entropy_std, nli_verification):
    """
    Combine the intrinsic and extrinsic scores into a final reliability score.
    """
    weights = {"self_consistency": 0.2, "confidence": 0.2, "entropy": 0.2, "nli_verification": 0.4}
    reliability_score = (
        (self_consistency * weights["self_consistency"]) +
        ((1 - confidence_std) * weights["confidence"]) +
        ((1 - entropy_std) * weights["entropy"]) +
        (nli_verification * weights["nli_verification"])
    )
    return reliability_score

# ------------------------------------------------------------------------------
# Step 6: Knowledge Graph Functions (Enhanced Visualization)
# ------------------------------------------------------------------------------
def build_heterogeneous_fact_graph(knowledge_base, similarity_threshold=0.5):
    """
    Build a knowledge graph where each node represents a fact fragment.
    Edges connect nodes with semantic similarity above the threshold.
    """
    if not knowledge_base:
        return nx.Graph()
    G = nx.Graph()
    fact_embeddings = embedding_model.embed_documents(knowledge_base)
    for i, fact in enumerate(knowledge_base):
        G.add_node(i, text=fact, embedding=fact_embeddings[i], type='fact')
        for j in range(i):
            similarity = cosine_similarity([fact_embeddings[i]], [fact_embeddings[j]])[0][0]
            if similarity > similarity_threshold:
                G.add_edge(i, j, weight=similarity)
    return G

# ----- Improved 3D Graph Visualization -----
def plot_3d_graph(knowledge_graph):
    """
    Visualize the knowledge graph in 3D using a Kamada-Kawai layout.
    The function uses larger figure dimensions, refined node sizes, and clear axis labels.
    """
    if not knowledge_graph.nodes():
        print("⚠️ No graph to display.")
        return
    pos = nx.kamada_kawai_layout(knowledge_graph, dim=3)
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    
    # Plot edges
    for edge in knowledge_graph.edges(data=True):
        x_coords, y_coords, z_coords = [], [], []
        for node in edge[:2]:
            x, y, z = pos[node]
            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)
        ax.plot(x_coords, y_coords, z_coords, color='gray', alpha=0.5, lw=1)
    
    # Plot nodes
    degrees = dict(knowledge_graph.degree())
    sizes = [300 + 150 * degrees[node] for node in knowledge_graph.nodes()]
    colors = [knowledge_graph.nodes[node].get('community', 0) for node in knowledge_graph.nodes()]
    xs = [pos[node][0] for node in knowledge_graph.nodes()]
    ys = [pos[node][1] for node in knowledge_graph.nodes()]
    zs = [pos[node][2] for node in knowledge_graph.nodes()]
    sc = ax.scatter(xs, ys, zs, s=sizes, c=colors, cmap='viridis', alpha=0.9)
    
    # Add node labels (optional)
    for node in knowledge_graph.nodes():
        x, y, z = pos[node]
        ax.text(x, y, z, f"{node}", fontsize=10, color='black')
    
    ax.set_title("3D Knowledge Graph Visualization", fontsize=16)
    ax.set_xlabel("X-axis", fontsize=12)
    ax.set_ylabel("Y-axis", fontsize=12)
    ax.set_zlabel("Z-axis", fontsize=12)
    fig.colorbar(sc, shrink=0.5, aspect=10, label="Community")
    plt.tight_layout()
    plt.show()

# ----- Improved 2D Graph Visualization -----
def plot_2d_graph(knowledge_graph):
    """
    Visualize the knowledge graph in 2D using a Kamada-Kawai layout.
    The function uses enhanced figure dimensions and refined node styles.
    """
    pos = nx.kamada_kawai_layout(knowledge_graph, dim=2)
    plt.figure(figsize=(14, 10))
    degrees = dict(knowledge_graph.degree())
    sizes = [300 + 150 * degrees[node] for node in knowledge_graph.nodes()]
    colors = [knowledge_graph.nodes[node].get('community', 0) for node in knowledge_graph.nodes()]
    nodes = nx.draw_networkx_nodes(knowledge_graph, pos, node_size=sizes, node_color=colors, cmap=plt.cm.viridis, alpha=0.9)
    nx.draw_networkx_edges(knowledge_graph, pos, alpha=0.5)
    nx.draw_networkx_labels(knowledge_graph, pos, font_size=10, font_color='black')
    plt.title("2D Knowledge Graph Visualization", fontsize=16)
    plt.xlabel("X-axis", fontsize=12)
    plt.ylabel("Y-axis", fontsize=12)
    plt.colorbar(nodes, label="Community")
    plt.tight_layout()
    plt.show()

def plot_heatmap(knowledge_base):
    """
    Plot a heatmap of semantic similarities between fact fragments.
    Enlarged figure and formatted numbers for clarity.
    """
    if len(knowledge_base) < 2:
        print("⚠️ Not enough facts for a heatmap.")
        return
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(knowledge_base)
    similarities = cosine_similarity(vectors)
    df = pd.DataFrame(similarities, 
                      columns=[f"Fact {i+1}" for i in range(len(knowledge_base))],
                      index=[f"Fact {i+1}" for i in range(len(knowledge_base))])
    plt.figure(figsize=(14, 10))
    sns.heatmap(df, annot=True, cmap="YlGnBu", linewidths=0.5, annot_kws={"size":10}, fmt=".2f")
    plt.title("Heatmap of Fact Similarity (Consistency)", fontsize=14)
    plt.xlabel("Fact Fragments", fontsize=12)
    plt.ylabel("Fact Fragments", fontsize=12)
    plt.show()

# ------------------------------------------------------------------------------
# Step 7: Main Pipeline: Generate and Evaluate the Answer
# ------------------------------------------------------------------------------
def generate_knowledge_base(query):
    print(f"\n Query: {query}")
    
    # STEP 7A: Generate LLM Output (Intrinsic-Only Mode)
    llm_output = generate_llm_output_no_retrieval(query)
    print(f"\n **LLM Output (No Retrieval):**\n{llm_output}")
    
    # STEP 7B: Perform Intrinsic Evaluation on the LLM Output
    self_consistency = chain_of_verification(llm_output)
    max_confidence, confidence_std = calculate_confidence_variance(llm_output)
    avg_entropy, entropy_std = entropy_analysis_multiple(llm_output)
    refined_response = iterative_self_refinement(llm_output)
    internal_consistency = detect_internal_contradictions(refined_response)
    in_context_ver = in_context_self_verification(llm_output)
    
    print(f"\n🔍 **Intrinsic Evaluation Scores:**")
    print(f"✅ Self-Consistency: {self_consistency}")
    print(f"✅ Confidence Score: {max_confidence:.2f} (Variance: {confidence_std:.2f})")
    print(f"✅ Average Entropy: {avg_entropy:.2f} (Variance: {entropy_std:.2f})")
    print(f"✅ Internal Consistency (No Contradictions): {internal_consistency}")
    print(f"✅ In-Context Self-Verification: {in_context_ver}")
    
    # Compute intrinsic reliability score (if high, assume extrinsic = 1)
    intrinsic_score = compute_reliability_score(self_consistency, confidence_std, entropy_std, 1)
    
    if intrinsic_score < 0.5:
        print("\n⚠️ Intrinsic evaluation failed. Proceeding with retrieval-based extrinsic verification...")
        chunks = generate_chunks(query)
        reranked_chunks = rerank_chunks_bm25(query, chunks)
        retrieved_docs = retrieve_data(query, reranked_chunks)
        display_retrieved_docs(retrieved_docs)
        faiss_docs = retrieve_with_faiss(query, retrieved_docs)
        knowledge_base_retrieved = [doc.page_content for doc in faiss_docs] if faiss_docs else [doc.page_content for doc in retrieved_docs]
        if knowledge_base_retrieved:
            extrinsic_scores = [1 if check_entailment(llm_output, fact) == "entailment" else 0 for fact in knowledge_base_retrieved]
            nli_verification = sum(extrinsic_scores) / len(extrinsic_scores)
        else:
            nli_verification = 0
    else:
        nli_verification = 1  # Assume high extrinsic reliability if intrinsic evaluation is high
    
    # STEP 7D: Compute Final Reliability Score and Print Confidence Level
    final_score = compute_reliability_score(self_consistency, confidence_std, entropy_std, nli_verification)
    print(f"\n🚦 **Final Reliability Score: {final_score:.2f}**")
    
    if final_score < 0.5:
        print("\n **Explanation for Low Confidence:**")
        if self_consistency == 0:
            print("- The LLM's response was inconsistent upon re-evaluation.")
        if max_confidence < 0.5:
            print("- The LLM's confidence in its response was low.")
        if avg_entropy > 0.7:
            print("- The LLM's response had high entropy, indicating uncertainty.")
        if nli_verification < 0.5:
            print("- The extrinsic validation (NLI) score was low, indicating a lack of supporting evidence.")
    elif final_score < 0.7:
        print("\n⚠️ **Medium Confidence: The response needs further review.**")
    else:
        print("\n✅ **High Confidence: The response is reliable.**")
    
    # STEP 7E: Build and Visualize the Knowledge Graph from the Refined Response
    knowledge_facts = [fact.strip() for fact in refined_response.split('.') if len(fact.strip()) > 20]
    
    if not knowledge_facts:
        print("⚠️ No valid factual fragments could be extracted.")
        return

    # Build the knowledge graph
    kg = build_heterogeneous_fact_graph(knowledge_facts)
    partition = community_louvain.best_partition(kg, weight='weight')
    nx.set_node_attributes(kg, partition, 'community')
    
    # Group factual fragments by community
    grouped_fragments = {}
    for node in kg.nodes():
        comm = kg.nodes[node].get('community', 0)
        fragment = kg.nodes[node].get('text', '')
        grouped_fragments.setdefault(comm, []).append(fragment)
    
    # Create a table from grouped fragments
    table_data = []
    for comm, fragments in sorted(grouped_fragments.items()):
        joined_fragments = "\n".join([f"{idx}. {frag}" for idx, frag in enumerate(fragments, start=1)])
        table_data.append({"Community": f"Community {comm}", "Fragments": joined_fragments})
    
    print("\n **Grouped Factual Fragments by Community:**")
    print(tabulate(table_data, headers="keys", tablefmt="fancy_grid"))
    
    print("\n--- 3D Graph Visualization ---")
    plot_3d_graph(kg)
    print("\n--- 2D Graph Visualization ---")
    plot_2d_graph(kg)
    print("\n--- Heatmap of Fact Consistency ---")
    plot_heatmap(knowledge_facts)

# ------------------------------------------------------------------------------
# Step 8: Run the Pipeline
# ------------------------------------------------------------------------------
query = input("Enter your query: ").strip()
generate_knowledge_base(query)
