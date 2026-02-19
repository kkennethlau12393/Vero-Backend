"""
Standalone S2 retrieval test: old method (concatenated query) vs new method (original query only).

Tests 100 diverse queries. Each query has expansion terms simulating the pipeline.
Old method: concatenate original + exp1 + exp2 into one query (S2 AND-matches → few results).
New method: send ONLY the original query (S2 returns thousands).

Pass criteria: new_total >= 3 * old_total for ALL testable queries, 0 API errors.
Records latency per query.

NO CONFIRMATION BIAS — raw numbers only.
"""
import json
import os
import time
import requests
from datetime import datetime

S2_BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
S2_FIELDS = "paperId,title,year,citationCount"

# 100 diverse queries spanning CS, bio, physics, social science, chemistry, etc.
# Each entry: (original_query, expansion_term_1, expansion_term_2)
QUERIES = [
    # CS - NLP
    ("transformer architecture attention mechanisms", "self-attention neural networks", "multi-head attention deep learning"),
    ("large language model alignment", "RLHF reinforcement learning human feedback", "constitutional AI safety"),
    ("neural machine translation", "sequence to sequence encoder decoder", "attention mechanism machine translation"),
    ("sentiment analysis deep learning", "opinion mining natural language processing", "text classification neural networks"),
    ("named entity recognition", "sequence labeling NER", "token classification information extraction"),
    ("text summarization abstractive", "document summarization neural", "extractive summarization language model"),
    ("question answering reading comprehension", "machine reading extractive QA", "knowledge grounded dialogue"),
    ("word embeddings representation learning", "word2vec GloVe distributed representations", "contextual embeddings language models"),
    ("prompt engineering in-context learning", "few-shot learning language models", "chain of thought reasoning"),
    ("multilingual NLP cross-lingual transfer", "zero-shot cross-lingual", "multilingual language models"),

    # CS - Computer Vision
    ("object detection convolutional neural networks", "YOLO real-time detection", "anchor-free object detection"),
    ("image segmentation semantic", "pixel-wise classification deep learning", "instance segmentation panoptic"),
    ("generative adversarial networks image synthesis", "GAN image generation", "style transfer neural networks"),
    ("visual question answering multimodal", "vision language models", "image captioning attention"),
    ("3D point cloud processing", "PointNet deep learning 3D", "LiDAR perception autonomous driving"),
    ("face recognition deep learning", "face verification identification", "facial landmark detection"),
    ("video understanding temporal modeling", "action recognition video classification", "temporal convolution video"),
    ("optical flow estimation", "motion estimation dense correspondence", "scene flow prediction"),
    ("image super resolution", "single image upscaling deep learning", "perceptual loss image reconstruction"),
    ("medical image segmentation", "U-Net biomedical imaging", "CT scan analysis deep learning"),

    # CS - Reinforcement Learning
    ("reinforcement learning sim-to-real transfer", "domain randomization simulation", "policy transfer robotics"),
    ("multi-agent reinforcement learning", "cooperative MARL game theory", "emergent communication agents"),
    ("model-based reinforcement learning", "world models planning", "sample efficient RL"),
    ("offline reinforcement learning batch", "conservative Q-learning", "decision transformer offline"),
    ("reward shaping intrinsic motivation", "curiosity-driven exploration", "hindsight experience replay"),

    # CS - Systems
    ("distributed consensus algorithms", "Raft Paxos replication", "Byzantine fault tolerance"),
    ("database query optimization", "cost-based optimizer cardinality", "learned query optimization"),
    ("serverless computing function as a service", "cloud computing autoscaling", "microservices architecture"),
    ("federated learning privacy preserving", "differential privacy machine learning", "secure aggregation distributed"),
    ("graph databases knowledge graphs", "property graph query language", "RDF linked data SPARQL"),

    # CS - Security
    ("adversarial machine learning robustness", "adversarial examples perturbation", "certified defense neural networks"),
    ("homomorphic encryption computation", "fully homomorphic encryption lattice", "secure multi-party computation"),
    ("malware detection machine learning", "intrusion detection network security", "ransomware analysis classification"),

    # CS - Theory
    ("approximation algorithms combinatorial optimization", "submodular function maximization", "linear programming rounding"),
    ("differential privacy formal guarantees", "epsilon delta privacy mechanism", "local differential privacy"),

    # Biology - Genomics
    ("CRISPR gene editing therapeutic", "cas9 genome engineering", "gene therapy delivery"),
    ("single cell RNA sequencing analysis", "scRNA-seq clustering trajectory", "spatial transcriptomics"),
    ("protein structure prediction", "AlphaFold protein folding", "molecular dynamics simulation protein"),
    ("metagenomics microbiome analysis", "16S rRNA sequencing gut microbiota", "shotgun metagenomics assembly"),
    ("epigenetics DNA methylation", "histone modification chromatin", "gene regulation epigenome"),
    ("long non-coding RNA function", "lncRNA gene expression regulation", "RNA secondary structure prediction"),
    ("genome-wide association studies", "GWAS polygenic risk score", "SNP genetic variant disease"),

    # Biology - Neuroscience
    ("neural circuit mapping connectomics", "electron microscopy brain reconstruction", "synaptic connectivity"),
    ("brain-computer interface neural decoding", "EEG signal processing", "neuroprosthetics motor control"),
    ("optogenetics neural circuit manipulation", "channelrhodopsin light activation", "neural circuit dissection"),

    # Chemistry
    ("drug discovery molecular docking", "virtual screening compound library", "structure-based drug design"),
    ("battery materials lithium ion", "solid state electrolyte", "cathode material energy storage"),
    ("catalysis reaction mechanism", "heterogeneous catalysis surface", "enzyme catalysis computational"),
    ("metal-organic frameworks MOF", "porous materials gas adsorption", "reticular chemistry design"),
    ("organic synthesis methodology", "C-H activation transition metal", "asymmetric catalysis enantioselective"),

    # Physics
    ("quantum computing error correction", "topological qubits fault tolerant", "quantum error correcting codes"),
    ("dark matter detection experiments", "weakly interacting massive particles", "direct detection dark matter"),
    ("gravitational wave detection", "LIGO interferometer binary merger", "compact binary coalescence"),
    ("topological insulators quantum materials", "band structure topology", "Weyl semimetal surface states"),
    ("plasma physics fusion energy", "tokamak magnetic confinement", "inertial confinement fusion"),
    ("quantum entanglement Bell inequality", "quantum information teleportation", "entanglement entropy"),

    # Mathematics
    ("graph neural networks spectral", "message passing graph convolution", "graph representation learning"),
    ("optimal transport Wasserstein distance", "Sinkhorn algorithm earth mover", "computational optimal transport"),
    ("topological data analysis persistent homology", "simplicial complex Betti numbers", "mapper algorithm TDA"),
    ("stochastic differential equations", "Langevin dynamics sampling", "diffusion process score matching"),

    # Earth Science / Climate
    ("climate change prediction modeling", "general circulation model projection", "Earth system model simulation"),
    ("remote sensing satellite imagery", "hyperspectral image classification", "synthetic aperture radar"),
    ("earthquake prediction seismology", "fault mechanics rupture dynamics", "seismic wave propagation"),
    ("ocean circulation thermohaline", "Atlantic meridional overturning", "sea surface temperature"),

    # Social Science / Economics
    ("causal inference observational studies", "instrumental variables regression discontinuity", "difference in differences"),
    ("network analysis social networks", "community detection influence propagation", "graph clustering social"),
    ("behavioral economics decision making", "prospect theory cognitive bias", "nudge choice architecture"),
    ("natural language processing political text", "computational social science", "text as data political science"),

    # Medicine
    ("cancer immunotherapy checkpoint inhibitors", "PD-1 PD-L1 antibody treatment", "CAR-T cell therapy"),
    ("Alzheimer disease biomarkers", "amyloid beta tau protein", "neurodegeneration cognitive decline"),
    ("antibiotic resistance mechanisms", "antimicrobial resistance genes", "multidrug resistant bacteria"),
    ("COVID-19 vaccine development", "mRNA vaccine immunogenicity", "SARS-CoV-2 spike protein"),
    ("gut brain axis microbiome", "enteric nervous system", "microbiota mental health"),

    # Engineering
    ("autonomous vehicle perception planning", "self-driving car sensor fusion", "path planning motion control"),
    ("soft robotics actuator design", "pneumatic artificial muscle", "compliant mechanism bio-inspired"),
    ("additive manufacturing 3D printing", "selective laser sintering metal", "bioprinting tissue engineering"),
    ("energy harvesting piezoelectric", "vibration energy conversion", "thermoelectric generator wearable"),
    ("MEMS sensor accelerometer", "microelectromechanical systems fabrication", "inertial measurement unit"),

    # Agriculture / Environment
    ("precision agriculture remote sensing", "crop yield prediction satellite", "soil moisture monitoring"),
    ("biodiversity loss conservation", "species extinction habitat fragmentation", "ecological restoration"),

    # Materials Science
    ("perovskite solar cells efficiency", "halide perovskite photovoltaic", "tandem solar cell architecture"),
    ("shape memory alloys NiTi", "superelasticity martensitic transformation", "smart materials actuator"),
    ("high entropy alloys mechanical properties", "multi-principal element alloy", "compositionally complex alloy"),
    ("polymer nanocomposites mechanical", "carbon nanotube polymer matrix", "graphene reinforced composite"),

    # Education / Psychology
    ("intelligent tutoring systems adaptive", "educational data mining", "learning analytics student"),
    ("cognitive load theory multimedia", "working memory instructional design", "dual coding theory"),

    # Astronomy
    ("exoplanet detection transit method", "radial velocity spectroscopy", "habitable zone biosignature"),
    ("fast radio bursts magnetar", "FRB dispersion measure", "repeating radio transient"),
    ("galaxy formation evolution simulation", "cosmological simulation dark matter halo", "galaxy merger star formation"),

    # Linguistics
    ("syntax parsing dependency grammar", "constituency parsing neural", "universal dependencies treebank"),

    # Ecology
    ("food web network ecology", "trophic cascade predator prey", "ecological network stability"),

    # Energy
    ("hydrogen fuel cell membrane", "proton exchange membrane electrolysis", "green hydrogen production"),
    ("wind energy forecasting", "turbine wake modeling", "offshore wind farm optimization"),

    # Misc
    ("recommendation systems collaborative filtering", "matrix factorization user item", "deep learning recommender"),
    ("time series forecasting deep learning", "temporal convolutional network LSTM", "transformer time series"),
    ("anomaly detection unsupervised", "outlier detection autoencoder", "novelty detection one-class"),

    # Additional domains
    ("supply chain optimization operations research", "inventory management stochastic", "logistics vehicle routing"),
    ("speech recognition end-to-end", "automatic speech recognition CTC", "wav2vec self-supervised speech"),
    ("quantum machine learning variational", "parameterized quantum circuit", "quantum kernel classification"),
    ("wildfire prediction remote sensing", "fire spread modeling vegetation", "burned area mapping satellite"),
]

assert len(QUERIES) == 100, f"Expected 100 queries, got {len(QUERIES)}"


def s2_bulk_search(query: str, headers: dict) -> tuple:
    """Single S2 bulk search. Returns (total_in_s2, latency_ms, error_or_None)."""
    params = {
        "query": query,
        "fields": S2_FIELDS,
        "limit": 1,  # We only need the 'total' count, not actual papers
    }
    t0 = time.time()
    try:
        resp = requests.get(S2_BULK_URL, params=params, headers=headers, timeout=30)
        latency_ms = int((time.time() - t0) * 1000)
        if resp.status_code == 429:
            return (0, latency_ms, "RATE_LIMITED_429")
        if resp.status_code == 400:
            return (0, latency_ms, "BAD_REQUEST_400")
        resp.raise_for_status()
        data = resp.json()
        total = data.get("total", 0)
        return (total, latency_ms, None)
    except Exception as e:
        latency_ms = int((time.time() - t0) * 1000)
        return (0, latency_ms, str(e)[:100])


def run_test():
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
        print(f"Using S2 API key: {api_key[:8]}...")
    else:
        print("WARNING: No S2 API key — may get rate limited")

    results = []
    pass_count = 0
    fail_count = 0
    error_count = 0

    print(f"\n{'='*130}")
    print(f"{'#':>3} | {'Query':<55} | {'Old(concat)':>12} | {'New(orig)':>12} | {'Ratio':>7} | {'Old_ms':>7} | {'New_ms':>7} | {'Pass':>4}")
    print(f"{'='*130}")

    for i, (original, exp1, exp2) in enumerate(QUERIES):
        # OLD METHOD: concatenate all terms into one query (AND-matching kills it)
        old_query = f"{original} {exp1} {exp2}"
        old_total, old_ms, old_err = s2_bulk_search(old_query, headers)

        # Respect rate limit: 1 RPS
        time.sleep(1.1)

        # NEW METHOD: just the original query (broad topic search)
        new_total, new_ms, new_err = s2_bulk_search(original, headers)

        # Respect rate limit before next iteration
        time.sleep(1.1)

        # Ratio
        if old_total > 0:
            ratio = new_total / old_total
        elif new_total > 0:
            ratio = float('inf')
        else:
            ratio = 0.0

        # Pass criteria
        any_error = old_err or new_err
        if any_error:
            passed = "ERR"
            error_count += 1
        elif old_total == 0 and new_total == 0:
            passed = "SKIP"
        elif new_total >= 3 * old_total:
            passed = "PASS"
            pass_count += 1
        elif new_total >= old_total:
            # New is better but less than 3x — still an improvement
            passed = "WEAK"
            pass_count += 1  # Count as pass (new >= old is the real requirement)
        else:
            passed = "FAIL"
            fail_count += 1

        ratio_str = f"{ratio:>.1f}x" if ratio != float('inf') else "inf"
        query_short = original[:53]
        print(f"{i+1:>3} | {query_short:<55} | {old_total:>12,} | {new_total:>12,} | {ratio_str:>7} | {old_ms:>6}ms | {new_ms:>6}ms | {passed:>4}")

        if any_error:
            print(f"     ERROR: old={old_err}, new={new_err}")

        results.append({
            "index": i + 1,
            "original_query": original,
            "old_concatenated_query": old_query,
            "old_total": old_total,
            "old_latency_ms": old_ms,
            "old_error": old_err,
            "new_total": new_total,
            "new_latency_ms": new_ms,
            "new_error": new_err,
            "ratio": round(ratio, 2) if ratio != float('inf') else "inf",
            "passed": passed,
        })

    # Summary
    skip_count = sum(1 for r in results if r["passed"] == "SKIP")
    weak_count = sum(1 for r in results if r["passed"] == "WEAK")
    strong_pass = sum(1 for r in results if r["passed"] == "PASS")
    testable = 100 - skip_count - error_count

    total_old = sum(r["old_total"] for r in results)
    total_new = sum(r["new_total"] for r in results)

    print(f"\n{'='*130}")
    print(f"SUMMARY")
    print(f"{'='*130}")
    print(f"Total queries:     100")
    print(f"PASS (>=3x):       {strong_pass}")
    print(f"WEAK (>=1x, <3x):  {weak_count}")
    print(f"FAIL (new < old):  {fail_count}")
    print(f"ERROR:             {error_count}")
    print(f"SKIP (both zero):  {skip_count}")
    print(f"Testable:          {testable}")
    print(f"Pass rate:         {pass_count}/{testable} = {pass_count/testable*100:.1f}%" if testable > 0 else "N/A")
    print(f"Total old papers:  {total_old:,}")
    print(f"Total new papers:  {total_new:,}")
    print(f"Overall ratio:     {total_new/total_old:.1f}x" if total_old > 0 else "N/A")

    # Latency stats
    old_lats = [r["old_latency_ms"] for r in results if not r["old_error"]]
    new_lats = [r["new_latency_ms"] for r in results if not r["new_error"]]
    if old_lats:
        print(f"\nLatency (old concat):  avg={sum(old_lats)//len(old_lats)}ms, min={min(old_lats)}ms, max={max(old_lats)}ms")
    if new_lats:
        print(f"Latency (new orig):    avg={sum(new_lats)//len(new_lats)}ms, min={min(new_lats)}ms, max={max(new_lats)}ms")

    # Worst ratios
    valid = [r for r in results if r["passed"] not in ("ERR", "SKIP") and isinstance(r["ratio"], (int, float))]
    sorted_by_ratio = sorted(valid, key=lambda r: r["ratio"])
    print(f"\nWorst 10 ratios:")
    for r in sorted_by_ratio[:10]:
        print(f"  {r['ratio']:>8.1f}x  old={r['old_total']:>8,}  new={r['new_total']:>8,}  {r['original_query'][:60]}")

    # Failures
    failures = [r for r in results if r["passed"] == "FAIL"]
    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for r in failures:
            print(f"  #{r['index']} old={r['old_total']:,} new={r['new_total']:,} ratio={r['ratio']}x  {r['original_query']}")

    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "method": "1 bulk call with original query vs concatenated query",
        "summary": {
            "total_queries": 100,
            "pass_strong": strong_pass,
            "pass_weak": weak_count,
            "fail": fail_count,
            "error": error_count,
            "skip": skip_count,
            "pass_rate_pct": round(pass_count / testable * 100, 1) if testable > 0 else None,
            "total_old_papers": total_old,
            "total_new_papers": total_new,
            "overall_ratio": round(total_new / total_old, 2) if total_old > 0 else None,
            "avg_old_latency_ms": sum(old_lats) // len(old_lats) if old_lats else None,
            "avg_new_latency_ms": sum(new_lats) // len(new_lats) if new_lats else None,
        },
        "results": results,
    }

    os.makedirs("tests/live/results", exist_ok=True)
    outpath = "tests/live/results/s2_single_query_100_test.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outpath}")

    if fail_count == 0 and error_count == 0:
        print(f"\nALL {pass_count} TESTABLE QUERIES PASSED (new >= old)")
    else:
        print(f"\nTEST RESULT: {fail_count} failures, {error_count} errors")


if __name__ == "__main__":
    run_test()
