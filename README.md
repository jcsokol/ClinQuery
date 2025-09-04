# Clinical Timeline Query System

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open-informational)](https://clinquery-live.onrender.com/)
[![Technical README](https://img.shields.io/badge/Technical-README-blue)](https://github.com/jcsokol/ClinQuery/blob/main/README_TECHNICAL.md)

I’m an MD trained in Massachusetts’ largest safety-net hospital (adult + pediatric emergency sub-internships) and an engineer (Stanford BS/MS). I built this end-to-end clinical reasoning engine — from NER to SQL to natural language querying — that answers questions most retrieval-based LLM systems can’t.

It turns raw EHR notes into a temporal, canonicalized database capable of answering reasoning-heavy queries like:

> “Which patients remained clinically stable after heart pump placement?”.

Where typical retrieval pipelines return text fragments that “look” relevant, this system:
- Models abstract clinical concepts (e.g. stability, watcher status) as structured proxies
- Links them to temporal event data
- Enables direct query execution over structured timelines

**Built in ~2.5 months** on 589 synthetic ICU notes generated from the open-source Synthea simulator. No real patient data used.

---

## Why Clinical Reasoning Can’t Be Prompt-Engineered

Real retrieval-based clinical reasoning needs structured logic over time. But most retrieval pipelines (including RAG) just pull sentences that “look” relevant missing the ability to:

- Interpret phrases like “improved perfusion” or “deteriorated after diuresis”
- Resolve relative time constraints like “≥2 days post-implant”
- Map abstractions like “hemodynamic instability” into concrete measurable proxies

What happens under the hood:

- **Clinical abstractions → structured proxies**
  _e.g. “stability” → unchanged pressors, stable lactate, no ICU escalation_
- **Entity, timestamp, and negation extraction**
  using GatorTron + BioLinkBERT, with custom rules
- **Temporal database construction**
  canonicalized events, baselines, and trends
- **Query interpretation**
  natural language → SQL → LLM synthesis

**This enables:**

- Complex, clinically meaningful queries over an entire database
- Highly reliable comparisons across sources
- Rapid extension to new clinical concepts by adding to the abstraction mapping

### Handling Undefined Concepts

When a query contains something not directly in the database (e.g. “respiratory distress”):

- Semantic match first: check against custom definitions in provided yaml fileset
- If unresolved, asks for clarification: 
  _e.g. “which vitals, labs, or events define respiratory distress in your case?”_

This approach keeps the system flexible, safe, and extensible even in this early prototype. 

---

## For Builders: Data + Assumptions

**Input expectations:** 
- Notes are plain text (e.g. discharge or progress notes)
- Tabular content uses pipe-delimited format
- Notes follow typical ICU formatting conventions  

**Modularity:**
- NER and relation extraction may require fine-tuning on your own data
- Concept normalization, timestamp/negation logic, timeline construction, and querying are modular and generalizable (see [full technical readme](https://github.com/jcsokol/ClinQuery/blob/main/README_TECHNICAL.md) for details)

If you're building your own version, reach out! I’m happy to discuss improvements, ideas, or adaptations.

---

## Built Independently, Inspired by Real Clinical Work

This project was inspired by challenges I encountered at Johnson & Johnson (Abiomed), but was independently built from scratch over ~2.5 months using synthetic ICU notes and fully open infrastructure. No proprietary assets were used.

A customized version was delivered to J&J for internal use on real EHR data.

- EHR notes are based on [Synthea](https://github.com/synthetichealth/synthea), an open-source synthetic patient simulator
- Concept normalized via [UMLS Metathesaurus](https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/index.html)
- Models: UFNLP/gatortron-base (Yang et al) and michiyasunaga/BioLinkBERT-large (Yasunaga et al)
- Query layer utilizes OpenAI API

---

## About Me

I bridge frontline clinical experience with production-ready ML engineering, with:

- Intuition for how physicians think
- State-of-the-art NLP/ML toolchains (GatorTron, BioLinkBERT, UMLS)
- Robust data engineering (temporal SQL schemas, scalable preprocessing)

This allows me to iterate both smart ***and fast***. If you’re also in this space, or know someone who is, I’d love to talk!

Contact: <span>jsokol</span><span>@</span><span>alumni.stanford.edu</span> | [linkedin](https://www.linkedin.com/in/jan-sokol-md-17215655/)
