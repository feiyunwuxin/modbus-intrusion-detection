
export const meta = {
  name: 'modbus-semantic-feature-extraction-research',
  description: 'Deep research on Modbus protocol semantic feature extraction for SCADA/ICS intrusion detection',
  phases: [
    { title: 'Scope: 5 search angles' },
    { title: 'Search: parallel agents' },
    { title: 'Fetch: top sources' },
    { title: 'Verify: adversarial' },
    { title: 'Synthesize: final report' },
  ],
}

// Phase 1: Scope — decompose into 5 search angles
const ANGLES = [
  {
    key: 'L1-L2-field-frame',
    label: 'L1-L2 字段/帧级语义',
    prompt: 'Search academic literature (2010-2025) on Modbus protocol field-level and frame-level semantic features for SCADA/ICS intrusion detection. Cover: function code distribution, register address patterns, one-hot encoding of function codes, payload length, payload entropy, Modbus exception codes, RTU vs TCP framing semantics. Find papers by authors like Robert Mitchell, Caroline Chan, Goldenberg/Wool, Linjama, Urias/Farwell. For each paper capture: title, authors, year, venue, dataset used, features defined, and key result. Output a structured list of at least 8-10 papers. Use web search and web fetch. Verify author names and venues from Google Scholar / arXiv / DOI pages — do not invent. Mark unverifiable ones as "未核实".',
  },
  {
    key: 'L3-session-temporal',
    label: 'L3 会话/时序语义',
    prompt: 'Search academic literature on Modbus session-level and temporal semantic features for ICS intrusion detection. Cover: command-response pairs, request-response timing/inter-arrival time, polling cycle patterns, sequence-of-frames semantics, Modbus transaction pairing, multi-frame correlation, time-difference features. Find papers from 2010-2025. Authors/groups to look for: Robert Mitchell (Honeywell), Wei Gao, Quanyan Zhu, Sridhar Adepu, Nils Ole Tippenhauer, and other ICS security researchers. Capture: title, authors, year, venue, dataset (especially Gas Pipeline dataset by iTrust/SUTD), features defined, results. Output at least 8-10 papers with verifiable citations. Use web search/fetch. Mark unverifiable as "未核实".',
  },
  {
    key: 'public-datasets',
    label: '公开数据集与基准',
    prompt: 'Search literature on Modbus/SCADA semantic feature extraction applied to PUBLIC datasets. Focus on: (1) iTrust SUTD Gas Pipeline dataset (used by Mathur & Tippenhauer 2016 "SWaT"-like gas pipeline), (2) SWaT and WADI datasets (iTrust Singapore), (3) BATADAL (BATtle of the Attack Detection ALgorithms — Cagliari/UNICA), (4) HIL-based ICS testbeds, (5) Lemay/SCADASandbox (Modbus-specific), (6) Mississippi State University SCADA datasets. For each dataset, identify: who created it, year, attack scenarios, what semantic features have been extracted from it, and benchmark results. Output structured list of at least 10 papers. Use web search and verify dataset origins and authors. Mark unverifiable as "未核实".',
  },
  {
    key: 'L4-process-physics',
    label: 'L4 工艺流/物理语义',
    prompt: 'Search academic literature on process-level and physical semantics for SCADA intrusion detection. Cover: mass balance equations, conservation laws, thermodynamic invariants, SCADA process state machines, cyber-physical consistency, semantic invariants derived from process physics, anomaly detection using physical relationships between registers. Look for: papers by Urias/Farwell (Sandia), Erez Karpas, Anita Wasilewska, Miguel A. Prada, approaches using "first principles" in ICS security, papers on Algeciras/Madrid process-aware detection. Search arXiv, IEEE Xplore, ACM, Springer for 2010-2025. Output at least 6-8 papers. Verify each citation. Mark unverifiable as "未核实".',
  },
  {
    key: 'deep-era-semantic',
    label: '深度/Transformer 时代语义建模',
    prompt: 'Search recent (2022-2025) literature on deep learning and Transformer-based semantic feature extraction for Modbus/SCADA intrusion detection. Cover: 1D-CNN, BiLSTM, GRU, TCN, Transformer encoder, graph neural networks (GNN) on Modbus traffic, contrastive learning for protocol semantics, self-supervised pretraining on ICS traffic, payload tokenization of Modbus bytes. Authors/groups: Sridhar Adepu, Eunsuk Kang, Cristina Nita-Rotaru, Alvaro Cardenas, ICCPS, USENIX Security, NDSS, IEEE TIFS, Computers & Security. Output at least 8-10 papers from top venues (NDSS, USENIX, CCS, S&P, TIFS). For each, verify the venue. Mark unverifiable as "未核实".',
  },
]

// Phase 2: Search — 5 parallel agents
const searchResults = await parallel(ANGLES.map(a => () =>
  agent(a.prompt, { label: `search:${a.key}`, phase: 'Search: parallel agents' })
))

// Phase 3: Fetch — collect all cited URLs and verify a sample
// First, instruct a synthesis agent to extract the most important sources
const allPapers = searchResults.filter(Boolean)
const extraction = await agent(`You are a research librarian. From the following 5 search outputs on Modbus semantic feature extraction, extract a DEDUPLICATED list of the most important 30+ papers/sources with: title, authors, year, venue, URL. Remove any duplicated papers across the 5 sources. Prefer papers from top venues (USENIX Security, NDSS, CCS, IEEE S&P, TIFS, Computers & Security, NDSS, ICCPS, Applied Sciences, Sensors). Order by relevance. Mark each as "verified" (you can confirm the venue/year from URL) or "unverified".

Here are the 5 search outputs (each separated):

=== ANGLE 1: L1-L2 字段/帧级语义 ===
${searchResults[0] || 'EMPTY'}

=== ANGLE 2: L3 会话/时序语义 ===
${searchResults[1] || 'EMPTY'}

=== ANGLE 3: 公开数据集与基准 ===
${searchResults[2] || 'EMPTY'}

=== ANGLE 4: L4 工艺流/物理语义 ===
${searchResults[3] || 'EMPTY'}

=== ANGLE 5: 深度/Transformer 时代语义建模 ===
${searchResults[4] || 'EMPTY'}

Return a JSON array of {title, authors, year, venue, url, verified, brief_relevance}.`, {
  label: 'extract:papers',
  phase: 'Fetch: top sources',
  schema: {
    type: 'object',
    properties: {
      papers: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            title: { type: 'string' },
            authors: { type: 'string' },
            year: { type: 'number' },
            venue: { type: 'string' },
            url: { type: 'string' },
            verified: { type: 'boolean' },
            brief_relevance: { type: 'string' },
          },
          required: ['title', 'authors', 'year', 'venue'],
        },
      },
    },
    required: ['papers'],
  },
})

const papers = extraction.papers || []
log(`extracted ${papers.length} papers`)

// Phase 4: Verify — adversarial 3-vote on a sample of the most controversial/unverifiable claims
// Focus on (a) the foundational papers (Mitchell, Goldenberg/Wool, Mathur/Tippenhauer)
// (b) recent claims about SOTA results
// (c) any paper not marked verified

const toVerify = papers.filter(p => !p.verified).slice(0, 12)
const verifiedPapers = papers.filter(p => p.verified)

const verdicts = await parallel(toVerify.map(p => () =>
  agent(`Verify this citation. Try to REFUTE it (i.e. find evidence the paper does NOT exist, or that the venue/year is wrong).
Title: ${p.title}
Authors: ${p.authors}
Year: ${p.year}
Venue: ${p.venue}
URL: ${p.url || 'N/A'}

Search the web (Google Scholar, arXiv, DBLP, IEEE Xplore) for the EXACT title. If you find it, confirm venue/year/authors. If you cannot find it, that is grounds for refutation.

Return: {refuted: boolean, evidence: string, actualVenue?: string, actualYear?: number}`, {
    label: `verify:${p.title?.slice(0, 30)}`,
    phase: 'Verify: adversarial',
    schema: {
      type: 'object',
      properties: {
        refuted: { type: 'boolean' },
        evidence: { type: 'string' },
        actualVenue: { type: 'string' },
        actualYear: { type: 'number' },
      },
      required: ['refuted', 'evidence'],
    },
  })
))

// Attach verdicts
const finalPapers = verifiedPapers.map(p => ({ ...p, status: 'verified' }))
toVerify.forEach((p, i) => {
  const v = verdicts[i]
  finalPapers.push({ ...p, status: v?.refuted ? 'refuted' : 'confirmed', evidence: v?.evidence })
})

log(`${finalPapers.filter(p => p.status !== 'refuted').length}/${finalPapers.length} papers survived verification`)

// Phase 5: Synthesize — final comprehensive report
const synthesis = await agent(`You are a senior ICS security researcher writing a comprehensive literature review on Modbus protocol semantic feature extraction for SCADA intrusion detection. Your output will be used in the Related Work / Background section of a Master's thesis.

Use the following VERIFIED paper list and your own knowledge to produce a structured report. The report must:

1. DEFINE "semantic" — what does "semantic feature" mean in Modbus context (5 levels: L1 字段, L2 帧, L3 会话, L4 工艺流, L5 跨会话/系统级)

2. LIST and GROUP representative papers/features by level. For each feature, give:
   - Feature name (Chinese + English)
   - Definition + calculation formula
   - What it captures (the "semantic meaning")
   - Representative paper (use ONLY the verified papers below; if a paper is "unverified" or "refuted", do NOT cite it)
   - Performance on what dataset (if known)

3. PROVIDE an "evolution timeline" 2010-2025 showing major milestones

4. COMPARE: semantic features vs statistical features vs deep-learned features

5. CONCLUDE with "gaps / your project's contribution" — what is missing in the field that the user's project (gas pipeline, 25+ models, TCN v4+SE, 8K-params MCU deploy) addresses

VERIFIED PAPER LIST (use only these):
${JSON.stringify(finalPapers.filter(p => p.status !== 'refuted'), null, 2)}

TOTAL: ${finalPapers.filter(p => p.status !== 'refuted').length} papers

Format: Use Chinese with English paper titles. Use markdown headers, tables, code blocks where appropriate. Length: 3500-5500 words. End with a "参考文献" list using [N] numbered citation style, where N maps to entries in the paper list. Do NOT fabricate any papers not in the verified list.`, {
  label: 'synthesize:report',
  phase: 'Synthesize: final report',
})

return {
  totalPapers: finalPapers.length,
  verifiedPapers: finalPapers.filter(p => p.status !== 'refuted').length,
  refutedPapers: finalPapers.filter(p => p.status === 'refuted').length,
  paperList: finalPapers,
  report: synthesis,
}
