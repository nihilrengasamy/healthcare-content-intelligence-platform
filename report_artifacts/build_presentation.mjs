import fs from "node:fs/promises";
import path from "node:path";
import {
  Presentation,
  PresentationFile,
} from "file:///C:/Users/Nihil%20Rengasamy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const FINAL_PPTX =
  "C:\\Users\\Nihil Rengasamy\\Documents\\Codex\\2026-06-27\\hi\\outputs\\healthcare-content-intelligence\\presentation.pptx";
const PREVIEW_DIR =
  "C:\\Users\\Nihil Rengasamy\\Documents\\Codex\\2026-06-27\\hi\\outputs\\healthcare-content-intelligence\\report_artifacts\\presentation_preview";

const SLIDE_W = 1280;
const SLIDE_H = 720;
const PAGE = { left: 54, top: 48, width: 1172, height: 620 };

const COLORS = {
  navy: "#123B66",
  blue: "#2E6FD8",
  blueDark: "#174EA6",
  blueLight: "#EAF2FD",
  teal: "#0F8B8D",
  green: "#1F9D74",
  amber: "#F5A524",
  orange: "#F97316",
  red: "#D92D20",
  purple: "#7C3AED",
  ink: "#1F2937",
  subtext: "#5B6472",
  panel: "#F5F7FA",
  panel2: "#EEF2F6",
  rule: "#D8E0EA",
  white: "#FFFFFF",
};

function bulletize(items) {
  return items.map((item) => `• ${item}`).join("\n");
}

async function writeBlob(filePath, blob) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

function addShape(slide, config) {
  return slide.shapes.add({
    geometry: config.geometry ?? "textbox",
    name: config.name,
    position: {
      left: config.left,
      top: config.top,
      width: config.width,
      height: config.height,
    },
    fill: config.fill ?? "none",
    line:
      config.line ?? { style: "solid", fill: "none", width: 0 },
    borderRadius: config.borderRadius,
    shadow: config.shadow,
  });
}

function addText(slide, config) {
  const shape = addShape(slide, {
    geometry: config.geometry ?? "textbox",
    left: config.left,
    top: config.top,
    width: config.width,
    height: config.height,
    fill: config.fill ?? "none",
    line: config.line ?? { style: "solid", fill: "none", width: 0 },
    borderRadius: config.borderRadius,
    shadow: config.shadow,
    name: config.name,
  });
  shape.text = config.text;
  shape.text.style = {
    fontSize: config.fontSize ?? 20,
    bold: config.bold ?? false,
    color: config.color ?? COLORS.ink,
    italic: config.italic ?? false,
    alignment: config.align ?? "left",
  };
  return shape;
}

function addHeader(slide, title, subtitle, slideNumber) {
  slide.background.fill = COLORS.white;
  addText(slide, {
    left: PAGE.left,
    top: 26,
    width: 720,
    height: 46,
    text: title,
    fontSize: 36,
    bold: true,
    color: COLORS.navy,
  });
  if (subtitle) {
    addText(slide, {
      left: PAGE.left,
      top: 72,
      width: 860,
      height: 24,
      text: subtitle,
      fontSize: 18,
      color: COLORS.subtext,
    });
  }
  addShape(slide, {
    geometry: "rect",
    left: PAGE.left,
    top: 108,
    width: PAGE.width,
    height: 2,
    fill: COLORS.rule,
    line: { style: "solid", fill: COLORS.rule, width: 0 },
  });
  addText(slide, {
    left: PAGE.left,
    top: 686,
    width: 420,
    height: 18,
    text: "Cotiviti Hackathon Assessment | AI-Powered Healthcare Content Intelligence Platform",
    fontSize: 12,
    color: COLORS.subtext,
  });
  addText(slide, {
    left: 1180,
    top: 684,
    width: 40,
    height: 18,
    text: String(slideNumber).padStart(2, "0"),
    fontSize: 12,
    color: COLORS.subtext,
    align: "right",
  });
}

function addPill(slide, left, top, width, text, fill, color = COLORS.ink) {
  addText(slide, {
    geometry: "roundRect",
    left,
    top,
    width,
    height: 28,
    text,
    fontSize: 14,
    bold: true,
    color,
    align: "center",
    fill,
    line: { style: "solid", fill, width: 0 },
    borderRadius: "rounded-full",
  });
}

function addCard(slide, config) {
  addShape(slide, {
    geometry: "roundRect",
    left: config.left,
    top: config.top,
    width: config.width,
    height: config.height,
    fill: config.fill ?? COLORS.panel,
    line: config.line ?? { style: "solid", fill: COLORS.rule, width: 1 },
    borderRadius: config.borderRadius ?? "rounded-xl",
    shadow: config.shadow,
  });
  if (config.title) {
    addText(slide, {
      left: config.left + 18,
      top: config.top + 14,
      width: config.width - 36,
      height: 24,
      text: config.title,
      fontSize: 18,
      bold: true,
      color: config.titleColor ?? COLORS.navy,
    });
  }
  if (config.body) {
    addText(slide, {
      left: config.left + 18,
      top: config.top + (config.title ? 46 : 18),
      width: config.width - 36,
      height: config.height - (config.title ? 58 : 36),
      text: config.body,
      fontSize: config.bodySize ?? 18,
      color: config.bodyColor ?? COLORS.ink,
    });
  }
}

function addMiniMetric(slide, left, top, width, title, value, subtitle, accent) {
  addCard(slide, {
    left,
    top,
    width,
    height: 116,
    fill: COLORS.white,
    line: { style: "solid", fill: COLORS.rule, width: 1 },
    title,
    body: `${value}\n${subtitle}`,
    bodySize: 16,
    titleColor: COLORS.subtext,
  });
  addShape(slide, {
    geometry: "rect",
    left,
    top,
    width,
    height: 5,
    fill: accent,
    line: { style: "solid", fill: accent, width: 0 },
  });
}

function addStepBox(slide, left, top, width, height, text, fill, color = COLORS.ink) {
  addText(slide, {
    geometry: "roundRect",
    left,
    top,
    width,
    height,
    text,
    fontSize: 18,
    bold: true,
    color,
    align: "center",
    fill,
    line: { style: "solid", fill: COLORS.rule, width: 1 },
    borderRadius: "rounded-xl",
  });
}

function addArrowText(slide, left, top, text = "→") {
  addText(slide, {
    left,
    top,
    width: 24,
    height: 28,
    text,
    fontSize: 26,
    bold: true,
    color: COLORS.blue,
    align: "center",
  });
}

function addPlaceholder(slide, left, top, width, height, title, caption) {
  addCard(slide, {
    left,
    top,
    width,
    height,
    fill: COLORS.panel,
    line: { style: "dashed", fill: COLORS.blueDark, width: 1.5 },
    title,
    body: caption,
    bodySize: 15,
    bodyColor: COLORS.subtext,
  });
  addText(slide, {
    left: left + 18,
    top: top + height - 34,
    width: width - 36,
    height: 18,
    text: "Insert working POC screenshot",
    fontSize: 12,
    color: COLORS.blueDark,
    italic: true,
    align: "right",
  });
}

function addRiskMitigationRow(slide, y, risk, mitigation) {
  addCard(slide, {
    left: 90,
    top: y,
    width: 400,
    height: 66,
    fill: "#FFF6F5",
    line: { style: "solid", fill: "#F2C7C2", width: 1 },
    title: risk,
    titleColor: COLORS.red,
  });
  addArrowText(slide, 522, y + 20, "→");
  addCard(slide, {
    left: 568,
    top: y,
    width: 620,
    height: 66,
    fill: "#F3FBF8",
    line: { style: "solid", fill: "#C7E7D8", width: 1 },
    title: mitigation,
    titleColor: COLORS.green,
  });
}

function addManualTable(slide, config) {
  const { left, top, width, header, rows, colWidths, rowHeight, headerFill } = config;
  let x = left;
  header.forEach((cell, i) => {
    addText(slide, {
      geometry: "rect",
      left: x,
      top,
      width: colWidths[i],
      height: rowHeight,
      text: cell,
      fontSize: 13,
      bold: true,
      color: COLORS.white,
      align: "center",
      fill: headerFill ?? COLORS.navy,
      line: { style: "solid", fill: COLORS.white, width: 1 },
    });
    x += colWidths[i];
  });
  rows.forEach((row, r) => {
    let cellX = left;
    const y = top + rowHeight * (r + 1);
    row.forEach((cell, c) => {
      addText(slide, {
        geometry: "rect",
        left: cellX,
        top: y,
        width: colWidths[c],
        height: rowHeight,
        text: String(cell),
        fontSize: 11.5,
        color: COLORS.ink,
        align: c === 0 ? "left" : "left",
        fill: r % 2 === 0 ? COLORS.white : COLORS.panel,
        line: { style: "solid", fill: COLORS.rule, width: 1 },
      });
      cellX += colWidths[c];
    });
  });
}

function setNotes(slide, lines) {
  slide.speakerNotes.textFrame.setText(lines);
  slide.speakerNotes.setVisible(true);
}

const presentation = Presentation.create({
  slideSize: { width: SLIDE_W, height: SLIDE_H },
});

// Slide 1
{
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.white;
  addShape(slide, {
    geometry: "rect",
    left: 0,
    top: 0,
    width: 22,
    height: SLIDE_H,
    fill: COLORS.navy,
    line: { style: "solid", fill: COLORS.navy, width: 0 },
  });
  addPill(slide, 70, 52, 250, "Cotiviti Hackathon | Report + POC", COLORS.blueLight, COLORS.blueDark);
  addText(slide, {
    left: 70,
    top: 120,
    width: 620,
    height: 170,
    text: "AI-Powered Healthcare Content Intelligence Platform",
    fontSize: 46,
    bold: true,
    color: COLORS.navy,
  });
  addText(slide, {
    left: 72,
    top: 268,
    width: 600,
    height: 84,
    text: "Transforming Healthcare Policies into Intelligent Decision Support Systems",
    fontSize: 22,
    color: COLORS.subtext,
  });
  addText(slide, {
    left: 72,
    top: 566,
    width: 420,
    height: 80,
    text: "Nihil Rengasamy\nUniversity at Albany\nJune 2026",
    fontSize: 20,
    color: COLORS.ink,
  });
  addCard(slide, {
    left: 760,
    top: 98,
    width: 420,
    height: 500,
    fill: COLORS.panel,
    line: { style: "solid", fill: COLORS.rule, width: 1 },
  });
  const heroSteps = [
    ["Policy PDFs", COLORS.blueLight, COLORS.blueDark],
    ["AI Intelligence Layer", "#EAF8F5", COLORS.green],
    ["Decision Support", "#FFF5E8", COLORS.orange],
  ];
  heroSteps.forEach(([label, fill, color], i) => {
    addStepBox(slide, 820, 150 + i * 118, 300, 70, label, fill, color);
    if (i < heroSteps.length - 1) {
      addArrowText(slide, 958, 228 + i * 118, "↓");
    }
  });
  addText(slide, {
    left: 800,
    top: 494,
    width: 340,
    height: 60,
    text: "Working POC covers classification, summarization, RAG, rule extraction, feature extraction, claim decision, explainability, and evaluation.",
    fontSize: 18,
    color: COLORS.subtext,
    align: "center",
  });
  addText(slide, {
    left: 70,
    top: 684,
    width: 420,
    height: 18,
    text: "Cotiviti Hackathon Assessment",
    fontSize: 12,
    color: COLORS.subtext,
  });
  setNotes(slide, [
    "This deck summarizes both my written report and my working proof of concept for Cotiviti.",
    "The topic is AI-powered healthcare content intelligence, which focuses on turning policy-heavy healthcare documents into structured, explainable decision support.",
    "I will start with the business problem, then show the proposed solution, the system architecture, the demonstration workflow, and the business value for Cotiviti.",
    "The key message is that this is not just an AI demo. It is a practical, human-in-the-loop platform concept that fits payment integrity workflows.",
  ]);
}

// Slide 2
{
  const slide = presentation.slides.add();
  addHeader(slide, "Problem Statement", "Healthcare policy review is manual, fragmented, and difficult to operationalize at scale.", 2);
  const problemCards = [
    ["Manual review", "Analysts read long PDFs and translate policy language into action."],
    ["Frequent changes", "Coverage criteria and coding rules change faster than teams can operationalize."],
    ["Slow implementation", "Policy updates take time to become usable rules and reference logic."],
    ["Payment leakage risk", "Misread or delayed policy interpretation can affect payment integrity."],
  ];
  problemCards.forEach(([title, body], i) => {
    addCard(slide, {
      left: 70,
      top: 148 + i * 100,
      width: 430,
      height: 82,
      fill: i % 2 === 0 ? COLORS.white : COLORS.panel,
      line: { style: "solid", fill: COLORS.rule, width: 1 },
      title,
      body,
      bodySize: 15,
    });
  });
  addText(slide, {
    left: 580,
    top: 148,
    width: 420,
    height: 26,
    text: "Current challenge flow",
    fontSize: 22,
    bold: true,
    color: COLORS.navy,
  });
  addStepBox(slide, 590, 205, 190, 62, "Policy PDF", COLORS.panel);
  addArrowText(slide, 806, 222);
  addStepBox(slide, 850, 205, 210, 62, "Manual analyst review", COLORS.panel);
  addArrowText(slide, 945, 292, "↓");
  addStepBox(slide, 850, 334, 210, 62, "Interpretation lag", "#FFF6F5", COLORS.red);
  addArrowText(slide, 806, 352, "←");
  addStepBox(slide, 590, 334, 190, 62, "Delayed rule usage", "#FFF6F5", COLORS.red);
  addCard(slide, {
    left: 575,
    top: 465,
    width: 520,
    height: 118,
    fill: COLORS.blueLight,
    line: { style: "solid", fill: "#CFE0FB", width: 1 },
    title: "Core implication for Cotiviti",
    body: bulletize([
      "Policy understanding stays person-dependent",
      "Scaling quality requires more effort, not more intelligence",
      "Decision support arrives too late in the workflow",
    ]),
    bodySize: 16,
    titleColor: COLORS.blueDark,
  });
  setNotes(slide, [
    "The problem starts with the nature of healthcare content itself. Policy documents are long, complex, and frequently updated.",
    "Today, much of the interpretation work still depends on manual reading, tribal knowledge, and delayed translation into operational logic.",
    "That creates risk in payment accuracy, workflow speed, and compliance. It also makes policy execution inconsistent across teams.",
    "This sets up the need for a simple but credible AI proof of concept that shows how documents can be transformed into usable intelligence.",
  ]);
}

// Slide 3
{
  const slide = presentation.slides.add();
  addHeader(slide, "Project Objective", "Move from document-heavy manual review to an AI-assisted, human-validated operating model.", 3);
  addText(slide, {
    left: 104,
    top: 146,
    width: 340,
    height: 28,
    text: "Current process",
    fontSize: 24,
    bold: true,
    color: COLORS.red,
    align: "center",
  });
  ["Read PDF", "Interpret policy", "Translate to rules", "Review exceptions"].forEach((step, i) => {
    addStepBox(slide, 118, 200 + i * 95, 310, 58, step, "#FFF6F5", COLORS.red);
    if (i < 3) addArrowText(slide, 262, 260 + i * 95, "↓");
  });
  addText(slide, {
    left: 830,
    top: 146,
    width: 340,
    height: 28,
    text: "AI-powered process",
    fontSize: 24,
    bold: true,
    color: COLORS.green,
    align: "center",
  });
  [
    "Upload & classify",
    "Summarize & compare",
    "Ask grounded questions",
    "Extract rules & features",
    "Support decisions",
  ].forEach((step, i) => {
    addStepBox(slide, 842, 180 + i * 75, 310, 52, step, "#F3FBF8", COLORS.green);
    if (i < 4) addArrowText(slide, 986, 232 + i * 75, "↓");
  });
  addArrowText(slide, 560, 326, "→");
  addCard(slide, {
    left: 468,
    top: 266,
    width: 310,
    height: 126,
    fill: COLORS.blueLight,
    line: { style: "solid", fill: "#CFE0FB", width: 1 },
    title: "Automation opportunities",
    body: bulletize([
      "Reduce repetitive document reading",
      "Standardize interpretation",
      "Surface explainable recommendations",
    ]),
    bodySize: 16,
    titleColor: COLORS.blueDark,
  });
  setNotes(slide, [
    "The objective of this project is not to replace analysts. It is to improve how analysts work.",
    "On the left is the current process, which is document-heavy and slow to scale. On the right is the target process, where AI accelerates ingestion, search, summarization, extraction, and decision support.",
    "The real value comes from shifting effort away from basic reading and toward higher-value review and judgment.",
    "That transition is what the proof of concept was designed to demonstrate end to end.",
  ]);
}

// Slide 4
{
  const slide = presentation.slides.add();
  addHeader(slide, "Why This Matters for Cotiviti", "The use case aligns directly with payment integrity, policy interpretation, and explainable operational decision support.", 4);
  const cards = [
    ["Reduced manual effort", "Less time spent reading and re-reading long policy PDFs.", COLORS.blueLight, COLORS.blueDark],
    ["Faster policy updates", "Shorter path from policy revision to usable decision support.", "#F3FBF8", COLORS.green],
    ["Improved payment integrity", "More consistent interpretation of coverage and coding logic.", "#FFF5E8", COLORS.orange],
    ["Lower operating cost", "Better analyst leverage without adding workflow complexity.", COLORS.panel, COLORS.ink],
    ["Explainable recommendations", "Human reviewers can see why a recommendation was made.", "#F4F0FF", COLORS.purple],
    ["Reusable intelligence layer", "Policies become a searchable, structured knowledge asset.", COLORS.white, COLORS.navy],
  ];
  cards.forEach(([title, body, fill, color], i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    addCard(slide, {
      left: 76 + col * 378,
      top: 160 + row * 170,
      width: 344,
      height: 132,
      fill,
      line: { style: "solid", fill: COLORS.rule, width: 1 },
      title,
      body,
      bodySize: 16,
      titleColor: color,
    });
  });
  setNotes(slide, [
    "This matters for Cotiviti because the business problem is not abstract. It sits directly inside payment integrity and policy-driven operational work.",
    "The platform can reduce manual effort, speed up policy interpretation, and improve consistency without removing the analyst from the loop.",
    "It also creates explainability and reuse, which are important differentiators in regulated healthcare environments.",
    "This slide connects the technical build to clear business outcomes before we go deeper into architecture.",
  ]);
}

// Slide 5
{
  const slide = presentation.slides.add();
  addHeader(slide, "Overall System Architecture", "End-to-end flow from healthcare documents to explainable decision support.", 5);
  addPill(slide, 76, 146, 160, "Input layer", COLORS.blueLight, COLORS.blueDark);
  addPill(slide, 300, 146, 180, "Ingestion layer", "#EEF8F6", COLORS.green);
  addPill(slide, 542, 146, 210, "Knowledge layer", "#FFF5E8", COLORS.orange);
  addPill(slide, 824, 146, 180, "Decision layer", "#F4F0FF", COLORS.purple);
  addPill(slide, 1060, 146, 130, "Trust layer", "#FFF6F5", COLORS.red);

  const row1 = ["Healthcare Documents", "PDF Loader", "Document Classifier", "Summarizer", "Version Comparison", "Embeddings", "FAISS"];
  row1.forEach((step, i) => {
    addStepBox(slide, 74 + i * 165, 220, 140, 58, step, i === 0 ? COLORS.blueLight : COLORS.white, i === 0 ? COLORS.blueDark : COLORS.ink);
    if (i < row1.length - 1) addArrowText(slide, 219 + i * 165, 236);
  });
  const row2 = ["RAG", "Rule Extraction", "Feature Extraction", "Rule Engine", "ML Model", "Claim Decision", "Explainability", "Evaluation Dashboard"];
  row2.forEach((step, i) => {
    addStepBox(slide, 68 + i * 146, 376, 130, 58, step, i >= 5 ? "#F4F0FF" : COLORS.white, i >= 5 ? COLORS.purple : COLORS.ink);
    if (i < row2.length - 1) addArrowText(slide, 204 + i * 146, 392);
  });
  addArrowText(slide, 1122, 300, "↓");
  addCard(slide, {
    left: 74,
    top: 500,
    width: 1118,
    height: 104,
    fill: COLORS.panel,
    line: { style: "solid", fill: COLORS.rule, width: 1 },
    title: "Architecture principle",
    body: "The platform converts unstructured healthcare documents into grounded answers, reusable rules, structured features, and explainable recommendations. The design intentionally keeps humans in the loop for high-stakes interpretation and decision review.",
    bodySize: 17,
  });
  setNotes(slide, [
    "This is the full system architecture for the proof of concept.",
    "The flow begins with multiple healthcare document types and then moves through ingestion, classification, summarization, comparison, embeddings, retrieval, extraction, decision support, and explainability.",
    "The architecture is modular by design, which means each stage can be evaluated separately while still contributing to an end-to-end workflow.",
    "That modularity is useful for Cotiviti because it supports phased adoption rather than requiring one large system rollout.",
  ]);
}

// Slide 6
{
  const slide = presentation.slides.add();
  addHeader(slide, "Technology Stack", "Grouped by business purpose rather than listed as a flat tool inventory.", 6);
  const stackCards = [
    ["Programming", "Python 3.11+\nModular OOP structure\nTyped production-style modules", COLORS.blueLight, COLORS.blueDark],
    ["AI & LLM", "Groq / GPT-compatible summarization\nPrompt-managed extraction\nPolicy Q&A patterns", "#EEF8F6", COLORS.green],
    ["Retrieval & NLP", "LangChain\nsentence-transformers\nRAG orchestration", "#FFF5E8", COLORS.orange],
    ["ML & Analytics", "Scikit-learn\nPandas\nNumPy", "#F4F0FF", COLORS.purple],
    ["Application Layer", "Streamlit multipage UI\nSession state workflow\nDownloadable outputs", COLORS.white, COLORS.navy],
    ["Storage & Parsing", "PyMuPDF\nFAISS\nJSON / local artifacts", COLORS.panel, COLORS.ink],
  ];
  stackCards.forEach(([title, body, fill, color], i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    addCard(slide, {
      left: 82 + col * 376,
      top: 160 + row * 206,
      width: 338,
      height: 164,
      fill,
      line: { style: "solid", fill: COLORS.rule, width: 1 },
      title,
      body,
      bodySize: 18,
      titleColor: color,
    });
  });
  setNotes(slide, [
    "I grouped the stack by purpose so the audience can see why each technology exists.",
    "Python and Streamlit support rapid application development. LangChain, embeddings, and FAISS support retrieval and search. Scikit-learn supports lightweight predictive signals. PyMuPDF and structured JSON outputs support document parsing and downstream reuse.",
    "The stack is intentionally practical for a hackathon-style proof of concept. It demonstrates capability without overcomplicating deployment.",
    "Next, I move from technology choice into the live workflow of the POC.",
  ]);
}

// Slide 7
{
  const slide = presentation.slides.add();
  addHeader(slide, "Proof of Concept Workflow", "Simple end-to-end demonstrator showing the main functions of the platform.", 7);
  const steps = [
    "Upload",
    "Classify",
    "Summarize",
    "Compare",
    "Chat",
    "Extract Rules",
    "Extract Features",
    "Rule Engine",
    "ML",
    "Decision",
    "Explanation",
    "Evaluation",
  ];
  steps.forEach((step, i) => {
    const row = i < 6 ? 0 : 1;
    const index = row === 0 ? i : i - 6;
    const x = 74 + index * 185;
    const y = row === 0 ? 214 : 408;
    addStepBox(slide, x, y, 150, 66, step, row === 0 ? COLORS.white : COLORS.blueLight, row === 0 ? COLORS.ink : COLORS.blueDark);
    if (index < 5) addArrowText(slide, x + 156, y + 18);
  });
  addArrowText(slide, 1134, 302, "↓");
  addArrowText(slide, 72, 494, "→");
  addCard(slide, {
    left: 74,
    top: 572,
    width: 1116,
    height: 52,
    fill: COLORS.panel,
    line: { style: "solid", fill: COLORS.rule, width: 1 },
    body: "Hackathon design principle: keep the workflow simple, visible, and demonstrable rather than over-optimizing for full production complexity.",
    bodySize: 17,
  });
  setNotes(slide, [
    "This is the live workflow that the proof of concept demonstrates.",
    "The user journey starts with document upload and continues through classification, summarization, version comparison, policy chat, rule extraction, feature extraction, rule execution, ML support, claim decisioning, explainability, and evaluation.",
    "The important point is that the POC proves the flow across modules, not just isolated technical features.",
    "That makes it a strong hackathon submission because it shows hands-on engineering and general capability without overcomplicating the demonstration.",
  ]);
}

// Slide 8
{
  const slide = presentation.slides.add();
  addHeader(slide, "Core AI Modules", "All fifteen modules are grouped into business-facing capabilities for the platform.", 8);
  addText(slide, {
    left: 74,
    top: 140,
    width: 520,
    height: 24,
    text: "Ingestion and intelligence modules",
    fontSize: 20,
    bold: true,
    color: COLORS.navy,
  });
  addText(slide, {
    left: 654,
    top: 140,
    width: 520,
    height: 24,
    text: "Decision, trust, and platform modules",
    fontSize: 20,
    bold: true,
    color: COLORS.navy,
  });
  const leftRows = [
    ["PDF Loader", "Parse policy pages", "PyMuPDF", "Creates analyzable input"],
    ["Document Classifier", "Identify document type", "Rules + LLM", "Routes correct workflow"],
    ["Summarizer", "Create structured summary", "LLM", "Speeds analyst review"],
    ["Version Comparison", "Compare old vs new policies", "Diff + LLM", "Highlights meaningful change"],
    ["Embeddings", "Vectorize text chunks", "Sentence Transformers", "Enables semantic retrieval"],
    ["Vector Store", "Store/search vectors", "FAISS", "Fast similarity search"],
    ["RAG", "Answer policy questions", "Retriever + LLM", "Grounded policy Q&A"],
    ["Prompt Manager", "Centralize prompts", "Template control", "Consistent prompt quality"],
  ];
  const rightRows = [
    ["Rule Extraction", "Convert policy to rules", "LLM + schema", "Draft executable logic"],
    ["Feature Extraction", "Create structured features", "LLM + regex", "Model-ready policy data"],
    ["Rule Engine", "Evaluate rules on features", "Deterministic Python", "Transparent decision checks"],
    ["ML Model", "Generate predictive signals", "Scikit-learn", "Adds probabilistic support"],
    ["Claim Decision", "Combine rule + ML signals", "Hybrid logic", "Final recommendation"],
    ["Explainability", "Explain why decision happened", "Traceable reasoning", "Builds user trust"],
    ["Evaluation", "Score system quality", "Rule-based QA", "Monitors groundedness"],
  ];
  addManualTable(slide, {
    left: 72,
    top: 174,
    width: 540,
    header: ["Module", "Purpose", "Technology", "Business Benefit"],
    rows: leftRows,
    colWidths: [118, 150, 120, 152],
    rowHeight: 34,
    headerFill: COLORS.navy,
  });
  addManualTable(slide, {
    left: 648,
    top: 174,
    width: 560,
    header: ["Module", "Purpose", "Technology", "Business Benefit"],
    rows: rightRows,
    colWidths: [118, 150, 120, 172],
    rowHeight: 34,
    headerFill: COLORS.blueDark,
  });
  setNotes(slide, [
    "This slide summarizes the full module inventory across the project.",
    "The left table covers ingestion, understanding, retrieval, and prompt governance. The right table covers extraction, decision logic, explainability, and evaluation.",
    "I framed each module in terms of business purpose and benefit so the audience can see how technical components map to operational value.",
    "This also reinforces that the platform is modular and extendable rather than being a one-off demo script.",
  ]);
}

// Slide 9
{
  const slide = presentation.slides.add();
  addHeader(slide, "POC Demonstration", "Suggested storyboard for the live demo and screenshot evidence.", 9);
  const placeholders = [
    ["Upload page", "Show policy upload and extraction"],
    ["Classification page", "Display document type and confidence"],
    ["Summary page", "Show executive summary and structured sections"],
    ["Policy Chat", "Grounded policy answer with sources"],
    ["Rule Extraction", "Normalized rules and conditions"],
    ["Claim Decision", "Hybrid decision with reasons and risks"],
    ["Analytics Dashboard", "End-to-end portfolio view"],
  ];
  const coords = [
    [74, 154, 260, 150],
    [362, 154, 260, 150],
    [650, 154, 260, 150],
    [938, 154, 260, 150],
    [170, 350, 260, 150],
    [510, 350, 260, 150],
    [850, 350, 260, 150],
  ];
  placeholders.forEach(([title, caption], i) => {
    const [x, y, w, h] = coords[i];
    addPlaceholder(slide, x, y, w, h, title, caption);
  });
  setNotes(slide, [
    "This slide is designed to support the demo portion of the presentation.",
    "Each placeholder can be replaced with a clean screenshot from the working Streamlit application. The most important screens are upload, classification, summarization, policy chat, rule extraction, claim decisioning, and analytics.",
    "During the live presentation, I would walk through the workflow in the same order shown earlier so the story remains easy to follow.",
    "This keeps the POC concrete and visually grounded for both technical and non-technical reviewers.",
  ]);
}

// Slide 10
{
  const slide = presentation.slides.add();
  addHeader(slide, "Business Value", "The proof of concept is designed to create operational leverage, not just technical novelty.", 10);
  addText(slide, {
    left: 88,
    top: 158,
    width: 250,
    height: 24,
    text: "Value infographic",
    fontSize: 20,
    bold: true,
    color: COLORS.navy,
  });
  addText(slide, {
    geometry: "ellipse",
    left: 132,
    top: 252,
    width: 250,
    height: 250,
    text: "Payment\nIntegrity",
    fontSize: 28,
    bold: true,
    color: COLORS.white,
    align: "center",
    fill: COLORS.navy,
    line: { style: "solid", fill: COLORS.navy, width: 0 },
  });
  const benefits = [
    [460, 176, "Reduced manual review", COLORS.blueLight, COLORS.blueDark],
    [820, 176, "Faster policy updates", "#EEF8F6", COLORS.green],
    [460, 320, "Improved consistency", "#FFF5E8", COLORS.orange],
    [820, 320, "Higher payment integrity", "#F4F0FF", COLORS.purple],
    [460, 464, "Scalable AI platform", COLORS.white, COLORS.navy],
    [820, 464, "Human-in-the-loop validation", COLORS.panel, COLORS.ink],
  ];
  benefits.forEach(([x, y, label, fill, color]) => {
    addCard(slide, {
      left: x,
      top: y,
      width: 300,
      height: 92,
      fill,
      line: { style: "solid", fill: COLORS.rule, width: 1 },
      title: label,
      titleColor: color,
    });
  });
  setNotes(slide, [
    "The business value is centered on better payment integrity supported by faster and more consistent policy intelligence.",
    "Instead of presenting aggressive ROI claims, this slide stays grounded in realistic value areas: reduced manual review, faster policy updates, better consistency, scalable platform design, and explainable human validation.",
    "That framing is important because this is a hackathon proof of concept. The goal is to demonstrate believable capability and strategic relevance, not to overclaim production results.",
    "Next, I address how the platform handles responsible AI concerns.",
  ]);
}

// Slide 11
{
  const slide = presentation.slides.add();
  addHeader(slide, "Responsible AI and Governance", "High-stakes healthcare workflows require grounded answers, transparent logic, and visible controls.", 11);
  addRiskMitigationRow(slide, 170, "Hallucination risk", "RAG grounds answers in retrieved policy text and source citations.");
  addRiskMitigationRow(slide, 258, "Policy ambiguity", "Human reviewers validate summaries, rules, and decision recommendations.");
  addRiskMitigationRow(slide, 346, "Low-confidence outputs", "Confidence scoring and evaluation signals surface uncertainty.");
  addRiskMitigationRow(slide, 434, "Opaque recommendations", "Explainability module shows reasons, evidence, and next actions.");
  addRiskMitigationRow(slide, 522, "Compliance exposure", "Audit trails, structured outputs, and HIPAA-aware handling support review.");
  setNotes(slide, [
    "Responsible AI is essential for this use case because healthcare content directly influences high-stakes operational decisions.",
    "The main risks are hallucination, ambiguity, low confidence, lack of transparency, and compliance exposure.",
    "The platform addresses those risks through grounded retrieval, confidence signals, structured outputs, explainability, evaluation, and human-in-the-loop review.",
    "This is also why the recommendation is to position the system as decision support rather than as a fully autonomous adjudication engine.",
  ]);
}

// Slide 12
{
  const slide = presentation.slides.add();
  addHeader(slide, "Future Enhancements", "A phased roadmap keeps the strategy practical while expanding the intelligence layer over time.", 12);
  const phases = [
    ["Now", bulletize(["Imaging policy rollout", "Prior authorization workflows", "Human-reviewed rule authoring"]), COLORS.blueLight, COLORS.blueDark],
    ["Next", bulletize(["FHIR integration", "Knowledge graph layer", "Cloud-hosted collaboration"]), "#EEF8F6", COLORS.green],
    ["Later", bulletize(["Graph RAG", "Multi-agent orchestration", "Real-time monitoring and feedback learning"]), "#F4F0FF", COLORS.purple],
  ];
  phases.forEach(([title, body, fill, color], i) => {
    addCard(slide, {
      left: 102 + i * 368,
      top: 212,
      width: 304,
      height: 250,
      fill,
      line: { style: "solid", fill: COLORS.rule, width: 1 },
      title,
      body,
      bodySize: 18,
      titleColor: color,
    });
    if (i < 2) addArrowText(slide, 418 + i * 368, 320, "→");
  });
  addCard(slide, {
    left: 102,
    top: 506,
    width: 1040,
    height: 90,
    fill: COLORS.panel,
    line: { style: "solid", fill: COLORS.rule, width: 1 },
    title: "Strategic posture",
    body: "Start with a narrow, high-value analyst copilot and expand only after the trust, retrieval quality, and rule-feature alignment are strong.",
    bodySize: 18,
  });
  setNotes(slide, [
    "This roadmap reflects the recommendation from the written report: start narrow, prove value, and then expand.",
    "The first phase focuses on use cases where policy interpretation matters immediately and where human review can stay closely involved.",
    "The next and later phases show how the platform could grow into richer interoperability, knowledge graph, and monitoring capabilities without changing the core business logic.",
    "That approach balances ambition with realism, which is important for both Cotiviti and the hackathon context.",
  ]);
}

// Slide 13
{
  const slide = presentation.slides.add();
  addHeader(slide, "Conclusion", "The project demonstrates a practical path from policy documents to explainable decision support.", 13);
  const takeaways = [
    ["Problem", "Healthcare policy review is complex, slow, and difficult to operationalize consistently."],
    ["Solution", "A modular AI platform converts documents into grounded summaries, answers, rules, features, and recommendations."],
    ["POC proof", "The working Streamlit demonstrator shows the full end-to-end flow across multiple AI modules."],
    ["Strategic value", "Cotiviti can explore this as a human-in-the-loop capability aligned with payment integrity workflows."],
  ];
  takeaways.forEach(([title, body], i) => {
    addCard(slide, {
      left: 82 + (i % 2) * 560,
      top: 170 + Math.floor(i / 2) * 170,
      width: 520,
      height: 130,
      fill: i % 2 === 0 ? COLORS.white : COLORS.panel,
      line: { style: "solid", fill: COLORS.rule, width: 1 },
      title,
      body,
      bodySize: 18,
      titleColor: COLORS.navy,
    });
  });
  addText(slide, {
    left: 120,
    top: 556,
    width: 1040,
    height: 46,
    text: "AI + Human Expertise = Better Healthcare Payment Integrity",
    fontSize: 30,
    bold: true,
    color: COLORS.blueDark,
    align: "center",
  });
  addText(slide, {
    left: 220,
    top: 610,
    width: 840,
    height: 22,
    text: "Detailed references are provided in the written report bibliography.",
    fontSize: 14,
    color: COLORS.subtext,
    align: "center",
  });
  setNotes(slide, [
    "To conclude, the core problem is real, the proposed solution is aligned to Cotiviti’s domain, and the proof of concept demonstrates the concept in a practical way.",
    "The platform is intentionally modular, explainable, and human-in-the-loop. That is what makes it more believable as a strategic direction than a generic AI demo.",
    "If Cotiviti wants to explore this area further, the best next step is a focused pilot around high-value policy use cases with clear evaluation criteria.",
    "The final takeaway is simple: AI plus human expertise can create better healthcare payment integrity outcomes.",
  ]);
}

await fs.mkdir(PREVIEW_DIR, { recursive: true });

for (const [index, slide] of presentation.slides.items.entries()) {
  const stem = `slide-${String(index + 1).padStart(2, "0")}`;
  const png = await presentation.export({ slide, format: "png", scale: 1 });
  await writeBlob(path.join(PREVIEW_DIR, `${stem}.png`), png);
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(PREVIEW_DIR, `${stem}.layout.json`), await layout.text());
}

const montage = await presentation.export({
  format: "webp",
  montage: true,
  scale: 1,
});
await writeBlob(path.join(PREVIEW_DIR, "deck-montage.webp"), montage);

const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(FINAL_PPTX);

console.log(FINAL_PPTX);
