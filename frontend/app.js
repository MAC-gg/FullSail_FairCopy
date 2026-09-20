// view at: http://127.0.0.1:8000
const API_BASE = "";

// main - nav
const main = document.getElementById("main");
const btnReset = document.getElementById("btnReset");

// form
const formBox = document.getElementById("form-box");
const form = document.getElementById("classify-form");
const textInput = document.getElementById("text-input");
const urlInput = document.getElementById("url-input");

// results
const results = document.getElementById("results-box");
const articleTitleEl = document.getElementById("article_title");
const dateEl = document.getElementById("date");
const subTypeEl = document.getElementById("submission-type");
const subContentEl = document.getElementById("submission-content");
const authorEl = document.getElementById("author");
const photoCreditsEl = document.getElementById("photo_credits");
const wordCountsEl = document.getElementById("word_counts");
const labelBars = document.getElementById("label-bars");
const sentenceCount = document.getElementById("sentence-count");
const sentenceList = document.getElementById("sentence-list");

// UI/UX
const loadMoreBtn = document.getElementById("load-more");
const loading = document.getElementById("loading");
const loadingLabel = document.getElementById("loadingLabel");
const errorEl = document.getElementById("error");
const errorLabel = document.getElementById("errorLabel");
const shareLinkInput = document.getElementById("share-link-input");
const btnCopyLink = document.getElementById("copy-link-btn");
const sharedBanner = document.getElementById("shared-banner");

let currentSessionId = null;
let nextOffset = 0;

// HISTORY
const BOOT_ID_KEY = "fairCopyServerBootId";
const HISTORY_KEY = "fairCopyHistory";
const MAX_HISTORY_ITEMS = 50;
const historyBox = document.getElementById("history-box");
const historyList = document.getElementById("history-list");

// INIT EVENTS
loadMoreBtn.addEventListener("click", () => loadNextSentenceBatch(10));
btnReset.addEventListener("click", btnResetFunc);
document.getElementById("copy-link-btn").addEventListener("click", btnCopyLinkFunc);
// init checks
(async () => {
  // server check - history clear if it was restarted
  await checkServerBootId();
  renderHistoryList();

  // shared check - see if ID is in URL
  const params = new URLSearchParams(window.location.search);
  const sharedId = params.get("shared");
  if (sharedId) {
    formBox.classList.add("hidden");
    loadingLabel.textContent = "Loading Shared Report...";
    loading.classList.remove("hidden");
    await loadSharedReport(sharedId);
  }
})();

function showError(msg) {
  errorLabel.textContent = msg;
  errorEl.classList.remove("hidden");
}

function clearError() {
  errorEl.classList.add("hidden");
  errorLabel.textContent = "";
}

function btnResetFunc() {
  // back to form from results
  results.classList.add("hidden");
  // rerender history for new item
  renderHistoryList();

  textInput.value = "";
  urlInput.value = "";
  sharedBanner.classList.add("hidden"); // if report was shared
  resetAnim();
  
  formBox.classList.remove("hidden");
}

const BAR_DETAILS = {
  "factual": {
    "color":"#379674",
    "label":"Factual",
    "cssClass":"fac",
  },
  "opinion": {
    "color":"#D52348",
    "label":"Opinion/Editorial",
    "cssClass":"opi",
  },
  "hyperpartisan": {
    "color":"#8042A7",
    "label":"Hyperpartisan",
    "cssClass":"hyp",
  },
  "clickbait": {
    "color":"#4177B8",
    "label":"Clickbait",
    "cssClass":"cli",
  },
};

function renderLabelBars(labels) {
  const byName = Object.fromEntries(labels.map(l => [l.name, Math.round(l.confidence * 100)]));
  const container = document.getElementById("label-bars");

  const factual = byName["factual"] ?? 0;
  const opinion = byName["opinion"] ?? 0;
  const hyperpartisan = byName["hyperpartisan"] ?? 0;
  const clickbait = byName["clickbait"] ?? 0;

  container.innerHTML = `
    <div class="bar-row">
      <div class="bar-row-header dual">
        <span class="${BAR_DETAILS["factual"].cssClass}">${BAR_DETAILS["factual"].label}${factual <= 8 ? ": " + factual + "%" : ""}</span>
        <span class="${BAR_DETAILS["opinion"].cssClass}">${BAR_DETAILS["opinion"].label}${opinion <= 8 ? ": " + opinion + "%" : ""}</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width: ${factual}%; background: ${BAR_DETAILS["factual"].color};justify-content:start;"><span>${factual > 8 ? factual + "%" : ""}</span></div>
        <div class="bar-fill" style="width: ${opinion}%; background: ${BAR_DETAILS["opinion"].color};"><span>${opinion > 8 ? opinion + "%" : ""}</span></div>
      </div>
    </div>

    <div class="bar-row">
      <div class="bar-row-header"><span class="${BAR_DETAILS["hyperpartisan"].cssClass}">${BAR_DETAILS["hyperpartisan"].label}${hyperpartisan <= 8 ? ": " + hyperpartisan + "%" : ""}</span></div>
      <div class="bar-track">
        <div class="bar-fill" style="width: ${hyperpartisan}%; background: ${BAR_DETAILS["hyperpartisan"].color};"><span>${hyperpartisan > 8 ? hyperpartisan + "%" : ""}</span></div>
      </div>
    </div>

    <div class="bar-row">
      <div class="bar-row-header"><span class="${BAR_DETAILS["clickbait"].cssClass}">${BAR_DETAILS["clickbait"].label}${clickbait <= 8 ? ": " + clickbait + "%" : ""}</span></div>
      <div class="bar-track">
        <div class="bar-fill" style="width: ${clickbait}%; background: ${BAR_DETAILS["clickbait"].color};"><span>${clickbait > 8 ? clickbait + "%" : ""}</span></div>
      </div>
    </div>
  `;
}

function renderSentenceBatch(batch) {
  batch.forEach(({ sentence, impacts }) => {
    const li = document.createElement("li");
    const impactText = Object.entries(impacts)
      .map(([label, delta]) => {
        const details = BAR_DETAILS[label];
        const displayName = details ? details.label : label;
        const cssClass = details ? details.cssClass : "";
        return `<span class="impact-tag${cssClass ? " " + cssClass : ""}">${displayName}: ${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)}%</span>`;
      })
      .join("");
    li.innerHTML = `
      <div class="sentence-content">
        ${sentence}
      </div>
      <div class="impact-tags">
        ${impactText}
      </div>
    `;
    sentenceList.appendChild(li);
  });
}

function renderWordCounts(wordCounts) {
  wordCountsEl.innerHTML = wordCounts.map(wc => `
    <div>
      ${wc.word}
      <span class="dots"></span>
      <span class="num">${wc.count}</span>
      </div>
  `).join("");
  wordCountsEl.parentElement.classList.remove("hidden");
}

function renderPhotoCredits(photoCredits) {
  photoCreditsEl.textContent = photoCredits.length > 0 ? photoCredits.join(", ") : "None";
  photoCreditsEl.parentElement.classList.remove("hidden");
}

function renderSubmissionContent(content) {
  subTypeEl.textContent = content.type;
  if (content.type === "URL") {
    subContentEl.innerHTML = `<a href="${content.content}" target="_blank" rel="noopener">${content.content}</a>`;
  } else {
    subContentEl.textContent = content.content;
  }
}

async function loadNextSentenceBatch(limit = 10) {
  if (nextOffset === null || !currentSessionId) return;
  const res = await fetch(`${API_BASE}/api/classify/${currentSessionId}/sentences?offset=${nextOffset}&limit=${limit}`);
  const data = await res.json();
  renderSentenceBatch(data.results);
  nextOffset = data.next_offset;
  loadMoreBtn.classList.toggle("hidden", nextOffset === null);
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  clearError();
  results.classList.add("hidden");
  sentenceList.innerHTML = "";
  loading.classList.remove("hidden");

  let txtValue = textInput.value.trim();
  let urlValue = urlInput.value.trim();
  if (!txtValue && !urlValue) {
    showError("Provide either 'text' or 'url'.");
    loading.classList.add("hidden");
    return;
  }

  const body = {};
  if (txtValue) body.text = txtValue;
  if (urlValue) body.url = urlValue;

  try {
    const res = await fetch(`${API_BASE}/api/classify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Classification failed.");
    }
    const data = await res.json();
    console.log(data);

    currentSessionId = data.id;
    nextOffset = 0;
    populateShareLink(data.id);

    // switch views
    formBox.classList.add("hidden");
    main.classList.add("results");
    results.classList.remove("hidden");

    // render results
    articleTitleEl.textContent = data.title || "N/A";
    articleTitleEl.parentElement.classList.remove("hidden");

    dateEl.textContent = data.date || "N/A";
    authorEl.textContent = data.author || "N/A";
    sentenceCount.textContent = data.sentence_count || "N/A";

    // set submission content
    let subContent = {
      type: urlValue ? "URL" : "Text",
      content: urlValue ? urlValue : txtValue,
    }
    renderSubmissionContent(subContent);
    renderWordCounts(data.word_counts);
    renderPhotoCredits(data.photo_credits);
    renderLabelBars(data.labels);

    await loadNextSentenceBatch(5);

    saveHistoryEntry(data);
  } catch (err) {
    showError(err.message);
  } finally {
    fadeReveal();
    loading.classList.add("hidden");
  }
});

// Anim stuff
function wait(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function resetAnim() {
  main.classList.remove("results");
  document.querySelectorAll(".fade-item").forEach(el => el.classList.remove("revealed"));
}

async function fadeReveal(selector = ".fade-item") {
  main.classList.add("results");
  results.classList.remove("hidden");
  await wait(300);
  const items = document.querySelectorAll(selector);
  for (const el of items) {
    el.classList.add("revealed");
    await wait(300);
  }
}

// history stuff
function loadHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch (e) {
    console.error("Failed to read history from localStorage:", e);
    return [];
  }
}

function saveHistoryEntry(data) {
  const history = loadHistory();

  // sub content
  let txtValue = textInput.value.trim();
  let urlValue = urlInput.value.trim();
  let submittedContent = {
    type: urlValue ? "URL" : "Text",
    content: urlValue ? urlValue : txtValue,
  }

  history.unshift({
    id: data.id,
    title: data.title,
    author: data.author,
    date: data.date,
    labels: data.labels,
    sentence_count: data.sentence_count,
    submitted_content: submittedContent,
    word_counts: data.word_counts,
    photo_credits: data.photo_credits,
    savedAt: new Date().toLocaleString(),
  });

  const trimmed = history.slice(0, MAX_HISTORY_ITEMS);

  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(trimmed));
  } catch (e) {
    console.error("Failed to save history (localStorage may be full):", e);
  }

  return trimmed;
}

function renderHistoryList() {
  const history = loadHistory();
  historyBox.classList.toggle("hidden", history.length === 0);

  historyList.innerHTML = history
    .map((item, i) => {
      const labelTags = item.labels
        .map(({ name, confidence }) => {
          const details = BAR_DETAILS[name];
          const displayName = details ? details.label : name;
          const cssClass = details ? details.cssClass : "";
          return `<span class="impact-tag${cssClass ? " " + cssClass : ""}">${displayName}: ${(confidence * 100).toFixed(1)}%</span>`;
        })
        .join("");

      return `<div class="history-item" data-index="${i}">
        <div class="history-content">
          <strong>${item.title || "Untitled"}</strong>
          <span class="history-meta">${item.savedAt}</span>
        </div>
        <div class="impact-tags">
          ${labelTags}
        </div>
      </div>`;
    })
    .join("");

  historyList.querySelectorAll(".history-item").forEach(el => {
    el.addEventListener("click", () => loadFromHistory(Number(el.dataset.index)));
  });
}

async function loadFromHistory(index) {
  const history = loadHistory();
  const item = history[index];
  if (!item) return;

  currentSessionId = item.id;
  nextOffset = 0;
  populateShareLink(item.id);
  sentenceList.innerHTML = "";

  // display results from history
  articleTitleEl.textContent = item.title || "N/A";
  articleTitleEl.parentElement.classList.remove("hidden");

  dateEl.textContent = item.date || "N/A";
  authorEl.textContent = item.author || "N/A";
  sentenceCount.textContent = item.sentence_count || "N/A";

  renderSubmissionContent(item.submitted_content)
  renderLabelBars(item.labels);
  renderWordCounts(item.word_counts);
  renderPhotoCredits(item.photo_credits);

  await loadNextSentenceBatch(5);

  formBox.classList.add("hidden");
  results.classList.remove("hidden");

  fadeReveal();
}

// force history clear on server restart
// check server boot token to see if history matches
async function checkServerBootId() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    const data = await res.json();

    const lastKnownBootId = localStorage.getItem(BOOT_ID_KEY);

    if (lastKnownBootId && lastKnownBootId !== data.boot_id) {
      localStorage.removeItem(HISTORY_KEY);
    }

    localStorage.setItem(BOOT_ID_KEY, data.boot_id);
  } catch (e) {
    console.error("Could not reach backend to check server boot id:", e);
  }
}

// share link stuff
function populateShareLink(sessionId) {
  const shareUrl = `${window.location.origin}${window.location.pathname}?shared=${sessionId}`;
  document.getElementById("share-link-input").value = shareUrl;
}

async function loadSharedReport(sessionId) {
  const res = await fetch(`${API_BASE}/api/classify/${sessionId}`);

  if (!res.ok) {
    formBox.classList.remove("hidden");
    showError("This shared report is no longer available -- the server may have restarted since it was created.");
    return;
  }

  const data = await res.json();
  currentSessionId = data.id;
  nextOffset = 0;
  populateShareLink(data.id);
  sentenceList.innerHTML = "";

  // display results from history
  articleTitleEl.textContent = data.title || "N/A";
  articleTitleEl.parentElement.classList.remove("hidden");

  dateEl.textContent = data.date || "N/A";
  authorEl.textContent = data.author || "N/A";
  sentenceCount.textContent = data.sentence_count || "N/A";

  let subContent = {
    type: data.submitted_type,
    content: data.submitted_content_raw,
  }
  renderSubmissionContent(subContent);
  renderLabelBars(data.labels);
  renderWordCounts(data.word_counts);
  renderPhotoCredits(data.photo_credits);

  await loadNextSentenceBatch(5);

  // reset loading
  loadingLabel.textContent = "Analyzing...";
  loading.classList.add("hidden");
  // show results
  results.classList.remove("hidden");
  sharedBanner.classList.remove("hidden");
  fadeReveal();
}

async function btnCopyLinkFunc() {
  try {
    await navigator.clipboard.writeText(shareLinkInput.value);
    btnCopyLink.textContent = "Copied!";
    btnCopyLink.classList.add("copied");
  } catch (e) {
    console.error("Clipboard write failed:", e);
    shareLinkInput.select();
  }

  setTimeout(() => {
    btnCopyLink.textContent = "Copy link";
    btnCopyLink.classList.remove("copied");
  }, 2000);
}