const API_BASE = "http://127.0.0.1:8000";

// form
const formBox = document.getElementById("form-box");
const form = document.getElementById("classify-form");
const textInput = document.getElementById("text-input");
const urlInput = document.getElementById("url-input");

// results
const results = document.getElementById("results");
const articleTitleEl = document.getElementById("article_title");
const photoCreditsEl = document.getElementById("photo_credits");
const wordCountsEl = document.getElementById("word_counts");
const badgesEl = document.getElementById("badges");
const sentenceList = document.getElementById("sentence-list");

// UI/UX
const loadMoreBtn = document.getElementById("load-more");
const loading = document.getElementById("loading");
const errorEl = document.getElementById("error");

let currentSessionId = null;
let nextOffset = 0;

function showError(msg) {
  errorEl.textContent = msg;
  errorEl.classList.remove("hidden");
}

function clearError() {
  errorEl.classList.add("hidden");
  errorEl.textContent = "";
}

function renderBadges(labels) {
  badgesEl.innerHTML = "";
  labels
    .sort((a, b) => b.confidence - a.confidence)
    .forEach(({ name, confidence }) => {
      const div = document.createElement("div");
      div.className = "badge";
      div.innerHTML = `<strong>${name}</strong> — ${(confidence * 100).toFixed(1)}%`;
      badgesEl.appendChild(div);
    });
}

function renderSentenceBatch(batch) {
  batch.forEach(({ sentence, impacts }) => {
    const li = document.createElement("li");
    const impactText = Object.entries(impacts)
      .map(([label, delta]) => `<span class="impact-tag">${label}: ${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)}%</span>`)
      .join("");
    li.innerHTML = `${sentence}<br>${impactText}`;
    sentenceList.appendChild(li);
  });
}

async function loadNextSentenceBatch() {
  if (nextOffset === null || !currentSessionId) return;
  const res = await fetch(`${API_BASE}/api/classify/${currentSessionId}/sentences?offset=${nextOffset}&limit=10`);
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

  const body = {};
  if (textInput.value.trim()) body.text = textInput.value.trim();
  if (urlInput.value.trim()) body.url = urlInput.value.trim();

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

    // switch views
    formBox.classList.add("hidden");
    results.classList.remove("hidden");

    // display results
    articleTitleEl.textContent = data.title || "N/A";
    articleTitleEl.parentElement.classList.remove("loading");

    photoCreditsEl.textContent = data.photo_credits.length > 0 ? data.photo_credits.join(", ") : "None";
    photoCreditsEl.parentElement.classList.remove("loading");

    wordCountsEl.innerHTML = data.word_counts.map(wc => `<div>${wc.word}: ${wc.count}</div>`).join("");
    wordCountsEl.parentElement.classList.remove("loading");

    renderBadges(data.labels);

    await loadNextSentenceBatch();
  } catch (err) {
    showError(err.message);
  } finally {
    loading.classList.add("hidden");
  }
});

loadMoreBtn.addEventListener("click", loadNextSentenceBatch);
