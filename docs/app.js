// ── Minimal Markdown Parser (no dependencies) ──
function renderMarkdown(md) {
    if (!md) return '<p>No analysis available.</p>';

    let html = '';
    const lines = md.split('\n');
    let inList = false;
    let listType = 'ul';

    for (let i = 0; i < lines.length; i++) {
        let line = lines[i];

        // Headers
        if (line.startsWith('### ')) {
            if (inList) { html += `</${listType}>`; inList = false; }
            html += `<h3>${inline(line.slice(4))}</h3>`;
        } else if (line.startsWith('## ')) {
            if (inList) { html += `</${listType}>`; inList = false; }
            html += `<h2>${inline(line.slice(3))}</h2>`;
        }
        // Unordered list
        else if (/^[-*]\s/.test(line)) {
            if (!inList) { html += '<ul>'; inList = true; listType = 'ul'; }
            html += `<li>${inline(line.slice(2).trim())}</li>`;
        }
        // Ordered list
        else if (/^\d+\.\s/.test(line)) {
            if (!inList) { html += '<ol>'; inList = true; listType = 'ol'; }
            html += `<li>${inline(line.replace(/^\d+\.\s/, '').trim())}</li>`;
        }
        // Indented sub-bullets
        else if (/^\s+[-*]\s/.test(line)) {
            if (!inList) { html += '<ul>'; inList = true; listType = 'ul'; }
            html += `<li>${inline(line.trim().slice(2).trim())}</li>`;
        }
        // Empty line
        else if (line.trim() === '') {
            if (inList) { html += `</${listType}>`; inList = false; }
        }
        // Paragraph
        else {
            if (inList) { html += `</${listType}>`; inList = false; }
            html += `<p>${inline(line)}</p>`;
        }
    }
    if (inList) html += `</${listType}>`;
    return html;
}

// Inline formatting: **bold**, *italic*, `code`
function inline(text) {
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/`(.+?)`/g, '<code>$1</code>');
}


// ── App State ──
let currentData = null;
let currentTopic = null;

// ── Init ──
async function init() {
    const manifest = await fetchJSON('data/index.json');
    if (!manifest || !manifest.weeks.length) {
        document.getElementById('report-content').innerHTML =
            '<div class="loading">No reports yet. Run the researcher first.</div>';
        return;
    }

    // Populate week selector
    const select = document.getElementById('week-select');
    manifest.weeks.forEach(w => {
        const opt = document.createElement('option');
        opt.value = w.date;
        opt.textContent = w.date;
        select.appendChild(opt);
    });

    // Check URL hash for initial state
    const hash = window.location.hash.slice(1);
    const [hashDate, hashTopic] = hash.split('/');

    if (hashDate && manifest.weeks.some(w => w.date === hashDate)) {
        select.value = hashDate;
    }

    select.addEventListener('change', () => loadWeek(select.value));
    window.addEventListener('hashchange', () => {
        const [d, t] = window.location.hash.slice(1).split('/');
        if (d && d !== select.value) {
            select.value = d;
            loadWeek(d, t);
        } else if (t && t !== currentTopic) {
            selectTopic(t);
        }
    });

    await loadWeek(select.value, hashTopic);
}

async function fetchJSON(url) {
    try {
        const res = await fetch(url);
        if (!res.ok) return null;
        return await res.json();
    } catch { return null; }
}

async function loadWeek(date, topicId) {
    const content = document.getElementById('report-content');
    content.innerHTML = '<div class="loading">Loading...</div>';

    const data = await fetchJSON(`data/${date}.json`);
    if (!data) {
        content.innerHTML = '<div class="loading">Failed to load report data.</div>';
        return;
    }

    currentData = data;

    // Build tabs
    const tabsEl = document.getElementById('topic-tabs');
    tabsEl.innerHTML = '';
    const topicIds = Object.keys(data.reports);
    topicIds.forEach(id => {
        const tab = document.createElement('button');
        tab.className = 'tab';
        tab.dataset.topic = id;
        tab.textContent = data.reports[id].title;
        tab.addEventListener('click', () => selectTopic(id));
        tabsEl.appendChild(tab);
    });

    // Select topic
    const initialTopic = topicId && data.reports[topicId] ? topicId : topicIds[0];
    selectTopic(initialTopic);
}

function selectTopic(topicId) {
    if (!currentData || !currentData.reports[topicId]) return;
    currentTopic = topicId;

    // Update hash without triggering reload
    const date = currentData.date;
    history.replaceState(null, '', `#${date}/${topicId}`);

    // Update active tab
    document.querySelectorAll('.tab').forEach(t => {
        t.classList.toggle('active', t.dataset.topic === topicId);
    });

    // Render report
    const report = currentData.reports[topicId];
    const content = document.getElementById('report-content');

    content.innerHTML = `
        <div class="report">
            <div class="report-title">${report.title}</div>

            <div class="questions">
                <h3>Key Research Questions</h3>
                <ol>
                    ${report.questions.map(q => `<li>${q}</li>`).join('')}
                </ol>
            </div>

            <div class="analysis">
                ${renderMarkdown(report.analysis)}
            </div>

            <div class="stats">
                <h3>Source Stats</h3>
                <div class="stats-grid">
                    ${Object.entries(currentData.scrape_stats.sources)
                        .map(([name, s]) => `
                            <div class="stat-item">
                                <span class="stat-name">${name}</span>
                                <span class="stat-${s.status}">${(s.chars / 1000).toFixed(1)}K</span>
                            </div>
                        `).join('')}
                </div>
            </div>
        </div>
    `;
}

// Go
init();
